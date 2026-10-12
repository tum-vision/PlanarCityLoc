"""Mesh rasterization: :class:`Renderer`, :class:`RendererGPU`, :class:`RendererCPU`.

Convention
----------
All rendering APIs take **w2c_batch** — (B, 4, 4) world-to-camera extrinsics (OpenCV).
Use ``B=1`` for a single pose. Pick :class:`RendererGPU` or :class:`RendererCPU` at
construction time; backends never fall back to each other internally.

Naming: ``render_<output>_batch(w2c_batch, ...)``
"""
from __future__ import annotations

import colorsys
import os
import sysconfig
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh as _trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sparse_cc
from scipy.spatial import ConvexHull

from utils.io import load_mesh_vertices_faces
from utils.pose import euler_to_camera_pose

_cuda_home = os.environ.get("CUDA_HOME", "")
if not os.path.isfile(os.path.join(_cuda_home, "include", "cuda_runtime.h")):
    _conda_cuda = os.path.join(sysconfig.get_path("data"), "targets", "x86_64-linux")
    if os.path.isfile(os.path.join(_conda_cuda, "include", "cuda_runtime.h")):
        os.environ["CUDA_HOME"] = _conda_cuda

import nvdiffrast.torch as dr  # noqa: E402

class Renderer:
    """Shared API — use :meth:`from_arrays` / :meth:`from_mesh_file` to pick backend."""

    SEMANTIC_BG = 0
    SEMANTIC_ROOF = 1
    SEMANTIC_WALL = 2
    SEMANTIC_OTHER = 3

    _SEMANTIC_LABEL_COLORS = {
        0: np.array([0, 0, 0], dtype=np.uint8),
        1: np.array([220, 20, 60], dtype=np.uint8),
        2: np.array([70, 130, 180], dtype=np.uint8),
        3: np.array([128, 128, 128], dtype=np.uint8),
    }

    _MESH_LABEL_TO_SEMANTIC = {
        "wall": 2, "roof": 1, "ground": 3, "closure": 3, "other": 3, "building": 3,
    }

    def __new__(cls, *args, **kwargs):
        if cls is Renderer and args and isinstance(args[0], (str, bytes)):
            mesh_path, K, image_size = args[0], args[1], args[2]
            kw = dict(kwargs)
            return Renderer.from_mesh_file(mesh_path, image_size, K=K, **kw)
        return super().__new__(cls)

    @classmethod
    def from_arrays(cls, vertices, faces, K, image_size, *, labels=None, building_ids=None,
                    near=0.5, far=10000.0, device="cuda:0", depth_flag=True,
                    minibatch_size=52, auto_cull=False):
        if "cuda" in str(device).lower():
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA device {device!r} requested but CUDA is unavailable")
            return RendererGPU(vertices, faces, K, image_size, labels=labels, building_ids=building_ids,
                near=near, far=far, device=device, depth_flag=depth_flag,
                minibatch_size=minibatch_size, auto_cull=auto_cull)
        return RendererCPU(vertices, faces, K, image_size, labels=labels, building_ids=building_ids,
            near=near, far=far, depth_flag=depth_flag, minibatch_size=minibatch_size, auto_cull=auto_cull)

    @classmethod
    def from_mesh_file(cls, mesh_path, image_size, K=None, *, near=1.0, far=2000.0,
                       device="cuda:0", depth_flag=False, minibatch_size=52, auto_cull=False):
        import trimesh as _tm
        mesh = _tm.load(str(mesh_path), force="mesh", process=False)
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if K is None:
            H, W = image_size
            K = np.array([[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]], dtype=np.float64)
        return cls.from_arrays(verts, faces, K, image_size, near=near, far=far, device=device,
            depth_flag=depth_flag, minibatch_size=minibatch_size, auto_cull=auto_cull)

    def _init_common(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        K: np.ndarray | None,
        image_size: tuple[int, int],
        *,
        near: float,
        far: float,
        labels: np.ndarray | None = None,
        building_ids: np.ndarray | None = None,
        depth_flag: bool = True,
        minibatch_size: int = 52,
        auto_cull: bool = False,
    ) -> None:
        H, W = image_size
        self.H, self.W = H, W
        self.near, self.far = near, far
        self.depth_flag = depth_flag
        self.minibatch_size = minibatch_size
        self.auto_cull = auto_cull
        self._proj_cache_key = None
        self._proj_dynamic = None

        vertices = np.asarray(vertices, dtype=np.float32)
        faces_np = np.asarray(faces, dtype=np.int32)
        self._verts_np = vertices
        self._faces_np = faces_np
        self._labels_np = np.asarray(labels) if labels is not None else None
        self._building_ids_np = (
            np.asarray(building_ids, dtype=np.int32) if building_ids is not None else None
        )
        v0, v1, v2 = vertices[faces_np[:, 0]], vertices[faces_np[:, 1]], vertices[faces_np[:, 2]]
        raw_n = np.cross(v1 - v0, v2 - v0)
        self._face_normals_np = (
            raw_n / np.linalg.norm(raw_n, axis=1, keepdims=True).clip(min=1e-12)
        ).astype(np.float32)

        if K is not None:
            K = np.asarray(K, dtype=np.float64)
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            n, f = near, far
            proj = np.zeros((4, 4), dtype=np.float64)
            proj[0, 0] = 2 * fx / W
            proj[1, 1] = 2 * fy / H
            proj[0, 2] = 1 - 2 * cx / W
            proj[1, 2] = 2 * cy / H - 1
            proj[2, 2] = -(f + n) / (f - n)
            proj[2, 3] = -2 * f * n / (f - n)
            proj[3, 2] = -1
            self._K = K.copy()
            self._proj_np = proj
        else:
            self._K = None
            self._proj_np = None

    def _ensure_K(self, K: np.ndarray | None = None) -> np.ndarray:
        K_use = K if K is not None else self._K
        if K_use is None:
            raise ValueError("intrinsics K required (pass at init or per call)")
        return np.asarray(K_use, dtype=np.float64)

    @property
    def zfar(self) -> float:
        return self.far

    @property
    def znear(self) -> float:
        return self.near

    @staticmethod
    def _as_w2c_batch(
        w2c_batch: "torch.Tensor | np.ndarray",
        device: "torch.device | str | None" = None,
    ) -> torch.Tensor:
        """Normalize input to a (B, 4, 4) float32 tensor."""
        if isinstance(w2c_batch, np.ndarray):
            arr = np.asarray(w2c_batch, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis]
            t = torch.from_numpy(arr)
        else:
            t = w2c_batch.float()
            if t.ndim == 2:
                t = t.unsqueeze(0)
        if t.shape[-2:] == (3, 4):
            bottom = torch.zeros(t.shape[0], 1, 4, dtype=t.dtype, device=t.device)
            bottom[:, :, 3] = 1.0
            t = torch.cat([t, bottom], dim=-2)
        if device is not None:
            t = t.to(device)
        return t

    def _iter_w2c_chunks(self, w2c_batch: torch.Tensor):
        """Yield w2c slices of at most ``minibatch_size`` poses."""
        B = w2c_batch.shape[0]
        for s in range(0, B, self.minibatch_size):
            yield w2c_batch[s:min(s + self.minibatch_size, B)]


class RendererGPU(Renderer):
    """nvdiffrast GPU mesh rasterizer."""

    """GPU-accelerated differentiable mesh rasterizer using nvdiffrast."""

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        K: np.ndarray | None,
        image_size: tuple[int, int],
        *,
        labels: np.ndarray | None = None,
        building_ids: np.ndarray | None = None,
        near: float = 0.5,
        far: float = 10000.0,
        device: str = "cuda:0",
        depth_flag: bool = True,
        minibatch_size: int = 52,
        auto_cull: bool = False,
    ) -> None:
        self._init_common(vertices, faces, K, image_size, labels=labels, building_ids=building_ids,
            near=near, far=far, depth_flag=depth_flag, minibatch_size=minibatch_size, auto_cull=auto_cull)
        self._dev = torch.device(device)
        self._init_gpu_backend()

    @property
    def dev(self) -> torch.device:
        return self._dev

    def _init_gpu_backend(self) -> None:
        dev = self._dev
        vertices = self._verts_np
        faces_np = self._faces_np
        v = torch.tensor(vertices, dtype=torch.float32, device=dev)
        self._verts = v
        self.verts_h = torch.cat([v, torch.ones(v.shape[0], 1, device=dev)], dim=1)
        self.faces = torch.tensor(faces_np, dtype=torch.int32, device=dev).contiguous()
        self._pos_attr = v.unsqueeze(0).contiguous()
        self._face_centroids_xy = v[self.faces.long()].mean(dim=1)[:, :2].contiguous()

        v0t, v1t, v2t = v[self.faces[:, 0]], v[self.faces[:, 1]], v[self.faces[:, 2]]
        raw_t = torch.cross(v1t - v0t, v2t - v0t, dim=1)
        self.face_normals = raw_t / raw_t.norm(dim=1, keepdim=True).clamp(min=1e-12)

        n_faces = faces_np.shape[0]
        exp_v = v[self.faces.long().reshape(-1)]
        self.exp_verts_h = torch.cat([exp_v, torch.ones(n_faces * 3, 1, device=dev)], dim=1)
        self.exp_faces = torch.arange(n_faces * 3, device=dev, dtype=torch.int32).reshape(n_faces, 3).contiguous()
        self.exp_face_normals = (
            self.face_normals.unsqueeze(1).expand(-1, 3, -1).reshape(n_faces * 3, 3).contiguous()
        )

        if self._proj_np is not None:
            self.proj = torch.tensor(self._proj_np, dtype=torch.float32, device=dev)
        else:
            self.proj = None

        try:
            self.glctx = dr.RasterizeCudaContext(device=dev)
        except Exception:
            self.glctx = dr.RasterizeGLContext(output_db=False)

    def _to_w2c(self, w2c: "torch.Tensor | np.ndarray") -> torch.Tensor:
        """Convert w2c to float32 4×4 tensor on the renderer device."""
        if isinstance(w2c, np.ndarray):
            t = torch.from_numpy(np.asarray(w2c, dtype=np.float32))
        else:
            t = w2c.float()
        if t.shape == (3, 4):
            bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=t.dtype)
            t = torch.cat([t, bottom], dim=0)
        return t.to(self._dev)

    def _build_view(self, w2c: torch.Tensor) -> torch.Tensor:
        """Apply Y/Z row flip to match OpenGL convention."""
        view = w2c.clone()
        view[1, :] = -view[1, :]
        view[2, :] = -view[2, :]
        return view

    def _ensure_K(self, K: np.ndarray | None = None) -> np.ndarray:
        K_use = K if K is not None else self._K
        if K_use is None:
            raise ValueError("intrinsics K required (pass at init or per call)")
        return np.asarray(K_use, dtype=np.float64)

    def _build_projection_matrix(self, K: torch.Tensor) -> torch.Tensor:
        """Build OpenGL projection from intrinsics; cache by fx,fy,cx,cy."""
        K = K.to(self._dev)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        key = (fx, fy, cx, cy)
        if self._proj_cache_key == key and self._proj_dynamic is not None:
            return self._proj_dynamic
        H, W = self.H, self.W
        n, f = self.near, self.far
        P = torch.zeros(4, 4, dtype=torch.float32, device=self._dev)
        P[0, 0] = 2.0 * fx / W
        P[1, 1] = 2.0 * fy / H
        P[0, 2] = 1.0 - 2.0 * cx / W
        P[1, 2] = 2.0 * cy / H - 1.0
        P[2, 2] = -(f + n) / (f - n)
        P[2, 3] = -2.0 * f * n / (f - n)
        P[3, 2] = -1.0
        self._proj_cache_key = key
        self._proj_dynamic = P
        return P

    def _proj_for_K(self, K: np.ndarray | torch.Tensor | None) -> torch.Tensor:
        """Return projection tensor for *K* (updates cached ``self.proj`` when K changes)."""
        if K is None:
            if self.proj is None:
                raise ValueError("intrinsics K required")
            return self.proj
        K_np = np.asarray(K, dtype=np.float64)
        if self._K is None or not np.array_equal(self._K, K_np):
            self._K = K_np.copy()
            fx, fy = float(K_np[0, 0]), float(K_np[1, 1])
            cx, cy = float(K_np[0, 2]), float(K_np[1, 2])
            n, f = self.near, self.far
            proj = np.zeros((4, 4), dtype=np.float64)
            proj[0, 0] = 2 * fx / self.W;  proj[1, 1] = 2 * fy / self.H
            proj[0, 2] = 1 - 2 * cx / self.W;  proj[1, 2] = 2 * cy / self.H - 1
            proj[2, 2] = -(f + n) / (f - n);  proj[2, 3] = -2 * f * n / (f - n)
            proj[3, 2] = -1
            self._proj_np = proj
            self.proj = torch.tensor(proj, dtype=torch.float32, device=self._dev)
        return self.proj

    def _w2c_batch_to_gl_view(self, w2c_b: torch.Tensor):
        """w2c (B,4,4) → OpenGL view matrix + OpenCV R_w2c, t_w2c."""
        R_w2c = w2c_b[:, :3, :3]
        t_w2c = w2c_b[:, :3, 3]
        B = w2c_b.shape[0]
        view = torch.zeros(B, 4, 4, dtype=torch.float32, device=self._dev)
        view[:, :3, :3] = R_w2c
        view[:, :3, 3] = t_w2c
        view[:, 3, 3] = 1.0
        view[:, 1, :] = -view[:, 1, :]
        view[:, 2, :] = -view[:, 2, :]
        return view, R_w2c, t_w2c

    def _cam_positions_from_w2c(self, w2c_b: torch.Tensor) -> torch.Tensor:
        """Camera centres in world frame, shape (B, 3)."""
        R_w2c = w2c_b[:, :3, :3]
        t_w2c = w2c_b[:, :3, 3]
        return -torch.bmm(R_w2c.transpose(1, 2), t_w2c.unsqueeze(-1)).squeeze(-1)

    def _render_path_batch(
        self,
        verts_h: torch.Tensor,
        faces: torch.Tensor,
        pos_attr: torch.Tensor,
        mvp: torch.Tensor,
        R_w2c: torch.Tensor,
        t_w2c: torch.Tensor,
        *,
        depth_flag: bool | None = None,
    ):
        """Inner batch rasterize → mask + optional camera-space Z depth."""
        depth_flag = self.depth_flag if depth_flag is None else depth_flag
        B = mvp.shape[0]
        masks_list = []
        depths_list = [] if depth_flag else None

        for s in range(0, B, self.minibatch_size):
            e = min(s + self.minibatch_size, B)
            bsz = e - s
            mvp_b = mvp[s:e]
            clip = torch.matmul(verts_h, mvp_b.transpose(1, 2)).contiguous()
            rast, _ = dr.rasterize(self.glctx, clip, faces, resolution=[self.H, self.W])
            mask = rast[..., 3] > 0
            masks_list.append(mask.flip(1))
            if depth_flag:
                pos_world, _ = dr.interpolate(pos_attr, rast, faces)
                p = pos_world.permute(0, 3, 1, 2)
                R_b = R_w2c[s:e]
                t_b = t_w2c[s:e]
                cam = torch.einsum("bij,bjhw->bihw", R_b, p) + t_b.view(bsz, 3, 1, 1)
                depths_list.append((cam[:, 2] * mask.float()).flip(1))

        masks_t = torch.cat(masks_list, dim=0)
        depths_t = torch.cat(depths_list, dim=0) if depths_list is not None else None
        return masks_t, depths_t

    def render_masks_batch(
        self,
        w2c_batch: torch.Tensor,
        K: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Batch-render binary silhouettes (+ optional depth) for w2c poses.

        Args:
            w2c_batch: (B, 4, 4) world-to-camera matrices.
            K: (3, 3) intrinsics.

        Returns:
            mask: (B, H, W) bool.
            depth: (B, H, W) float32 camera-space Z in metres, or None.
        """
        if self.auto_cull:
            return self.render_masks_batch_cropped(w2c_batch, K)

        B = w2c_batch.shape[0]
        if B == 0:
            empty = torch.empty((0, self.H, self.W), dtype=torch.bool, device=self._dev)
            dep = torch.empty((0, self.H, self.W), dtype=torch.float32, device=self._dev) if self.depth_flag else None
            return empty, dep

        w2c_b = w2c_batch.to(self._dev).float()
        K_b = K.to(self._dev).float()
        view, R_w2c, t_w2c = self._w2c_batch_to_gl_view(w2c_b)
        P = self._build_projection_matrix(K_b)
        mvp = P.unsqueeze(0) @ view
        return self._render_path_batch(
            self.verts_h, self.faces, self._pos_attr, mvp, R_w2c, t_w2c,
        )

    def render_masks_batch_cropped(
        self,
        w2c_batch: torch.Tensor,
        K: torch.Tensor,
        crop_radius: float | None = None,
        crop_center: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Batch-render silhouettes with XY footprint culling for large city meshes."""
        B = w2c_batch.shape[0]
        if B == 0:
            return self.render_masks_batch(w2c_batch, K)

        w2c_b = w2c_batch.to(self._dev).float()
        K_b = K.to(self._dev).float()
        cam_pos = self._cam_positions_from_w2c(w2c_b)
        if crop_center is None:
            crop_center = cam_pos.mean(0)[:2]
        else:
            crop_center = crop_center.to(self._dev).float()

        if crop_radius is None:
            fx_K = float(K_b[0, 0])
            ground_z = float(self._verts[:, 2].min())
            mean_cam_z = float(cam_pos[:, 2].mean())
            altitude = abs(mean_cam_z - ground_z)
            diag = float((self.H ** 2 + self.W ** 2) ** 0.5)
            half_fov = float(np.arctan(diag / (2.0 * fx_K)))
            crop_radius = float(altitude * np.tan(half_fov) * 2.0 + 100.0)

        diff = self._face_centroids_xy - crop_center.unsqueeze(0)
        keep = (diff * diff).sum(dim=1) <= (crop_radius * crop_radius)
        n_kept = int(keep.sum())
        if n_kept == 0:
            empty_m = torch.zeros(B, self.H, self.W, dtype=torch.bool, device=self._dev)
            empty_d = torch.zeros(B, self.H, self.W, dtype=torch.float32, device=self._dev) if self.depth_flag else None
            return empty_m, empty_d

        faces_kept = self.faces[keep]
        flat_idx = faces_kept.reshape(-1).long()
        verts_sub = self._verts[flat_idx]
        ones = torch.ones(3 * n_kept, 1, dtype=torch.float32, device=self._dev)
        verts_sub_h = torch.cat([verts_sub, ones], dim=1).contiguous()
        faces_sub = torch.arange(3 * n_kept, device=self._dev, dtype=torch.int32).reshape(-1, 3)
        pos_attr_sub = verts_sub.unsqueeze(0).contiguous()

        view, R_w2c, t_w2c = self._w2c_batch_to_gl_view(w2c_b)
        P = self._build_projection_matrix(K_b)
        mvp = P.unsqueeze(0) @ view
        return self._render_path_batch(
            verts_sub_h, faces_sub, pos_attr_sub, mvp, R_w2c, t_w2c,
        )

    def render_mask_batch(
        self,
        w2c_batch: torch.Tensor,
        *,
        antialias: bool = True,
    ) -> torch.Tensor:
        """Render binary silhouettes for a batch of w2c poses.

        Args:
            w2c_batch: (B, 4, 4) world-to-camera tensor (requires_grad OK).
            antialias: Whether to apply silhouette antialiasing for gradient flow.

        Returns:
            mask: (B, 1, H, W) float tensor in [0, 1], differentiable when antialias=True.
        """
        w2c_batch = self._as_w2c_batch(w2c_batch, self._dev)
        masks = []
        for chunk in self._iter_w2c_chunks(w2c_batch):
            masks.append(self._render_mask_chunk(chunk, antialias=antialias))
        return torch.cat(masks, dim=0)

    def _render_mask_chunk(
        self,
        w2c_chunk: torch.Tensor,
        *,
        antialias: bool = True,
    ) -> torch.Tensor:
        """Render antialiased silhouettes for up to ``minibatch_size`` poses."""
        proj = self._proj_for_K(None)
        chunk_masks = []
        for i in range(w2c_chunk.shape[0]):
            w2c = w2c_chunk[i]
            view = self._build_view(w2c)
            mvp = proj @ view
            verts_clip = torch.matmul(self.verts_h, mvp.T).unsqueeze(0).contiguous()
            rast_out, _ = dr.rasterize(self.glctx, verts_clip, self.faces, resolution=[self.H, self.W])
            mask = (rast_out[..., 3:4] > 0).float()
            if antialias:
                mask = dr.antialias(mask, rast_out, verts_clip, self.faces)
            chunk_masks.append(mask.flip(1).permute(0, 3, 1, 2))
        return torch.cat(chunk_masks, dim=0)

    def render_edges_from_xyz_ypr(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        yaw: torch.Tensor,
        pitch: torch.Tensor,
        roll: torch.Tensor,
        *,
        antialias: bool = True,
        sobel_scale: float = 8.0,
    ) -> torch.Tensor:
        """Render differentiable edge map from (x,y,z,yaw,pitch,roll) pose params.

        Returns:
            edges: (1, 1, H, W) float tensor in [0, 1].
        """
        w2c = self._pose_params_to_w2c(x, y, z, yaw, pitch, roll)
        return self.render_edges_batch(w2c.unsqueeze(0), antialias=antialias, sobel_scale=sobel_scale)[0:1]

    def _pose_params_to_w2c(self, x, y, z, yaw, pitch, roll):
        """Convert individual pose params to a 4x4 w2c matrix."""
        # Build R_c2w from ZXY Euler: R = Rz(yaw) @ Rx(pitch) @ Ry(roll)
        cy, sy = torch.cos(yaw).reshape(1), torch.sin(yaw).reshape(1)
        cp, sp = torch.cos(pitch).reshape(1), torch.sin(pitch).reshape(1)
        cr, sr = torch.cos(roll).reshape(1), torch.sin(roll).reshape(1)

        zero = torch.zeros(1, device=x.device, dtype=x.dtype)
        one = torch.ones(1, device=x.device, dtype=x.dtype)

        # Build each rotation as a flat 9-element vector, then reshape
        Rz = torch.cat([cy, -sy, zero, sy, cy, zero, zero, zero, one]).reshape(3, 3)
        Rx = torch.cat([one, zero, zero, zero, cp, -sp, zero, sp, cp]).reshape(3, 3)
        Ry = torch.cat([cr, zero, sr, zero, one, zero, -sr, zero, cr]).reshape(3, 3)

        R_c2w = Rz @ Rx @ Ry  # (3, 3)
        R_w2c = R_c2w.T

        pos = torch.stack([x, y, z])
        t_w2c = -(R_w2c @ pos)

        # Build w2c without in-place ops that break autograd
        top = torch.cat([R_w2c, t_w2c.unsqueeze(1)], dim=1)  # (3, 4)
        bottom = torch.tensor([[0, 0, 0, 1]], device=x.device, dtype=x.dtype)
        w2c = torch.cat([top, bottom], dim=0)  # (4, 4)

        return w2c

    def render_face_ids_and_depth_batch(
        self,
        w2c_batch: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render face ID + metric depth maps for a batch of w2c poses.

        Returns:
            face_ids: (B, H, W) int32 numpy, -1 = background.
            depth:    (B, H, W) float32 numpy in metres, 0 = background.
        """
        w2c_batch = self._as_w2c_batch(w2c_batch, self._dev)
        face_ids_list, depth_list = [], []
        for chunk in self._iter_w2c_chunks(w2c_batch):
            fids_c, depth_c = self._render_face_ids_and_depth_chunk(chunk)
            face_ids_list.append(fids_c)
            depth_list.append(depth_c)
        return np.concatenate(face_ids_list, axis=0), np.concatenate(depth_list, axis=0)

    def _render_face_ids_and_depth_chunk(
        self,
        w2c_chunk: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        face_ids_list, depth_list = [], []
        for i in range(w2c_chunk.shape[0]):
            w2c = w2c_chunk[i]
            view = self._build_view(w2c)
            mvp = self.proj @ view
            verts_clip = torch.matmul(self.verts_h, mvp.T).unsqueeze(0).contiguous()
            rast_out, _ = dr.rasterize(self.glctx, verts_clip, self.faces,
                                        resolution=[self.H, self.W])
            face_ids = rast_out[0, :, :, 3].int() - 1
            face_ids = face_ids.flip(0).cpu().numpy().astype(np.int32)
            z_ndc = rast_out[..., 2:3]
            visible = rast_out[..., 3:4] > 0
            n, f = self.near, self.far
            denom = (f + n) - (f - n) * z_ndc
            depth_t = torch.where(visible, 2.0 * f * n / denom.clamp(min=1e-6),
                                  torch.zeros_like(z_ndc))
            depth = depth_t[0, :, :, 0].flip(0).cpu().numpy().astype(np.float32)
            face_ids_list.append(face_ids)
            depth_list.append(depth)
        return np.stack(face_ids_list, axis=0), np.stack(depth_list, axis=0)

    def render_structural_edges_and_depth_batch(
        self,
        w2c_batch: torch.Tensor,
        *,
        normal_cos_thresh: float = 0.999,
        depth_disc_thresh: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render structural edge + depth maps for a batch of w2c poses.

        Returns:
            edge_map: (B, H, W) float32 numpy in {0, 1}.
            depth:    (B, H, W) float32 numpy in metres, 0 = background.
        """
        w2c_batch = self._as_w2c_batch(w2c_batch, self._dev)
        edges_list, depth_list = [], []
        for chunk in self._iter_w2c_chunks(w2c_batch):
            edges_c, depth_c = self._render_structural_edges_and_depth_chunk(
                chunk, normal_cos_thresh=normal_cos_thresh, depth_disc_thresh=depth_disc_thresh)
            edges_list.append(edges_c)
            depth_list.append(depth_c)
        return np.concatenate(edges_list, axis=0), np.concatenate(depth_list, axis=0)

    def _render_structural_edges_and_depth_chunk(
        self,
        w2c_chunk: torch.Tensor,
        *,
        normal_cos_thresh: float = 0.999,
        depth_disc_thresh: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        edges_list, depth_list = [], []
        for i in range(w2c_chunk.shape[0]):
            w2c = w2c_chunk[i]
            view = self._build_view(w2c)
            mvp = self.proj @ view
            verts_clip = torch.matmul(self.verts_h, mvp.T).unsqueeze(0).contiguous()
            rast_out, _ = dr.rasterize(self.glctx, verts_clip, self.faces,
                                        resolution=[self.H, self.W])
            face_ids = (rast_out[0, :, :, 3].int() - 1).flip(0)
            z_ndc = rast_out[..., 2:3]
            visible = rast_out[..., 3:4] > 0
            n, f = self.near, self.far
            denom = (f + n) - (f - n) * z_ndc
            depth_t = torch.where(visible, 2.0 * f * n / denom.clamp(min=1e-6),
                                  torch.zeros_like(z_ndc))
            depth_img = depth_t[0, :, :, 0].flip(0)
            edges = self._compute_structural_edges(face_ids, normal_cos_thresh)
            inv_d = 1.0 / depth_img.clamp(min=1.0)
            edge_h = (depth_img[:, :-1] - depth_img[:, 1:]).abs() * inv_d[:, :-1] > depth_disc_thresh
            edge_v = (depth_img[:-1, :] - depth_img[1:, :]).abs() * inv_d[:-1, :] > depth_disc_thresh
            edges[:, :-1] = torch.where(edge_h, torch.ones_like(edges[:, :-1]), edges[:, :-1])
            edges[:-1, :] = torch.where(edge_v, torch.ones_like(edges[:-1, :]), edges[:-1, :])
            edges_list.append(edges.cpu().numpy().astype(np.float32))
            depth_list.append(depth_img.cpu().numpy().astype(np.float32))
        return np.stack(edges_list, axis=0), np.stack(depth_list, axis=0)

    def render_edges_batch(
        self,
        w2c_batch: torch.Tensor,
        *,
        antialias: bool = True,
        sobel_scale: float = 8.0,
    ) -> torch.Tensor:
        """Differentiable edge maps via Sobel on batched normal maps.

        Returns:
            edges: (B, 1, H, W) float tensor in [0, 1].
        """
        del antialias  # normals path already antialiased in render_normals_batch
        normals = self.render_normals_batch(w2c_batch)
        Kx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            device=normals.device, dtype=torch.float32,
        ).view(1, 1, 3, 3)
        Ky = Kx.transpose(-1, -2).contiguous()
        gx = F.conv2d(normals, Kx.expand(3, -1, -1, -1), groups=3, padding=1)
        gy = F.conv2d(normals, Ky.expand(3, -1, -1, -1), groups=3, padding=1)
        edge_mag = (gx.pow(2) + gy.pow(2)).sum(dim=1, keepdim=True).clamp(min=1e-8).sqrt()
        return torch.sigmoid(sobel_scale * edge_mag - 4.0)

    def render_depth_batch(
        self,
        w2c_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Render metric depth maps for a batch of w2c poses.

        Returns:
            depth: (B, 1, H, W) float tensor in metres (0 = background).
        """
        w2c_batch = self._as_w2c_batch(w2c_batch, self._dev)
        if w2c_batch.shape[0] == 0:
            return torch.empty((0, 1, self.H, self.W), dtype=torch.float32, device=self._dev)
        depth_chunks = [self._render_depth_chunk(chunk) for chunk in self._iter_w2c_chunks(w2c_batch)]
        return torch.cat(depth_chunks, dim=0).unsqueeze(1)

    def _render_depth_chunk(self, w2c_chunk: torch.Tensor) -> torch.Tensor:
        """Metric depth for up to ``minibatch_size`` poses (compact mesh)."""
        N = w2c_chunk.shape[0]
        H, W = self.H, self.W
        n_f, f_f = self.near, self.far
        proj = self._proj_for_K(None)
        views = w2c_chunk.clone()
        views[:, 1, :] = -views[:, 1, :]
        views[:, 2, :] = -views[:, 2, :]
        proj_b = proj.unsqueeze(0).expand(N, -1, -1)
        mvps = torch.bmm(proj_b, views)
        verts_clip = torch.bmm(
            self.verts_h.unsqueeze(0).expand(N, -1, -1),
            mvps.permute(0, 2, 1),
        ).contiguous()
        rast_out, _ = dr.rasterize(self.glctx, verts_clip, self.faces, resolution=[H, W])
        z_ndc = rast_out[..., 2:3]
        visible = rast_out[..., 3:4] > 0
        denom = (f_f + n_f) - (f_f - n_f) * z_ndc
        metric_depth = 2.0 * f_f * n_f / denom.clamp_min(1e-6)
        metric_depth = torch.where(visible, metric_depth, torch.zeros_like(metric_depth))
        return metric_depth.flip(1).squeeze(-1)

    def _render_normals_chunk(self, w2c_chunk: torch.Tensor) -> torch.Tensor:
        """Flat-shaded camera-space normals for up to ``minibatch_size`` poses."""
        N = w2c_chunk.shape[0]
        H, W = self.H, self.W
        proj = self._proj_for_K(None)
        views = w2c_chunk.clone()
        views[:, 1, :] = -views[:, 1, :]
        views[:, 2, :] = -views[:, 2, :]
        proj_b = proj.unsqueeze(0).expand(N, -1, -1)
        mvps = torch.bmm(proj_b, views)
        exp_verts_clip = torch.bmm(
            self.exp_verts_h.unsqueeze(0).expand(N, -1, -1),
            mvps.permute(0, 2, 1),
        ).contiguous()
        rast_exp, _ = dr.rasterize(self.glctx, exp_verts_clip, self.exp_faces, resolution=[H, W])
        fn_attr = self.exp_face_normals.unsqueeze(0).expand(N, -1, -1).contiguous()
        normals_world_raw, _ = dr.interpolate(fn_attr, rast_exp.contiguous(), self.exp_faces)
        vis_mask_n = (rast_exp[..., 3:4] > 0).float()
        normals_world_raw = normals_world_raw * vis_mask_n
        normals_world = normals_world_raw.flip(1)
        R_w2c = w2c_chunk[:, :3, :3]
        nw = normals_world.reshape(N, H * W, 3)
        nc = torch.bmm(nw, R_w2c.transpose(1, 2))
        normals = nc.reshape(N, H, W, 3)
        nrm = normals.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        has_val = (normals != 0).any(dim=-1, keepdim=True)
        normals = torch.where(has_val, normals / nrm, normals)
        return normals

    def render_depth_normals_batch(
        self,
        w2c_batch: torch.Tensor,
    ) -> tuple:
        """Batch-render depth + camera-space normals, chunked by ``minibatch_size``.

        Args:
            w2c_batch: (N, 4, 4) float32 CUDA tensor of world-to-camera matrices.

        Returns:
            depth:   (N, H, W) float32 CUDA tensor, metric depth (0 = background).
            normals: (N, H, W, 3) float32 CUDA tensor, camera-space unit normals
                     (zero for background pixels).
        """
        w2c_batch = self._as_w2c_batch(w2c_batch, self._dev)
        if w2c_batch.shape[0] == 0:
            empty_d = torch.empty((0, self.H, self.W), dtype=torch.float32, device=self._dev)
            empty_n = torch.empty((0, self.H, self.W, 3), dtype=torch.float32, device=self._dev)
            return empty_d, empty_n
        depth_chunks, normal_chunks = [], []
        for chunk in self._iter_w2c_chunks(w2c_batch):
            depth_chunks.append(self._render_depth_chunk(chunk))
            normal_chunks.append(self._render_normals_chunk(chunk))
        return torch.cat(depth_chunks, dim=0), torch.cat(normal_chunks, dim=0)

    def render_normals_batch(
        self,
        w2c_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Render per-pixel camera-space normals for a batch of w2c poses.

        Returns:
            normals: (B, 3, H, W) float tensor. Background pixels are zero.
        """
        w2c_batch = self._as_w2c_batch(w2c_batch, self._dev)
        if w2c_batch.shape[0] == 0:
            return torch.empty((0, 3, self.H, self.W), dtype=torch.float32, device=self._dev)
        normal_chunks = [self._render_normals_chunk(chunk) for chunk in self._iter_w2c_chunks(w2c_batch)]
        normals = torch.cat(normal_chunks, dim=0)
        return normals.permute(0, 3, 1, 2).contiguous()

    def render_planes_batch(
        self,
        w2c_batch: torch.Tensor,
        *,
        normal_cos_thresh: float = 0.995,
        depth_rel_thresh: float = 0.02,
        min_pixels: int = 50,
    ) -> list[dict]:
        """Extract planar regions for each pose in ``w2c_batch``."""
        w2c_batch = self._as_w2c_batch(w2c_batch)
        results = []
        for chunk in self._iter_w2c_chunks(w2c_batch):
            for i in range(chunk.shape[0]):
                results.append(self._render_planes_one(
                    chunk[i],
                    normal_cos_thresh=normal_cos_thresh,
                    depth_rel_thresh=depth_rel_thresh,
                    min_pixels=min_pixels,
                ))
        return results

    def _render_planes_one(
        self,
        w2c: torch.Tensor,
        *,
        normal_cos_thresh: float = 0.995,
        depth_rel_thresh: float = 0.02,
        min_pixels: int = 50,
    ) -> dict:
        """Extract planar regions from the rendered normal and depth maps.

        Differentiable: plane normal vectors and depth offsets carry gradients
        back to ``w2c`` via the rendered normal map and depth map.

        Strategy
        --------
        1. Render the normal map and metric depth map (both differentiable
           through nvdiffrast).
        2. Segment pixels into planar regions using connected components on
           **detached** copies (normal cosine similarity + relative depth check).
           The hard segment masks are kept outside the autograd graph.
        3. For each segment, aggregate the *live* (grad-enabled) normal and
           depth tensors through the hard mask:
               normal_cam[p] = normalize( sum_{i in p}  nmap[:,i] )
               d_cam[p]      = dot( normal_cam[p],  mean_{i in p}  xyz_cam[:,i] )
           where ``xyz_cam`` is the depth-unprojected 3-D point cloud
           (differentiable in depth).

        Gradient flow
        -------------
        pose → rendered normals  → plane normal vectors   (∂normal_cam / ∂pose)
        pose → rendered depth    → unprojected xyz        (∂d_cam / ∂pose)

        Args:
            w2c:               (4, 4) world-to-camera matrix (requires_grad OK).
            normal_cos_thresh: cosine threshold for grouping adjacent pixels
                               into the same plane (0.995 ≈ 5.7°).
            depth_rel_thresh:  maximum relative depth step between neighbours
                               (0.02 = 2 %).
            min_pixels:        minimum segment size (pixels) to keep.

        Returns:
            dict with keys:

            ``'normal_cam'``  – (P, 3) unit plane normals in camera space.
                                Differentiable w.r.t. ``w2c``.
            ``'d_cam'``       – (P,)   plane offsets (n · x = d) in camera
                                space.  Differentiable w.r.t. ``w2c``.
            ``'masks'``       – (P, H, W) float32 binary segment masks on the
                                same device as ``w2c`` (no gradient).
            ``'pixel_count'`` – list[int] of length P.

            P = 0 tensors are returned when no valid planes are found.
        """

        H, W = self.H, self.W
        if isinstance(w2c, torch.Tensor):
            device = w2c.device
        else:
            w2c = torch.tensor(np.asarray(w2c, dtype=np.float32), dtype=torch.float32)
            device = w2c.device

        R_w2c_mat = w2c[:3, :3]

        w2c = self._to_w2c(w2c)
        proj = self._proj_for_K(None)
        view = self._build_view(w2c)
        mvp  = proj @ view
        exp_clip = torch.matmul(self.exp_verts_h, mvp.T).unsqueeze(0).contiguous()
        rast_exp, _ = dr.rasterize(self.glctx, exp_clip, self.exp_faces,
                                   resolution=[H, W])
        fn_attr = self.exp_face_normals.unsqueeze(0)
        normals_world_raw, _ = dr.interpolate(fn_attr, rast_exp, self.exp_faces)
        vis_mask = (rast_exp[..., 3:4] > 0).float()
        normals_world_raw = normals_world_raw * vis_mask
        normals_world = normals_world_raw.flip(1).permute(0, 3, 1, 2)
        nw_flat  = normals_world[0].reshape(3, -1)
        nc_flat  = R_w2c_mat @ nw_flat
        nmap_hw  = nc_flat.reshape(3, H, W)

        dmap = self.render_depth_batch(w2c.unsqueeze(0)).to(device)
        d_hw = dmap[0, 0]

        # ── 2. Hard plane segmentation on detached numpy arrays ───────────
        nmap_np = nmap_hw.detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
        dmap_np = d_hw.detach().cpu().numpy()                         # (H, W)

        valid = dmap_np > 0
        nlen_sq = (nmap_np ** 2).sum(axis=2)
        valid &= nlen_sq > 0.25
        safe_nlen = np.sqrt(np.maximum(nlen_sq, 1e-16))[..., None]
        normals_n = nmap_np / safe_nlen  # (H, W, 3) unit normals

        N = H * W
        flat_valid = valid.ravel()
        valid_indices = np.where(flat_valid)[0]
        n_valid = len(valid_indices)

        _empty = {
            'normal_cam':  torch.zeros(0, 3, device=device),
            'd_cam':       torch.zeros(0, device=device),
            'masks':       torch.zeros(0, H, W, device=device),
            'pixel_count': [],
        }
        if n_valid < min_pixels:
            return _empty

        compact_id = np.full(N, -1, dtype=np.int32)
        compact_id[valid_indices] = np.arange(n_valid, dtype=np.int32)

        all_px   = np.arange(N, dtype=np.int32)
        col_idx  = all_px % W
        h_a = all_px[col_idx < W - 1];  h_b = h_a + 1
        v_a = all_px[all_px < N - W];   v_b = v_a + W

        ea = np.concatenate([h_a, v_a])
        eb = np.concatenate([h_b, v_b])
        both = flat_valid[ea] & flat_valid[eb]
        ea, eb = ea[both], eb[both]

        normals_flat = normals_n.reshape(N, 3)
        dots = (normals_flat[ea] * normals_flat[eb]).sum(axis=1)
        depth_flat = dmap_np.ravel()
        da, db  = depth_flat[ea], depth_flat[eb]
        keep = (dots >= normal_cos_thresh) & (
            np.abs(da - db) < depth_rel_thresh * np.maximum(da, db))
        ea, eb = ea[keep], eb[keep]

        ea_c = compact_id[ea];  eb_c = compact_id[eb]
        data = np.ones(len(ea_c) * 2, dtype=np.int8)
        rows_sp = np.concatenate([ea_c, eb_c])
        cols_sp = np.concatenate([eb_c, ea_c])
        adj = csr_matrix((data, (rows_sp, cols_sp)), shape=(n_valid, n_valid))
        _, cc_compact = sparse_cc(adj, directed=False)

        cc_flat = np.full(N, -1, dtype=cc_compact.dtype)
        cc_flat[valid_indices] = cc_compact
        cc_flat[~flat_valid]   = -1

        max_label = int(cc_flat.max()) + 1 if cc_flat.max() >= 0 else 0
        if max_label == 0:
            return _empty

        counts    = np.bincount(cc_flat[cc_flat >= 0], minlength=max_label)
        big_labels = np.where(counts >= min_pixels)[0]
        if len(big_labels) == 0:
            return _empty

        # ── 3. Differentiable plane parameter aggregation ─────────────────
        # Unproject depth to 3-D camera-space points (differentiable in d_hw)
        K  = self._K
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        ys = torch.arange(H, device=d_hw.device, dtype=torch.float32)
        xs = torch.arange(W, device=d_hw.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)

        X_cam = (grid_x - cx) / fx * d_hw   # (H, W) diff
        Y_cam = (grid_y - cy) / fy * d_hw   # (H, W) diff
        Z_cam = d_hw                          # (H, W) diff
        pts_cam = torch.stack([X_cam, Y_cam, Z_cam], dim=0)  # (3, H, W) diff


        normals_out:  list[torch.Tensor]          = []
        d_out:        list[torch.Tensor]          = []
        masks_out:    list[torch.Tensor]          = []
        pixel_counts: list[int]                   = []
        polygons_out: list[np.ndarray | None]     = []

        for lbl in big_labels:
            pix  = np.where(cc_flat == lbl)[0]
            r, c = pix // W, pix % W

            # Hard binary mask — no gradient
            m = torch.zeros(H, W, device=d_hw.device, dtype=torch.float32)
            m[r, c] = 1.0

            # Differentiable mean normal (weighted sum through live nmap)
            n_sum  = (nmap_hw * m.unsqueeze(0)).sum(dim=(1, 2))  # (3,)
            n_unit = F.normalize(n_sum.unsqueeze(0), dim=1).squeeze(0)  # (3,) diff

            # Differentiable mean 3-D point via depth unproject
            pt_sum  = (pts_cam * m.unsqueeze(0)).sum(dim=(1, 2))  # (3,) diff
            pt_mean = pt_sum / float(len(pix))                      # (3,) diff

            # Plane offset d = n · centroid  (differentiable)
            d_val = (n_unit * pt_mean).sum()  # scalar diff

            # 2-D convex hull of pixel coordinates (col, row) — no gradient
            pts_2d = np.stack([c, r], axis=1).astype(np.float32)  # (N, 2) (x, y)
            if len(pts_2d) >= 3:
                try:
                    hull = ConvexHull(pts_2d)
                    poly = pts_2d[hull.vertices].astype(np.float32)  # (K, 2)
                except Exception:
                    poly = None
            else:
                poly = None

            normals_out.append(n_unit)
            d_out.append(d_val)
            masks_out.append(m)
            pixel_counts.append(int(len(pix)))
            polygons_out.append(poly)

        return {
            'normal_cam':  torch.stack(normals_out),   # (P, 3) diff
            'd_cam':       torch.stack(d_out),          # (P,)   diff
            'masks':       torch.stack(masks_out),      # (P, H, W) no grad
            'pixel_count': pixel_counts,
            'polygon_2d':  polygons_out,               # list[ndarray(K,2)|None]
        }

    def _compute_structural_edges(self, face_ids, normal_cos_thresh=0.999):
        """Compute structural edge map from face ID map on GPU.

        Detects silhouette edges (face vs background) and structural edges
        (adjacent faces with different normals). Skips internal triangulation
        edges (coplanar adjacent faces).

        Args:
            face_ids: (H, W) int tensor, -1 = background.
            normal_cos_thresh: cosine threshold above which faces are coplanar.

        Returns:
            edge_map: (H, W) float tensor in {0, 1}.
        """
        H, W = face_ids.shape
        edge = torch.zeros(H, W, device=face_ids.device, dtype=torch.float32)

        # Horizontal pairs
        fid_l = face_ids[:, :-1]  # (H, W-1)
        fid_r = face_ids[:, 1:]   # (H, W-1)
        # Vertical pairs
        fid_t = face_ids[:-1, :]  # (H-1, W)
        fid_b = face_ids[1:, :]   # (H-1, W)

        # Silhouette: face vs background
        sil_h = (fid_l >= 0) != (fid_r >= 0)
        sil_v = (fid_t >= 0) != (fid_b >= 0)

        # Structural: different non-coplanar faces
        diff_h = (fid_l != fid_r) & (fid_l >= 0) & (fid_r >= 0)
        diff_v = (fid_t != fid_b) & (fid_t >= 0) & (fid_b >= 0)

        n_faces = self.face_normals.shape[0]

        if diff_h.any():
            fa = fid_l[diff_h].clamp(0, n_faces - 1).long()
            fb = fid_r[diff_h].clamp(0, n_faces - 1).long()
            cos = (self.face_normals[fa] * self.face_normals[fb]).sum(dim=1).abs()
            struct_mask = cos <= normal_cos_thresh
            struct_h = torch.zeros_like(diff_h)
            struct_h[diff_h] = struct_mask
        else:
            struct_h = diff_h

        if diff_v.any():
            fa = fid_t[diff_v].clamp(0, n_faces - 1).long()
            fb = fid_b[diff_v].clamp(0, n_faces - 1).long()
            cos = (self.face_normals[fa] * self.face_normals[fb]).sum(dim=1).abs()
            struct_mask = cos <= normal_cos_thresh
            struct_v = torch.zeros_like(diff_v)
            struct_v[diff_v] = struct_mask
        else:
            struct_v = diff_v

        # Combine
        edge[:, :-1] = torch.where(sil_h | struct_h,
                                     torch.ones_like(edge[:, :-1]), edge[:, :-1])
        edge[:-1, :] = torch.where(sil_v | struct_v,
                                     torch.ones_like(edge[:-1, :]), edge[:-1, :])

        return edge


    def render_face_ids_batch(self, w2c_batch: "torch.Tensor | np.ndarray") -> np.ndarray:
        """Render per-pixel face index maps for a batch of w2c poses.

        Returns:
            face_ids: (B, H, W) int32 numpy array, -1 for background pixels.
        """
        w2c_batch = self._as_w2c_batch(w2c_batch, self._dev)
        out = []
        for chunk in self._iter_w2c_chunks(w2c_batch):
            out.append(self._render_face_ids_chunk(chunk))
        return np.concatenate(out, axis=0)

    def _render_face_ids_chunk(self, w2c_chunk: torch.Tensor) -> np.ndarray:
        chunk_out = []
        proj = self._proj_for_K(None)
        for i in range(w2c_chunk.shape[0]):
            w2c = w2c_chunk[i]
            view = self._build_view(w2c)
            mvp = proj @ view
            verts_clip = torch.matmul(self.verts_h, mvp.T).unsqueeze(0).contiguous()
            rast_out, _ = dr.rasterize(self.glctx, verts_clip, self.faces,
                                        resolution=[self.H, self.W])
            fids = rast_out[0, :, :, 3].int() - 1
            chunk_out.append(fids.flip(0).cpu().numpy().astype(np.int32))
        return np.stack(chunk_out, axis=0)

    def render_shaded(
        self,
        w2c: "torch.Tensor | np.ndarray",
        *,
        device: str = "gpu",
        light_dir_cam: np.ndarray | None = None,
        ambient: float = 0.20,
        base_color: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render Phong-shaded RGB image.

        Args:
            device: ``'gpu'`` or ``'cpu'``.
            light_dir_cam: (3,) unit light direction in camera coords (towards surface).
            ambient: ambient intensity [0, 1].
            base_color: (3,) RGB base colour [0-255].  Default warm concrete gray.

        Returns:
            shaded: (H, W, 3) uint8 RGB image.  Background is black.
        """
        w2c_t = self._to_w2c(w2c)
        normals_t = self.render_normals_batch(w2c_t.unsqueeze(0))
        normals = normals_t[0].permute(1, 2, 0).cpu().numpy()

        if light_dir_cam is None:
            l = np.array([0.3, -0.6, -0.7], dtype=np.float32)
            light_dir_cam = l / np.linalg.norm(l)
        if base_color is None:
            base_color = np.array([200, 195, 185], dtype=np.float32)
        else:
            base_color = np.asarray(base_color, dtype=np.float32)

        mask = np.linalg.norm(normals, axis=2) > 0.3
        n = normals.copy()
        n_len = np.linalg.norm(n, axis=2, keepdims=True).clip(min=1e-6)
        n = n / n_len

        ndotl = np.clip(-np.sum(n * light_dir_cam, axis=2), 0.0, 1.0)
        diffuse = 0.65 * ndotl
        view_v = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        half_vec = -light_dir_cam + (-view_v)
        half_vec = half_vec / (np.linalg.norm(half_vec) + 1e-8)
        ndoth = np.clip(np.sum(n * half_vec, axis=2), 0.0, 1.0)
        specular = 0.15 * (ndoth ** 16)

        shade = np.clip(ambient + diffuse + specular, 0.0, 1.0)
        img = np.zeros((self.H, self.W, 3), dtype=np.float32)
        for c in range(3):
            img[:, :, c] = shade * base_color[c]
        img[~mask] = 0
        return np.clip(img, 0, 255).astype(np.uint8)

    def render_semantic(
        self,
        w2c: "torch.Tensor | np.ndarray",
        *,
        label_source: str = "normals",
        device: str = "gpu",
        face_ids: np.ndarray | None = None,
        roof_nz_thresh: float = 0.5,
        labels: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render per-pixel semantic label map (roof / wall / other).

        Args:
            label_source: ``'normals'`` (classify from z-normal) or
                          ``'mesh'`` (use per-face labels stored in renderer).
            device: ``'gpu'`` or ``'cpu'``.
            face_ids: pre-computed (H, W) int32 face-id map; rendered if None.
            roof_nz_thresh: |n_z| threshold (only for ``label_source='normals'``).
            labels: override face label array; falls back to ``self._labels_np``.

        Returns:
            label_map: (H, W) uint8 — 0=bg, 1=roof, 2=wall, 3=other.
            color_map: (H, W, 3) uint8 — per-pixel colour.
        """
        if face_ids is None:
            w2c_t = self._to_w2c(w2c)
            face_ids = self.render_face_ids_batch(w2c_t.unsqueeze(0))[0]

        H, W = self.H, self.W
        n_faces = len(self._faces_np)
        fid_safe = np.clip(face_ids, 0, n_faces - 1)
        valid = face_ids >= 0

        label_map = np.zeros((H, W), dtype=np.uint8)

        if label_source == "normals":
            fn = self._face_normals_np          # (F,3) world-space flat normals
            nz = np.abs(fn[fid_safe, 2])
            label_map[valid & (nz > roof_nz_thresh)] = self.SEMANTIC_ROOF
            label_map[valid & (nz <= roof_nz_thresh)] = self.SEMANTIC_WALL

        elif label_source == "mesh":
            lbl_arr = labels if labels is not None else self._labels_np
            if lbl_arr is None:
                raise ValueError("label_source='mesh' requires labels (pass labels= or build Renderer with labels=)")
            face_sem = np.full(n_faces, self.SEMANTIC_OTHER, dtype=np.uint8)
            for lbl_str, sid in self._MESH_LABEL_TO_SEMANTIC.items():
                face_sem[lbl_arr == lbl_str] = sid
            label_map[valid] = face_sem[fid_safe[valid]]
        else:
            raise ValueError(f"label_source must be 'normals' or 'mesh', got '{label_source}'")

        color_map = np.zeros((H, W, 3), dtype=np.uint8)
        for sid, color in self._SEMANTIC_LABEL_COLORS.items():
            mask = label_map == sid
            if mask.any():
                color_map[mask] = color

        return label_map, color_map

    def render_segmentation(
        self,
        w2c: "torch.Tensor | np.ndarray",
        *,
        mode: str = "semantic",
        device: str = "gpu",
        face_ids: np.ndarray | None = None,
        building_ids: np.ndarray | None = None,
        labels: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render building segmentation (semantic or instance).

        Args:
            mode: ``'semantic'`` — per-pixel class (roof/wall/other).
                  ``'instance'`` — per-pixel building instance ID.
            building_ids: override per-face building ID array; falls back to
                          ``self._building_ids_np``.
            labels:       override face labels (for semantic mesh mode).

        Returns:
            color_map: (H, W, 3) uint8 — coloured segmentation.
            id_map:    (H, W) int32 — class IDs (semantic) or building IDs, -1=bg.
        """
        H, W = self.H, self.W
        if face_ids is None:
            w2c_t = self._to_w2c(w2c)
            face_ids = self.render_face_ids_batch(w2c_t.unsqueeze(0))[0]

        if mode == "semantic":
            label_map, color_map = self.render_semantic(
                w2c, label_source="mesh", device=device,
                face_ids=face_ids, labels=labels,
            )
            id_map = label_map.astype(np.int32)
            id_map[face_ids < 0] = -1
            return color_map, id_map

        elif mode == "instance":
            bid_arr = building_ids if building_ids is not None else self._building_ids_np
            if bid_arr is None:
                raise ValueError("mode='instance' requires building_ids")

            lbl_arr = labels if labels is not None else self._labels_np
            _BUILDING_LABELS = {'wall', 'roof', 'ground', 'closure', 'building'}

            n_faces = len(self._faces_np)
            if lbl_arr is not None and len(lbl_arr) == n_faces:
                is_building_face = np.array(
                    [str(l) in _BUILDING_LABELS for l in lbl_arr], dtype=bool
                )
            else:
                is_building_face = np.ones(n_faces, dtype=bool)

            color_map = np.zeros((H, W, 3), dtype=np.uint8)
            id_map = np.full((H, W), -1, dtype=np.int32)
            valid = face_ids >= 0
            fid_safe = np.clip(face_ids, 0, n_faces - 1)
            bids_valid = bid_arr[fid_safe[valid]]
            is_bldg_px = is_building_face[fid_safe[valid]]
            bids_valid[~is_bldg_px] = -1
            id_map[valid] = bids_valid

            unique_bids = np.unique(bids_valid[bids_valid >= 0])
            golden_ratio_conjugate = 0.618033988749895
            h_hue = 0.0
            bid_to_color = {}
            for idx, bid in enumerate(unique_bids):
                s = 0.7 + 0.3 * (idx % 2)
                v_val = 0.95 - 0.15 * (idx % 3)
                r, g, b = colorsys.hsv_to_rgb(h_hue, s, v_val)
                bid_to_color[bid] = np.array([int(r*255), int(g*255), int(b*255)], dtype=np.uint8)
                h_hue = (h_hue + golden_ratio_conjugate) % 1.0
            for bid, color in bid_to_color.items():
                color_map[id_map == bid] = color
            return color_map, id_map
        else:
            raise ValueError(f"mode must be 'semantic' or 'instance', got '{mode}'")

    def render_wireframe(
        self,
        w2c: "torch.Tensor | np.ndarray",
        *,
        interval: float = 2.0,
        thickness: int = 1,
        color: tuple[int, int, int] = (0, 255, 0),
        depth: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Sample points along unique mesh edges and project as a wireframe.

        Deduplicates edges by vertex-index pairs, then samples only unique edges.
        Frustum-culls edges before sampling to avoid wasting time on off-screen geometry.

        Args:
            interval:  Sampling distance in metres along each edge.
            thickness: Pixel radius for drawn wireframe points.
            color:     RGB colour of wireframe dots on the returned overlay.
            depth:     optional pre-rendered depth (H, W); if None, will render.

        Returns:
            wireframe_img: (H, W, 3) uint8 image — coloured dots on black background.
            wire_info: dict with ``'pts_3d'`` (N,3), ``'normals'`` (N,3),
                       ``'uv'`` (N,2), ``'depth'`` (N,), ``'visible_mask'`` (N,).
        """

        H, W = self.H, self.W
        w2c_np = w2c.detach().cpu().numpy() if isinstance(w2c, torch.Tensor) else np.asarray(w2c, dtype=np.float64)
        K = self._ensure_K().astype(np.float64)

        verts = self._verts_np.astype(np.float64)   # (V, 3) absolute world-space
        faces = self._faces_np                        # (F, 3)
        fn = self._face_normals_np                    # (F, 3) world-space

        # 1. Extract ALL face edges and deduplicate by sorted vertex pairs
        e0 = faces[:, [0, 1]]
        e1 = faces[:, [1, 2]]
        e2 = faces[:, [2, 0]]
        all_edges = np.concatenate([e0, e1, e2], axis=0)  # (3F, 2)
        face_idx_all = np.repeat(np.arange(len(faces)), 3)
        sorted_edges = np.sort(all_edges, axis=1).astype(np.int64)
        max_v = int(sorted_edges.max()) + 1
        edge_keys = sorted_edges[:, 0] * max_v + sorted_edges[:, 1]
        _, uniq_idx = np.unique(edge_keys, return_index=True)
        edges = all_edges[uniq_idx]        # (E, 2)
        face_idx = face_idx_all[uniq_idx]

        # 2. Frustum-cull edges
        ones_v = np.ones((len(verts), 1), dtype=verts.dtype)
        v_cam = (w2c_np[:3] @ np.hstack([verts, ones_v]).T).T.astype(np.float32)
        v_z = v_cam[:, 2]
        inv_vz = np.where(v_z > 1e-6, 1.0 / v_z, 0.0).astype(np.float32)
        v_u = K[0, 0] * v_cam[:, 0] * inv_vz + K[0, 2]
        v_v = K[1, 1] * v_cam[:, 1] * inv_vz + K[1, 2]
        margin = max(H, W) * 0.3
        v_in = (v_z > 0) & (v_u > -margin) & (v_u < W + margin) & (v_v > -margin) & (v_v < H + margin)
        e_visible = v_in[edges[:, 0]] | v_in[edges[:, 1]]
        edges = edges[e_visible]
        face_idx = face_idx[e_visible]

        # 3. Prepare edge endpoints; filter zero-length
        p0 = verts[edges[:, 0]].astype(np.float32)
        p1 = verts[edges[:, 1]].astype(np.float32)
        n0 = fn[face_idx]
        lengths = np.linalg.norm(p1 - p0, axis=1)
        valid = lengths > 0
        p0, p1, n0, lengths = p0[valid], p1[valid], n0[valid], lengths[valid]

        # 4. Vectorised sampling along all edges
        n_samples = np.maximum(np.ceil(lengths / interval).astype(np.int32) + 1, 2)
        total_pts = int(n_samples.sum())
        offsets = np.zeros(len(n_samples) + 1, dtype=np.int64)
        np.cumsum(n_samples, out=offsets[1:])

        edge_idx = np.repeat(np.arange(len(n_samples), dtype=np.int32), n_samples)
        local_idx = np.arange(total_pts, dtype=np.float32) - offsets[edge_idx].astype(np.float32)
        denom = (n_samples[edge_idx] - 1).clip(min=1).astype(np.float32)
        t = (local_idx / denom)[:, None]

        pts_3d = p0[edge_idx] + t * (p1[edge_idx] - p0[edge_idx])
        normals_out = n0[edge_idx]

        # 5. Project 3D → 2D
        ones = np.ones((len(pts_3d), 1), dtype=np.float32)
        cam = (w2c_np[:3] @ np.hstack([pts_3d, ones]).T).astype(np.float32)  # (3, N)
        z = cam[2]
        inv_z = np.float32(1.0) / np.maximum(z, np.float32(1e-12))
        uv = np.empty((len(pts_3d), 2), dtype=np.float32)
        uv[:, 0] = K[0, 0] * cam[0] * inv_z + K[0, 2]
        uv[:, 1] = K[1, 1] * cam[1] * inv_z + K[1, 2]
        in_frame = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)

        # 6. Depth-based occlusion culling
        if depth is None:
            w2c_t = self._to_w2c(w2c_np)
            depth = self.render_depth_batch(w2c_t.unsqueeze(0))[0, 0].detach().cpu().numpy()

        uv_safe = np.clip(np.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0), -1, max(W, H))
        px = np.clip(uv_safe[:, 0].astype(np.int32), 0, W - 1)
        py = np.clip(uv_safe[:, 1].astype(np.int32), 0, H - 1)
        rendered_z = depth[py, px]
        depth_tol = np.float32(0.3)
        vis = in_frame & (rendered_z > 0) & (z <= rendered_z + depth_tol)

        # 7. Draw wireframe
        wireframe_img = np.zeros((H, W, 3), dtype=np.uint8)
        uv_vis = uv[vis].astype(np.int32)
        if len(uv_vis) > 0:
            if thickness <= 1:
                wireframe_img[uv_vis[:, 1], uv_vis[:, 0]] = color
            else:
                for i in range(len(uv_vis)):
                    cv2.circle(wireframe_img, (uv_vis[i, 0], uv_vis[i, 1]), thickness, color, -1)

        return wireframe_img, {
            'pts_3d': pts_3d,
            'normals': normals_out,
            'uv': uv,
            'depth': z,
            'visible_mask': vis,
        }


class RendererCPU(Renderer):
    """CPU cv2 z-buffer backend."""

    def __init__(self, vertices, faces, K, image_size, *, labels=None, building_ids=None,
                 near=0.5, far=10000.0, depth_flag=True, minibatch_size=52, auto_cull=False):
        self._init_common(vertices, faces, K, image_size, labels=labels, building_ids=building_ids,
            near=near, far=far, depth_flag=depth_flag, minibatch_size=minibatch_size, auto_cull=auto_cull)
        self._dev = torch.device("cpu")
        self.proj = torch.tensor(self._proj_np, dtype=torch.float32) if self._proj_np is not None else None

    @property
    def dev(self):
        return self._dev

    def _ensure_K(self, K: np.ndarray | None = None) -> np.ndarray:
        K_use = K if K is not None else self._K
        if K_use is None:
            raise ValueError("intrinsics K required (pass at init or per call)")
        return np.asarray(K_use, dtype=np.float64)

    def _to_w2c(self, w2c):
        if isinstance(w2c, np.ndarray):
            t = torch.from_numpy(np.asarray(w2c, dtype=np.float32))
        else:
            t = w2c.float()
        if t.shape == (3, 4):
            t = torch.cat([t, torch.tensor([[0., 0., 0., 1.]], dtype=t.dtype)], dim=0)
        return t

    def _rasterize(self, w2c):
        return _rasterize_cpu(w2c, verts=self._verts_np, faces=self._faces_np,
            face_normals=self._face_normals_np, K=self._ensure_K(), H=self.H, W=self.W, near=self.near)

    def _render_masks_batch_cpu(
        self,
        w2c_batch: torch.Tensor,
        K: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        K_np = K.detach().cpu().numpy() if isinstance(K, torch.Tensor) else np.asarray(K)
        if self._K is None or not np.allclose(self._K, K_np):
            self._proj_for_K(K_np)
        masks, depths = [], [] if self.depth_flag else None
        w2c_np = self._as_w2c_batch(w2c_batch).detach().cpu().numpy()
        for s in range(0, w2c_np.shape[0], self.minibatch_size):
            e = min(s + self.minibatch_size, w2c_np.shape[0])
            for i in range(s, e):
                out = self._rasterize(w2c_np[i])
                masks.append(out['mask'] > 0)
                if depths is not None:
                    depths.append(out['depth'])
        mask_t = torch.from_numpy(np.stack(masks, axis=0))
        dep_t = torch.from_numpy(np.stack(depths, axis=0)).float() if depths is not None else None
        return mask_t, dep_t

    def render_mask_batch(self, w2c_batch, *, antialias=True):
        del antialias
        w2c_batch = self._as_w2c_batch(w2c_batch)
        masks = []
        for chunk in self._iter_w2c_chunks(w2c_batch):
            for i in range(chunk.shape[0]):
                out = self._rasterize(chunk[i].numpy())
                masks.append(torch.from_numpy(out["mask"].astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0))
        return torch.cat(masks, dim=0)

    def render_depth_batch(self, w2c_batch):
        w2c_batch = self._as_w2c_batch(w2c_batch)
        depth_chunks = []
        for chunk in self._iter_w2c_chunks(w2c_batch):
            chunk_depths, _ = self._render_depth_normals_chunk_cpu(chunk)
            depth_chunks.append(chunk_depths)
        return torch.cat(depth_chunks, dim=0).unsqueeze(1)

    def render_normals_batch(self, w2c_batch):
        w2c_batch = self._as_w2c_batch(w2c_batch)
        normal_chunks = []
        for chunk in self._iter_w2c_chunks(w2c_batch):
            _, chunk_normals = self._render_depth_normals_chunk_cpu(chunk)
            normal_chunks.append(chunk_normals)
        normals = torch.cat(normal_chunks, dim=0)
        return normals.permute(0, 3, 1, 2).contiguous()

    def render_face_ids_batch(self, w2c_batch):
        w2c_batch = self._as_w2c_batch(w2c_batch)
        out = []
        for chunk in self._iter_w2c_chunks(w2c_batch):
            for i in range(chunk.shape[0]):
                out.append(self._rasterize(chunk[i].numpy())["face_ids"])
        return np.stack(out, axis=0)

    def render_masks_batch(self, w2c_batch, K):
        return self._render_masks_batch_cpu(w2c_batch, K)

    def render_masks_batch_cropped(self, w2c_batch, K, crop_radius=None, crop_center=None):
        return self._render_masks_batch_cpu(w2c_batch, K)

    def warmup(self, K=None):
        return

    def _proj_for_K(self, K=None):
        if K is None:
            if self.proj is None:
                raise ValueError("intrinsics K required")
            return self.proj
        Kn = np.asarray(K, dtype=np.float64)
        if self._K is not None and np.allclose(self._K, Kn) and self.proj is not None:
            return self.proj
        H, W = self.H, self.W
        fx, fy, cx, cy = Kn[0, 0], Kn[1, 1], Kn[0, 2], Kn[1, 2]
        n, f = self.near, self.far
        proj = np.zeros((4, 4), dtype=np.float64)
        proj[0, 0] = 2 * fx / W
        proj[1, 1] = 2 * fy / H
        proj[0, 2] = 1 - 2 * cx / W
        proj[1, 2] = 2 * cy / H - 1
        proj[2, 2] = -(f + n) / (f - n)
        proj[2, 3] = -2 * f * n / (f - n)
        proj[3, 2] = -1
        self._K, self._proj_np = Kn.copy(), proj
        self.proj = torch.tensor(proj, dtype=torch.float32)
        return self.proj

    def _render_depth_normals_chunk_cpu(self, w2c_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        depths, normals = [], []
        for i in range(w2c_chunk.shape[0]):
            out = self._rasterize(w2c_chunk[i].detach().cpu().numpy())
            depths.append(out["depth"])
            normals.append(out["nmap"])
        return (
            torch.from_numpy(np.stack(depths)).float(),
            torch.from_numpy(np.stack(normals)).float(),
        )

    def render_depth_normals_batch(self, w2c_batch):
        w2c_batch = self._as_w2c_batch(w2c_batch)
        depth_chunks, normal_chunks = [], []
        for chunk in self._iter_w2c_chunks(w2c_batch):
            d_c, n_c = self._render_depth_normals_chunk_cpu(chunk)
            depth_chunks.append(d_c)
            normal_chunks.append(n_c)
        return torch.cat(depth_chunks, dim=0), torch.cat(normal_chunks, dim=0)

    def render_face_ids_and_depth_batch(self, w2c):
        raise RuntimeError("render_face_ids_and_depth_batch requires RendererGPU")

    def render_structural_edges_and_depth_batch(self, w2c, **kwargs):
        raise RuntimeError("render_structural_edges_and_depth_batch requires RendererGPU")

    def render_edges_batch(self, w2c_batch, **kwargs):
        return self.render_normals_batch(w2c_batch)[:, :1] * 0  # CPU: no diff edges; return zeros
