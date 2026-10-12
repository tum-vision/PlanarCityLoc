"""
utils/lod.py
============
LoD mesh loading and manipulation utilities.

Supports loading LoD meshes from CityGML/OBJ/PLY and optional ``.npz`` dumps.

``LoD`` is the unified mesh type: load **.obj / .ply** (metric/world meshes),
optional ``.npz`` (``from_npz``), footprint cropping,
plane / feature-edge helpers, and depth/mask rendering on CPU or GPU via
:class:`~utils.renderer.Renderer`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time as _time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as _torch
import trimesh
from matplotlib.patches import Patch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sparse_cc
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation

from utils.io import load_mesh_vertices_faces
from utils.pose import CameraPose, load_poses_csv
from utils.renderer import Renderer
from utils.timing import GpuTimer

Device = Literal["cpu", "gpu"]
ImageSize = Union[Tuple[int, int], int]


class LoD:
    """Triangle mesh: .obj/.ply/.npz, cropping, planes, depth/mask render."""

    @staticmethod
    def _lod_compute_building_ids(faces: np.ndarray, num_vertices: int) -> np.ndarray:
        """Per-face building IDs: faces sharing an edge belong to the same building.

        Vectorised face-adjacency connected components using scipy sparse CC.
        """

        faces = np.asarray(faces, dtype=np.int64)
        F = len(faces)
        if F == 0:
            return np.zeros(0, dtype=np.int32)

        # All 3 edges per face (sorted vertex pairs), vectorised
        # edges_raw: (F*3, 2)
        edges_raw = np.stack([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ], axis=1).reshape(-1, 2)
        edges_sorted = np.sort(edges_raw, axis=1)

        # Unique edge key for grouping
        edge_keys = edges_sorted[:, 0] * (num_vertices + 1) + edges_sorted[:, 1]

        # Face index for each edge entry
        face_of_edge = np.repeat(np.arange(F, dtype=np.int64), 3)

        # Sort by edge key to find shared edges
        order = np.argsort(edge_keys)
        sorted_keys = edge_keys[order]
        sorted_faces = face_of_edge[order]

        # Consecutive entries with same key → faces share that edge
        same = sorted_keys[:-1] == sorted_keys[1:]
        fa = sorted_faces[:-1][same]
        fb = sorted_faces[1:][same]

        if len(fa) == 0:
            return np.arange(F, dtype=np.int32)

        # Build sparse face adjacency and run CC
        data = np.ones(len(fa) * 2, dtype=np.int8)
        row = np.concatenate([fa, fb])
        col = np.concatenate([fb, fa])
        adj = csr_matrix((data, (row, col)), shape=(F, F))
        _, labels = sparse_cc(adj, directed=False)
        return labels.astype(np.int32)

    @staticmethod
    def _lod_component_xy_samples(
        vertices: np.ndarray,
        faces: np.ndarray,
        face_component_ids: np.ndarray,
        component_ids: np.ndarray,
        *,
        pts_per_comp: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """XY sample points for a set of mesh components (centroid + bbox + subsample)."""
        centroids_xy = LoD._lod_face_centroids(vertices, faces)[:, :2]
        samples: List[np.ndarray] = []
        cids: List[int] = []
        for cid in component_ids:
            mask = face_component_ids == cid
            if not mask.any():
                continue
            xy = centroids_xy[mask]
            pts: List[np.ndarray] = [xy.mean(axis=0)]
            mn, mx = xy.min(axis=0), xy.max(axis=0)
            pts.extend([mn, mx, np.array([mn[0], mx[1]]), np.array([mx[0], mn[1]])])
            n_extra = max(0, pts_per_comp - 5)
            if len(xy) > n_extra:
                idx = np.linspace(0, len(xy) - 1, n_extra, dtype=np.int64)
                pts.extend(xy[idx])
            else:
                pts.extend(xy[:n_extra])
            for p in pts[:pts_per_comp]:
                samples.append(p)
                cids.append(int(cid))
        if not samples:
            return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.int32)
        return np.asarray(samples, dtype=np.float64), np.asarray(cids, dtype=np.int32)

    @staticmethod
    def _lod_merge_building_parts(
        faces: np.ndarray,
        vertices: np.ndarray,
        face_component_ids: np.ndarray,
        labels: np.ndarray,
        dist_thresh: float = 2.0,
    ) -> np.ndarray:
        """Merge building components that are connected in the horizontal (XY) plane.

        Two separate mesh components belong to the same building if their triangle
        surfaces come within *dist_thresh* metres in XY.  Surface proximity is
        approximated by sampling each face at 7 points (3 vertices, 3 edge
        midpoints, 1 centroid) so that large triangles whose interiors are close
        are correctly detected even when no shared vertex is nearby.
        Vertical separation (balconies, terraces, rooftop structures) is ignored.

        A building is a separate instance only when it has no XY-close surface to
        any other building — i.e. it is genuinely spatially disconnected.
        """

        _BUILDING_LABELS = {"wall", "roof", "ground", "closure", "building"}
        n_comps = int(face_component_ids.max()) + 1
        if n_comps < 2:
            return face_component_ids.copy()

        # --- Identify building components ---
        face_is_building = np.array(
            [str(l) in _BUILDING_LABELS for l in labels], dtype=bool
        )
        bldg_count = np.bincount(
            face_component_ids, weights=face_is_building.astype(np.float64), minlength=n_comps
        )
        is_building_comp = bldg_count > 0
        if is_building_comp.sum() < 2:
            return face_component_ids.copy()

        bldg_face_mask = is_building_comp[face_component_ids]
        bldg_face_ids = np.where(bldg_face_mask)[0]
        bldg_cids = face_component_ids[bldg_face_ids].astype(np.int32)
        n_bldg_faces = len(bldg_face_ids)

        if n_bldg_faces > 200_000:
            # Large mesh: multi-point XY samples per component (memory-safe).
            bldg_comp_idx = np.flatnonzero(is_building_comp)
            sample_pts, all_cids = LoD._lod_component_xy_samples(
                vertices, faces, face_component_ids, bldg_comp_idx,
            )
        else:
            # --- Collect face surface samples (verts + edge midpoints + centroid) ---
            bv_xy = vertices[faces[bldg_face_ids].ravel(), :2].reshape(-1, 3, 2)
            v0 = bv_xy[:, 0, :]
            v1 = bv_xy[:, 1, :]
            v2 = bv_xy[:, 2, :]
            sample_pts = np.concatenate([
                v0, v1, v2,
                (v0 + v1) * 0.5,
                (v1 + v2) * 0.5,
                (v0 + v2) * 0.5,
                (v0 + v1 + v2) / 3.0,
            ], axis=0)
            all_cids = np.tile(bldg_cids, 7)

        # --- XY proximity: find all sample pairs within dist_thresh ---
        tree = cKDTree(sample_pts)
        kd_pairs = tree.query_pairs(dist_thresh, output_type='ndarray')

        # --- Build component adjacency graph and run connected components ---
        if len(kd_pairs) > 0:
            cid_a = all_cids[kd_pairs[:, 0]]
            cid_b = all_cids[kd_pairs[:, 1]]
            diff = cid_a != cid_b
            ea, eb = cid_a[diff], cid_b[diff]
            data = np.ones(len(ea) * 2, dtype=np.int8)
            row = np.concatenate([ea, eb]).astype(np.int32)
            col = np.concatenate([eb, ea]).astype(np.int32)
            comp_adj = csr_matrix((data, (row, col)), shape=(n_comps, n_comps))
            _, labels_out = sparse_cc(comp_adj, directed=False)
        else:
            labels_out = np.arange(n_comps, dtype=np.int32)

        # --- Remap to dense final IDs ---
        _, inv = np.unique(labels_out, return_inverse=True)
        return inv[face_component_ids].astype(np.int32)

    @staticmethod
    def _lod_face_centroids(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        v = np.asarray(vertices, dtype=np.float64)
        f = np.asarray(faces, dtype=np.int64)
        return (v[f[:, 0]] + v[f[:, 1]] + v[f[:, 2]]) / 3.0

    @staticmethod
    def _lod_crop_vertices_faces(
        vertices: np.ndarray,
        faces: np.ndarray,
        pose: Any,
        K: np.ndarray,
        image_size: Tuple[int, int],
        margin: float = 1.5,
        use_frustum: bool = False,
        along_view_extend_m: Optional[float] = None,
        footprint_extend_m: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Crop mesh to camera footprint; building-aware (full buildings).

        FOV-derived XY disc around the ground footprint, plus optional extras:
        ``footprint_extend_m``, image frustum test (``use_frustum``), and forward-only
        cap ``along_view_extend_m``.
        """

        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        if len(faces) == 0:
            return vertices[:0].copy(), faces[:0].copy(), np.zeros((0,), dtype=np.int64)

        if isinstance(pose, np.ndarray):
            pose = CameraPose.from_w2c(np.asarray(pose, dtype=np.float64))
        w2c = pose.to_w2c()
        H, W = int(image_size[0]), int(image_size[1])
        K = np.asarray(K, dtype=np.float64)

        cam_pos = pose.position
        R_c2w = pose.R_c2w
        look_dir = R_c2w[:, 2]

        ground_z = float(np.percentile(vertices[:, 2], 10))
        dz = look_dir[2]
        if abs(dz) > 1e-6:
            t_ground = (ground_z - cam_pos[2]) / dz
            if t_ground > 0:
                footprint_center = cam_pos + t_ground * look_dir
            else:
                footprint_center = cam_pos.copy()
                footprint_center[2] = ground_z
        else:
            footprint_center = cam_pos.copy()
            footprint_center[2] = ground_z

        altitude = abs(cam_pos[2] - ground_z)
        fx = float(K[0, 0])
        img_scale = max(H, W)
        half_fov = np.arctan(img_scale / (2.0 * fx))
        visible_radius = altitude * np.tan(half_fov) * np.sqrt(2.0)
        offset_xy = float(
            np.hypot(
                footprint_center[0] - cam_pos[0],
                footprint_center[1] - cam_pos[1],
            )
        )
        crop_radius = max((visible_radius + offset_xy) * margin, 100.0) + float(
            footprint_extend_m
        )

        centroids = LoD._lod_face_centroids(vertices, faces)
        dist_xy = np.sqrt(
            (centroids[:, 0] - footprint_center[0]) ** 2
            + (centroids[:, 1] - footprint_center[1]) ** 2
        )
        face_mask = dist_xy < crop_radius

        if use_frustum:
            ones = np.ones((len(centroids), 1), dtype=np.float64)
            pts_h = np.hstack([centroids, ones])
            X_cam = (w2c @ pts_h.T).T[:, :3]
            z = X_cam[:, 2]
            proj = (K @ X_cam.T).T
            zp = np.where(np.abs(z) > 1e-9, z, 1e9)
            u = proj[:, 0] / zp
            vv = proj[:, 1] / zp
            face_mask &= (
                (z > 0.0)
                & (u >= 0.0)
                & (u < float(W))
                & (vv >= 0.0)
                & (vv < float(H))
            )

        if along_view_extend_m is not None:
            # Horizontal projection onto camera forward: s=0 at footprint, +s = "ahead"
            # along the view on the ground. Foreground buildings (between camera and
            # footprint) have s << 0; a lower bound on s incorrectly removed them.
            # Only cap forward distance; do not cut the "behind footprint" half-plane.
            dir_xy = R_c2w[:2, 2].astype(np.float64)
            nlen = float(np.linalg.norm(dir_xy))
            if nlen < 1e-9:
                dir_xy = np.array([1.0, 0.0], dtype=np.float64)
            else:
                dir_xy /= nlen
            fp_xy = footprint_center[:2]
            rel = centroids[:, :2] - fp_xy
            s = rel @ dir_xy
            face_mask &= s <= float(along_view_extend_m)

        building_ids = LoD._lod_compute_building_ids(faces, len(vertices))
        visible_bids = np.unique(building_ids[face_mask])
        if len(visible_bids) == 0:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
            )

        building_mask = np.isin(building_ids, visible_bids)
        kept_faces = faces[building_mask]
        if len(kept_faces) == 0:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
            )

        orig_face_indices = np.flatnonzero(building_mask)
        used_verts = np.unique(kept_faces.ravel())
        vert_remap = np.full(len(vertices), -1, dtype=np.int64)
        vert_remap[used_verts] = np.arange(len(used_verts))
        new_vertices = vertices[used_verts]
        new_faces = vert_remap[kept_faces].astype(np.int64)
        return new_vertices, new_faces, orig_face_indices.astype(np.int64)

    @staticmethod
    def _lod_w2c_batch(w2c) -> torch.Tensor:
        """Single w2c → (1, 4, 4) float tensor for batch renderer APIs."""
        t = w2c if isinstance(w2c, torch.Tensor) else torch.tensor(np.asarray(w2c, dtype=np.float32))
        if t.ndim == 2:
            t = t.unsqueeze(0)
        return t.float()

    @staticmethod
    def _lod_as_hw(image_size: ImageSize) -> Tuple[int, int]:
        if isinstance(image_size, int):
            return image_size, image_size
        h, w = image_size
        return int(h), int(w)

    @staticmethod
    def _lod_load_mesh_vertices_faces(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        return load_mesh_vertices_faces(path)

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path).resolve()
        self.vertices, self.faces = LoD._lod_load_mesh_vertices_faces(self.path)
        self.utm_offset = np.zeros(3, dtype=np.float64)
        self.labels = np.full((len(self.faces),), "building", dtype=object)
        self.polygon_ids: Optional[np.ndarray] = None
        self._building_ids: Optional[np.ndarray] = None
        self._centroids_abs: Optional[np.ndarray] = None
        self._renderer: Optional[Renderer] = None
        self._render_key: Optional[tuple] = None
        self._batch_size: int = 52

    def set_batch_size(self, batch_size: int) -> None:
        """Default GPU batch size for this LoD (applied on init and rebuild)."""
        self._batch_size = int(batch_size)
        if self._renderer is not None:
            self._renderer.minibatch_size = self._batch_size

    def _init_renderer(
        self,
        K: np.ndarray,
        image_size: ImageSize,
        device: Device = "gpu",
    ) -> Renderer:
        """Build (or rebuild) the mesh renderer for this LoD crop."""
        H, W = LoD._lod_as_hw(image_size)
        K = np.asarray(K, dtype=np.float64)
        if device == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("GPU rendering requested but CUDA is unavailable")
            dev = "cuda:0"
        else:
            dev = "cpu"
        self._renderer = Renderer.from_arrays(
            self.vertices_abs,
            self.faces,
            K,
            (H, W),
            labels=self.labels,
            building_ids=self.get_building_ids(),
            device=dev,
            minibatch_size=self._batch_size,
        )
        self._render_key = (K.tobytes(), H, W, device, self.num_vertices, self.num_faces)
        
        if getattr(self, '_ignored_buildings', None):
            self._renderer.ignore_buildings(self._ignored_buildings)
            
        return self._renderer

    def init_renderer(
        self,
        K: np.ndarray,
        image_size: ImageSize,
        device: Device = "gpu",
    ) -> Renderer:
        """Create the renderer once for fixed intrinsics / resolution."""
        return self._init_renderer(K, image_size, device)

    def _ensure_renderer(
        self,
        K: np.ndarray,
        image_size: ImageSize,
        device: Device = "gpu",
    ) -> Renderer:
        """Return the cached renderer, rebuilding only if K/H/W/device/mesh changed."""
        H, W = LoD._lod_as_hw(image_size)
        K = np.asarray(K, dtype=np.float64)
        key = (K.tobytes(), H, W, device, self.num_vertices, self.num_faces)
        if self._renderer is None or self._render_key != key:
            return self._init_renderer(K, image_size, device)
        return self._renderer

    @property
    def renderer(self) -> Renderer:
        if self._renderer is None:
            raise RuntimeError(
                "Renderer not initialized — call crop_to_camera_pose() or init_renderer(K, image_size)"
            )
        return self._renderer

    @property
    def vertices_abs(self) -> np.ndarray:
        """World coordinates (local storage + UTM offset)."""
        return self.vertices + self.utm_offset

    def face_normals(self) -> np.ndarray:
        """Per-face unit normals in world coordinates. Shape (F, 3)."""
        v = self.vertices_abs
        v0, v1, v2 = v[self.faces[:, 0]], v[self.faces[:, 1]], v[self.faces[:, 2]]
        raw = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        return raw / norms

    def face_centroids(self) -> np.ndarray:
        """Per-face centroids in world coordinates. Shape (F, 3)."""
        if self._centroids_abs is None:
            v = self.vertices_abs
            self._centroids_abs = (
                v[self.faces[:, 0]] + v[self.faces[:, 1]] + v[self.faces[:, 2]]
            ) / 3.0
        return self._centroids_abs

    def get_building_ids(
        self,
        *,
        merge_parts: bool = True,
        merge_dist: float = 2.0,
    ) -> np.ndarray:
        """Per-face building IDs (edge-CC + optional XY proximity merge). Cached."""
        if self._building_ids is None:
            raw = LoD._lod_compute_building_ids(self.faces, self.num_vertices)
            n_comps = int(raw.max()) + 1 if len(raw) else 0
            if merge_parts and n_comps <= 50_000:
                self._building_ids = LoD._lod_merge_building_parts(
                    self.faces,
                    self.vertices_abs,
                    raw,
                    self.labels,
                    dist_thresh=merge_dist,
                )
            else:
                self._building_ids = raw
        return self._building_ids

    def ignore_buildings(self, bids: list[int]) -> None:
        """Temporarily drop specific buildings from rendering."""
        self._ignored_buildings = bids
        if self._renderer is not None:
            self._renderer.ignore_buildings(bids)

    @classmethod
    def from_npz(cls, npz_path: Union[str, Path]) -> "LoD":
        """Load an ``.npz`` mesh dump (vertices, faces, optional labels / utm_offset)."""
        npz_path = Path(npz_path)
        data = np.load(npz_path, allow_pickle=True)
        inst = object.__new__(cls)
        inst.path = npz_path.resolve()
        inst.vertices = data["vertices"].astype(np.float64)
        inst.faces = data["faces"].astype(np.int64)
        raw_lab = data.get("labels")
        if raw_lab is None or len(raw_lab) == 0:
            inst.labels = np.full((len(inst.faces),), "building", dtype=object)
        else:
            inst.labels = np.asarray(raw_lab)
        inst.utm_offset = (
            data["utm_offset"].astype(np.float64)
            if "utm_offset" in data
            else np.zeros(3, dtype=np.float64)
        )
        inst.polygon_ids = None
        inst._building_ids = None
        inst._centroids_abs = None
        inst._renderer = None
        inst._render_key = None
        inst._batch_size = 52
        return inst

    @classmethod
    def from_sequence(cls, sequence_dir: Union[str, Path], lod_level: str = "2") -> "LoD":
        """Load ``lod{lod_level}.npz`` from a sequence directory."""
        seq_dir = Path(sequence_dir)
        npz_path = seq_dir / f"lod{lod_level}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"LoD{lod_level} mesh not found: {npz_path}")
        return cls.from_npz(npz_path)

    @classmethod
    def from_arrays(
        cls,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        labels: Optional[np.ndarray] = None,
        utm_offset: Optional[np.ndarray] = None,
        polygon_ids: Optional[np.ndarray] = None,
        source_path: Optional[Union[str, Path]] = None,
        K: Optional[np.ndarray] = None,
        image_size: Optional[ImageSize] = None,
        device: Device = "gpu",
        renderer_minibatch: int = 52,
        ignored_buildings: Optional[list[int]] = None,
    ) -> "LoD":
        """Build LoD from arrays (e.g. after :meth:`crop_to_camera_pose`)."""
        inst = object.__new__(cls)
        inst.path = Path(source_path).resolve() if source_path else Path()
        inst.vertices = np.asarray(vertices, dtype=np.float64)
        inst.faces = np.asarray(faces, dtype=np.int64)
        inst.utm_offset = (
            np.asarray(utm_offset, dtype=np.float64)
            if utm_offset is not None
            else np.zeros(3, dtype=np.float64)
        )
        if labels is not None and len(labels) > 0:
            inst.labels = np.asarray(labels)
        elif len(inst.faces) > 0:
            inst.labels = np.full((len(inst.faces),), "building", dtype=object)
        else:
            inst.labels = np.array([], dtype=object)
        inst.polygon_ids = (
            np.asarray(polygon_ids) if polygon_ids is not None else None
        )
        inst._building_ids = None
        inst._centroids_abs = None
        inst._renderer = None
        inst._render_key = None
        inst._batch_size = int(renderer_minibatch)
        if ignored_buildings is not None:
            inst._ignored_buildings = ignored_buildings

        if K is not None and image_size is not None:
            inst._init_renderer(K, image_size, device)

        return inst

    def crop_to_camera_pose(
        self,
        pose: Any,
        K: np.ndarray,
        image_size: ImageSize,
        *,
        margin: float = 1.5,
        use_frustum: bool = False,
        along_view_extend_m: Optional[float] = None,
        footprint_extend_m: float = 0.0,
        device: Device = "gpu",
    ) -> "LoD":
        """Crop mesh around a camera pose and initialize the renderer.

        This is the **only** mesh-crop entry point.  Radius is derived from altitude,
        intrinsics, and image size (FOV footprint on the ground), scaled by ``margin``,
        plus ``footprint_extend_m`` for GPS error / particle-filter search slack.
        Building-aware: any face inside the disc pulls in its whole building cluster.

        Args:
            pose: w2c (4×4), :class:`~utils.pose.CameraPose`, or compatible object.
            K: (3, 3) intrinsics at ``image_size``.
            image_size: (H, W) render resolution.
            margin: multiplicative safety factor on the auto footprint radius.
            use_frustum: also require face centroids to project inside the image.
            along_view_extend_m: cap forward extent along the horizontal view axis (m).
            footprint_extend_m: extra meters added to the crop radius.
            device: ``'gpu'`` or ``'cpu'`` for the renderer created on the crop.
        """
        H, W = LoD._lod_as_hw(image_size)
        v, f, fi = LoD._lod_crop_vertices_faces(
            self.vertices_abs,
            self.faces,
            pose,
            K,
            (H, W),
            margin=margin,
            use_frustum=use_frustum,
            along_view_extend_m=along_view_extend_m,
            footprint_extend_m=footprint_extend_m,
        )
        if len(f) == 0:
            return LoD.from_arrays(
                v,
                f,
                labels=np.array([], dtype=object),
                utm_offset=np.zeros(3),
                polygon_ids=None,
                source_path=self.path,
                ignored_buildings=getattr(self, '_ignored_buildings', None),
            )
        labels_new = (
            self.labels[fi] if len(self.labels) == len(self.faces) else None
        )
        poly_new = (
            self.polygon_ids[fi]
            if self.polygon_ids is not None and len(self.polygon_ids) == len(self.faces)
            else None
        )
        return LoD.from_arrays(
            v,
            f,
            labels=labels_new,
            utm_offset=np.zeros(3),
            polygon_ids=poly_new,
            source_path=self.path,
            K=K,
            image_size=image_size,
            device=device,
            renderer_minibatch=self._batch_size,
            ignored_buildings=getattr(self, '_ignored_buildings', None),
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save mesh as ``.obj`` or ``.ply`` (from suffix)."""
        path = Path(path)
        suf = path.suffix.lower()
        if suf not in {".obj", ".ply"}:
            raise ValueError(f"Unsupported extension {suf}; use .obj or .ply")
        if len(self.faces) == 0:
            raise ValueError("Cannot save empty mesh")
        mesh = trimesh.Trimesh(
            vertices=np.asarray(self.vertices, dtype=np.float64),
            faces=np.asarray(self.faces, dtype=np.int64),
        )
        mesh.export(str(path))

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])

    def render_face_id_map(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: ImageSize,
    ) -> np.ndarray:
        """Render per-pixel face ID map using GPU rasterization (nvdiffrast).

        Returns:
            face_ids: (H, W) int32, -1 = background.
        """
        H, W = LoD._lod_as_hw(image_size)
        renderer = self._ensure_renderer(K, image_size, device="gpu")
        w2c_t = torch.tensor(np.asarray(w2c, dtype=np.float32))
        face_ids, _ = renderer.render_face_ids_and_depth_batch(LoD._lod_w2c_batch(w2c_t))
        return face_ids[0]

    def render_face_ids_and_depth(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: ImageSize,
    ) -> tuple:
        """Render face ID map and metric depth in a single GPU pass.

        Returns:
            face_ids: (H, W) int32, -1 = background.
            depth:    (H, W) float32 in metres, 0 = background.
        """
        H, W = LoD._lod_as_hw(image_size)
        renderer = self._ensure_renderer(K, image_size, device="gpu")
        w2c_t = torch.tensor(np.asarray(w2c, dtype=np.float32))
        fids, depth = renderer.render_face_ids_and_depth_batch(LoD._lod_w2c_batch(w2c_t))
        return fids[0], depth[0]

    def render_structural_edges_and_depth(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: ImageSize,
        *,
        normal_cos_thresh: float = 0.999,
        depth_disc_thresh: float = 0.05,
    ) -> tuple:
        """Render structural edge map and metric depth in a single GPU pass.

        Returns:
            edge_map: (H, W) float32 in {0, 1}.
            depth:    (H, W) float32 in metres, 0 = background.
        """
        H, W = LoD._lod_as_hw(image_size)
        renderer = self._ensure_renderer(K, image_size, device="gpu")
        w2c_t = torch.tensor(np.asarray(w2c, dtype=np.float32))
        edges, depth = renderer.render_structural_edges_and_depth_batch(
            LoD._lod_w2c_batch(w2c_t),
            normal_cos_thresh=normal_cos_thresh,
            depth_disc_thresh=depth_disc_thresh,
        )
        return edges[0], depth[0]

    def render_mask(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: ImageSize,
        *,
        device: Device = "gpu",
        near_plane: float = 0.5,
    ) -> np.ndarray:
        """Render binary building mask.

        Args:
            device: ``'gpu'`` uses nvdiffrast (CUDA tensor path);
                    ``'cpu'`` forces the CPU rasterisation path even when CUDA
                    is available.

        Returns:
            mask: (H, W) uint8 in {0, 255}.
        """
        H, W = LoD._lod_as_hw(image_size)
        w2c = np.asarray(w2c, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        renderer = self._ensure_renderer(K, image_size, device=device)
        w2c_t = torch.tensor(np.asarray(w2c, dtype=np.float32))
        mask_t = renderer.render_mask_batch(LoD._lod_w2c_batch(w2c_t))
        return (mask_t.squeeze().cpu().numpy() * 255).astype(np.uint8)

    def render_normals(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: ImageSize,
        *,
        device: Device = "gpu",
        near_plane: float = 0.5,
        cpu_max_side: Optional[int] = 1536,
    ) -> np.ndarray:
        """Render per-pixel unit normals in camera coordinates.

        Returns:
          normals: (H, W, 3) float32. Uncovered pixels are zero.
        """
        H, W = LoD._lod_as_hw(image_size)
        w2c = np.asarray(w2c, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        renderer = self._ensure_renderer(K, image_size, device=device)
        w2c_t = torch.tensor(np.asarray(w2c, dtype=np.float32))
        normals_t = renderer.render_normals_batch(LoD._lod_w2c_batch(w2c_t))
        return normals_t[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)

    def render_depth(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: ImageSize,
        *,
        device: Device = "gpu",
        near_plane: float = 0.5,
        cpu_max_side: Optional[int] = 1536,
    ) -> np.ndarray:
        """Render metric depth map (metres, 0 where no mesh).

        Args:
            device: ``'gpu'`` uses nvdiffrast; ``'cpu'`` forces CPU path.

        Returns:
            depth: (H, W) float32 in metres.
        """
        H, W = LoD._lod_as_hw(image_size)
        w2c = np.asarray(w2c, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        renderer = self._ensure_renderer(K, image_size, device=device)
        w2c_t = torch.tensor(np.asarray(w2c, dtype=np.float32))
        depth_t = renderer.render_depth_batch(LoD._lod_w2c_batch(w2c_t))
        return depth_t[0, 0].cpu().numpy().astype(np.float32)

    def render_depth_normals_batch(
        self,
        w2c_list: list,
        K: np.ndarray,
        image_size: "ImageSize",
    ) -> tuple:
        """Batch-render depth + camera-space normals for N poses in one GPU call.

        Args:
            w2c_list: list of N (4, 4) float64 ndarray world-to-camera matrices.
            K: (3, 3) intrinsic matrix.
            image_size: (H, W) or single int.

        Returns:
            depths:  list of N (H, W) float32 numpy arrays, metric depth (0=bg).
            normals: list of N (H, W, 3) float32 numpy arrays, camera-space normals.
        """
        H, W = LoD._lod_as_hw(image_size)
        K = np.asarray(K, dtype=np.float64)
        dev_name = "gpu" if torch.cuda.is_available() else "cpu"
        renderer = self._ensure_renderer(K, image_size, device=dev_name)
        dev = renderer.dev
        w2c_batch = torch.stack([
            torch.tensor(np.asarray(w, dtype=np.float32), device=dev)
            for w in w2c_list
        ])
        depth_t, normals_t = renderer.render_depth_normals_batch(w2c_batch)
        depth_np = depth_t.cpu().numpy().astype(np.float32)
        normal_np = normals_t.cpu().numpy().astype(np.float32)
        depths_np = [depth_np[i] for i in range(len(w2c_list))]
        normals_np = [normal_np[i] for i in range(len(w2c_list))]
        return depths_np, normals_np

    def render_masks_batch(
        self,
        w2c_batch: "np.ndarray | torch.Tensor | list",
        K: np.ndarray,
        image_size: "ImageSize",
        *,
        device: Device = "gpu",
    ) -> tuple:
        """Batch-render binary silhouettes (+ optional depth) via the cached renderer."""
        K = np.asarray(K, dtype=np.float64)
        renderer = self._ensure_renderer(K, image_size, device=device)
        if isinstance(w2c_batch, list):
            w2c_t = torch.stack([
                torch.tensor(np.asarray(w, dtype=np.float32)) for w in w2c_batch
            ])
        else:
            w2c_t = Renderer._as_w2c_batch(w2c_batch)
        w2c_t = w2c_t.to(renderer.dev).float()
        K_t = torch.tensor(K, dtype=torch.float32, device=renderer.dev)
        return renderer.render_masks_batch(w2c_t, K_t)

    def render_depth_maps_batch(
        self,
        w2c_batch: "np.ndarray | torch.Tensor | list",
        K: np.ndarray,
        image_size: "ImageSize",
        *,
        device: Device = "gpu",
    ) -> torch.Tensor:
        """Batch-render metric depth maps, shape (B, H, W)."""
        K = np.asarray(K, dtype=np.float64)
        renderer = self._ensure_renderer(K, image_size, device=device)
        w2c_t = (
            torch.stack([torch.tensor(np.asarray(w, dtype=np.float32)) for w in w2c_batch])
            if isinstance(w2c_batch, list) else Renderer._as_w2c_batch(w2c_batch)
        )
        w2c_t = w2c_t.to(renderer.dev).float()
        return renderer.render_depth_batch(w2c_t)[:, 0]

    def render_edges_batch(
        self,
        w2c_batch: "np.ndarray | torch.Tensor | list",
        K: np.ndarray,
        image_size: "ImageSize",
        *,
        device: Device = "gpu",
        sobel_scale: float = 8.0,
    ) -> list:
        """Batch-render structural edge maps (Sobel on rasterized normals).

        Returns:
            list of B (H, W) float32 arrays in [0, 1].
        """
        K = np.asarray(K, dtype=np.float64)
        renderer = self._ensure_renderer(K, image_size, device=device)
        w2c_t = (
            torch.stack([torch.tensor(np.asarray(w, dtype=np.float32)) for w in w2c_batch])
            if isinstance(w2c_batch, list) else Renderer._as_w2c_batch(w2c_batch)
        )
        w2c_t = w2c_t.to(renderer.dev).float()
        with torch.no_grad():
            out = renderer.render_edges_batch(w2c_t, sobel_scale=sobel_scale)
        return [out[i, 0].cpu().numpy().astype(np.float32) for i in range(out.shape[0])]

    def render_shaded(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: ImageSize,
        *,
        device: Device = "gpu",
        near_plane: float = 0.5,
        light_dir_cam: Optional[np.ndarray] = None,
        ambient: float = 0.20,
        base_color: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render Phong-shaded RGB image. Delegates to :class:`~utils.renderer.Renderer`.

        Returns:
            shaded: (H, W, 3) uint8 RGB image.  Background is black.
        """
        H, W = LoD._lod_as_hw(image_size)
        renderer = self._ensure_renderer(K, image_size, device=device)
        return renderer.render_shaded(
            np.asarray(w2c, dtype=np.float64),
            device=device,
            light_dir_cam=light_dir_cam,
            ambient=ambient,
            base_color=base_color,
        )

    # ────────── Semantic & instance segmentation ──────────

    def render_segmentation(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: "ImageSize",
        *,
        mode: str = "semantic",
        face_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render GT building segmentation. Delegates to :class:`~utils.renderer.Renderer`.

        Returns:
            color_map: (H, W, 3) uint8 — colored segmentation.
            id_map:    (H, W) int32 — class IDs (semantic) or building IDs (instance),
                       -1 for background.
        """
        H, W = LoD._lod_as_hw(image_size)
        renderer = self._ensure_renderer(K, image_size, device="gpu")
        return renderer.render_segmentation(
            np.asarray(w2c, dtype=np.float64),
            mode=mode,
            face_ids=face_ids,
        )

    def get_plane_segmentation(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: "ImageSize",
        *,
        device: "Device" = "gpu",
        normal_cos_thresh: float = 0.995,
        depth_rel_thresh: float = 0.02,
        min_pixels: int = 50,
        depth: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        gpu_render_time_s: Optional[list] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Render depth + normals, then segment into planes via region-growing.

        Two pixels are grouped if:
          1. Normal cosine similarity >= normal_cos_thresh
          2. Relative depth difference < depth_rel_thresh

        Uses vectorized edge computation + scipy sparse connected components
        for speed (~100× faster than pure-Python BFS).

        Args:
            w2c: 4x4 world-to-camera matrix
            K: 3x3 intrinsics
            image_size: (H, W) or single int for square
            device: "gpu" or "cpu"
            normal_cos_thresh: cosine threshold (0.995 ~ 5.7 deg)
            depth_rel_thresh: max relative depth step between neighbors (0.02 = 2%)
            min_pixels: minimum connected-component size to keep
            depth: optional pre-rendered depth (H, W); if None, will render
            normals: optional pre-rendered normals (H, W, 3); if None, will render

        Returns:
            depth: (H, W) float32 rendered depth
            normals: (H, W, 3) float32 camera-space normals
            label_map: (H, W) int32 plane IDs (-1 = background/discarded)
            segments: list of dicts per plane with keys:
                'plane_id': int
                'normal_cam': (3,) mean unit normal in camera space
                'pixel_count': int
                'pixel_indices': (N, 2) int (row, col)
        """

        H, W = LoD._lod_as_hw(image_size)
        w2c = np.asarray(w2c, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)

        if depth is None or normals is None:
            renderer = self._ensure_renderer(K, (H, W), device=device)
            w2c_t = torch.tensor(w2c.astype(np.float32), device=renderer.dev).unsqueeze(0)
            with GpuTimer() as gt:
                depth_t, normals_t = renderer.render_depth_normals_batch(w2c_t)
            if gpu_render_time_s is not None:
                gpu_render_time_s.append(gt.elapsed_s)
            if depth is None:
                depth = depth_t[0].detach().cpu().numpy().astype(np.float32)
            if normals is None:
                normals = normals_t[0].detach().cpu().numpy().astype(np.float32)

        valid = (depth > 0)
        nlen_sq = np.einsum('ijk,ijk->ij', normals, normals)
        valid &= (nlen_sq > 0.25)
        safe_nlen_sq = np.maximum(nlen_sq, 1e-16)
        normals_n = normals * (1.0 / np.sqrt(safe_nlen_sq))[..., None]

        N = H * W
        flat_valid = valid.ravel()

        # Map valid pixels to a compact index space [0, n_valid)
        # so the sparse matrix is n_valid × n_valid instead of N × N.
        valid_indices = np.where(flat_valid)[0]  # flat indices of valid pixels
        n_valid = len(valid_indices)
        # full→compact mapping (-1 for invalid)
        compact_id = np.full(N, -1, dtype=np.int32)
        compact_id[valid_indices] = np.arange(n_valid, dtype=np.int32)

        # ----- Vectorized edge computation (flat indexing, no repeat/tile) ----
        # Flat pixel indices: horizontal pairs (i, i+1) skipping column W-1
        all_px = np.arange(N, dtype=np.int32)
        col_idx = all_px % W
        # Horizontal: every pixel except last column
        h_mask = col_idx < (W - 1)
        h_a = all_px[h_mask]
        h_b = h_a + 1
        # Vertical: every pixel except last row
        v_mask = all_px < (N - W)
        v_a = all_px[v_mask]
        v_b = v_a + W

        ea = np.concatenate([h_a, v_a])
        eb = np.concatenate([h_b, v_b])

        # Both ends must be valid
        both_valid = flat_valid[ea] & flat_valid[eb]
        ea = ea[both_valid]
        eb = eb[both_valid]

        # Normal cosine similarity + depth check (fused)
        normals_flat = normals_n.reshape(N, 3)
        dots = np.einsum('ij,ij->i', normals_flat[ea], normals_flat[eb])
        depth_flat = depth.ravel()
        da = depth_flat[ea]
        db = depth_flat[eb]
        keep = (dots >= normal_cos_thresh) & (np.abs(da - db) < depth_rel_thresh * np.maximum(da, db))
        ea = ea[keep]
        eb = eb[keep]

        # Remap to compact indices and build sparse adjacency
        ea_c = compact_id[ea]
        eb_c = compact_id[eb]
        data = np.ones(len(ea_c) * 2, dtype=np.int8)
        row = np.concatenate([ea_c, eb_c])
        col = np.concatenate([eb_c, ea_c])
        adj = csr_matrix((data, (row, col)), shape=(n_valid, n_valid))
        n_cc, cc_labels_compact = sparse_cc(adj, directed=False)

        # Map compact CC labels back to full image
        cc_labels = np.full(N, -1, dtype=cc_labels_compact.dtype)
        cc_labels[valid_indices] = cc_labels_compact

        # ----- Fast segment collection via bincount + argsort -----
        cc_flat = cc_labels.copy()
        cc_flat[~flat_valid] = -1

        # Count pixels per label and filter by min_pixels
        max_label = int(cc_flat.max()) + 1 if cc_flat.max() >= 0 else 0
        counts = np.bincount(cc_flat[cc_flat >= 0], minlength=max_label)
        big_labels = np.where(counts >= min_pixels)[0]

        if len(big_labels) == 0:
            label_map = np.full((H, W), -1, dtype=np.int32)
            return depth, normals, label_map, []

        # Sort all valid pixels by their CC label for O(N) grouping
        valid_mask = cc_flat >= 0
        valid_idx = np.where(valid_mask)[0]
        valid_labels = cc_flat[valid_idx]
        order = np.argsort(valid_labels, kind='mergesort')
        sorted_labels = valid_labels[order]
        sorted_idx = valid_idx[order]

        # Find boundaries between label groups
        breaks = np.concatenate([[0], np.where(np.diff(sorted_labels) != 0)[0] + 1, [len(sorted_labels)]])
        label_at_break = sorted_labels[breaks[:-1]]

        # Build a set of big labels for fast lookup
        big_set = np.zeros(max_label, dtype=bool)
        big_set[big_labels] = True

        label_map = np.full((H, W), -1, dtype=np.int32)
        normals_flat3 = normals_n.reshape(N, 3)
        segments = []
        plane_id = 0

        for i in range(len(breaks) - 1):
            cc_id = label_at_break[i]
            if not big_set[cc_id]:
                continue
            pix = sorted_idx[breaks[i]:breaks[i + 1]]
            rows = (pix // W).astype(np.int32)
            cols = (pix % W).astype(np.int32)
            avg_n = normals_flat3[pix].mean(axis=0)
            avg_n /= np.linalg.norm(avg_n) + 1e-12
            label_map[rows, cols] = plane_id
            segments.append({
                'plane_id': plane_id,
                'normal_cam': avg_n.astype(np.float32),
                'pixel_count': len(pix),
                'pixel_indices': np.stack([rows, cols], axis=1),
            })
            plane_id += 1

        return depth, normals, label_map, segments

    def create_3d_planes_from_2d(
        self,
        segments: List[Dict],
        depth: np.ndarray,
        K: np.ndarray,
        w2c: np.ndarray,
        *,
        min_3d_area: float = 1.0,
        min_extent: float = 0.3,
        max_aspect_ratio: float = 25.0,
    ) -> List[Dict]:
        """Convert 2D plane segments into 3D planes with robust filtering.

        Back-projects each segment's pixels to 3D, fits a plane, and filters
        out tiny, thin, or degenerate planes.

        Filtering criteria:
          - min_3d_area: minimum convex-hull-like area in 3D (m^2)
          - min_extent: minimum extent along the two principal axes (m)
          - max_aspect_ratio: max ratio of major to minor extent;
            rejects extreme slivers (e.g., 25 = 25:1 elongation)

        Args:
            segments: list of segment dicts from get_plane_segmentation
            depth: (H, W) float32 depth map
            K: 3x3 intrinsics
            w2c: 4x4 world-to-camera

        Returns:
            planes: list of dicts with keys:
                'plane_id': int (from segment)
                'normal_cam': (3,) unit normal in camera space
                'normal_world': (3,) unit normal in world space
                'centroid_cam': (3,) centroid in camera space
                'centroid_world': (3,) centroid in world space
                'd_world': float, plane offset (n . x = d)
                'area_3d': float, approximate 3D area (m^2)
                'extent_major': float, extent along 1st principal axis (m)
                'extent_minor': float, extent along 2nd principal axis (m)
                'pixel_count': int
                'pixel_indices': (N, 2) int (row, col)
        """
        w2c = np.asarray(w2c, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        R_w2c = w2c[:3, :3]
        t_w2c = w2c[:3, 3]
        R_c2w = R_w2c.T

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        planes = []
        n_rejected = {'tiny': 0, 'thin_extent': 0, 'sliver': 0}

        for seg in segments:
            rows = seg['pixel_indices'][:, 0]
            cols = seg['pixel_indices'][:, 1]
            z = depth[rows, cols].astype(np.float64)

            # Back-project to camera space
            x = (cols - cx) * z / fx
            y = (rows - cy) * z / fy
            pts_cam = np.stack([x, y, z], axis=1)  # (N, 3)

            # Transform to world space
            pts_world = (R_c2w @ (pts_cam - t_w2c).T).T  # (N, 3)

            # Centroid
            centroid_cam = pts_cam.mean(axis=0)
            centroid_world = pts_world.mean(axis=0)

            # Plane normal in world (area-weighted from segment)
            n_cam = seg['normal_cam'].astype(np.float64)
            n_world = R_c2w @ n_cam
            n_world /= np.linalg.norm(n_world) + 1e-12

            # Plane offset
            d_world = float(np.dot(n_world, centroid_world))

            # --- Robust filtering via SVD of centered points ---
            centered = pts_world - centroid_world
            if len(centered) < 3:
                n_rejected['tiny'] += 1
                continue

            # SVD to get principal extents
            _, S, Vt = np.linalg.svd(centered, full_matrices=False)
            # S values are proportional to spread along each axis
            # Scale to approximate half-extents: S[i] / sqrt(N)
            n_pts = len(centered)
            extents = S / np.sqrt(n_pts)  # approximate half-extents

            extent_major = 2.0 * extents[0]  # full extent along 1st axis
            extent_minor = 2.0 * extents[1]  # full extent along 2nd axis
            thickness = 2.0 * extents[2] if len(extents) > 2 else 0.0

            # Approximate 3D area as extent_major * extent_minor (bounding rectangle)
            area_3d = extent_major * extent_minor

            # Filter: minimum area
            if area_3d < min_3d_area:
                n_rejected['tiny'] += 1
                continue

            # Filter: minimum extent (reject very narrow planes)
            if extent_minor < min_extent:
                n_rejected['thin_extent'] += 1
                continue

            # Filter: reject extreme slivers (major/minor aspect ratio)
            if extent_minor > 1e-6 and extent_major / extent_minor > max_aspect_ratio:
                n_rejected['sliver'] += 1
                continue

            planes.append({
                'plane_id': seg['plane_id'],
                'normal_cam': n_cam.astype(np.float32),
                'normal_world': n_world.astype(np.float32),
                'centroid_cam': centroid_cam.astype(np.float32),
                'centroid_world': centroid_world.astype(np.float32),
                'd_world': d_world,
                'area_3d': float(area_3d),
                'extent_major': float(extent_major),
                'extent_minor': float(extent_minor),
                'pixel_count': seg['pixel_count'],
                'pixel_indices': seg['pixel_indices'],
            })

        print(f"  create_3d_planes_from_2d: {len(planes)} kept, "
              f"rejected {n_rejected['tiny']} tiny, "
              f"{n_rejected['thin_extent']} thin-extent, "
              f"{n_rejected['sliver']} slivers")
        return planes

    def get_plane_polygons_2d(
        self,
        w2c: "np.ndarray | torch.Tensor",
        K: np.ndarray,
        image_size: "ImageSize",
        *,
        device: "Device" = "gpu",
        normal_cos_thresh: float = 0.98,
        depth_rel_thresh: float = 0.02,
        min_pixels: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Extract 2D plane polygons from rendered normal + depth maps.

        Operates in image space — no 3D mesh BFS.  Each plane is a spatially-
        connected region of pixels whose normals and depths are mutually
        consistent.

        This method delegates to :meth:`~utils.renderer.Renderer.render_planes`
        which is **fully differentiable** on GPU (nvdiffrast).  The segmentation
        (connected components) is performed on detached arrays outside the
        autograd graph, but the returned ``normal_cam`` and ``d_cam`` tensors
        carry gradients back to ``w2c`` through the rendered normal/depth maps.

        Args:
            w2c:               4×4 world-to-camera matrix (numpy or torch.Tensor).
                               Pass a ``requires_grad=True`` tensor to enable
                               gradient flow.
            K:                 3×3 camera intrinsics (numpy).
            image_size:        (H, W) or a single int for square images.
            device:            ``'gpu'`` (nvdiffrast, differentiable) or ``'cpu'``.
            normal_cos_thresh: minimum cosine similarity to link two adjacent
                               pixels (0.98 ≈ 11°, 0.995 ≈ 5.7°).
            depth_rel_thresh:  maximum relative depth step between neighbours
                               (0.02 = 2 %).
            min_pixels:        minimum segment size to keep.

        Returns:
            depth:    (H, W) float32 numpy array — metric depth in metres.
            normals:  (H, W, 3) float32 numpy array — camera-space unit normals.
            planes:   list of dicts, one per plane, with keys:

                ``'plane_id'``      – int, sequential ID.

                ``'normal_cam'``    – (3,) float32 **torch.Tensor**, mean unit
                                      normal in camera space.
                                      **Differentiable** w.r.t. ``w2c`` on GPU.

                ``'d_cam'``         – scalar float32 **torch.Tensor**, plane
                                      offset (n · x = d) in camera space.
                                      **Differentiable** w.r.t. ``w2c`` on GPU.

                ``'pixel_count'``   – int.

                ``'pixel_indices'`` – (N, 2) int numpy array of (row, col).

                ``'polygon_2d'``    – (K, 2) float32 numpy array — convex-hull
                                      vertices in (col, row) = (x, y) image
                                      coordinates, or ``None`` if < 3 pixels.
        """

        H, W = LoD._lod_as_hw(image_size)
        K = np.asarray(K, dtype=np.float64)
        renderer = self._ensure_renderer(K, image_size, device=device)

        if isinstance(w2c, np.ndarray):
            w2c_t = torch.tensor(w2c.astype(np.float32))
        else:
            w2c_t = w2c  # caller may have requires_grad=True

        w2c_b = LoD._lod_w2c_batch(w2c_t)

        result = renderer.render_planes_batch(
            w2c_b,
            normal_cos_thresh=normal_cos_thresh,
            depth_rel_thresh=depth_rel_thresh,
            min_pixels=min_pixels,
        )[0]

        # Extract numpy depth + normals for convenience
        depth_t = renderer.render_depth_batch(w2c_b)
        depth_np = depth_t[0, 0].detach().cpu().numpy().astype(np.float32)
        normals_t = renderer.render_normals_batch(w2c_b)
        normals_np = normals_t[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

        # Build planes list in the expected format
        planes = []
        n_normal = result['normal_cam']   # (P, 3) diff tensor
        n_d      = result['d_cam']        # (P,) diff tensor
        masks    = result['masks']        # (P, H, W) no grad
        polygons = result['polygon_2d']   # list[ndarray|None]

        for idx in range(n_normal.shape[0]):
            m_np = masks[idx].cpu().numpy()  # (H, W)
            ys, xs = np.where(m_np > 0)
            planes.append({
                'plane_id':     idx,
                'normal_cam':   result['normal_cam'][idx],   # torch, diff
                'd_cam':        result['d_cam'][idx],         # torch, diff
                'pixel_count':  result['pixel_count'][idx],
                'pixel_indices': np.stack([ys, xs], axis=1) if len(ys) else np.zeros((0, 2), dtype=np.int32),
                'polygon_2d':   polygons[idx],
            })

        return depth_np, normals_np, planes


    def get_3d_planes_from_3d(
        self,
        *,
        min_area: float = 1.0,
        normal_threshold: float = 0.9998,
        d_threshold: float = 0.5,
    ) -> List[Dict]:
        """Extract planes from mesh geometry (no rendering, viewpoint-independent).

        Groups coplanar adjacent triangles via BFS on the mesh adjacency graph.
        Edge-adjacency is built with vectorised sort+duplicate detection;
        grouping uses seed-comparison BFS (each face compared to its group's
        seed normal/d, matching ``mesh_to_planes`` semantics exactly).

        For large meshes, crop first via ``crop_to_camera_pose`` with a
        generous margin to reduce the face count before calling this method.

        Args:
            min_area: minimum total triangle area (m²) to keep a plane.
            normal_threshold: cosine similarity for coplanarity (default 0.9998 ≈ 1.1°).
            d_threshold: max |Δd| in metres between adjacent faces.

        Returns:
            List of dicts with keys:
                'normal': (3,) float64 area-weighted unit normal (world frame)
                'd': float, plane offset (n · x = d)
                'centroid': (3,) float64 area-weighted centroid (world frame)
                'area': float, total triangle area (m²)
                'label': majority per-face label
                'face_indices': ndarray of int, mesh face indices in this plane
        """
        vertices = np.asarray(self.vertices_abs, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.int64)
        labels = self.labels
        M = len(faces)
        if M == 0:
            return []

        # ── Per-triangle normals, areas, d values (vectorised) ──
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        area_2x = np.linalg.norm(cross, axis=1)
        tri_areas = area_2x * 0.5
        valid = area_2x > 1e-12
        safe_norms = np.where(area_2x[:, None] > 1e-12, area_2x[:, None], 1.0)
        face_normals = cross / safe_norms
        tri_centroids = (v0 + v1 + v2) / 3.0
        d_vals = np.einsum("ij,ij->i", face_normals, tri_centroids)

        # ── Vectorised edge → face adjacency (CSR for O(1) neighbor lookup) ──
        e_all = np.concatenate([
            faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]
        ], axis=0)                                          # (3M, 2)
        e_all = np.sort(e_all, axis=1)
        face_idx = np.tile(np.arange(M, dtype=np.int64), 3)

        N_verts = len(vertices)
        edge_ids = e_all[:, 0].astype(np.int64) * N_verts + e_all[:, 1]

        order = np.argsort(edge_ids, kind='mergesort')
        sorted_ids = edge_ids[order]
        sorted_faces = face_idx[order]

        # Find runs of equal edge_ids; keep only manifold edges (exactly 2 faces)
        run_starts = np.concatenate([[0], np.where(np.diff(sorted_ids) != 0)[0] + 1])
        run_lengths = np.diff(np.concatenate([run_starts, [len(sorted_ids)]]))
        manifold = run_lengths == 2
        m_starts = run_starts[manifold]
        fa = sorted_faces[m_starts]
        fb = sorted_faces[m_starts + 1]

        # Build CSR neighbor structure: for face i, neighbors are
        # nb_dst[nb_ptr[i]:nb_ptr[i+1]]
        src = np.concatenate([fa, fb])
        dst = np.concatenate([fb, fa])
        sort_ord = np.argsort(src, kind='mergesort')
        src_sorted = src[sort_ord]
        nb_dst = dst[sort_ord]
        nb_ptr = np.searchsorted(src_sorted, np.arange(M + 1))

        # ── BFS grouping (seed-comparison, matching mesh_to_planes) ──
        visited = np.zeros(M, dtype=bool)
        groups: List[np.ndarray] = []

        # Seed BFS from largest faces first so that stable, well-defined faces
        # become group references — tiny triangulation fragments are absorbed into
        # existing groups rather than spawning their own noisy plane.
        seed_order = np.argsort(-tri_areas)  # descending area

        for start in seed_order:
            if visited[start]:
                continue
            visited[start] = True
            if not valid[start]:
                continue
            n_ref = face_normals[start]
            d_ref = d_vals[start]
            group = [start]
            qi = 0
            while qi < len(group):
                fi = group[qi]
                qi += 1
                for nb in nb_dst[nb_ptr[fi]:nb_ptr[fi + 1]]:
                    if visited[nb]:
                        continue
                    if not valid[nb]:
                        visited[nb] = True
                        continue
                    dot = np.dot(n_ref, face_normals[nb])
                    # Use abs(dot) so residual flipped faces on the same geometric
                    # plane are still grouped; sign-adjust d before comparing.
                    if (abs(dot) >= normal_threshold
                            and abs(np.sign(dot) * d_vals[nb] - d_ref) <= d_threshold):
                        visited[nb] = True
                        group.append(nb)
            groups.append(np.array(group, dtype=np.int64))

        # ── Per-group plane parameters ──
        result: List[Dict] = []
        for idx in groups:
            w = tri_areas[idx]
            total_area = float(w.sum())
            if total_area < min_area:
                continue
            w_norm = w / total_area
            avg_normal = (face_normals[idx] * w_norm[:, None]).sum(axis=0)
            nn = float(np.linalg.norm(avg_normal))
            if nn < 1e-8:
                continue
            avg_normal /= nn
            avg_centroid = (tri_centroids[idx] * w_norm[:, None]).sum(axis=0)
            d = float(np.dot(avg_normal, avg_centroid))

            if labels is not None and len(labels) == len(faces):
                grp_lbl = labels[idx]
                uq, cnt = np.unique(grp_lbl, return_counts=True)
                majority_label = uq[np.argmax(cnt)]
            else:
                majority_label = None

            result.append({
                "normal": avg_normal,
                "d": d,
                "centroid": avg_centroid,
                "area": total_area,
                "label": majority_label,
                "face_indices": idx,
            })

        return result

    # ────────── Semantic rendering (unified) ──────────

    def render_semantic(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: "ImageSize",
        *,
        label_source: str = "normals",
        device: "Device" = "gpu",
        face_ids: Optional[np.ndarray] = None,
        roof_nz_thresh: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render per-pixel semantic label map. Delegates to :class:`~utils.renderer.Renderer`.

        Returns:
            label_map: (H, W) uint8 — 0=bg, 1=roof, 2=wall, 3=other.
            color_map: (H, W, 3) uint8 — per-pixel colour.
        """
        H, W = LoD._lod_as_hw(image_size)
        renderer = self._ensure_renderer(K, image_size, device=device)
        return renderer.render_semantic(
            np.asarray(w2c, dtype=np.float64),
            label_source=label_source,
            device=device,
            face_ids=face_ids,
            roof_nz_thresh=roof_nz_thresh,
        )

    # ────────── Edge map & wireframe rendering ──────────

    def render_edges(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: "ImageSize",
        *,
        device: "Device" = "gpu",
        sobel_scale: float = 8.0,
    ) -> np.ndarray:
        """Render structural edge map from the LoD mesh.

        Delegates to :class:`~utils.renderer.Renderer` (expanded-mesh
        Sobel-on-flat-normals via nvdiffrast on GPU, or cv2 z-buffer + torch Sobel
        on CPU).  GPU/CPU selection follows the ``device`` parameter:
        ``'gpu'`` passes a CUDA tensor (requires CUDA); ``'cpu'`` forces the CPU
        rasterisation path even when CUDA is available.

        Args:
            sobel_scale: Sigmoid sensitivity (default 8.0; higher = fewer edges).

        Returns:
            edge_map: (H, W) float32 in [0, 1].
        """
        H, W = LoD._lod_as_hw(image_size)
        w2c = np.asarray(w2c, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)

        renderer = self._ensure_renderer(K, image_size, device=device)

        w2c_f = w2c.astype(np.float32)
        if w2c_f.shape == (3, 4):
            w2c_f = np.vstack([w2c_f, [0, 0, 0, 1]])
        w2c_t = torch.tensor(w2c_f, dtype=torch.float32)

        with torch.no_grad():
            out = renderer.render_edges_batch(LoD._lod_w2c_batch(w2c_t), sobel_scale=sobel_scale)
        return out[0, 0].cpu().numpy()

    def render_wireframe(
        self,
        w2c: np.ndarray,
        K: np.ndarray,
        image_size: "ImageSize",
        *,
        interval: float = 2.0,
        thickness: int = 1,
        color: Tuple[int, int, int] = (0, 255, 0),
        depth: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render wireframe. Delegates to :class:`~utils.renderer.Renderer`."""
        H, W = LoD._lod_as_hw(image_size)
        renderer = self._ensure_renderer(K, image_size, device="gpu")
        return renderer.render_wireframe(
            np.asarray(w2c, dtype=np.float64),
            interval=interval,
            thickness=thickness,
            color=color,
            depth=depth,
        )
