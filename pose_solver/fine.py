"""
method/fine.py
==============
Multi-scale particle filter for pose estimation.

Progressively refines pose through 3 stages at increasing resolution,
combining silhouette IoU and depth NCC scoring.

When CUDA is available, uses GPU batch rendering via :class:`~utils.lod.LoD`
(~10-20x speedup: all candidates rendered in a single batch call per iteration).
"""

import torch.nn.functional as F

import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict

import torch

from typing import Any

from utils.lod import LoD
from utils.pose import pose_to_w2c
from utils.metrics import compute_iou
from .edge_scoring import score_mask_edges_vs_dt
from .scoring import BatchAlignmentScorer

def _w2c_batch_from_euler(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    pitch: float,
    roll: float,
    device: torch.device | str,
) -> torch.Tensor:
    """Build (N, 4, 4) w2c matrices from batched position + ZXY Euler angles."""
    R_w2c, t_w2c = _batch_euler_to_w2c(positions, yaws, pitch, roll, device)
    n = positions.shape[0]
    w2c = torch.zeros(n, 4, 4, device=positions.device, dtype=torch.float32)
    w2c[:, :3, :3] = R_w2c
    w2c[:, :3, 3] = t_w2c
    w2c[:, 3, 3] = 1.0
    return w2c


# Default 3-stage configuration
DEFAULT_STAGES = [
    {
        'name': 'coarse',
        'mask_size': (112, 151),
        'sigma_trans': 30.0,
        'sigma_yaw': np.radians(15),
        'n_iter': 60,
        'n_beams': 4,
        'n_candidates': 120,
        'gamma': 0.15,
    },
    {
        'name': 'medium',
        'mask_size': (224, 301),
        'sigma_trans': 8.0,
        'sigma_yaw': np.radians(5),
        'n_iter': 40,
        'n_beams': 2,
        'n_candidates': 60,
        'gamma': 0.2,
    },
    {
        'name': 'fine',
        'mask_size': (448, 602),
        'sigma_trans': 2.0,
        'sigma_yaw': np.radians(2),
        'n_iter': 30,
        'n_beams': 2,
        'n_candidates': 52,
        'gamma': 0.3,
    },
]


# Fast stages: reduced iterations + dims multiple of 8 for nvdiffrast
FAST_STAGES = [
    {
        'name': 'coarse',
        'mask_size': (112, 152),
        'sigma_trans': 30.0,
        'sigma_yaw': np.radians(15),
        'n_iter': 20,
        'n_beams': 8,
        'n_candidates': 120,
        'gamma': 0.15,
    },
    {
        'name': 'medium',
        'mask_size': (224, 304),
        'sigma_trans': 8.0,
        'sigma_yaw': np.radians(5),
        'n_iter': 15,
        'n_beams': 4,
        'n_candidates': 60,
        'gamma': 0.2,
    },
    {
        'name': 'fine',
        'mask_size': (448, 608),
        'sigma_trans': 2.0,
        'sigma_yaw': np.radians(2),
        'n_iter': 10,
        'n_beams': 2,
        'n_candidates': 52,
        'gamma': 0.3,
    },
]

# Stages for PF when cost-volume already provides a good initial pose.
# Skips σ=30m coarse PF stage; starts from medium (σ=8m).
FAST_STAGES_NO_COARSE = [
    {
        'name': 'medium',
        'mask_size': (224, 304),
        'sigma_trans': 8.0,
        'sigma_yaw': np.radians(5),
        'n_iter': 15,
        'n_beams': 4,
        'n_candidates': 60,
        'gamma': 0.2,
    },
    {
        'name': 'fine',
        'mask_size': (336, 456),  # reduced from (448, 608) — 44% fewer pixels/call
        'sigma_trans': 2.0,
        'sigma_yaw': np.radians(2),
        'n_iter': 6,              # reduced from 10 — early termination handles easy frames
        'n_beams': 2,
        'n_candidates': 52,
        'gamma': 0.3,
    },
]


def _batched_nelder_mead(score_fn, starts, sigma_t, sigma_y, max_iter=150):
    """Run Nelder-Mead for multiple starts simultaneously with batched scoring.

    Each start has its own simplex (5 vertices in 4D). At each iteration,
    all simplex operations (reflect, expand, contract) are collected into
    a single batch and scored together, amortizing GPU overhead.

    Args:
        score_fn: callable(params_array) -> scores_array, where params_array
                  is (N, 4) float32 and returns (N,) float scores.
        starts: (N_starts, 4) array of starting points
        sigma_t, sigma_y: initial simplex scale
        max_iter: max iterations

    Returns:
        best_params: (N_starts, 4) best point per start
        best_scores: (N_starts,) best score per start
    """
    N = len(starts)
    D = 4  # x, y, z, yaw

    # Initialize simplexes: (N, D+1, D) — N simplexes, each with D+1 vertices
    simplexes = np.zeros((N, D + 1, D), dtype=np.float64)
    for i in range(N):
        simplexes[i, 0] = starts[i]
        for d in range(D):
            simplexes[i, d + 1] = starts[i].copy()
            scale = sigma_y if d == 3 else sigma_t
            simplexes[i, d + 1, d] += scale * 0.5

    # Score all initial vertices: N*(D+1) evaluations
    flat_init = simplexes.reshape(-1, D).astype(np.float32)
    scores_init = score_fn(flat_init)
    f_vals = scores_init.reshape(N, D + 1)

    # NM parameters (adaptive for D=4)
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink

    for it in range(max_iter):
        # Sort each simplex by score (descending — we maximize)
        for i in range(N):
            order = np.argsort(f_vals[i])[::-1]
            simplexes[i] = simplexes[i][order]
            f_vals[i] = f_vals[i][order]

        # Centroid of all vertices except worst
        centroids = simplexes[:, :-1, :].mean(axis=1)  # (N, D)

        # 1. Reflection: x_r = centroid + alpha*(centroid - x_worst)
        x_worst = simplexes[:, -1, :]      # (N, D)
        x_r = centroids + alpha * (centroids - x_worst)

        # Score reflections
        f_r = score_fn(x_r.astype(np.float32))

        # Classify: for each start, decide action
        # f_best = f_vals[:, 0], f_second_worst = f_vals[:, -2], f_worst = f_vals[:, -1]
        f_best = f_vals[:, 0]
        f_sw = f_vals[:, -2]

        # Expansion candidates (where f_r > f_best)
        need_expand = f_r > f_best
        # Outside contraction (where f_sw <= f_r <= f_best)
        need_out_contract = (~need_expand) & (f_r >= f_sw)
        # Inside contraction (where f_r < f_sw)
        need_in_contract = (~need_expand) & (~need_out_contract)

        # 2. Expansion: x_e = centroid + gamma*(x_r - centroid) where f_r > f_best
        to_eval = []
        expand_idx = np.where(need_expand)[0]
        x_e = centroids + gamma * (x_r - centroids)
        for idx in expand_idx:
            to_eval.append((idx, 'expand', x_e[idx]))

        # 3. Outside contraction: x_oc = centroid + rho*(x_r - centroid)
        oc_idx = np.where(need_out_contract)[0]
        x_oc = centroids + rho * (x_r - centroids)
        for idx in oc_idx:
            to_eval.append((idx, 'oc', x_oc[idx]))

        # 4. Inside contraction: x_ic = centroid - rho*(centroid - x_worst)
        ic_idx = np.where(need_in_contract)[0]
        x_ic = centroids - rho * (centroids - x_worst)
        for idx in ic_idx:
            to_eval.append((idx, 'ic', x_ic[idx]))

        if not to_eval:
            break

        # Batch evaluate all secondary candidates
        eval_pts = np.array([t[2] for t in to_eval], dtype=np.float32)
        eval_scores = score_fn(eval_pts)

        # Apply results
        shrink_needed = []
        for k, (idx, action, pt) in enumerate(to_eval):
            f_pt = eval_scores[k]
            if action == 'expand':
                if f_pt > f_r[idx]:
                    # Accept expansion
                    simplexes[idx, -1] = pt
                    f_vals[idx, -1] = f_pt
                else:
                    # Accept reflection
                    simplexes[idx, -1] = x_r[idx]
                    f_vals[idx, -1] = f_r[idx]
            elif action == 'oc':
                if f_pt >= f_r[idx]:
                    simplexes[idx, -1] = pt
                    f_vals[idx, -1] = f_pt
                else:
                    shrink_needed.append(idx)
            elif action == 'ic':
                if f_pt > f_vals[idx, -1]:
                    simplexes[idx, -1] = pt
                    f_vals[idx, -1] = f_pt
                else:
                    shrink_needed.append(idx)

        # 5. Shrink: move all vertices towards the best vertex
        if shrink_needed:
            shrink_pts = []
            for idx in shrink_needed:
                x_best = simplexes[idx, 0]
                for j in range(1, D + 1):
                    simplexes[idx, j] = x_best + sigma * (simplexes[idx, j] - x_best)
                    shrink_pts.append(simplexes[idx, j])

            if shrink_pts:
                shrink_pts = np.array(shrink_pts, dtype=np.float32)
                shrink_scores = score_fn(shrink_pts)
                k = 0
                for idx in shrink_needed:
                    for j in range(1, D + 1):
                        f_vals[idx, j] = shrink_scores[k]
                        k += 1

        # Check convergence: if all simplexes have small spread
        spreads = np.max(simplexes, axis=1) - np.min(simplexes, axis=1)  # (N, D)
        max_spread_t = spreads[:, :3].max()
        max_spread_y = spreads[:, 3].max()
        if max_spread_t < 0.05 and max_spread_y < np.radians(0.1):
            break

    # Extract best vertex from each simplex
    best_params = np.empty((N, D), dtype=np.float64)
    best_scores = np.empty(N, dtype=np.float64)
    for i in range(N):
        best_j = f_vals[i].argmax()
        best_params[i] = simplexes[i, best_j]
        best_scores[i] = f_vals[i, best_j]

    return best_params, best_scores


def deterministic_refine(
    query_mask: np.ndarray,
    lod: LoD,
    K: np.ndarray,
    starts: List[Tuple[float, float, float, float]],
    pitch: float,
    roll: float,
    pred_depth: Optional[np.ndarray] = None,
    depth_confidence: Optional[np.ndarray] = None,
    iou_weight: float = 1.0,
    depth_weight: float = 0.4,
    plane_weight: float = 0.0,
    edge_weight: float = 0.0,
    edge_dt: Optional[np.ndarray] = None,
    alignment_result: Any = None,
    stages: Optional[List[Dict]] = None,
    image_size: Optional[Tuple[int, int]] = None,
) -> Tuple[float, float, float, float, float, float, float, float, float, list]:
    """Deterministic pose refinement using batched multi-start Nelder-Mead.

    Multi-resolution, multi-start. At each resolution stage, runs a custom
    batched Nelder-Mead that evaluates all simplex operations for all starts
    in a single GPU batch call. Fully deterministic.

    Args:
        query_mask: (H, W) building mask (uint8, 255=building)
        lod: LoD mesh
        K: (3, 3) camera intrinsics
        starts: list of (x, y, z, yaw) start positions from coarse stage
        pitch, roll: fixed orientation
        pred_depth, depth_confidence, iou_weight, depth_weight, plane_weight,
        edge_weight, edge_dt, alignment_result: scoring parameters
        stages: resolution stages (default: FAST_STAGES_NO_COARSE)

    Returns:
        (x, y, z, yaw, pitch, roll, score, depth_scale, depth_bias, all_results)
    """
    if stages is None:
        stages = FAST_STAGES_NO_COARSE

    gpu_path = torch.cuda.is_available()
    if not gpu_path:
        raise RuntimeError("deterministic_refine requires CUDA")

    ctx = _BatchGPUContext(lod, device='cuda')
    if alignment_result is not None and plane_weight > 0:
        ctx.setup_planes(alignment_result)

    _ih, _iw = image_size if image_size is not None else query_mask.shape[:2]

    current_starts = np.array(starts, dtype=np.float64)

    for si, stage in enumerate(stages):
        ms = stage['mask_size']
        q_mask = cv2.resize(query_mask, (ms[1], ms[0]),
                            interpolation=cv2.INTER_NEAREST)
        sx, sy = ms[1] / _iw, ms[0] / _ih
        K_s = K.copy()
        K_s[0, :] *= sx
        K_s[1, :] *= sy

        pd_s = None
        if pred_depth is not None:
            pd_s = cv2.resize(pred_depth, (ms[1], ms[0]),
                              interpolation=cv2.INTER_LINEAR)
        dc_s = None
        if depth_confidence is not None:
            dc_s = cv2.resize(depth_confidence, (ms[1], ms[0]),
                              interpolation=cv2.INTER_LINEAR)

        ctx.setup_stage(K_s, ms, q_mask, pred_depth=pd_s,
                        depth_confidence=dc_s)

        if edge_weight > 0 and edge_dt is not None:
            edge_dt_s = cv2.resize(edge_dt, (ms[1], ms[0]),
                                   interpolation=cv2.INTER_LINEAR)
            ctx.setup_edge_scoring(edge_dt_s)

        sigma_t = stage['sigma_trans']
        sigma_y = stage['sigma_yaw']

        n_faces = ctx.faces.shape[0]
        chunk = _effective_batch_size(ctx.batch_size, n_faces)

        def batch_score(params):
            """Score a batch of (N, 4) poses. Returns (N,) scores."""
            N = len(params)
            pos = torch.tensor(params[:, :3], dtype=torch.float32, device='cuda')
            yaw = torch.tensor(params[:, 3], dtype=torch.float32, device='cuda')
            all_sc = torch.empty(N, device='cuda')
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                all_sc[s:e] = ctx.batch_render_and_score(
                    pos[s:e], yaw[s:e], pitch, roll,
                    iou_weight, depth_weight, plane_weight,
                    edge_weight)
            return all_sc.cpu().numpy()

        max_iter = stage.get('n_iter', 15) * 10
        best_params, best_scores = _batched_nelder_mead(
            batch_score, current_starts, sigma_t, sigma_y, max_iter=max_iter)

        # Keep all starts, sorted by score, for next stage
        order = np.argsort(best_scores)[::-1]
        current_starts = best_params[order]

    scores = best_scores[order]
    poses = current_starts

    best_idx = 0  # already sorted
    bx, by, bz, byaw = poses[best_idx]

    # Estimate depth scale/bias
    depth_scale, depth_bias = 1.0, 0.0
    if ctx.has_depth and depth_weight > 0 and ctx.n_depth_valid >= 50:
        pos_t = torch.tensor([[bx, by, bz]], dtype=torch.float32, device='cuda')
        yaw_t = torch.tensor([byaw], dtype=torch.float32, device='cuda')
        ctx.batch_render_and_score(pos_t, yaw_t, pitch, roll,
                                   iou_weight, depth_weight, plane_weight)
        depth_scale = float(ctx._depth_scales[0].cpu())
        depth_bias = float(ctx._depth_biases[0].cpu())

    all_results = [poses[i].tolist() for i in range(len(poses))]
    return (float(bx), float(by), float(bz), float(byaw),
            pitch, roll, float(scores[best_idx]), depth_scale, depth_bias, all_results)


# ── GPU batch rendering + scoring ────────────────────────────────────────

def _round_up_8(x: int) -> int:
    return ((x + 7) // 8) * 8


def _effective_batch_size(batch_size: int, n_faces: int) -> int:
    """Cap batch size by mesh size to avoid raster OOM."""
    return max(4, min(batch_size, 20_000_000 // max(n_faces, 1)))


def _batch_euler_to_w2c(positions, yaws, pitch, roll, device):
    """Vectorised ZXY Euler → w2c (batch).

    pitch/roll are constant scalars; positions (N,3) and yaws (N,) vary.
    """
    cp, sp = float(np.cos(pitch)), float(np.sin(pitch))
    cr, sr = float(np.cos(roll)), float(np.sin(roll))

    # Rx(pitch) @ Ry(roll) — constant across batch
    R_pr = torch.tensor([
        [cr,      0.0,   sr],
        [sp * sr, cp,    -sp * cr],
        [-cp * sr, sp,   cp * cr],
    ], dtype=torch.float32, device=device)

    cos_y = torch.cos(yaws)
    sin_y = torch.sin(yaws)

    r0, r1, r2 = R_pr[0], R_pr[1], R_pr[2]
    N = len(yaws)

    R_c2w = torch.empty(N, 3, 3, device=device, dtype=torch.float32)
    R_c2w[:, 0, :] = cos_y.unsqueeze(1) * r0 - sin_y.unsqueeze(1) * r1
    R_c2w[:, 1, :] = sin_y.unsqueeze(1) * r0 + cos_y.unsqueeze(1) * r1
    R_c2w[:, 2, :] = r2

    R_w2c = R_c2w.transpose(1, 2)
    t_w2c = -torch.bmm(R_w2c, positions.unsqueeze(2)).squeeze(2)
    return R_w2c, t_w2c


class _BatchGPUContext:
    """Batch GPU rendering + scoring for the particle filter (via LoD renderer)."""

    NEAR_PLANE = 0.5

    def __init__(self, lod: LoD, device: str = 'cuda', batch_size: int = 256):
        self.lod = lod
        self.device = device
        self.batch_size = int(batch_size)
        self._renderer = None
        self._K_np: np.ndarray | None = None

    def setup_stage(self, K, mask_size, query_mask, pred_depth=None,
                    depth_confidence=None, query_mask_float=None,
                    mask_iou_mode: str = "union"):
        self.mask_iou_mode = mask_iou_mode
        H, W = mask_size
        self.H = _round_up_8(H)
        self.W = _round_up_8(W)

        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])
        self._K_np = np.asarray(K, dtype=np.float64)
        self.lod._ensure_renderer(self._K_np, (self.H, self.W), device='gpu')
        self._renderer = self.lod.renderer
        self._renderer.minibatch_size = self.batch_size

        # Pad query mask to rounded dims
        q = np.zeros((self.H, self.W), dtype=np.uint8)
        q[:H, :W] = query_mask
        self.q_mask = torch.tensor(q > 0, dtype=torch.bool, device=self.device)

        # Optional float query mask for soft NCC scoring (e.g. edge-enhanced mask).
        # When provided, enables an additional NCC term in scoring so that
        # float-valued features (edges, soft probabilities) can be compared
        # against the binary rendered depth mask via normalised cross-correlation.
        self.has_float_mask = False
        if query_mask_float is not None:
            qf = np.zeros((self.H, self.W), dtype=np.float32)
            qf[:H, :W] = query_mask_float.astype(np.float32) / 255.0 \
                if query_mask_float.dtype == np.uint8 else query_mask_float.astype(np.float32)
            qf_t = torch.tensor(qf, device=self.device, dtype=torch.float32)
            # Precompute mean and std over the building region for fast NCC
            valid = self.q_mask
            self.q_float_flat = qf_t[valid]          # (M,)
            m = self.q_float_flat.mean()
            s = self.q_float_flat.std().clamp(min=1e-6)
            self.q_float_norm = (self.q_float_flat - m) / s  # (M,) zero-mean unit-var
            self.q_float_valid = valid                # (H, W) bool
            self.n_float_valid = int(valid.sum().item())
            self.has_float_mask = self.n_float_valid >= 50

        # Plane alignment data
        self.has_planes = False
        self.has_edges = False
        self.plane_n_det = None
        self.plane_d_det = None
        self.plane_n_world = None
        self.plane_centroid = None

        # Depth scoring data
        self.has_depth = pred_depth is not None
        self.has_confidence = False
        self.conf_flat = None
        if self.has_depth:
            pd = np.zeros((self.H, self.W), dtype=np.float32)
            pd[:H, :W] = pred_depth
            pd_t = torch.tensor(pd, dtype=torch.float32, device=self.device)
            self.depth_valid = (pd_t > 0.1) & self.q_mask
            self.pred_depth_flat = pd_t[self.depth_valid]
            self.n_depth_valid = int(self.depth_valid.sum().item())

            # Per-pixel depth confidence weights
            if depth_confidence is not None:
                dc = np.zeros((self.H, self.W), dtype=np.float32)
                dc[:H, :W] = depth_confidence
                dc_t = torch.tensor(dc, dtype=torch.float32, device=self.device)
                self.conf_flat = dc_t[self.depth_valid]
                # Normalize so mean weight = 1 (preserves score magnitude)
                cmean = self.conf_flat.mean()
                if cmean > 1e-8:
                    self.conf_flat = self.conf_flat / cmean
                self.has_confidence = True

    def setup_planes(self, alignment_result):
        """Store matched plane data from ALGW for plane-aware scoring.

        For each matched pair, we store:
          - n_det: detected normal in camera frame (3,)
          - d_det: detected offset in camera frame
          - n_world: LoD plane normal in world frame (3,)
          - centroid: LoD plane centroid in world frame (3,)

        At scoring time, for each candidate R_w2c, t_w2c:
          n_cam = R_w2c @ n_world
          d_cam = n_cam · (R_w2c @ centroid + t_w2c)
          score = mean( cos(n_cam, n_det) * exp(-|d_cam - d_det| / tau) )
        """
        if alignment_result is None or not alignment_result.success:
            self.has_planes = False
            return

        matches = alignment_result.matches
        det_planes = alignment_result.det_planes
        lod_planes = alignment_result.lod_planes

        if len(matches) < 1:
            self.has_planes = False
            return

        n_det_list, d_det_list, n_world_list, centroid_list = [], [], [], []
        weight_list = []
        for m in matches:
            dp = det_planes[m.det_idx]
            lp = lod_planes[m.lod_idx]
            n_det_list.append(dp.normal_cam)
            d_det_list.append(dp.d_cam)
            n_world_list.append(lp.normal)
            centroid_list.append(lp.centroid)
            # Confidence: sqrt(pixel area) / |d_cam|, geometric mean with LoD area
            conf = np.sqrt(dp.area_pixels) / max(abs(dp.d_cam), 1.0)
            area_w = np.sqrt(dp.area_3d * lp.area) if dp.area_3d > 0 and lp.area > 0 else 1.0
            weight_list.append(conf * area_w)

        # Normalize weights to sum to 1
        weights = np.array(weight_list, dtype=np.float32)
        weights = weights / (weights.sum() + 1e-12)

        device = self.device
        self.plane_n_det = torch.tensor(
            np.stack(n_det_list), dtype=torch.float32, device=device)  # (P, 3)
        self.plane_d_det = torch.tensor(
            d_det_list, dtype=torch.float32, device=device)            # (P,)
        self.plane_n_world = torch.tensor(
            np.stack(n_world_list), dtype=torch.float32, device=device)  # (P, 3)
        self.plane_centroid = torch.tensor(
            np.stack(centroid_list), dtype=torch.float32, device=device)  # (P, 3)
        self.plane_weights = torch.tensor(
            weights, dtype=torch.float32, device=device)               # (P,)
        self.has_planes = True

    def setup_edge_scoring(self, edge_dt):
        """Store pre-computed image edge distance transform for edge scoring.

        Args:
            edge_dt: (H, W) float32 distance transform from image edges.
                     Resized and padded to match the current stage resolution.
        """
        self.has_edges = False
        if edge_dt is None:
            return
        dt = np.zeros((self.H, self.W), dtype=np.float32)
        h, w = edge_dt.shape[:2]
        dt[:h, :w] = edge_dt
        self.edge_dt = torch.tensor(dt, dtype=torch.float32, device=self.device)
        self.has_edges = True

    def setup_sdf_scoring(self, sdf: np.ndarray):
        """Store the mask SDF for batch SDF-cost computation.

        Args:
            sdf: (H, W) float32/64 signed-distance-to-mask image at the
                 current stage render resolution (clipped to ±50).
        """
        h, w = sdf.shape[:2]
        buf = np.zeros((self.H, self.W), dtype=np.float32)
        buf[:h, :w] = sdf.astype(np.float32)
        self.sdf_map = torch.tensor(buf, dtype=torch.float32, device=self.device)

    def _batch_sdf_cost(self, depth_maps: torch.Tensor) -> torch.Tensor:
        """SDF cost for a batch of rendered depth maps.

        Implements the same asymmetric squared-distance loss as
        ``_eval_cost_combined`` in test_lodloc5_best.py:
          cost_i = mean_{rendered pixels p} (sdf[p]>0 ? sdf[p]^2 : 0.25*sdf[p]^2)
        Normalised by (H/4)^2 so the value is ~[0, 1].

        Returns:
            (N,) float tensor — lower is better.
        """
        N, H, W = depth_maps.shape
        masks = depth_maps > 0                   # (N, H, W) bool

        sdf = self.sdf_map[:H, :W]               # (H, W)
        sdf_sq = sdf ** 2
        # Asymmetric: outside (sdf>0) → full penalty; inside → ¼ penalty
        w_map = torch.where(sdf > 0, sdf_sq, 0.25 * sdf_sq)  # (H, W)

        w_map_b = w_map.unsqueeze(0).expand(N, -1, -1)        # (N, H, W)
        n_pix = masks.sum(dim=(1, 2)).float().clamp(min=1)     # (N,)
        cost = (w_map_b * masks.float()).sum(dim=(1, 2)) / n_pix  # (N,)
        norm = max((H / 4.0) ** 2, 1.0)
        return cost / norm                                     # (N,)

    def _batch_sdf_edge_cost(
        self,
        depth_maps: torch.Tensor,
        edge_weight: float,
        edge_dt_scale: float = 75.0,
    ) -> torch.Tensor:
        """Combined SDF + edge-DT cost for batch renders.

        Mirrors ``_eval_cost_combined``.  Returns lower-is-better cost
        in ~[0, 1].
        """
        N, H, W = depth_maps.shape
        mask_norm = self._batch_sdf_cost(depth_maps)

        if not self.has_edges or edge_weight <= 0.0:
            return mask_norm


        masks = (depth_maps > 0).float().unsqueeze(1)   # (N,1,H,W)
        eroded = -F.max_pool2d(-masks, kernel_size=3, stride=1, padding=1)
        contour = ((masks - eroded).squeeze(1) > 0.5)   # (N,H,W) bool

        # Scale edge-DT from original resolution to render resolution
        h0, w0 = self.edge_dt.shape
        scale_factor = H / h0
        scaled_dt_scale = edge_dt_scale * scale_factor

        dt = self.edge_dt[:H, :W]
        dt_b = dt.unsqueeze(0).expand(N, -1, -1)        # (N,H,W)
        n_cont = contour.sum(dim=(1, 2)).float().clamp(min=1)
        edge_norm = (dt_b * contour.float()).sum(dim=(1, 2)) / n_cont
        edge_norm = edge_norm / max(scaled_dt_scale, 1.0)

        return (1.0 - edge_weight) * mask_norm + edge_weight * edge_norm

    def batch_render_depth(
        self,
        positions: torch.Tensor,
        yaws: torch.Tensor,
        pitch: float,
        roll: float,
    ) -> torch.Tensor:
        """Render depth maps for N poses in one batch via LoD renderer."""
        w2c_batch = _w2c_batch_from_euler(positions, yaws, pitch, roll, self.device)
        return self._renderer.render_depth_batch(w2c_batch)[:, 0]

    def _batch_edge_score(self, masks_bhw, tau=None, min_res=0):
        """Silhouette boundary vs query DT — no extra normal raster."""
        h = masks_bhw.shape[1]
        if h < min_res:
            return torch.zeros(masks_bhw.shape[0], device=masks_bhw.device)
        return score_mask_edges_vs_dt(masks_bhw, self.edge_dt, tau=tau)

    @torch.no_grad()
    def batch_render_and_score(self, positions, yaws, pitch, roll,
                               iou_weight, depth_weight, plane_weight=0.0,
                               edge_weight=0.0, ncc_weight=0.0):
        """Render + score all N candidates in one GPU batch call.

        Auto-chunks large batches based on mesh size to avoid OOM on 12GB GPUs.
        """
        N = len(positions)
        # Store ncc_weight so _batch_render_and_score_impl can pick it up
        self._ncc_weight = ncc_weight
        n_faces = len(self.lod.faces)
        max_batch = _effective_batch_size(self.batch_size, n_faces)
        if N > max_batch:
            all_scores = torch.empty(N, device=self.device)
            for start in range(0, N, max_batch):
                end = min(start + max_batch, N)
                all_scores[start:end] = self._batch_render_and_score_impl(
                    positions[start:end], yaws[start:end], pitch, roll,
                    iou_weight, depth_weight, plane_weight, edge_weight)
            return all_scores
        return self._batch_render_and_score_impl(
            positions, yaws, pitch, roll, iou_weight, depth_weight,
            plane_weight, edge_weight)

    def _batch_render_and_score_impl(self, positions, yaws, pitch, roll,
                                     iou_weight, depth_weight, plane_weight=0.0,
                                     edge_weight=0.0):
        """Inner implementation: render + score a single chunk."""
        N = len(positions)
        device = self.device

        R_w2c, t_w2c = _batch_euler_to_w2c(positions, yaws, pitch, roll, device)
        w2c_batch = _w2c_batch_from_euler(positions, yaws, pitch, roll, device)
        K_t = torch.tensor(self._K_np, dtype=torch.float32, device=device)
        masks, depth_maps = self._renderer.render_masks_batch(w2c_batch, K_t)
        if depth_maps is None:
            depth_maps = self._renderer.render_depth_batch(w2c_batch)[:, 0]

        # GPU IoU
        q = self.q_mask.unsqueeze(0)
        inter = (q & masks).sum(dim=(1, 2)).float()
        if getattr(self, 'mask_iou_mode', 'union') == 'pred_recall':
            ious = inter / q.sum(dim=(1, 2)).float().clamp(min=1.0)
        else:
            union = (q | masks).sum(dim=(1, 2)).float()
            ious = inter / union.clamp(min=1)

        scores = iou_weight * ious

        if depth_weight > 0 and self.has_depth and self.n_depth_valid >= 50:
            depth_scores = self._batch_depth_score(depth_maps)
            scores = scores + depth_weight * depth_scores

        if plane_weight > 0 and self.has_planes:
            plane_scores = self._batch_plane_score(R_w2c, t_w2c)
            scores = scores + plane_weight * plane_scores

        if edge_weight > 0 and self.has_edges:
            edge_scores = self._batch_edge_score(masks)
            scores = scores + edge_weight * edge_scores

        if hasattr(self, '_ncc_weight') and self._ncc_weight > 0:
            ncc_scores = self._batch_binary_ncc(masks)
            scores = scores + self._ncc_weight * ncc_scores

        return scores

    def _batch_binary_ncc(self, masks):
        """NCC between binary query mask and each rendered binary mask.

        Pearson correlation coefficient between q_mask.float() and masks[i].float()
        over all pixels.  Higher = better overlap.  Returned in [0, 1] (negative
        correlations clipped to 0).

        Args:
            masks: (N, H, W) bool tensor on device.
        Returns:
            (N,) float32 tensor, values in [0, 1].
        """
        N = masks.shape[0]
        q = self.q_mask.float().view(1, -1)  # (1, H*W)
        r = masks.float().view(N, -1)        # (N, H*W)
        M = q.shape[1]

        q_mean = q.mean(dim=1, keepdim=True)             # (1, 1)
        r_mean = r.mean(dim=1, keepdim=True)             # (N, 1)
        q_zm = q - q_mean                               # (1, H*W)
        r_zm = r - r_mean                               # (N, H*W)

        q_std = q_zm.pow(2).mean().sqrt().clamp(min=1e-6)   # scalar
        r_std = r_zm.pow(2).mean(dim=1).sqrt().clamp(min=1e-6)  # (N,)

        ncc = (q_zm * r_zm).mean(dim=1) / (q_std * r_std)  # (N,)
        return ncc.clamp(min=0.0)

    def _batch_float_mask_ncc(self, depth_maps):
        """NCC between float query mask and rendered binary silhouette.

        For each candidate, extracts the float query values at rendered
        building pixels and computes their NCC against the query float mask
        values at the same pixels.  This is the Pearson correlation between
        the float query map and the binary rendered mask, restricted to the
        query-mask region.

        High NCC means the rendered silhouette overlaps the high-response
        regions of the query float map (e.g. edge locations), which is a
        strong discriminator even at large spatial offsets.

        Returns:
            (N,) float tensor in [-1, 1].  Clipped to [0, 1] for use as
            an additive score (negative correlations are treated as 0).
        """
        N = depth_maps.shape[0]
        # rendered building pixels in the query-mask region: (N, M)
        rend = depth_maps[:, self.q_float_valid].float()  # (N, M)
        ok = rend > 0.1                                   # (N, M) bool

        q_norm = self.q_float_norm.unsqueeze(0).expand(N, -1)  # (N, M)

        cnt = ok.sum(dim=1).float().clamp(min=1)  # (N,)
        r_masked = rend * ok.float()

        # Normalise rendered mask values per candidate
        r_mean = r_masked.sum(dim=1) / cnt         # (N,)
        r_zm = rend - r_mean.unsqueeze(1)           # (N, M)
        r_zm = r_zm * ok.float()

        r_std = (r_zm.pow(2).sum(dim=1) / cnt).sqrt().clamp(min=1e-6)  # (N,)
        r_norm = r_zm / r_std.unsqueeze(1)          # (N, M)

        # Pearson correlation over the valid pixels
        ncc = (q_norm * r_norm * ok.float()).sum(dim=1) / cnt  # (N,)
        return ncc.clamp(min=0.0)

    def _batch_depth_score(self, depth_maps):
        """Depth score with per-particle affine alignment.

        For each particle, estimates optimal scale s and bias b such that
        (s * pred + b) best matches the rendered depth via weighted least
        squares.  Scores based on NCC (shape correlation) and residual
        RMSE after affine correction.

        This makes depth scoring robust to unknown depth scale and bias,
        which is common with monocular depth estimators.
        """
        N = depth_maps.shape[0]
        pred_flat = self.pred_depth_flat         # (M,)
        rend = depth_maps[:, self.depth_valid]   # (N, M)
        ok = rend > 0.1

        p = pred_flat.unsqueeze(0).expand(N, -1)
        cnt = ok.sum(dim=1).float().clamp(min=1)

        scores = torch.zeros(N, device=self.device)
        good_mask = cnt.squeeze(-1) >= 50

        if not good_mask.any():
            self._depth_scales = torch.ones(N, device=self.device)
            self._depth_biases = torch.zeros(N, device=self.device)
            return scores

        # Confidence weights (mean-normalized so mean(w)=1)
        if self.has_confidence:
            w = self.conf_flat.unsqueeze(0).expand(N, -1) * ok  # (N, M)
            w_sum = w.sum(dim=1).clamp(min=1e-8)  # (N,)
        else:
            w = ok.float()
            w_sum = cnt

        # Weighted means
        p_mean = (p * w).sum(dim=1) / w_sum  # (N,)
        r_mean = (rend * w).sum(dim=1) / w_sum  # (N,)

        # Centered
        pc = (p - p_mean.unsqueeze(1)) * ok  # (N, M)
        rc = (rend - r_mean.unsqueeze(1)) * ok  # (N, M)

        # Weighted variance and covariance
        if self.has_confidence:
            p_var = (w * pc ** 2).sum(dim=1)
            r_var = (w * rc ** 2).sum(dim=1)
            cov = (w * pc * rc).sum(dim=1)
        else:
            p_var = (pc ** 2).sum(dim=1)
            r_var = (rc ** 2).sum(dim=1)
            cov = (pc * rc).sum(dim=1)

        # --- NCC (already scale+bias invariant) ---
        denom = (p_var * r_var).sqrt()
        nccs = torch.zeros(N, device=self.device)
        ncc_good = good_mask & (denom > 1e-6)
        nccs[ncc_good] = (cov[ncc_good] / denom[ncc_good]).clamp(min=0)

        # --- Per-particle affine alignment: corrected = s * pred + b ---
        safe = good_mask & (p_var > 1e-6)
        s = torch.ones(N, device=self.device)
        b = torch.zeros(N, device=self.device)
        s[safe] = (cov[safe] / p_var[safe]).clamp(0.1, 10.0)
        b[safe] = r_mean[safe] - s[safe] * p_mean[safe]

        # Residual RMSE after affine correction
        corrected = s.unsqueeze(1) * p + b.unsqueeze(1)  # (N, M)
        residual_sq = (corrected - rend) ** 2 * w
        rmse = (residual_sq.sum(dim=1) / w_sum).sqrt()  # (N,)

        TAU = 10.0  # metres — tolerance for residual RMSE
        align_scores = torch.exp(-rmse / TAU)

        # Combine: NCC (shape) + alignment residual (fit quality)
        scores[good_mask] = 0.5 * nccs[good_mask] + 0.5 * align_scores[good_mask]

        # Store per-particle affine params for downstream retrieval
        self._depth_scales = s
        self._depth_biases = b

        return scores

    def _batch_plane_score(self, R_w2c, t_w2c):
        """Score candidates by plane alignment consistency.

        For each candidate pose and each matched plane:
          n_cam = R_w2c @ n_world
          d_cam = n_cam . (R_w2c @ centroid + t_w2c)
          normal_score = cos(n_cam, n_det)  [clamped >= 0]
          offset_score = exp(-|d_cam - d_det| / tau)
          per-plane score = normal_score * offset_score

        Returns mean over all planes, shape (N,).
        """
        N = R_w2c.shape[0]
        P = self.plane_n_world.shape[0]
        TAU = 5.0  # metres — tolerance for offset mismatch

        # Rotate LoD normals to camera: (N,3,3) @ (P,3,1) → (N,P,3)
        n_cam = torch.bmm(
            R_w2c.unsqueeze(1).expand(N, P, 3, 3).reshape(N * P, 3, 3),
            self.plane_n_world.unsqueeze(0).expand(N, P, 3).reshape(N * P, 3, 1)
        ).reshape(N, P, 3)

        # Centroid in camera: R_w2c @ centroid + t_w2c
        c_cam = torch.bmm(
            R_w2c.unsqueeze(1).expand(N, P, 3, 3).reshape(N * P, 3, 3),
            self.plane_centroid.unsqueeze(0).expand(N, P, 3).reshape(N * P, 3, 1)
        ).reshape(N, P, 3) + t_w2c.unsqueeze(1).expand(N, P, 3)

        # d_cam = n_cam . c_cam (per plane)
        d_cam = (n_cam * c_cam).sum(dim=2)  # (N, P)

        # Normal alignment: cos(n_cam, n_det)
        n_det = self.plane_n_det.unsqueeze(0).expand(N, P, 3)  # (N, P, 3)
        cos_angle = (n_cam * n_det).sum(dim=2).clamp(min=0)   # (N, P)

        # Offset alignment: exp(-|d_cam - d_det| / tau)
        d_det = self.plane_d_det.unsqueeze(0).expand(N, P)     # (N, P)
        offset_score = torch.exp(-(d_cam - d_det).abs() / TAU) # (N, P)

        # Combined per-plane score, weighted by confidence
        per_plane = cos_angle * offset_score  # (N, P)
        w = self.plane_weights.unsqueeze(0).expand(N, P)  # (N, P)
        return (per_plane * w).sum(dim=1)     # (N,) — weights sum to 1

    @torch.no_grad()
    def score_components_batch(self, positions, yaws, pitch, roll):
        """Return individual score components for analysis.

        Returns dict with tensors (N,):
            'ious', 'nccs', 'rmses', 'align_scores', 'depth_combined'
        All moved to CPU as numpy arrays.
        """
        N = len(positions)
        R_w2c, t_w2c = _batch_euler_to_w2c(positions, yaws, pitch, roll, self.device)
        w2c_batch = _w2c_batch_from_euler(positions, yaws, pitch, roll, self.device)
        K_t = torch.tensor(self._K_np, dtype=torch.float32, device=self.device)
        masks, depth_maps = self._renderer.render_masks_batch(w2c_batch, K_t)
        if depth_maps is None:
            depth_maps = self._renderer.render_depth_batch(w2c_batch)[:, 0]

        q = self.q_mask.unsqueeze(0)
        inter = (q & masks).sum(dim=(1, 2)).float()
        union = (q | masks).sum(dim=(1, 2)).float()
        ious = inter / union.clamp(min=1)

        nccs = torch.zeros(N, device=self.device)
        rmses = torch.full((N,), float('nan'), device=self.device)
        align_scores = torch.zeros(N, device=self.device)
        depth_combined = torch.zeros(N, device=self.device)

        if self.has_depth and self.n_depth_valid >= 50:
            pred_flat = self.pred_depth_flat
            rend = depth_maps[:, self.depth_valid]
            ok = rend > 0.1
            p = pred_flat.unsqueeze(0).expand(N, -1)
            cnt = ok.sum(dim=1).float().clamp(min=1)
            good_mask = cnt >= 50

            if self.has_confidence:
                w = self.conf_flat.unsqueeze(0).expand(N, -1) * ok
                w_sum = w.sum(dim=1).clamp(min=1e-8)
            else:
                w = ok.float()
                w_sum = cnt

            p_mean = (p * w).sum(dim=1) / w_sum
            r_mean = (rend * w).sum(dim=1) / w_sum
            pc = (p - p_mean.unsqueeze(1)) * ok
            rc = (rend - r_mean.unsqueeze(1)) * ok
            if self.has_confidence:
                p_var = (w * pc ** 2).sum(dim=1)
                r_var = (w * rc ** 2).sum(dim=1)
                cov = (w * pc * rc).sum(dim=1)
            else:
                p_var = (pc ** 2).sum(dim=1)
                r_var = (rc ** 2).sum(dim=1)
                cov = (pc * rc).sum(dim=1)

            denom = (p_var * r_var).sqrt()
            ncc_good = good_mask & (denom > 1e-6)
            nccs[ncc_good] = (cov[ncc_good] / denom[ncc_good]).clamp(min=0)

            safe = good_mask & (p_var > 1e-6)
            s = torch.ones(N, device=self.device)
            b = torch.zeros(N, device=self.device)
            s[safe] = (cov[safe] / p_var[safe]).clamp(0.1, 10.0)
            b[safe] = r_mean[safe] - s[safe] * p_mean[safe]
            corrected = s.unsqueeze(1) * p + b.unsqueeze(1)
            residual_sq = (corrected - rend) ** 2 * w
            rmse_vals = (residual_sq.sum(dim=1) / w_sum).sqrt()
            rmses[good_mask] = rmse_vals[good_mask]
            TAU = 10.0
            align_scores[good_mask] = torch.exp(-rmse_vals[good_mask] / TAU)
            depth_combined[good_mask] = 0.5 * nccs[good_mask] + 0.5 * align_scores[good_mask]

        return {
            'ious': ious.cpu().numpy(),
            'nccs': nccs.cpu().numpy(),
            'rmses': rmses.cpu().numpy(),
            'align_scores': align_scores.cpu().numpy(),
            'depth_combined': depth_combined.cpu().numpy(),
            'plane_scores': self._batch_plane_score(R_w2c, t_w2c).cpu().numpy()
                            if self.has_planes else None,
        }


def _run_pf_stage_gpu(ctx, q_mask, K_s, mask_size, pred_depth_s,
                      x0, y0, z0, yaw0, pitch, roll,
                      sigma_trans, sigma_yaw,
                      n_iter, n_beams, n_candidates, gamma,
                      iou_weight, depth_weight, plane_weight=0.0,
                      edge_weight=0.0,
                      depth_confidence_s=None, beam_init=None,
                      edge_dt_s=None, mask_iou_mode: str = "union"):
    """GPU-batched particle filter stage.

    Each outer iteration performs ``sub_steps`` mini-batches, re-sampling
    from updated beam centres after each step. This recovers sequential
    improvement chaining that would otherwise be lost with pure batch
    evaluation.
    """
    device = ctx.device
    n_per_beam = n_candidates // n_beams

    ctx.setup_stage(K_s, mask_size, q_mask, pred_depth=pred_depth_s,
                    depth_confidence=depth_confidence_s,
                    mask_iou_mode=mask_iou_mode)
    if edge_weight > 0:
        ctx.setup_edge_scoring(edge_dt_s)

    # Seed beams: use diverse coarse candidates if provided, else replicate
    if beam_init is not None and len(beam_init) > 1:
        init_poses = []
        for i in range(n_beams):
            if i < len(beam_init):
                p = beam_init[i]
                init_poses.append([float(p[0]), float(p[1]), float(p[2]), float(p[3])])
            else:
                init_poses.append([x0, y0, z0, yaw0])
        beams = torch.tensor(init_poses, dtype=torch.float32, device=device)
        # Score each beam at its own starting point
        beam_scores = ctx.batch_render_and_score(
            beams[:, :3], beams[:, 3], pitch, roll,
            iou_weight, depth_weight, plane_weight, edge_weight)
    else:
        beams = torch.tensor(
            [[x0, y0, z0, yaw0]] * n_beams,
            dtype=torch.float32, device=device)
        # Score initial pose
        init_pos = torch.tensor([[x0, y0, z0]], dtype=torch.float32, device=device)
        init_yaw = torch.tensor([yaw0], dtype=torch.float32, device=device)
        init_sc = ctx.batch_render_and_score(
            init_pos, init_yaw, pitch, roll, iou_weight, depth_weight,
            plane_weight, edge_weight)
        beam_scores = init_sc.expand(n_beams).clone()

    st, sy = sigma_trans, sigma_yaw
    patience = max(3, n_iter // 5)
    stale = 0
    prev_best = float(beam_scores.max())

    for _ in range(n_iter):
        noise = torch.randn(n_beams, n_per_beam, 4, device=device)
        noise[:, :, :3] *= st
        noise[:, :, 3] *= sy

        cands = beams.unsqueeze(1) + noise           # (B, C, 4)
        flat = cands.reshape(-1, 4)

        scores = ctx.batch_render_and_score(
            flat[:, :3], flat[:, 3], pitch, roll, iou_weight, depth_weight,
            plane_weight, edge_weight)
        
        # Penalize distance from initialization to prevent perceptual aliasing drift
        init_pos_penalty = torch.tensor([[x0, y0, z0]], dtype=torch.float32, device=device)
        dist_m = torch.norm(flat[:, :3] - init_pos_penalty, dim=-1)
        scores = scores - 0.002 * dist_m

        scores = scores.reshape(n_beams, n_per_beam)

        best_val, best_idx = scores.max(dim=1)
        improved = best_val > beam_scores
        for b in range(n_beams):
            if improved[b]:
                beams[b] = cands[b, best_idx[b]]
                beam_scores[b] = best_val[b]

        st *= (1 - gamma)
        sy *= (1 - gamma)

        # Early termination
        cur_best = float(beam_scores.max())
        if cur_best <= prev_best + 1e-6:
            stale += 1
            if stale >= patience:
                break
        else:
            stale = 0
            prev_best = cur_best

    best = beam_scores.argmax()
    bx, by, bz, byaw = beams[best].cpu().tolist()

    # Estimate depth scale+bias at converged pose
    depth_scale, depth_bias = 1.0, 0.0
    if ctx.has_depth and depth_weight > 0 and ctx.n_depth_valid >= 50:
        best_pos = beams[best:best+1, :3]
        best_yaw = beams[best, 3:4]
        ctx.batch_render_and_score(
            best_pos, best_yaw, pitch, roll,
            iou_weight, depth_weight, plane_weight, edge_weight)
        depth_scale = float(ctx._depth_scales[0].cpu())
        depth_bias = float(ctx._depth_biases[0].cpu())

    # Return all beam poses sorted by score (for multi-start refinement)
    sorted_idx = beam_scores.argsort(descending=True)
    all_beams = beams[sorted_idx].cpu().tolist()

    return bx, by, bz, byaw, float(beam_scores[best].cpu()), depth_scale, depth_bias, all_beams


def _multi_scale_pf_gpu(query_mask, lod, K,
                        x0, y0, z0, yaw0, pitch, roll,
                        pred_depth, iou_weight, depth_weight, stages,
                        alignment_result=None, plane_weight=0.0,
                        edge_weight=0.0, edge_dt=None,
                        depth_confidence=None, beam_init=None,
                        image_size=None, batch_size: int = 256,
                        mask_iou_mode: str = "union"):
    """GPU-accelerated multi-scale particle filter."""
    ctx = _BatchGPUContext(lod, device='cuda', batch_size=batch_size)
    if alignment_result is not None and plane_weight > 0:
        ctx.setup_planes(alignment_result)
    _ih, _iw = image_size if image_size is not None else query_mask.shape[:2]
    cx, cy, cz, cyaw = x0, y0, z0, yaw0

    for si, stage in enumerate(stages):
        ms = stage['mask_size']
        q_mask = cv2.resize(query_mask, (ms[1], ms[0]),
                            interpolation=cv2.INTER_NEAREST)
        sx, sy = ms[1] / _iw, ms[0] / _ih
        K_s = K.copy()
        K_s[0, :] *= sx
        K_s[1, :] *= sy

        pd_s = None
        if pred_depth is not None:
            pd_s = cv2.resize(pred_depth, (ms[1], ms[0]),
                              interpolation=cv2.INTER_LINEAR)

        dc_s = None
        if depth_confidence is not None:
            dc_s = cv2.resize(depth_confidence, (ms[1], ms[0]),
                              interpolation=cv2.INTER_LINEAR)

        # Resize edge distance transform for this stage
        edt_s = None
        if edge_dt is not None and edge_weight > 0:
            edt_s = cv2.resize(edge_dt, (ms[1], ms[0]),
                               interpolation=cv2.INTER_LINEAR)
            # Scale DT values by resolution ratio (distances in pixels)
            edt_s = edt_s * (ms[1] / _iw)

        # Only seed beams from coarse diverse candidates on the first stage
        stage_beam_init = beam_init if si == 0 else None

        cx, cy, cz, cyaw, score, d_scale, d_bias, all_beams = _run_pf_stage_gpu(
            ctx, q_mask, K_s, ms, pd_s,
            cx, cy, cz, cyaw, pitch, roll,
            sigma_trans=stage['sigma_trans'],
            sigma_yaw=stage['sigma_yaw'],
            n_iter=stage['n_iter'],
            n_beams=stage['n_beams'],
            n_candidates=stage['n_candidates'],
            gamma=stage['gamma'],
            iou_weight=iou_weight,
            depth_weight=depth_weight,
            plane_weight=plane_weight,
            edge_weight=edge_weight,
            depth_confidence_s=dc_s,
            beam_init=stage_beam_init,
            edge_dt_s=edt_s,
            mask_iou_mode=mask_iou_mode,
        )

    return (cx, cy, cz, cyaw, pitch, roll, score, d_scale, d_bias, all_beams)


# ── CPU fallback (original implementation) ───────────────────────────────

def _run_particle_filter_stage(
    q_mask: np.ndarray,
    lod: LoD,
    K_scaled: np.ndarray,
    mask_size: Tuple[int, int],
    scorer: Optional[BatchAlignmentScorer],
    x0: float, y0: float, z0: float, yaw0: float,
    pitch: float, roll: float,
    sigma_trans: float,
    sigma_yaw: float,
    n_iter: int,
    n_beams: int,
    n_candidates: int,
    gamma: float,
    iou_weight: float,
    depth_weight: float,
    normal_weight: float,
) -> Tuple[float, float, float, float, float]:
    """Run a single particle filter stage.

    Returns:
        (x, y, z, yaw, best_score)
    """
    def _score(w2c):
        hyp_mask = lod.render_mask(w2c, K_scaled, mask_size)
        iou = compute_iou(q_mask, hyp_mask)
        if scorer is not None:
            depth_map = lod.render_depth(w2c, K_scaled, mask_size)
            sc = scorer.score_batch(
                np.array([iou]), [depth_map],
                iou_weight, depth_weight, normal_weight)
            return float(sc[0])
        return iou

    n_per_beam = n_candidates // n_beams

    # Initialize all beams at starting pose
    beams = [(x0, y0, z0, yaw0)] * n_beams
    beam_scores = [0.0] * n_beams

    # Evaluate initial pose
    for b in range(n_beams):
        w2c = pose_to_w2c(x0, y0, z0, yaw0, pitch, roll)
        beam_scores[b] = _score(w2c)

    cur_st, cur_sy = sigma_trans, sigma_yaw

    for _ in range(n_iter):
        for b in range(n_beams):
            bx, by, bz, byaw = beams[b]
            best = (bx, by, bz, byaw)
            best_sc = beam_scores[b]

            for _ in range(n_per_beam):
                cx = bx + np.random.normal(0, cur_st)
                cy = by + np.random.normal(0, cur_st)
                cz = bz + np.random.normal(0, cur_st)
                cyaw = byaw + np.random.normal(0, cur_sy)

                w2c = pose_to_w2c(cx, cy, cz, cyaw, pitch, roll)
                sc = _score(w2c)

                if sc > best_sc:
                    best_sc = sc
                    best = (cx, cy, cz, cyaw)

            beams[b] = best
            beam_scores[b] = best_sc

        cur_st *= (1 - gamma)
        cur_sy *= (1 - gamma)

    best_idx = int(np.argmax(beam_scores))
    bx, by, bz, byaw = beams[best_idx]
    return bx, by, bz, byaw, beam_scores[best_idx]


def multi_scale_particle_filter(
    query_mask: np.ndarray,
    lod: LoD,
    K: np.ndarray,
    x0: float, y0: float, z0: float,
    yaw0: float, pitch: float, roll: float,
    pred_depth: Optional[np.ndarray] = None,
    pred_normals: Optional[np.ndarray] = None,
    depth_confidence: Optional[np.ndarray] = None,
    iou_weight: float = 0.5,
    depth_weight: float = 0.5,
    normal_weight: float = 0.0,
    plane_weight: float = 0.0,
    edge_weight: float = 0.0,
    edge_dt: Optional[np.ndarray] = None,
    alignment_result: Any = None,
    stages: Optional[List[Dict]] = None,
    use_gpu: bool = True,
    seed: Optional[int] = None,
    beam_init: Optional[List[np.ndarray]] = None,
    image_size: Optional[Tuple[int, int]] = None,
    batch_size: Optional[int] = None,
    mask_iou_mode: str = "union",
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Multi-scale particle filter for pose estimation.

    Runs 3 progressive stages at increasing resolution:
      1. Coarse (112px): large σ=30m, 4 beams, slow cooling — broad exploration
      2. Medium (224px): σ=8m, 2 beams — narrows down
      3. Fine (448px): σ=2m, 2 beams — precise refinement

    At each stage, combines silhouette IoU + depth NCC for scoring.

    When ``use_gpu=True`` and CUDA is available, uses GPU batch rendering via LoD.
    with nvdiffrast for ~10-20x speedup and defaults to FAST_STAGES.

    Args:
        query_mask: (H, W) building mask (uint8, 255=building)
        lod: LoD mesh instance
        K: (3, 3) camera intrinsics at full resolution
        x0, y0, z0: initial position (e.g. GPS prior)
        yaw0, pitch, roll: initial orientation (radians, ZXY Euler)
        pred_depth: (H, W) predicted metric depth (None → IoU-only)
        pred_normals: (H, W, 3) predicted camera-frame normals
        iou_weight: weight for silhouette IoU scoring
        depth_weight: weight for depth NCC scoring
        normal_weight: weight for normal consistency scoring
        plane_weight: weight for plane alignment scoring (from ALGW)
        alignment_result: AlignmentResult from PlaneAligner.align() (for plane scoring)
        stages: list of stage config dicts (default: auto-selected)
        use_gpu: use GPU batch rendering via LoD when CUDA is available
        seed: optional integer seed for numpy and torch RNGs (makes PF deterministic)
        beam_init: list of (x,y,z,yaw) arrays from coarse diverse candidates.
                   When provided, the first PF stage seeds each beam from a
                   different coarse candidate instead of all from (x0,y0,z0,yaw0).

    Returns:
        (x, y, z, yaw, pitch, roll, best_score, depth_scale, depth_bias)
        depth_scale, depth_bias: affine params such that
        (depth_scale * pred_depth + depth_bias) ≈ metric depth.
        Returns (1.0, 0.0) when depth scoring is disabled.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    gpu_path = use_gpu and torch.cuda.is_available()

    if batch_size is None:
        batch_size = int(getattr(lod, '_batch_size', 256))

    if stages is None:
        stages = FAST_STAGES if gpu_path else FAST_STAGES_NO_COARSE

    if gpu_path:
        return _multi_scale_pf_gpu(
            query_mask, lod, K,
            x0, y0, z0, yaw0, pitch, roll,
            pred_depth, iou_weight, depth_weight, stages,
            alignment_result=alignment_result,
            plane_weight=plane_weight,
            edge_weight=edge_weight,
            edge_dt=edge_dt,
            depth_confidence=depth_confidence,
            beam_init=beam_init,
            image_size=image_size,
            batch_size=batch_size,
            mask_iou_mode=mask_iou_mode)  # returns 10-tuple: (..., all_beams)

    # ── CPU fallback ──
    use_alignment = (pred_depth is not None and pred_normals is not None)
    _ih, _iw = image_size if image_size is not None else query_mask.shape[:2]
    cx, cy, cz, cyaw = x0, y0, z0, yaw0

    for stage in stages:
        ms = stage['mask_size']

        # Scale mask and intrinsics
        q_mask = cv2.resize(query_mask, (ms[1], ms[0]),
                            interpolation=cv2.INTER_NEAREST)
        sx, sy = ms[1] / _iw, ms[0] / _ih
        K_s = K.copy()
        K_s[0, :] *= sx
        K_s[1, :] *= sy

        # Build scorer for this resolution
        scorer = None
        if use_alignment:
            pd_s = cv2.resize(pred_depth, (ms[1], ms[0]),
                              interpolation=cv2.INTER_LINEAR)
            pn_s = cv2.resize(pred_normals, (ms[1], ms[0]),
                              interpolation=cv2.INTER_LINEAR)
            nlen = np.linalg.norm(pn_s, axis=-1, keepdims=True)
            pn_s /= np.where(nlen > 1e-8, nlen, 1.0)
            scorer = BatchAlignmentScorer(pd_s, pn_s, q_mask, K_s)

        cx, cy, cz, cyaw, score = _run_particle_filter_stage(
            q_mask, lod, K_s, ms, scorer,
            cx, cy, cz, cyaw, pitch, roll,
            sigma_trans=stage['sigma_trans'],
            sigma_yaw=stage['sigma_yaw'],
            n_iter=stage['n_iter'],
            n_beams=stage['n_beams'],
            n_candidates=stage['n_candidates'],
            gamma=stage['gamma'],
            iou_weight=iou_weight,
            depth_weight=depth_weight,
            normal_weight=normal_weight,
        )

    return (cx, cy, cz, cyaw, pitch, roll, score, 1.0, 0.0)


def robust_pose_basin_score(
    query_mask: np.ndarray,
    lod: LoD,
    K: np.ndarray,
    x: float,
    y: float,
    z: float,
    yaw: float,
    pitch: float,
    roll: float,
    pred_depth: Optional[np.ndarray] = None,
    depth_confidence: Optional[np.ndarray] = None,
    iou_weight: float = 1.0,
    depth_weight: float = 0.0,
    plane_weight: float = 0.0,
    alignment_result: Any = None,
    mask_size: Tuple[int, int] = (448, 608),
    sigma_trans: float = 0.75,
    sigma_yaw: float = np.radians(0.5),
    n_samples: int = 17,
    reducer: str = 'p25',
    seed: int = 0,
    image_size: Optional[Tuple[int, int]] = None,
) -> float:
    """Score pose robustness by averaging scores over small local jitter.

    This is intended as a reranker, not as a PF objective. The center pose is
    included in the sample set so the score remains anchored to the candidate.
    """
    gpu_path = torch.cuda.is_available()

    if n_samples <= 1:
        center = multi_scale_particle_filter(
            query_mask, lod, K,
            x, y, z, yaw, pitch, roll,
            pred_depth=pred_depth,
            depth_confidence=depth_confidence,
            iou_weight=iou_weight,
            depth_weight=depth_weight,
            plane_weight=plane_weight,
            alignment_result=alignment_result,
            stages=[{
                'name': 'score',
                'mask_size': mask_size,
                'sigma_trans': 0.0,
                'sigma_yaw': 0.0,
                'n_iter': 1,
                'n_beams': 1,
                'n_candidates': 1,
                'gamma': 0.0,
            }],
            use_gpu=gpu_path,
            seed=seed,
        )
        return float(center[6])

    _ih, _iw = image_size if image_size is not None else query_mask.shape[:2]
    q_mask = cv2.resize(query_mask, (mask_size[1], mask_size[0]), interpolation=cv2.INTER_NEAREST)
    sx, sy = mask_size[1] / _iw, mask_size[0] / _ih
    K_s = K.copy()
    K_s[0, :] *= sx
    K_s[1, :] *= sy

    pred_depth_s = None
    if pred_depth is not None:
        pred_depth_s = cv2.resize(pred_depth, (mask_size[1], mask_size[0]), interpolation=cv2.INTER_LINEAR)

    depth_conf_s = None
    if depth_confidence is not None:
        depth_conf_s = cv2.resize(depth_confidence, (mask_size[1], mask_size[0]), interpolation=cv2.INTER_LINEAR)

    if gpu_path:
        ctx = _BatchGPUContext(lod, device='cuda')
        ctx.setup_stage(K_s, mask_size, q_mask, pred_depth=pred_depth_s, depth_confidence=depth_conf_s)
        if alignment_result is not None and plane_weight > 0:
            ctx.setup_planes(alignment_result)

        rng = np.random.default_rng(seed)
        positions = np.empty((n_samples, 3), dtype=np.float32)
        yaws = np.empty((n_samples,), dtype=np.float32)
        positions[0] = [x, y, z]
        yaws[0] = yaw
        if n_samples > 1:
            positions[1:] = np.array([x, y, z], dtype=np.float32) + rng.normal(
                0.0, sigma_trans, size=(n_samples - 1, 3)
            ).astype(np.float32)
            yaws[1:] = yaw + rng.normal(0.0, sigma_yaw, size=(n_samples - 1,)).astype(np.float32)

        pos_t = torch.tensor(positions, dtype=torch.float32, device='cuda')
        yaw_t = torch.tensor(yaws, dtype=torch.float32, device='cuda')
        scores = ctx.batch_render_and_score(
            pos_t, yaw_t, pitch, roll,
            iou_weight=iou_weight,
            depth_weight=depth_weight,
            plane_weight=plane_weight,
        ).detach().cpu().numpy()
    else:
        scorer = None
        if pred_depth_s is not None:
            pred_normals_s = np.zeros((*pred_depth_s.shape, 3), dtype=np.float32)
            scorer = BatchAlignmentScorer(pred_depth_s, pred_normals_s, q_mask, K_s)

        rng = np.random.default_rng(seed)
        scores = []
        for sample_i in range(n_samples):
            if sample_i == 0:
                cx, cy, cz, cyaw = x, y, z, yaw
            else:
                cx = x + float(rng.normal(0.0, sigma_trans))
                cy = y + float(rng.normal(0.0, sigma_trans))
                cz = z + float(rng.normal(0.0, sigma_trans))
                cyaw = yaw + float(rng.normal(0.0, sigma_yaw))
            w2c = pose_to_w2c(cx, cy, cz, cyaw, pitch, roll)
            hyp_mask = lod.render_mask(w2c, K_s, mask_size)
            iou = compute_iou(q_mask, hyp_mask)
            score = float(iou_weight * iou)
            if scorer is not None and depth_weight > 0:
                depth_map = lod.render_depth(w2c, K_s, mask_size)
                sc = scorer.score_batch(
                    np.array([iou]), [depth_map],
                    iou_weight, depth_weight, 0.0,
                )
                score = float(sc[0])
            scores.append(score)
        scores = np.asarray(scores, dtype=np.float32)

    if reducer == 'mean':
        return float(scores.mean())
    if reducer == 'center_plus_p25':
        return float(0.5 * scores[0] + 0.5 * np.percentile(scores, 25))
    return float(np.percentile(scores, 25))


# ── Wireframe-based PF refinement ─────────────────────────────────────


def _extract_structural_edge_points(
    lod: LoD,
    interval: float = 1.0,
    normal_cos_thresh: float = 0.999,
) -> np.ndarray:
    """Extract 3D points along structural (non-coplanar) mesh edges.

    Returns (N, 3) float32 world-coordinate points sampled along edges
    between non-coplanar adjacent faces. These are the LoD building
    boundary lines that should align with image edges.
    """
    verts = lod.vertices_abs.astype(np.float32)
    faces = lod.faces
    fn = lod.face_normals()

    # Build edge→face adjacency
    e0 = faces[:, [0, 1]]
    e1 = faces[:, [1, 2]]
    e2 = faces[:, [2, 0]]
    all_edges = np.concatenate([e0, e1, e2], axis=0)  # (3F, 2)
    face_idx_all = np.repeat(np.arange(len(faces)), 3)

    sorted_edges = np.sort(all_edges, axis=1).astype(np.int64)
    max_v = int(sorted_edges.max()) + 1
    edge_keys = sorted_edges[:, 0] * max_v + sorted_edges[:, 1]

    # For each unique edge, find both adjacent faces
    order = np.argsort(edge_keys)
    sorted_keys = edge_keys[order]
    # Edges shared by 2 faces appear consecutively after sorting
    pair_mask = sorted_keys[:-1] == sorted_keys[1:]
    pair_idx = np.where(pair_mask)[0]

    if len(pair_idx) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    fa = face_idx_all[order[pair_idx]]
    fb = face_idx_all[order[pair_idx + 1]]

    # Keep only non-coplanar pairs (structural edges)
    cos = np.abs(np.einsum('ij,ij->i', fn[fa], fn[fb]))
    structural = cos <= normal_cos_thresh
    edge_a = all_edges[order[pair_idx[structural]]]  # (S, 2) vertex indices

    if len(edge_a) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Sample points along each structural edge
    p0 = verts[edge_a[:, 0]]
    p1 = verts[edge_a[:, 1]]
    lengths = np.linalg.norm(p1 - p0, axis=1)
    valid = lengths > 1e-6
    p0, p1, lengths = p0[valid], p1[valid], lengths[valid]

    n_samples = np.maximum(np.ceil(lengths / interval).astype(np.int32) + 1, 2)
    total = int(n_samples.sum())
    offsets = np.zeros(len(n_samples) + 1, dtype=np.int64)
    np.cumsum(n_samples, out=offsets[1:])

    edge_idx = np.repeat(np.arange(len(n_samples)), n_samples)
    local = np.arange(total, dtype=np.float32) - offsets[edge_idx].astype(np.float32)
    denom = (n_samples[edge_idx] - 1).clip(min=1).astype(np.float32)
    t = (local / denom)[:, None]

    pts = p0[edge_idx] + t * (p1[edge_idx] - p0[edge_idx])
    return pts  # (N, 3) float32


def _project_points(
    pts_3d: np.ndarray, w2c: np.ndarray, K: np.ndarray, H: int, W: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points → 2D pixel coords. Returns (uv, in_frame_mask)."""
    ones = np.ones((len(pts_3d), 1), dtype=np.float32)
    cam = (w2c[:3] @ np.hstack([pts_3d, ones]).T).astype(np.float32)
    z = cam[2]
    inv_z = np.where(z > 1e-6, 1.0 / z, 0.0).astype(np.float32)
    u = K[0, 0] * cam[0] * inv_z + K[0, 2]
    v = K[1, 1] * cam[1] * inv_z + K[1, 2]
    uv = np.stack([u, v], axis=1)
    in_frame = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return uv, in_frame


def wireframe_pf_refine(
    lod: LoD,
    K: np.ndarray,
    image_size: Tuple[int, int],
    x0: float, y0: float, z0: float,
    yaw0: float, pitch: float, roll: float,
    image_line_dt: np.ndarray,
    query_mask: Optional[np.ndarray] = None,
    n_iter: int = 20,
    n_candidates: int = 40,
    sigma_trans: float = 1.5,
    sigma_yaw: float = np.radians(1.5),
    gamma: float = 0.15,
    edge_tau: float = 5.0,
    iou_weight: float = 0.3,
    edge_weight: float = 0.7,
    seed: Optional[int] = None,
) -> Tuple[float, float, float, float, float]:
    """Final PF refinement using LoD edge → image line alignment.

    Precomputes 3D structural edge points from the LoD mesh, then for
    each candidate pose simply reprojects them (matrix multiply) and
    looks up distances in the image line distance transform.  This is
    ~100x faster than calling render_edges per candidate.

    Args:
        lod: LoD mesh instance.
        K: (3,3) camera intrinsics at full resolution.
        image_size: (H, W) output image size.
        x0..roll: starting pose (from upstream PF).
        image_line_dt: (H, W) float32 distance transform from image lines.
        query_mask: (H, W) building mask for optional IoU scoring.
        n_iter: PF iterations.
        n_candidates: candidates per iteration.
        sigma_trans: initial position noise (metres).
        sigma_yaw: initial yaw noise (radians).
        gamma: noise decay per iteration.
        edge_tau: distance scale for exponential edge scoring (pixels).
        iou_weight: weight for IoU component.
        edge_weight: weight for edge alignment component.
        seed: RNG seed for reproducibility.

    Returns:
        (x, y, z, yaw, best_score)
    """
    H, W = image_size
    rng = np.random.default_rng(seed)

    # Precompute 3D structural edge points (once)
    edge_pts_3d = _extract_structural_edge_points(lod, interval=1.0)
    if len(edge_pts_3d) < 10:
        return x0, y0, z0, yaw0, 0.0

    # Check that enough points are visible at the starting pose (signal quality)
    w2c_check = pose_to_w2c(x0, y0, z0, yaw0, pitch, roll)
    _, m_check = _project_points(edge_pts_3d, w2c_check, K, H, W)
    if int(m_check.sum()) < 200:
        # Too few edge points visible — score landscape too noisy, skip refinement
        return x0, y0, z0, yaw0, 0.0

    def _score(cx, cy, cz, cyaw):
        w2c = pose_to_w2c(cx, cy, cz, cyaw, pitch, roll)
        uv, mask = _project_points(edge_pts_3d, w2c, K, H, W)
        n_vis = int(mask.sum())
        if n_vis < 10:
            return 0.0

        # Look up distance transform at projected edge points
        ui = uv[mask, 0].astype(np.int32).clip(0, W - 1)
        vi = uv[mask, 1].astype(np.int32).clip(0, H - 1)
        dists = image_line_dt[vi, ui]
        mean_chamfer = float(np.clip(dists, 0, edge_tau * 5).mean())
        e_score = float(np.exp(-mean_chamfer / edge_tau))

        # Optional IoU
        i_score = 0.0
        if query_mask is not None and iou_weight > 0:
            mask_render = lod.render_mask(w2c, K, (H, W))
            i_score = compute_iou(query_mask, mask_render)

        return iou_weight * i_score + edge_weight * e_score

    bx, by, bz, byaw = x0, y0, z0, yaw0
    best_score = _score(bx, by, bz, byaw)

    st, sy = sigma_trans, sigma_yaw
    patience = max(3, n_iter // 4)
    stale = 0

    for _ in range(n_iter):
        improved = False
        for _ in range(n_candidates):
            cx = bx + float(rng.normal(0, st))
            cy = by + float(rng.normal(0, st))
            cz = bz + float(rng.normal(0, st))
            cyaw = byaw + float(rng.normal(0, sy))

            sc = _score(cx, cy, cz, cyaw)
            if sc > best_score:
                best_score = sc
                bx, by, bz, byaw = cx, cy, cz, cyaw
                improved = True

        st *= (1 - gamma)
        sy *= (1 - gamma)

        if not improved:
            stale += 1
            if stale >= patience:
                break
        else:
            stale = 0

    return bx, by, bz, byaw, best_score
