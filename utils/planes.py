"""
utils/planes.py
===============
Plane segmentation from normal maps and plane matching utilities.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sparse_cc
from scipy.spatial import ConvexHull


def extract_plane_polygons_from_normals(
    normals: np.ndarray,
    depth: np.ndarray | None = None,
    *,
    normal_cos_thresh: float = 0.98,
    depth_rel_thresh: float = 0.02,
    min_pixels: int = 100,
) -> list[dict]:
    """Extract 2D plane polygons from any normal map (e.g. moge2 predictions).

    Segments the image into planar regions via connected components on pixel
    adjacency (horizontal + vertical 4-connectivity), then computes a convex
    hull polygon for each region.

    Works on any camera-frame normal map — rendered LoD mesh normals, moge2
    predicted normals, GT raycasted normals, etc.  No mesh or renderer needed.

    Args:
        normals:           (H, W, 3) float32 camera-space unit normals.
                           Background / invalid pixels should be zero.
        depth:             (H, W) float32 metric depth in metres (optional).
                           When provided, two pixels are only linked if their
                           relative depth difference is < ``depth_rel_thresh``.
        normal_cos_thresh: minimum cosine similarity to connect two adjacent
                           pixels (0.98 ≈ 11°, 0.995 ≈ 5.7°).
        depth_rel_thresh:  max relative depth step between neighbours when
                           ``depth`` is provided (0.02 = 2 %).
        min_pixels:        minimum region size (pixels) to keep.

    Returns:
        list of dicts, one per plane, sorted by ``pixel_count`` descending:

        ``'plane_id'``      – int, sequential (0, 1, …).

        ``'normal_cam'``    – (3,) float32 mean unit normal in camera space.

        ``'pixel_count'``   – int.

        ``'pixel_indices'`` – (N, 2) int32 array of (row, col) coordinates.

        ``'polygon_2d'``    – (K, 2) float32 convex-hull vertices in
                              (col, row) = (x, y) image coordinates,
                              or ``None`` if < 3 pixels.
    """

    H, W = normals.shape[:2]
    N = H * W

    # Valid pixels: non-zero normals (and optionally positive depth)
    nlen_sq = (normals ** 2).sum(axis=2)
    valid = nlen_sq > 0.25
    if depth is not None:
        valid &= (depth > 0)

    # Normalise
    safe = np.maximum(nlen_sq, 1e-16)
    normals_n = normals / np.sqrt(safe)[..., None]  # (H, W, 3)

    flat_valid = valid.ravel()
    valid_indices = np.where(flat_valid)[0]
    n_valid = len(valid_indices)

    if n_valid < min_pixels:
        return []

    compact_id = np.full(N, -1, dtype=np.int32)
    compact_id[valid_indices] = np.arange(n_valid, dtype=np.int32)

    # Build 4-connected edges (horizontal + vertical)
    all_px   = np.arange(N, dtype=np.int32)
    col_idx  = all_px % W
    row_idx  = all_px // W

    h_mask = col_idx < (W - 1)
    v_mask = row_idx < (H - 1)
    h_a, h_b = all_px[h_mask], all_px[h_mask] + 1
    v_a, v_b = all_px[v_mask], all_px[v_mask] + W

    ea = np.concatenate([h_a, v_a])
    eb = np.concatenate([h_b, v_b])

    both = flat_valid[ea] & flat_valid[eb]
    ea, eb = ea[both], eb[both]

    nf = normals_n.reshape(N, 3)
    dots = (nf[ea] * nf[eb]).sum(axis=1)
    keep = dots >= normal_cos_thresh

    if depth is not None:
        df = depth.ravel()
        da, db = df[ea], df[eb]
        keep &= np.abs(da - db) < depth_rel_thresh * np.maximum(da, db).clip(min=1e-6)

    ea, eb = ea[keep], eb[keep]

    # Sparse CC
    ea_c = compact_id[ea]
    eb_c = compact_id[eb]
    data = np.ones(len(ea_c) * 2, dtype=np.int8)
    rows_sp = np.concatenate([ea_c, eb_c])
    cols_sp = np.concatenate([eb_c, ea_c])
    adj = csr_matrix((data, (rows_sp, cols_sp)), shape=(n_valid, n_valid))
    _, cc_compact = sparse_cc(adj, directed=False)

    cc_flat = np.full(N, -1, dtype=cc_compact.dtype)
    cc_flat[valid_indices] = cc_compact
    cc_flat[~flat_valid] = -1

    max_label = int(cc_flat.max()) + 1 if cc_flat.max() >= 0 else 0
    if max_label == 0:
        return []

    counts    = np.bincount(cc_flat[cc_flat >= 0], minlength=max_label)
    big_labels = np.where(counts >= min_pixels)[0]
    if len(big_labels) == 0:
        return []

    planes = []
    for plane_id, lbl in enumerate(big_labels):
        pix = np.where(cc_flat == lbl)[0]
        r, c = pix // W, pix % W

        avg_n = nf[pix].mean(axis=0)
        norm_ = np.linalg.norm(avg_n)
        avg_n = avg_n / (norm_ + 1e-12)

        pts_2d = np.stack([c, r], axis=1).astype(np.float32)  # (N, 2) (x, y)
        poly = None
        if len(pts_2d) >= 3:
            try:
                hull = ConvexHull(pts_2d)
                poly = pts_2d[hull.vertices].astype(np.float32)
            except Exception:
                poly = None

        planes.append({
            'plane_id':     plane_id,
            'normal_cam':   avg_n.astype(np.float32),
            'pixel_count':  int(len(pix)),
            'pixel_indices': np.stack([r, c], axis=1).astype(np.int32),
            'polygon_2d':   poly,
        })

    # Sort by pixel count descending
    planes.sort(key=lambda p: p['pixel_count'], reverse=True)
    for i, p in enumerate(planes):
        p['plane_id'] = i

    return planes


def segment_normals_into_planes(
    normals: np.ndarray,
    depth: np.ndarray,
    building_mask: np.ndarray,
    normal_cos_thresh: float = 0.995,
    depth_rel_thresh: float = 0.05,
    min_pixels: int = 50,
):
    """Region-growing plane segmentation on predicted normals.

    Returns
    -------
    label_map : (H, W) int32  -- plane IDs, -1 = background
    segments  : list of dicts with keys: plane_id, normal_cam, pixel_count
    """
    if isinstance(normals, torch.Tensor):
        normals = normals.cpu().numpy()
    if depth is not None and isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()
    if isinstance(building_mask, torch.Tensor):
        building_mask = building_mask.cpu().numpy()

    H, W = normals.shape[:2]
    N = H * W

    nlen = np.linalg.norm(normals, axis=2, keepdims=True)
    valid = (nlen[..., 0] > 0.1) & building_mask
    normals_n = normals / np.maximum(nlen, 1e-16)

    flat_valid = valid.ravel()
    valid_indices = np.where(flat_valid)[0]
    n_valid = len(valid_indices)

    if n_valid < min_pixels:
        return np.full((H, W), -1, dtype=np.int32), []

    compact_id = np.full(N, -1, dtype=np.int32)
    compact_id[valid_indices] = np.arange(n_valid, dtype=np.int32)

    all_px = np.arange(N, dtype=np.int32)
    h_a = all_px[all_px % W < (W - 1)]
    h_b = h_a + 1
    v_a = all_px[all_px < (N - W)]
    v_b = v_a + W
    ea = np.concatenate([h_a, v_a])
    eb = np.concatenate([h_b, v_b])
    both_valid = flat_valid[ea] & flat_valid[eb]
    ea, eb = ea[both_valid], eb[both_valid]
    if len(ea) == 0:
        return np.full((H, W), -1, dtype=np.int32), []

    normals_flat = normals_n.reshape(N, 3)
    dots = np.einsum('ij,ij->i', normals_flat[ea], normals_flat[eb])
    keep = dots >= normal_cos_thresh

    if depth is not None:
        depth_flat = depth.ravel()
        da, db = depth_flat[ea], depth_flat[eb]
        keep &= np.abs(da - db) < depth_rel_thresh * np.maximum(np.abs(da), np.abs(db))

    ea, eb = ea[keep], eb[keep]
    if len(ea) == 0:
        return np.full((H, W), -1, dtype=np.int32), []

    ea_c = compact_id[ea]
    eb_c = compact_id[eb]
    data = np.ones(len(ea_c) * 2, dtype=np.int8)
    row = np.concatenate([ea_c, eb_c])
    col = np.concatenate([eb_c, ea_c])
    adj = csr_matrix((data, (row, col)), shape=(n_valid, n_valid))
    _, cc_compact = sparse_cc(adj, directed=False)

    cc_labels = np.full(N, -1, dtype=cc_compact.dtype)
    cc_labels[valid_indices] = cc_compact

    pos_mask = cc_labels >= 0
    if not pos_mask.any():
        return np.full((H, W), -1, dtype=np.int32), []

    max_label = int(cc_labels[pos_mask].max()) + 1
    counts = np.bincount(cc_labels[pos_mask], minlength=max_label)
    big_labels = np.where(counts >= min_pixels)[0]
    if len(big_labels) == 0:
        return np.full((H, W), -1, dtype=np.int32), []

    big_set = np.zeros(max_label, dtype=bool)
    big_set[big_labels] = True

    valid_idx_all = np.where(pos_mask)[0]
    valid_labels_all = cc_labels[valid_idx_all]
    order = np.argsort(valid_labels_all, kind='mergesort')
    sorted_labels = valid_labels_all[order]
    sorted_idx = valid_idx_all[order]
    breaks = np.concatenate([[0],
                             np.where(np.diff(sorted_labels) != 0)[0] + 1,
                             [len(sorted_labels)]])
    label_at_break = sorted_labels[breaks[:-1]]

    label_map = np.full((H, W), -1, dtype=np.int32)
    segments = []
    plane_id = 0

    for i in range(len(breaks) - 1):
        cc_id = label_at_break[i]
        if not big_set[cc_id]:
            continue
        pix = sorted_idx[breaks[i]:breaks[i + 1]]
        rows = (pix // W).astype(np.int32)
        cols = (pix % W).astype(np.int32)
        avg_n = normals_flat[pix].mean(axis=0)
        nrm = np.linalg.norm(avg_n)
        avg_n = avg_n / (nrm + 1e-12)
        label_map[rows, cols] = plane_id
        segments.append({
            'plane_id': plane_id,
            'normal_cam': avg_n.astype(np.float32),
            'pixel_count': len(pix),
            'pixel_indices': np.stack([rows, cols], axis=1).astype(np.int32),
        })
        plane_id += 1

    return label_map, segments


# ======================================================================
# Plane matching (from plane_matching.py)
# ======================================================================

def depth_scale_factor(
    depth_pred: np.ndarray,   # (H, W) predicted depth (any relative scale)
    mask_pred:  np.ndarray,   # (H, W) bool — pixels to sample from pred
    depth_gt:   np.ndarray,   # (H, W) GT depth (metric, e.g. metres)
    mask_gt:    np.ndarray,   # (H, W) bool — pixels to sample from GT
) -> float:
    """Median scale factor to bring *pred* depth into GT metric scale.

    Computes::

        scale = median(depth_gt[valid_gt]) / median(depth_pred[valid_pred])

    where valid = mask & depth > 0.

    Apply as: ``depth_pred_metric = depth_pred * scale``

    Parameters
    ----------
    depth_pred, depth_gt : (H, W) float arrays
    mask_pred, mask_gt   : (H, W) bool arrays (e.g. building segmentation masks)

    Returns
    -------
    float  scale factor (> 0), or 1.0 if either median is zero / no valid pixels.
    """
    valid_pred = mask_pred & (depth_pred > 0)
    valid_gt   = mask_gt   & (depth_gt   > 0)

    if not valid_pred.any() or not valid_gt.any():
        return 1.0

    med_pred = float(np.median(depth_pred[valid_pred]))
    med_gt   = float(np.median(depth_gt[valid_gt]))

    if med_pred < 1e-9:
        return 1.0

    return med_gt / med_pred


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Plane pair quality score
# ═══════════════════════════════════════════════════════════════════════════════

def plane_pair_score(match: dict) -> float:
    """Composite plane-pair quality score.

    score = iou_2d × (1 − |n_lod × n_pred|) × Q_pred × Q_lod

    Normal similarity uses cross-product magnitude (= sin θ), giving
    1 − sin θ which decays linearly near θ=0 — more discriminative than
    cos θ (zero gradient at perfect alignment).

    Q = mean |cos(θᵢ)| = normal_quality, the mean per-pixel alignment to
    the plane's own mean normal (first-power).

    Falls back gracefully: missing components treated as 1.0.
    """
    iou = match.get('vote_frac', match.get('iou', float('nan')))
    if np.isnan(iou):
        return float('nan')

    # Normal similarity via cross product
    n_lod  = match.get('lod',  {}).get('normal_mean')
    n_pred = match.get('pred', {}).get('normal_mean')
    if n_lod is not None and n_pred is not None:
        n_lod  = np.asarray(n_lod,  dtype=np.float64)
        n_pred = np.asarray(n_pred, dtype=np.float64)
        n_lod  = n_lod  / (np.linalg.norm(n_lod)  + 1e-12)
        n_pred = n_pred / (np.linalg.norm(n_pred) + 1e-12)
        sin_theta = float(np.linalg.norm(np.cross(n_lod, n_pred)))
        n_sim = float(np.clip(1.0 - sin_theta, 0.0, 1.0))
    else:
        raw = match.get('normal_sim', float('nan'))
        n_sim = 1.0 if np.isnan(raw) else float(np.abs(raw))

    # Plane quality: mean |cos(θᵢ)| (first-power alignment)
    q_pred = match.get('pred', {}).get('normal_quality', float('nan'))
    q_lod  = match.get('lod',  {}).get('normal_quality', float('nan'))
    q_pred = 1.0 if np.isnan(q_pred) else float(np.clip(q_pred, 0.0, 1.0))
    q_lod  = 1.0 if np.isnan(q_lod)  else float(np.clip(q_lod,  0.0, 1.0))

    return float(iou) * n_sim * q_pred * q_lod


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Plane attribute enrichment
# ═══════════════════════════════════════════════════════════════════════════════

def enrich_planes(
    segments:   List[Dict],
    normals:    np.ndarray,              # (H, W, 3) camera-frame unit normals
    depth:      np.ndarray,              # (H, W) depth in metric scale
    K:          np.ndarray,              # (3, 3) camera intrinsics
    image_bgr:  Optional[np.ndarray] = None,  # (H, W, 3) uint8 for colour
) -> List[Dict]:
    """Enrich each plane segment with geometric and photometric attributes.

    For every segment dict the following keys are added (in-place copy returned):

    ``normal_mean``   : (3,) float32  — mean unit normal in camera frame.
    ``normal_var``    : float          — mean squared deviation of per-pixel
                                         normals from ``normal_mean`` (scalar).
    ``centroid_2d``   : (2,) float32  — image centroid [u, v] (col, row).
    ``mean_color_bgr``: (3,) float32  — mean BGR colour (or None if no image).
    ``has_3d``        : bool — True if at least one valid depth pixel exists.
    ``centroid_3d``   : (3,) float32  — mean 3D point in camera frame.
    ``area_2d_px``    : int            — number of valid pixels.
    ``area_3d_m2``    : float          — estimated 3D surface area (metres²).

    3-D unprojection uses the standard pinhole model::

        X = (u - cx) * d / fx
        Y = (v - cy) * d / fy
        Z = d

    3-D area is approximated as the sum of per-pixel footprint areas::

        area_3d ≈ Σ  d² / (fx · fy)          [metres² per pixel]

    This is exact for a fronto-parallel plane; for tilted planes it is a lower
    bound (cos θ correction would be needed for full accuracy).

    Parameters
    ----------
    segments  : list of dicts from LoD.get_plane_segmentation (or equivalent).
    normals   : (H, W, 3) float32 — per-pixel camera-space unit normals.
    depth     : (H, W) float32    — per-pixel metric depth (Z in camera frame).
    K         : (3, 3)            — camera intrinsic matrix.
    image_bgr : optional (H, W, 3) uint8 source image.

    Returns
    -------
    List of new dicts (copies) with the additional keys added.
    """

    is_torch = False
    if isinstance(normals, torch.Tensor):
        is_torch = True
        normals = normals.cpu().numpy()
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()
    if image_bgr is not None and isinstance(image_bgr, torch.Tensor):
        image_bgr = image_bgr.cpu().numpy()

    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    fx_fy  = fx * fy

    # Pre-build grid for unprojection
    us = np.arange(W, dtype=np.float32)   # (W,)
    vs = np.arange(H, dtype=np.float32)   # (H,)

    normals_flat = normals.reshape(-1, 3)  # (H*W, 3)
    depth_flat   = depth.ravel()           # (H*W,)

    enriched = []
    for seg in segments:
        out = dict(seg)   # shallow copy
        pix = seg['pixel_indices']   # (N, 2): col 0 = row, col 1 = col
        rows = pix[:, 0]
        cols = pix[:, 1]
        flat_idx = rows * W + cols

        # ── Normal stats ────────────────────────────────────────────────
        pix_normals = normals_flat[flat_idx]   # (N, 3)
        n_mean = pix_normals.mean(axis=0)
        nrm    = np.linalg.norm(n_mean)
        n_mean_unit = n_mean / (nrm + 1e-12)
        # Variance: mean squared angular deviation (1 - cos²)
        dots           = np.clip((pix_normals * n_mean_unit).sum(axis=1), -1.0, 1.0)
        normal_var     = float(1.0 - (dots ** 2).mean())
        # Quality: mean |cos(θᵢ)| — first-power alignment (more discriminative than 1 - var)
        normal_quality = float(np.abs(dots).mean())
        # Quality: mean |cos(θᵢ)| — first-power alignment (more discriminative)
        normal_quality = float(np.abs(dots).mean())

        # ── 2-D centroid (median — robust to non-convex masks) ──────────
        u_c = float(np.median(cols))
        v_c = float(np.median(rows))

        # ── Colour ──────────────────────────────────────────────────────
        mean_color = None
        if image_bgr is not None:
            mean_color = image_bgr[rows, cols].astype(np.float32).mean(axis=0)

        # ── 3-D unprojection ────────────────────────────────────────────
        d_pix = depth_flat[flat_idx]   # (N,)
        valid_d = d_pix > 0

        if valid_d.any():
            r_v = rows[valid_d]
            c_v = cols[valid_d]
            d_v = d_pix[valid_d]

            # 3D centroid: unproject the 2D centroid pixel.
            # Look up depth at the (rounded) centroid pixel; fall back to
            # median depth if the centroid pixel has no valid depth.
            uc_i = int(np.clip(int(round(u_c)), 0, W - 1))
            vc_i = int(np.clip(int(round(v_c)), 0, H - 1))
            d_at_centroid = float(depth[vc_i, uc_i])
            if d_at_centroid <= 0:
                d_at_centroid = float(np.median(d_v))
            centroid_3d = np.array([
                (u_c - cx) * d_at_centroid / fx,
                (v_c - cy) * d_at_centroid / fy,
                d_at_centroid,
            ], dtype=np.float32)

            area_3d = float((d_v ** 2).sum() / fx_fy)
            has_3d  = True
        else:
            centroid_3d = np.zeros(3, dtype=np.float32)
            area_3d     = 0.0
            has_3d      = False

        out.update({
            'normal_mean':    n_mean_unit.astype(np.float32),
            'normal_var':     normal_var,
            'normal_quality': normal_quality,
            'centroid_2d':    np.array([u_c, v_c], dtype=np.float32),
            'mean_color_bgr': mean_color,
            'has_3d':         has_3d,
            'centroid_3d':    centroid_3d.astype(np.float32),
            'area_2d_px':     int(seg['pixel_count']),
            'area_3d_m2':     area_3d,
        })
        enriched.append(out)

    return enriched


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Plane matching
# ═══════════════════════════════════════════════════════════════════════════════

def _iou_matrix(
    gt_label_map:   np.ndarray,   # (H, W) int32, -1 = bg
    pred_label_map: np.ndarray,   # (H, W) int32, -1 = bg
    gt_ids:   List[int],
    pred_ids: List[int],
) -> np.ndarray:
    """Compute IoU matrix of shape (n_gt, n_pred) efficiently."""
    n_gt, n_pred = len(gt_ids), len(pred_ids)
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred), dtype=np.float32)

    gt_flat   = gt_label_map.ravel()
    pred_flat = pred_label_map.ravel()

    gt_idx   = {g: i for i, g in enumerate(gt_ids)}
    pred_idx = {p: i for i, p in enumerate(pred_ids)}

    both = (gt_flat >= 0) & (pred_flat >= 0)
    gf = np.array([gt_idx.get(int(g), -1)   for g in gt_flat[both]], dtype=np.int32)
    pf = np.array([pred_idx.get(int(p), -1)  for p in pred_flat[both]], dtype=np.int32)
    valid = (gf >= 0) & (pf >= 0)

    inter = np.zeros((n_gt, n_pred), dtype=np.float64)
    if valid.any():
        keys = gf[valid] * n_pred + pf[valid]
        cnt  = np.bincount(keys, minlength=n_gt * n_pred)
        inter = cnt.reshape(n_gt, n_pred)

    gt_counts   = np.array([(gt_flat == g).sum()   for g in gt_ids],   dtype=np.float64)
    pred_counts = np.array([(pred_flat == p).sum() for p in pred_ids], dtype=np.float64)
    union = gt_counts[:, None] + pred_counts[None, :] - inter
    iou   = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)


def match_planes(
    gt_planes:      List[Dict],
    pred_planes:    List[Dict],
    gt_label_map:   np.ndarray,   # (H, W) int32
    pred_label_map: np.ndarray,   # (H, W) int32
    iou_thresh:     float = 0.5,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Match GT and predicted plane sets using greedy IoU matching.

    Parameters
    ----------
    gt_planes, pred_planes : enriched plane lists (output of :func:`enrich_planes`).
    gt_label_map, pred_label_map : (H, W) instance label maps (-1 = background).
    iou_thresh : minimum IoU to accept a match.

    Returns
    -------
    matched_pairs : list of dicts — one per TP pair with keys:

        ``gt``          : full GT plane dict
        ``pred``        : full pred plane dict
        ``iou``         : float  overlap IoU
        ``dist_2d_px``  : float  Euclidean distance between 2D centroids (pixels)
        ``dist_3d_m``   : float  Euclidean distance between 3D centroids (metres);
                                  0 if either plane lacks 3D points.
        ``normal_sim``  : float  cosine similarity of mean normals in camera frame
                                  (1 = identical orientation, -1 = opposite).

    unmatched_gt : list of GT plane dicts that were not matched (FN).

    unmatched_pred : list of pred plane dicts that were not matched (FP).
    """
    gt_ids   = [int(p['plane_id']) for p in gt_planes]
    pred_ids = [int(p['plane_id']) for p in pred_planes]

    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return [], list(gt_planes), list(pred_planes)

    iou_mat = _iou_matrix(gt_label_map, pred_label_map, gt_ids, pred_ids)

    # Greedy: sort all (i, j) pairs by IoU descending
    pairs = sorted(
        [(iou_mat[i, j], i, j)
         for i in range(len(gt_ids))
         for j in range(len(pred_ids))],
        key=lambda x: -x[0],
    )

    matched_gi, matched_pi = set(), set()
    matched_pairs  = []

    gt_by_id   = {p['plane_id']: p for p in gt_planes}
    pred_by_id = {p['plane_id']: p for p in pred_planes}

    for iou_val, gi, pi in pairs:
        if iou_val < iou_thresh:
            break
        if gi in matched_gi or pi in matched_pi:
            continue
        matched_gi.add(gi)
        matched_pi.add(pi)

        gp = gt_by_id[gt_ids[gi]]
        pp = pred_by_id[pred_ids[pi]]

        # 2D centroid distance
        gc2 = gp.get('centroid_2d')
        pc2 = pp.get('centroid_2d')
        dist_2d = float(np.linalg.norm(gc2 - pc2)) if (gc2 is not None and pc2 is not None) else 0.0

        # 3D centroid distance
        gc3 = gp.get('centroid_3d')
        pc3 = pp.get('centroid_3d')
        if (gc3 is not None and pc3 is not None
                and gp.get('has_3d') and pp.get('has_3d')):
            dist_3d = float(np.linalg.norm(gc3 - pc3))
        else:
            dist_3d = 0.0

        # Normal cosine similarity
        gn = gp.get('normal_mean', gp.get('normal_cam'))
        pn = pp.get('normal_mean', pp.get('normal_cam'))
        if gn is not None and pn is not None:
            normal_sim = float(np.clip(np.dot(gn, pn), -1.0, 1.0))
        else:
            normal_sim = float('nan')

        matched_pairs.append({
            'gt':           gp,
            'pred':         pp,
            'iou':          float(iou_val),
            'dist_2d_px':   dist_2d,
            'dist_3d_m':    dist_3d,
            'normal_sim':   normal_sim,
        })

    unmatched_gt   = [gt_by_id[gt_ids[i]]   for i in range(len(gt_ids))   if i not in matched_gi]
    unmatched_pred = [pred_by_id[pred_ids[j]] for j in range(len(pred_ids)) if j not in matched_pi]

    return matched_pairs, unmatched_gt, unmatched_pred


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Normal-based plane matching (for GPS prior / translation-invariant)
# ═══════════════════════════════════════════════════════════════════════════════

def match_planes_by_normal(
    lod_planes:   List[Dict],    # enriched LoD planes (rendered at GPS prior)
    pred_planes:  List[Dict],    # enriched pred planes (from moge2ext)
    normal_cos_thresh: float = 0.995,  # min cosine similarity to accept match
    size_ratio_max: float = 4.0,       # reject if area ratio > this (FP guard)
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Match LoD planes to predicted planes by camera-frame normal similarity.

    This is **translation-invariant**: camera-frame normal directions are
    determined almost entirely by camera rotation.  With a GPS prior that has
    low rotation error (~1°) but large translation error (~10–30 m), IoU-based
    matching fails (masks don't overlap) but normal-based matching succeeds.

    Algorithm
    ---------
    1. Build (N_lod × N_pred) cosine similarity matrix between ``normal_mean``
       vectors in camera frame.
    2. Greedy matching: sort all pairs by similarity descending, greedily assign
       the best unmatched pair above ``normal_cos_thresh``.
    3. Optional size guard: reject if area_2d_px ratio > ``size_ratio_max``
       (prevents large-plane ↔ small-plane mismatches with similar normals).

    Parameters
    ----------
    lod_planes, pred_planes : enriched plane lists (from :func:`enrich_planes`).
        Must contain ``normal_mean`` (3,) float32.
    normal_cos_thresh : minimum cosine similarity to accept a match.
    size_ratio_max : max allowed ratio max(area_i, area_j) / min(area_i, area_j).

    Returns
    -------
    matched_pairs : list of dicts with keys:
        ``lod``          : full LoD plane dict
        ``pred``         : full pred plane dict
        ``normal_sim``   : float  cosine similarity of mean normals
        ``dist_2d_px``   : float  2D centroid distance in pixels
        ``size_ratio``   : float  area_2d_px ratio (max/min)
    unmatched_lod  : unmatched LoD planes (FN for pred)
    unmatched_pred : unmatched pred planes (FP)
    """
    if not lod_planes or not pred_planes:
        return [], list(lod_planes), list(pred_planes)

    lod_normals  = np.stack([p['normal_mean'] for p in lod_planes],  axis=0)  # (L, 3)
    pred_normals = np.stack([p['normal_mean'] for p in pred_planes], axis=0)  # (P, 3)

    # Normalise (should already be unit, but be safe)
    lod_normals  = lod_normals  / (np.linalg.norm(lod_normals,  axis=1, keepdims=True) + 1e-12)
    pred_normals = pred_normals / (np.linalg.norm(pred_normals, axis=1, keepdims=True) + 1e-12)

    # (L, P) cosine similarity matrix — take absolute value since normal sign
    # can flip depending on which side of the surface is visible
    sim_mat = np.abs(lod_normals @ pred_normals.T)   # (L, P)

    lod_areas  = np.array([p.get('area_2d_px', 1) for p in lod_planes],  dtype=np.float32)
    pred_areas = np.array([p.get('area_2d_px', 1) for p in pred_planes], dtype=np.float32)

    # Greedy matching by descending similarity
    pairs = sorted(
        [(float(sim_mat[i, j]), i, j)
         for i in range(len(lod_planes))
         for j in range(len(pred_planes))],
        key=lambda x: -x[0],
    )

    matched_li, matched_pi = set(), set()
    matched_pairs: List[Dict] = []

    for sim_val, li, pi in pairs:
        if sim_val < normal_cos_thresh:
            break
        if li in matched_li or pi in matched_pi:
            continue

        # Size ratio guard
        a_lod  = max(lod_areas[li],  1.0)
        a_pred = max(pred_areas[pi], 1.0)
        ratio  = float(max(a_lod, a_pred) / min(a_lod, a_pred))
        if ratio > size_ratio_max:
            continue

        matched_li.add(li)
        matched_pi.add(pi)

        lp = lod_planes[li]
        pp = pred_planes[pi]

        c_lod  = lp.get('centroid_2d')
        c_pred = pp.get('centroid_2d')
        dist_2d = float(np.linalg.norm(c_lod - c_pred)) if (c_lod is not None and c_pred is not None) else float('nan')

        matched_pairs.append({
            'lod':        lp,
            'pred':       pp,
            'normal_sim': sim_val,
            'dist_2d_px': dist_2d,
            'size_ratio': ratio,
        })

    unmatched_lod  = [lod_planes[i]  for i in range(len(lod_planes))  if i not in matched_li]
    unmatched_pred = [pred_planes[j] for j in range(len(pred_planes)) if j not in matched_pi]

    return matched_pairs, unmatched_lod, unmatched_pred


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Relative-normal matching (rotation-invariant)
# ═══════════════════════════════════════════════════════════════════════════════

def match_planes_by_relative_normal(
    lod_planes:   List[Dict],    # enriched LoD planes
    pred_planes:  List[Dict],    # enriched pred planes
    sig_dist_thresh: float = 0.15,   # max L2 distance between normalised signatures
    size_ratio_max:  float = 4.0,    # reject if area ratio > this
    top_k_sig:       int   = 20,     # signature = top-K nearest-normal similarities
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Match LoD planes to predicted planes using **rotation-invariant** signatures.

    Core idea
    ---------
    The pairwise dot product ``|n_i . n_j|`` between two plane normals in the
    *same* camera frame is determined by the world-space angle between those
    planes — it is completely invariant to the camera orientation.

    For each plane ``i``, we define a descriptor as the **sorted vector of
    pairwise similarities** to all other planes in the same set::

        sig(i) = sorted([|n_i . n_k|  for k != i], descending)[:top_k]

    Two planes in different renders describe the *same physical plane* when they
    have the same angular relationship to their neighbours — i.e. when
    ``||sig_lod(i) - sig_pred(j)||`` is small.

    Compared to match_planes_by_normal
    ------------------------------------
    * Robust to **any rotation error** (not just small ones).
    * Slightly weaker when rotation is near-zero (adds noise from finite set size).
    * More expensive: O(L^2) + O(P^2) to build signatures, then O(L*P) to match.

    Parameters
    ----------
    lod_planes, pred_planes : enriched plane lists (from enrich_planes).
    sig_dist_thresh : max L2 distance between normalised (unit) signature vectors.
    size_ratio_max  : area ratio guard (same as match_planes_by_normal).
    top_k_sig       : number of nearest-neighbour similarities to keep per plane.

    Returns
    -------
    matched_pairs, unmatched_lod, unmatched_pred  (same schema as other matchers)
    Each matched pair dict has keys: lod, pred, sig_dist, normal_sim, dist_2d_px, size_ratio
    """
    if not lod_planes or not pred_planes:
        return [], list(lod_planes), list(pred_planes)

    lod_norms  = np.stack([p['normal_mean'] for p in lod_planes])   # (L, 3)
    pred_norms = np.stack([p['normal_mean'] for p in pred_planes])  # (P, 3)

    lod_norms  = lod_norms  / (np.linalg.norm(lod_norms,  axis=1, keepdims=True) + 1e-12)
    pred_norms = pred_norms / (np.linalg.norm(pred_norms, axis=1, keepdims=True) + 1e-12)

    L = len(lod_planes)
    P = len(pred_planes)

    # ── Intra-set pairwise similarities ─────────────────────────────────────
    sim_lod  = np.abs(lod_norms  @ lod_norms.T)    # (L, L)
    sim_pred = np.abs(pred_norms @ pred_norms.T)   # (P, P)

    K = min(L - 1, P - 1, top_k_sig)
    if K <= 0:
        return [], list(lod_planes), list(pred_planes)

    def _top_k_sorted(sim_mat: np.ndarray, k: int) -> np.ndarray:
        """(N, N) -> (N, k) sorted descending, excluding diagonal."""
        N = sim_mat.shape[0]
        np.fill_diagonal(sim_mat, -1.0)
        idx  = np.argpartition(-sim_mat, k, axis=1)[:, :k]
        sigs = np.take_along_axis(sim_mat, idx, axis=1)
        sigs = np.sort(sigs, axis=1)[:, ::-1]
        np.fill_diagonal(sim_mat, 1.0)
        return sigs.astype(np.float32)

    sigs_lod  = _top_k_sorted(sim_lod,  K)   # (L, K)
    sigs_pred = _top_k_sorted(sim_pred, K)   # (P, K)

    # Normalise signatures to unit vectors for stable L2 comparison
    sigs_lod  = sigs_lod  / (np.linalg.norm(sigs_lod,  axis=1, keepdims=True) + 1e-12)
    sigs_pred = sigs_pred / (np.linalg.norm(sigs_pred, axis=1, keepdims=True) + 1e-12)

    # Cross-set L2 distance: ||a - b||^2 = 2 - 2*a.b  (unit vectors)
    cross_dot  = sigs_lod @ sigs_pred.T          # (L, P)
    cross_dist = np.sqrt(np.clip(2.0 - 2.0 * cross_dot, 0.0, None))  # (L, P)

    # ── Greedy matching by ascending signature distance ────────────────────
    lod_areas  = np.array([p.get('area_2d_px', 1) for p in lod_planes],  dtype=np.float32)
    pred_areas = np.array([p.get('area_2d_px', 1) for p in pred_planes], dtype=np.float32)

    flat_i, flat_j = np.unravel_index(np.argsort(cross_dist, axis=None), cross_dist.shape)
    matched_li, matched_pi = set(), set()
    matched_pairs: List[Dict] = []

    for i_idx in range(len(flat_i)):
        li, pi    = int(flat_i[i_idx]), int(flat_j[i_idx])
        dist_val  = float(cross_dist[li, pi])
        if dist_val > sig_dist_thresh:
            break
        if li in matched_li or pi in matched_pi:
            continue

        a_lod  = max(lod_areas[li],  1.0)
        a_pred = max(pred_areas[pi], 1.0)
        ratio  = float(max(a_lod, a_pred) / min(a_lod, a_pred))
        if ratio > size_ratio_max:
            continue

        matched_li.add(li)
        matched_pi.add(pi)

        lp = lod_planes[li]
        pp = pred_planes[pi]

        c_lod  = lp.get('centroid_2d')
        c_pred = pp.get('centroid_2d')
        dist_2d = float(np.linalg.norm(c_lod - c_pred)) if (c_lod is not None and c_pred is not None) else float('nan')

        n_sim = float(np.abs(np.dot(
            lp['normal_mean'] / (np.linalg.norm(lp['normal_mean']) + 1e-12),
            pp['normal_mean'] / (np.linalg.norm(pp['normal_mean']) + 1e-12),
        )))

        matched_pairs.append({
            'lod':        lp,
            'pred':       pp,
            'sig_dist':   dist_val,
            'normal_sim': n_sim,
            'dist_2d_px': dist_2d,
            'size_ratio': ratio,
        })

    unmatched_lod  = [lod_planes[i]  for i in range(L) if i not in matched_li]
    unmatched_pred = [pred_planes[j] for j in range(P) if j not in matched_pi]

    return matched_pairs, unmatched_lod, unmatched_pred


# ═══════════════════════════════════════════════════════════════════════════════
# 6. World-frame normal matching (best for GPS→pred across different viewpoints)
# ═══════════════════════════════════════════════════════════════════════════════

def match_planes_by_world_normal(
    lod_planes:   List[Dict],         # enriched LoD planes (camera-frame normals at GPS pose)
    pred_planes:  List[Dict],         # enriched pred planes (camera-frame normals at GT pose)
    R_lod:        np.ndarray,         # (3,3) rotation: world→camera for LoD (w2c[:3,:3])
    R_pred:       np.ndarray,         # (3,3) rotation: world→camera for pred (w2c[:3,:3])
    normal_cos_thresh: float = 0.95,  # minimum world-frame cosine similarity
    size_ratio_max:    float = 4.0,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Match LoD planes to predicted planes by comparing **world-frame** normals.

    Camera-frame normals differ between the GPS render and the GT/pred render
    because the cameras have different orientations (~2-10° for GPS).  Converting
    both to world frame removes this dependency, giving near-perfect normal
    similarity for truly corresponding planes.

    World-frame normal:  n_world = R^T @ n_cam   (R is the w2c rotation)
    """
    if not lod_planes or not pred_planes:
        return [], list(lod_planes), list(pred_planes)

    # Convert camera-frame normals to world frame
    Rt_lod  = R_lod.T.astype(np.float64)   # c2w rotation for LoD
    Rt_pred = R_pred.T.astype(np.float64)  # c2w rotation for pred

    def _world_normals(planes, Rt):
        cams = np.stack([p['normal_mean'] for p in planes], axis=0).astype(np.float64)  # (N,3)
        cams = cams / (np.linalg.norm(cams, axis=1, keepdims=True) + 1e-12)
        return (Rt @ cams.T).T  # (N,3) world-frame normals

    lod_wn  = _world_normals(lod_planes,  Rt_lod)   # (L,3)
    pred_wn = _world_normals(pred_planes, Rt_pred)  # (P,3)

    # Normalise (should be unit already)
    lod_wn  = lod_wn  / (np.linalg.norm(lod_wn,  axis=1, keepdims=True) + 1e-12)
    pred_wn = pred_wn / (np.linalg.norm(pred_wn, axis=1, keepdims=True) + 1e-12)

    # |n_lod . n_pred| — absolute cosine (normals may flip sign)
    sim_mat = np.abs(lod_wn @ pred_wn.T)  # (L, P)

    lod_areas  = np.array([p.get('area_2d_px', 1) for p in lod_planes],  dtype=np.float32)
    pred_areas = np.array([p.get('area_2d_px', 1) for p in pred_planes], dtype=np.float32)

    # Greedy matching by descending similarity
    L, P = len(lod_planes), len(pred_planes)
    pairs = sorted(
        [(float(sim_mat[i, j]), i, j) for i in range(L) for j in range(P)],
        key=lambda x: -x[0],
    )

    matched_li, matched_pi = set(), set()
    matched_pairs: List[Dict] = []

    for sim_val, li, pi in pairs:
        if sim_val < normal_cos_thresh:
            break
        if li in matched_li or pi in matched_pi:
            continue
        a_lod  = max(lod_areas[li],  1.0)
        a_pred = max(pred_areas[pi], 1.0)
        ratio  = float(max(a_lod, a_pred) / min(a_lod, a_pred))
        if ratio > size_ratio_max:
            continue
        matched_li.add(li)
        matched_pi.add(pi)
        lp = lod_planes[li]
        pp = pred_planes[pi]
        c_lod  = lp.get('centroid_2d')
        c_pred = pp.get('centroid_2d')
        dist_2d = float(np.linalg.norm(c_lod - c_pred)) if (c_lod is not None and c_pred is not None) else float('nan')
        matched_pairs.append({
            'lod':        lp,
            'pred':       pp,
            'normal_sim': sim_val,
            'dist_2d_px': dist_2d,
            'size_ratio': ratio,
        })

    unmatched_lod  = [lod_planes[i] for i in range(L) if i not in matched_li]
    unmatched_pred = [pred_planes[j] for j in range(P) if j not in matched_pi]

    return matched_pairs, unmatched_lod, unmatched_pred
