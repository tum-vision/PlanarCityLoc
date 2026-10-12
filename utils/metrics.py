"""
utils/metrics.py
================
General-purpose scoring metrics for mask comparison, depth alignment,
and normal consistency. Used by the pose estimation pipeline and evaluation.
"""

import cv2
import numpy as np


def depth_sobel_edge_map(depth: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """Structural edges from a metric depth map (no extra GPU raster)."""
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy + 1e-8)
    return (mag > thresh).astype(np.float32)


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU between two binary masks."""
    a = mask_a > 0
    b = mask_b > 0
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def normals_from_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Compute camera-frame surface normals from a depth map.

    Unprojects depth to 3D via K, then finite-differences for normals.

    Args:
        depth: (H, W) float32 metric depth
        K: (3, 3) intrinsics

    Returns:
        (H, W, 3) float32 unit normals in camera frame (zero where invalid)
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    z = depth.astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)
    points[depth <= 0] = np.nan

    # Finite differences
    dx = np.zeros_like(points)
    dy = np.zeros_like(points)
    dx[:, :-1] = points[:, 1:] - points[:, :-1]
    dy[:-1, :] = points[1:, :] - points[:-1, :]

    normals = np.cross(dx, dy)
    nlen = np.linalg.norm(normals, axis=-1, keepdims=True)
    nlen = np.where(nlen > 1e-8, nlen, 1.0)
    normals = normals / nlen
    invalid = ~np.isfinite(normals).all(axis=-1)
    normals[invalid] = 0
    return normals.astype(np.float32)


def compute_depth_ncc(
    pred_depth: np.ndarray,
    rendered_depth: np.ndarray,
    mask: np.ndarray,
    min_pixels: int = 50,
) -> float:
    """Normalized cross-correlation between predicted and rendered depth.

    Affine-invariant: handles scale/shift bias from monocular estimators.

    Args:
        pred_depth: (H, W) predicted depth
        rendered_depth: (H, W) LoD-rendered depth
        mask: (H, W) binary overlap region
        min_pixels: minimum valid pixels for a meaningful score

    Returns:
        NCC in [-1, 1] (higher = better), or -1.0 if insufficient data
    """
    valid = (rendered_depth > 0.1) & (pred_depth > 0.1) & (mask > 0)
    n = int(valid.sum())
    if n < min_pixels:
        return -1.0

    r = rendered_depth[valid].astype(np.float64)
    o = pred_depth[valid].astype(np.float64)

    r_mean, o_mean = r.mean(), o.mean()
    r_std, o_std = r.std(), o.std()
    if r_std < 1e-6 or o_std < 1e-6:
        return 0.0

    return float(((r - r_mean) * (o - o_mean)).mean() / (r_std * o_std))


def compute_normal_consistency(
    pred_normals: np.ndarray,
    rendered_depth: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
    min_pixels: int = 50,
) -> float:
    """Mean cosine similarity between predicted and LoD-derived normals.

    Derives normals from the rendered depth map, then computes mean cosine
    similarity on the building overlap region.

    Args:
        pred_normals: (H, W, 3) predicted camera-frame normals
        rendered_depth: (H, W) LoD-rendered metric depth
        K: (3, 3) intrinsics (at the rendering resolution)
        mask: (H, W) binary overlap mask
        min_pixels: minimum valid pixels

    Returns:
        Mean cosine similarity in [-1, 1], or -1.0 if insufficient data
    """
    rendered_normals = normals_from_depth(rendered_depth, K)

    valid = ((mask > 0)
             & (rendered_depth > 0.1)
             & (np.linalg.norm(rendered_normals, axis=-1) > 0.5)
             & (np.linalg.norm(pred_normals, axis=-1) > 0.5))
    n = int(valid.sum())
    if n < min_pixels:
        return -1.0

    rn = rendered_normals[valid]
    pn = pred_normals[valid]

    cos_sim = (rn * pn).sum(axis=-1)
    return float(np.clip(cos_sim, -1.0, 1.0).mean())


def normal_angular_error_deg(
    pred_normals: np.ndarray,
    ref_normals: np.ndarray,
    valid: np.ndarray,
    *,
    undirected: bool = True,
) -> np.ndarray:
    """Per-pixel angular error (degrees) between normal maps on ``valid`` pixels."""
    pn = pred_normals[valid].astype(np.float64)
    rn = ref_normals[valid].astype(np.float64)
    pn_norm = np.linalg.norm(pn, axis=-1)
    rn_norm = np.linalg.norm(rn, axis=-1)
    keep = (pn_norm > 1e-6) & (rn_norm > 1e-6)
    if not np.any(keep):
        return np.array([], dtype=np.float64)
    pn = pn[keep] / pn_norm[keep, None]
    rn = rn[keep] / rn_norm[keep, None]
    cos = np.sum(pn * rn, axis=-1)
    if undirected:
        cos = np.abs(cos)
    cos = np.clip(cos, 0.0, 1.0)
    return np.degrees(np.arccos(cos))


def summarize_normal_angular_error(
    pred_normals: np.ndarray,
    ref_normals: np.ndarray,
    valid: np.ndarray,
    *,
    undirected: bool = True,
) -> dict[str, float]:
    """Summary stats for MoGe-vs-reference normal angular error (degrees)."""
    errs = normal_angular_error_deg(
        pred_normals, ref_normals, valid, undirected=undirected,
    )
    if errs.size == 0:
        return {
            'mean_deg': float('nan'),
            'median_deg': float('nan'),
            'p90_deg': float('nan'),
            'frac_gt_30deg': float('nan'),
            'n_valid': 0,
        }
    return {
        'mean_deg': float(np.mean(errs)),
        'median_deg': float(np.median(errs)),
        'p90_deg': float(np.percentile(errs, 90)),
        'frac_gt_30deg': float(np.mean(errs > 30.0)),
        'n_valid': int(errs.size),
    }

# ── Dense-prediction metrics ──────────────────────────────────────────────────

def depth_silog(pred_d: np.ndarray, gt_d: np.ndarray) -> float:
    """Scale-invariant log RMSE (SILog).

    SILog = sqrt( E[d²] − E[d]² )  where d_i = log(pred_i) − log(gt_i).
    No prior scale alignment is required.
    """
    valid = np.isfinite(gt_d) & (gt_d > 0) & np.isfinite(pred_d) & (pred_d > 0)
    if valid.sum() < 10:
        return float('nan')
    ld = np.log(pred_d[valid].astype(np.float64)) - np.log(gt_d[valid].astype(np.float64))
    return float(np.sqrt(max(0.0, np.mean(ld**2) - np.mean(ld)**2)))


def ang_err(pred_n: np.ndarray, gt_n: np.ndarray, gt_hit: np.ndarray) -> float:
    """Mean angular error (degrees) on hit pixels."""
    mask = gt_hit.astype(bool)
    if not mask.any():
        return float('nan')
    p = pred_n[mask]
    g = gt_n[mask]
    p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
    g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)
    cos = np.clip((p * g).sum(axis=1), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def mask_iou(pred_prob: np.ndarray, gt_mask: np.ndarray, thresh: float = 0.5) -> float:
    """IoU of binarised prediction vs GT binary mask."""
    pred_b = pred_prob > thresh
    gt_b   = gt_mask.astype(bool)
    inter  = (pred_b & gt_b).sum()
    union  = (pred_b | gt_b).sum()
    return float(inter / union) if union > 0 else float('nan')


def edge_f1(pred_prob: np.ndarray, gt_edge: np.ndarray, thresh: float = 0.5) -> float:
    """F1 score of binarised edge prediction vs GT edge map."""
    pred_b = pred_prob > thresh
    gt_b   = gt_edge.astype(bool)
    tp = (pred_b & gt_b).sum()
    fp = (pred_b & ~gt_b).sum()
    fn = (~pred_b & gt_b).sum()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    return float(2 * prec * rec / (prec + rec + 1e-8))


def compute_depth_metrics(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    min_pixels: int = 10,
) -> dict:
    """Compute depth quality metrics without creating any plot.

    Returns a dict with keys ``'silog'``, ``'rmse'``, ``'mae'`` (all float).
    Scale alignment: least-squares median scaling for RMSE/MAE.
    """
    metrics = {}
    metrics['silog'] = depth_silog(pred_depth, gt_depth)

    vp = np.isfinite(pred_depth) & (pred_depth > 0)
    vg = np.isfinite(gt_depth)   & (gt_depth   > 0)
    valid = vp & vg
    if valid.sum() >= min_pixels:
        p = pred_depth[valid].astype(np.float64)
        g = gt_depth[valid].astype(np.float64)
        s = float(np.dot(p, g) / (np.dot(p, p) + 1e-8))
        diff = s * p - g
        metrics['rmse'] = float(np.sqrt(np.mean(diff ** 2)))
        metrics['mae']  = float(np.mean(np.abs(diff)))
    else:
        metrics['rmse'] = float('nan')
        metrics['mae']  = float('nan')
    return metrics



# ── Batch IoU (merged from utils/lodloc/metrics.py) ──────────────────────────

import torch as _torch


def iou_binary(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Binary mask IoU (numpy, scalar)."""
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def iou_binary_batch(mask_gt: _torch.Tensor, masks_pred: _torch.Tensor) -> _torch.Tensor:
    """Batch IoU: mask_gt (H,W), masks_pred (N,H,W) → (N,) scores."""
    mask_gt_bool = mask_gt.bool().unsqueeze(0)
    masks_pred_bool = masks_pred.bool()
    intersection = (mask_gt_bool & masks_pred_bool).sum(dim=(1, 2)).float()
    union = (mask_gt_bool | masks_pred_bool).sum(dim=(1, 2)).float()
    return intersection / (union + 1e-8)


def iou_pred_recall_batch(query_mask: _torch.Tensor, renders: _torch.Tensor) -> _torch.Tensor:
    """Fraction of query (pred) mask pixels covered by render — ignores extra mesh area."""
    query_bool = query_mask.bool().unsqueeze(0)
    renders_bool = renders.bool()
    intersection = (query_bool & renders_bool).sum(dim=(1, 2)).float()
    query_px = query_bool.sum(dim=(1, 2)).float().clamp(min=1.0)
    return intersection / query_px


def mask_overlap_batch(
    mode: str,
    query_mask: _torch.Tensor,
    renders: _torch.Tensor,
) -> _torch.Tensor:
    if mode == "pred_recall":
        return iou_pred_recall_batch(query_mask, renders)
    return iou_binary_batch(query_mask, renders)


def iou_multiclass_batch(target: _torch.Tensor,
                         hyps: _torch.Tensor,
                         num_classes: int,
                         ignore_index: int = 0) -> _torch.Tensor:
    """Batch mean-IoU over foreground classes: target (H,W), hyps (N,H,W) → (N,)."""
    if target.dtype != hyps.dtype:
        target = target.to(hyps.dtype)
    ious_per_class = []
    for c in range(num_classes):
        if c == ignore_index:
            continue
        tgt_c = (target == c).unsqueeze(0)
        hyp_c = (hyps == c)
        inter = (tgt_c & hyp_c).sum(dim=(1, 2)).float()
        union = (tgt_c | hyp_c).sum(dim=(1, 2)).float()
        iou_c = _torch.where(union > 0, inter / union, _torch.ones_like(union))
        ious_per_class.append(iou_c)
    if not ious_per_class:
        raise ValueError(f"No foreground classes (num_classes={num_classes}, ignore_index={ignore_index})")
    return _torch.stack(ious_per_class, dim=0).mean(dim=0)


# ── Plane instance segmentation metrics ───────────────────────────────────────

DEFAULT_PLANE_IOU_THRESH = 0.5


def _compute_plane_iou_matrix(gt_label_map, pred_label_map, gt_ids, pred_ids):
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
    n_gt, n_pred = len(gt_ids), len(pred_ids)
    gt_flat = gt_label_map.ravel()
    pred_flat = pred_label_map.ravel()
    gt_id_to_idx = {g: i for i, g in enumerate(gt_ids)}
    pred_id_to_idx = {p: i for i, p in enumerate(pred_ids)}
    both = (gt_flat >= 0) & (pred_flat >= 0)
    gf = np.array([gt_id_to_idx.get(g, -1) for g in gt_flat[both]])
    pf = np.array([pred_id_to_idx.get(p, -1) for p in pred_flat[both]])
    valid_pair = (gf >= 0) & (pf >= 0)
    inter_flat_idx = (gf[valid_pair] * n_pred + pf[valid_pair]).astype(np.int64)
    inter = np.bincount(inter_flat_idx, minlength=n_gt * n_pred).reshape(n_gt, n_pred)
    gt_counts = np.array([(gt_flat == g).sum() for g in gt_ids], dtype=np.float64)
    pred_counts = np.array([(pred_flat == p).sum() for p in pred_ids], dtype=np.float64)
    union = gt_counts[:, None] + pred_counts[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _greedy_plane_match(iou_mat, iou_thresh):
    matched_gt, matched_pred = set(), set()
    pairs = [(iou_mat[i, j], i, j)
             for i in range(iou_mat.shape[0])
             for j in range(iou_mat.shape[1])]
    pairs.sort(key=lambda x: -x[0])
    result = []
    for iou, gi, pi in pairs:
        if iou < iou_thresh:
            break
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        result.append((gi, pi, float(iou)))
    return result


def compute_panoptic_quality(
    gt_label_map,
    pred_label_map,
    iou_thresh: float = DEFAULT_PLANE_IOU_THRESH,
):
    """PQ, SQ, RQ for a single plane-instance label map pair."""
    gt_ids = [int(x) for x in np.unique(gt_label_map) if x >= 0]
    pred_ids = [int(x) for x in np.unique(pred_label_map) if x >= 0]
    n_gt, n_pred = len(gt_ids), len(pred_ids)
    if n_gt == 0 and n_pred == 0:
        return dict(PQ=1.0, SQ=1.0, RQ=1.0, TP=0, FP=0, FN=0,
                    n_gt=0, n_pred=0, iou_matrix=np.zeros((0, 0)),
                    gt_ids=gt_ids, pred_ids=pred_ids, matched_pairs=[])
    if n_gt == 0:
        return dict(PQ=0.0, SQ=0.0, RQ=0.0, TP=0, FP=n_pred, FN=0,
                    n_gt=0, n_pred=n_pred, iou_matrix=np.zeros((0, n_pred)),
                    gt_ids=gt_ids, pred_ids=pred_ids, matched_pairs=[])
    if n_pred == 0:
        return dict(PQ=0.0, SQ=0.0, RQ=0.0, TP=0, FP=0, FN=n_gt,
                    n_gt=n_gt, n_pred=0, iou_matrix=np.zeros((n_gt, 0)),
                    gt_ids=gt_ids, pred_ids=pred_ids, matched_pairs=[])
    iou_mat = _compute_plane_iou_matrix(gt_label_map, pred_label_map, gt_ids, pred_ids)
    matched = _greedy_plane_match(iou_mat, iou_thresh)
    tp = len(matched)
    fp = n_pred - tp
    fn = n_gt - tp
    sq = float(np.mean([m[2] for m in matched])) if tp > 0 else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    return dict(PQ=sq * rq, SQ=sq, RQ=rq, TP=tp, FP=fp, FN=fn,
                n_gt=n_gt, n_pred=n_pred,
                iou_matrix=iou_mat, gt_ids=gt_ids, pred_ids=pred_ids,
                matched_pairs=matched)


def compute_ap_at_iou(
    gt_label_map,
    pred_label_map,
    pred_sizes,
    iou_thresh: float = DEFAULT_PLANE_IOU_THRESH,
):
    """AP@iou_thresh for one image (11-point interpolation)."""
    gt_ids = [int(x) for x in np.unique(gt_label_map) if x >= 0]
    pred_ids = [int(x) for x in np.unique(pred_label_map) if x >= 0]
    if not gt_ids or not pred_ids:
        return 0.0
    iou_mat = _compute_plane_iou_matrix(gt_label_map, pred_label_map, gt_ids, pred_ids)
    pred_order = sorted(range(len(pred_ids)), key=lambda i: -pred_sizes[i])
    matched_gt = set()
    tp = fp = 0
    precisions, recalls = [], []
    for pi_rank in pred_order:
        col = iou_mat[:, pi_rank]
        best_gi = int(np.argmax(col))
        if col[best_gi] >= iou_thresh and best_gi not in matched_gt:
            tp += 1
            matched_gt.add(best_gi)
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / len(gt_ids))
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    return float(sum(precisions[recalls >= t].max() if (recalls >= t).any() else 0.0
                     for t in np.linspace(0, 1, 11)) / 11.0)


if __name__ == "__main__":
    # Smoke test
    H, W = 64, 64
    mask_a = np.zeros((H, W), dtype=np.uint8)
    mask_b = np.zeros((H, W), dtype=np.uint8)
    mask_a[10:40, 10:40] = 255
    mask_b[20:50, 20:50] = 255
    iou = compute_iou(mask_a, mask_b)
    print(f"IoU test: {iou:.3f} (expected ~0.2)")

    depth = np.random.rand(H, W).astype(np.float32) * 50 + 10
    K = np.array([[500, 0, W / 2], [0, 500, H / 2], [0, 0, 1]])
    normals = normals_from_depth(depth, K)
    print(f"Normals shape: {normals.shape}, range [{normals.min():.2f}, {normals.max():.2f}]")

    ncc = compute_depth_ncc(depth, depth * 1.5 + 3.0, mask_a)
    print(f"NCC (affine copy): {ncc:.3f} (expected ~1.0)")

    print("All metrics smoke tests passed.")