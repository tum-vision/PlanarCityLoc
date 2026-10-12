import colorsys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from utils.metrics import compute_ap_at_iou, compute_panoptic_quality
from utils.planes import enrich_planes, match_planes, match_planes_by_normal, plane_pair_score


def render_by_colormap(label_map, id_to_bgr):
    """Color a label map using per-id BGR tuples and alpha values (OpenCV BGR order)."""
    H, W = label_map.shape
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    for lid, (bgr, alpha) in id_to_bgr.items():
        mask = (label_map == lid)
        if not mask.any():
            continue
        color_bgr = np.array(bgr, dtype=np.float32)
        vis[mask] = (alpha * color_bgr + (1 - alpha) * vis[mask]).astype(np.uint8)
    return vis


def nmap_to_rgb(nmap):
    """Convert normal map (H, W, 3) in [-1, 1] to RGB (H, W, 3) in [0, 255]."""
    if nmap is None:
        return None
    norm = np.linalg.norm(nmap, axis=2, keepdims=True)
    norm[norm == 0] = 1.0
    nmap = nmap / norm
    return ((nmap + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def build_planes_colormap(label_map):
    """Generate a standalone colormap for a label map (pred row)."""
    golden = 0.618033988749895
    h = 0.45
    cmap = {}
    for cid in [int(i) for i in np.unique(label_map) if i >= 0]:
        r, g, b = colorsys.hsv_to_rgb(h, 0.80, 0.90)
        cmap[cid] = ((int(b * 255), int(g * 255), int(r * 255)), 0.85)
        h = (h + golden) % 1.0
    return cmap


def build_pred_colormap(pred_ids):
    """Stable pred-plane-id → BGR colormap (shared across pose rows)."""
    golden = 0.618033988749895
    h = 0.15
    cmap = {}
    for pid in sorted(int(p) for p in pred_ids):
        r, g, b = colorsys.hsv_to_rgb(h, 0.82, 0.92)
        cmap[pid] = ((int(b * 255), int(g * 255), int(r * 255)), 0.90)
        h = (h + golden) % 1.0
    return cmap


_BUILDING_LABELS = frozenset({"wall", "roof", "ground", "closure", "building"})
_NON_BUILDING_RGBA = np.array([140, 140, 140, 255], dtype=np.uint8)
_INSTANCE_COLOR_MULT = 0x9E3779B1  # odd → bijective mod 2^24


def building_id_to_rgb(building_id: int) -> np.ndarray:
    """Bijective instance ID → RGB (24-bit permuted, visually spread)."""
    i = ((int(building_id) + 1) * _INSTANCE_COLOR_MULT) & 0xFFFFFF
    return np.array([(i >> 16) & 0xFF, (i >> 8) & 0xFF, i & 0xFF], dtype=np.uint8)


def building_ids_to_rgb(building_ids: np.ndarray) -> np.ndarray:
    """Vectorized :func:`building_id_to_rgb`. Shape (N, 3) uint8."""
    ids = np.asarray(building_ids, dtype=np.int64)
    i = ((ids + 1) * _INSTANCE_COLOR_MULT) & 0xFFFFFF
    return np.stack([
        ((i >> 16) & 0xFF).astype(np.uint8),
        ((i >> 8) & 0xFF).astype(np.uint8),
        (i & 0xFF).astype(np.uint8),
    ], axis=1)


def build_instance_rgb_colormap(instance_ids: np.ndarray) -> dict[int, np.ndarray]:
    """Map each building instance ID to a unique RGB uint8 color."""
    unique_bids = np.unique(instance_ids[instance_ids >= 0])
    return {int(bid): building_id_to_rgb(bid) for bid in unique_bids}


def face_colors_from_building_ids(
    building_ids: np.ndarray,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Per-face RGBA colors from building instance IDs. Shape (F, 4) uint8."""
    building_ids = np.asarray(building_ids, dtype=np.int32)
    n_faces = len(building_ids)
    if labels is not None and len(labels) == n_faces:
        is_building = np.array([str(l) in _BUILDING_LABELS for l in labels], dtype=bool)
    else:
        is_building = np.ones(n_faces, dtype=bool)

    rgb = building_ids_to_rgb(building_ids)
    face_colors = np.tile(_NON_BUILDING_RGBA, (n_faces, 1))
    face_colors[is_building, :3] = rgb[is_building]
    return face_colors


_UNMATCHED_BGR = ((60, 60, 60), 1.0)


def make_shared_colormap(gt_ids, pred_ids, matched_pairs):
    golden = 0.618033988749895
    h = 0.08
    colors_bgr = []
    for _ in range(len(matched_pairs)):
        r, g, b = colorsys.hsv_to_rgb(h, 0.88, 0.95)
        colors_bgr.append((int(b * 255), int(g * 255), int(r * 255)))
        h = (h + golden) % 1.0
    gt_id_to_bgr = {gid: ((70, 70, 70), 1.0) for gid in gt_ids}
    pred_id_to_bgr = {pid: ((60, 90, 130), 1.0) for pid in pred_ids}
    for k, (gi, pi, iou_val) in enumerate(matched_pairs):
        brightness = float(np.clip(0.45 + 0.55 * (float(iou_val) - 0.5) / 0.5, 0.1, 1.0))
        col = colors_bgr[k]
        gt_id_to_bgr[gt_ids[gi]] = (col, brightness)
        pred_id_to_bgr[pred_ids[pi]] = (col, brightness)
    return gt_id_to_bgr, pred_id_to_bgr


def _clip_pair_score(score) -> float | None:
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return None
    return float(np.clip(float(score), 0.0, 1.0))


def _format_pair_score_label(score) -> str:
    """Circle label: pair quality in [0, 1] (vote_frac, IoU, or composite)."""
    s = _clip_pair_score(score)
    if s is None:
        return "M"
    return f"{s:.2f}"


def draw_match_circles(canvas_bgr, label_map, ids, matched_set, score_labels):
    for gid in ids:
        ys, xs = np.where(label_map == gid)
        if not len(ys):
            continue
        cy_px, cx_px = int(ys.mean()), int(xs.mean())
        color = (0, 200, 0) if gid in matched_set else (0, 0, 220)
        label = score_labels.get(gid, "TP" if gid in matched_set else "FP")
        cv2.circle(canvas_bgr, (cx_px, cy_px), 5, color, -1)
        cv2.putText(canvas_bgr, label, (cx_px + 6, cy_px + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)
    return canvas_bgr


def _match_score_dict(match):
    """Normalize match dict keys for :func:`plane_pair_score`."""
    lod = match.get('lod', match.get('gt'))
    pred = match.get('pred')
    iou = match.get('vote_frac', match.get('iou', float('nan')))
    return {'lod': lod, 'pred': pred, 'vote_frac': iou, 'iou': iou,
            'normal_sim': match.get('normal_sim', float('nan'))}


def _match_lod_to_pred(lod_planes_e, pred_planes_e, lod_label_map, pred_label_map):
    """Match rendered LoD planes to pred planes (IoU, then normal fallback)."""
    matches, _, _ = match_planes(
        lod_planes_e, pred_planes_e, lod_label_map, pred_label_map, iou_thresh=0.20,
    )
    if not matches and lod_planes_e and pred_planes_e:
        matches, _, _ = match_planes_by_normal(
            lod_planes_e, pred_planes_e, normal_cos_thresh=0.90,
        )
    return matches


def _matches_to_plane_pairs(matches):
    """Convert match dicts to (lod_id, pred_id, score) tuples."""
    pairs = []
    for m in matches:
        lod_p = m.get('lod', m.get('gt'))
        pred_p = m['pred']
        score = m.get('vote_frac', m.get('iou', float('nan')))
        pairs.append((int(lod_p['plane_id']), int(pred_p['plane_id']), float(score)))
    return pairs


def render_lod_planes_from_pairs(lod_label_map, plane_pairs, pred_id_to_bgr):
    """Color LoD planes using pred-plane colors from explicit PnP/oracle pairs."""
    lod_id_to_bgr = {}
    lod_score_labels = {}
    for lod_id, pred_id, score in plane_pairs:
        lod_id = int(lod_id)
        pred_id = int(pred_id)
        lod_id_to_bgr[lod_id] = pred_id_to_bgr.get(pred_id, _UNMATCHED_BGR)
        lod_score_labels[lod_id] = _format_pair_score_label(score)

    for lid in [int(i) for i in np.unique(lod_label_map) if i >= 0]:
        if lid not in lod_id_to_bgr:
            lod_id_to_bgr[lid] = _UNMATCHED_BGR

    vis_bgr = render_by_colormap(lod_label_map, lod_id_to_bgr)
    lod_ids = [int(i) for i in np.unique(lod_label_map) if i >= 0]
    matched_lod = set(lod_score_labels.keys())
    draw_match_circles(vis_bgr, lod_label_map, lod_ids, matched_lod, lod_score_labels)
    return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB), len(lod_score_labels)


def _show_planes_column(p_info):
    """Whether to render the planes column for this pose row."""
    if 'show_planes' in p_info:
        return bool(p_info['show_planes'])
    name = p_info['name']
    if name in ('GPS', 'Cost Volume Selection', 'Particle Filter') or name.startswith('Final'):
        return False
    if p_info.get('is_gt'):
        return True
    return name.startswith(('A-iter', 'B-iter'))


def render_lod_planes_matched_to_pred(
    lod_label_map,
    lod_segments,
    lod_normals,
    lod_depth,
    pred_planes_e,
    pred_label_map,
    pred_id_to_bgr,
    K,
):
    """Render LoD planes using pred-plane colors; annotate pair scores on LoD planes."""
    lod_planes_e = enrich_planes(lod_segments, lod_normals, lod_depth, K)
    matches = _match_lod_to_pred(lod_planes_e, pred_planes_e, lod_label_map, pred_label_map)

    lod_id_to_bgr = {}
    lod_score_labels = {}
    for m in matches:
        lod_p = m.get('lod', m.get('gt'))
        pred_p = m['pred']
        lod_id = int(lod_p['plane_id'])
        pred_id = int(pred_p['plane_id'])
        lod_id_to_bgr[lod_id] = pred_id_to_bgr.get(pred_id, _UNMATCHED_BGR)
        score = plane_pair_score(_match_score_dict(m))
        lod_score_labels[lod_id] = _format_pair_score_label(score)

    for lid in [int(i) for i in np.unique(lod_label_map) if i >= 0]:
        if lid not in lod_id_to_bgr:
            lod_id_to_bgr[lid] = _UNMATCHED_BGR

    vis_bgr = render_by_colormap(lod_label_map, lod_id_to_bgr)
    lod_ids = [int(i) for i in np.unique(lod_label_map) if i >= 0]
    matched_lod = set(lod_score_labels.keys())
    draw_match_circles(vis_bgr, lod_label_map, lod_ids, matched_lod, lod_score_labels)
    return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB), len(matches)


def _pose_title(name, t_err=None, r_err=None, *, is_gt=False, timing_s=None):
    if is_gt:
        return f"{name}\n(reference)"
    t_str = f"{t_err:.2f}m" if t_err is not None and not np.isnan(t_err) else "—"
    r_str = f"{r_err:.2f}°" if r_err is not None and not np.isnan(r_err) else "—"
    title = f"{name}\nt={t_str}  r={r_str}"
    if timing_s is not None and not np.isnan(timing_s):
        title += f"\n{timing_s:.1f}s"
    return title


def _silhouette_boundary_mask(depth_mask: np.ndarray) -> np.ndarray:
    """Morphological gradient of a depth silhouette (building outlines)."""
    mask_u8 = depth_mask.astype(np.uint8)
    if not mask_u8.any():
        return mask_u8
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel)


def _overlay_mask_on_uav(uav_bgr, depth_mask, *, tint=(0, 0, 220), alpha=0.45):
    """Semi-transparent depth silhouette on the UAV image (LoD-Loc style)."""
    panel = uav_bgr.copy().astype(np.float32)
    m = depth_mask.astype(np.float32)[..., None]
    color = np.array(tint, dtype=np.float32)
    return np.clip(panel * (1.0 - alpha * m) + color * alpha * m, 0, 255).astype(np.uint8)


def _overlay_pose_on_uav(uav_bgr, depth_mask, edge_map=None):
    """Draw rendered silhouette outlines on the UAV image for pose rows."""
    panel = uav_bgr.copy()
    mask_u8 = depth_mask.astype(np.uint8)
    if not mask_u8.any():
        return panel
    boundary = _silhouette_boundary_mask(mask_u8).astype(bool)
    cov = float(mask_u8.mean())
    if cov > 0.85 and edge_map is not None:
        boundary |= edge_map > 0.15
    elif cov > 0.85:
        boundary |= cv2.Canny(mask_u8 * 255, 50, 150).astype(bool)
    else:
        boundary = cv2.dilate(boundary.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
    panel[boundary] = (0, 255, 200)
    return panel


def _compute_metrics(pred_mask, pred_edges, pred_normals, pred_planes,
                     target_mask, target_edges, target_normals, target_planes):
    """Compute IoU, EDT, Normal Sim, PQ/SQ/RQ/AP."""
    metrics = {}

    if pred_mask is not None and target_mask is not None:
        inter = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        metrics['IoU'] = float(inter) / float(union) if union > 0 else 0.0

    if pred_edges is not None and target_edges is not None:
        edge_binary = (pred_edges > 0.3).astype(bool)
        if edge_binary.any():
            edge_dt = distance_transform_edt(~edge_binary).astype(np.float32)
            target_edge_binary = (target_edges > 0.5).astype(bool)
            if target_edge_binary.any():
                metrics['EDT'] = float(edge_dt[target_edge_binary].mean())
                
            pred_edges_relaxed = edge_dt < 2.0
            n_pred_edges = pred_edges_relaxed.sum()
            overlap = (target_edge_binary & pred_edges_relaxed).sum()
            metrics['eRec'] = float(overlap) / float(max(n_pred_edges, 1.0))

    if pred_normals is not None and target_normals is not None and target_mask is not None:
        valid_n = (
            target_mask
            & (np.linalg.norm(pred_normals, axis=-1) > 0.1)
            & (np.linalg.norm(target_normals, axis=-1) > 0.1)
        )
        if valid_n.any():
            cos = np.clip(
                np.sum(pred_normals[valid_n] * target_normals[valid_n], axis=-1)
                / (np.linalg.norm(pred_normals[valid_n], axis=-1)
                   * np.linalg.norm(target_normals[valid_n], axis=-1) + 1e-8),
                -1.0, 1.0,
            )
            metrics['AngErr'] = float(np.degrees(np.arccos(cos)).mean())

    if pred_planes is not None and target_planes is not None:
        pq_metrics = compute_panoptic_quality(target_planes, pred_planes)
        metrics['PQ'] = pq_metrics['PQ']
        metrics['SQ'] = pq_metrics['SQ']
        metrics['RQ'] = pq_metrics['RQ']
        pred_ids = [int(i) for i in np.unique(pred_planes) if i >= 0]
        pred_sizes = [(pred_planes == pid).sum() for pid in pred_ids]
        metrics['AP'] = compute_ap_at_iou(target_planes, pred_planes, pred_sizes)
        metrics['N'] = len(pred_ids)

    return metrics


def plot_pipeline_results(
    out_path,
    uav_img,
    pred_mask,
    pred_edges,
    pred_normals,
    pred_planes,
    poses_info,
    lod,
    K,
    rH,
    rW,
    *,
    pred_planes_e=None,
    moge_timing_s=None,
    seg_normal_cos=0.995,
    seg_depth_rel=0.02,
    seg_min_pixels=50,
):
    """
    Multi-row pipeline figure.

    Row 0: UAV + moge2 predictions.
    Rows 1..N-1: rendered LoD attributes per pose (with t/r errors in titles).
    Last row: GT pose render (reference).

    ``poses_info`` entries: ``name``, ``pose`` (w2c), optional ``t_err``, ``r_err``,
    optional ``timing_s`` (stage wall time in seconds),
    optional ``is_gt`` (last row should be GT with ``is_gt=True``),
    optional ``show_planes`` (default: PnP stages + GT only),
    optional ``plane_pairs`` — list of ``(lod_id, pred_id, score)`` from PnP.
    """
    if poses_info and not poses_info[-1].get('is_gt'):
        raise ValueError("poses_info last entry must be GT with is_gt=True")

    pred_ids = [int(i) for i in np.unique(pred_planes) if i >= 0]
    pred_id_to_bgr = build_pred_colormap(pred_ids)

    num_rows = 1 + len(poses_info)
    fig, axes = plt.subplots(num_rows, 5, figsize=(20, 4 * num_rows))
    if num_rows == 1:
        axes = axes[np.newaxis, :]

    for ax in axes.flat:
        ax.axis('off')

    # Row 0 — predictions
    axes[0, 0].imshow(cv2.cvtColor(uav_img, cv2.COLOR_BGR2RGB))
    title0 = "UAV Image"
    if moge_timing_s is not None:
        title0 += f"\n{moge_timing_s:.1f}s (moge)"
    axes[0, 0].set_title(title0)

    axes[0, 1].imshow(pred_mask, cmap='gray')
    axes[0, 1].set_title("Pred Mask")

    axes[0, 2].imshow(pred_edges, cmap='magma', vmin=0.0, vmax=1.0)
    axes[0, 2].set_title("Pred Edges")

    axes[0, 3].imshow(nmap_to_rgb(pred_normals))
    axes[0, 3].set_title("Pred Normals")

    pred_planes_vis = render_by_colormap(pred_planes, pred_id_to_bgr)
    axes[0, 4].imshow(cv2.cvtColor(pred_planes_vis, cv2.COLOR_BGR2RGB))
    axes[0, 4].set_title(f"Pred Planes\nN={len(pred_ids)}")

    pred_m_bool = pred_mask > 0.5
    if pred_planes_e is None:
        raise ValueError("pred_planes_e is required for plane-match visualization")

    all_w2cs = [p_info['pose'] for p_info in poses_info]
    depths_batch, normals_batch = lod.render_depth_normals_batch(all_w2cs, K, (rH, rW))
    edges_batch = lod.render_edges_batch(all_w2cs, K, (rH, rW), sobel_scale=8.0)

    for i, p_info in enumerate(poses_info):
        row = i + 1
        name = p_info['name']
        w2c = p_info['pose']
        is_gt = bool(p_info.get('is_gt', False))

        r_depth = depths_batch[i]
        r_normals = normals_batch[i]
        r_mask = r_depth > 0
        r_edges = edges_batch[i] * r_mask.astype(np.float32)

        _, _, r_planes, r_segments = lod.get_plane_segmentation(
            w2c, K, (rH, rW),
            normal_cos_thresh=seg_normal_cos,
            depth_rel_thresh=seg_depth_rel,
            min_pixels=seg_min_pixels,
            depth=r_depth, normals=r_normals,
        )

        metrics = _compute_metrics(
            pred_m_bool, pred_edges, pred_normals, pred_planes,
            r_mask, r_edges, r_normals, r_planes,
        )

        pose_panel = _overlay_pose_on_uav(uav_img, r_mask, edge_map=r_edges)
        axes[row, 0].imshow(cv2.cvtColor(pose_panel, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_title(_pose_title(
            name, p_info.get('t_err'), p_info.get('r_err'), is_gt=is_gt,
            timing_s=p_info.get('timing_s'),
        ), fontsize=11)

        mask_panel = _overlay_mask_on_uav(uav_img, r_mask)
        axes[row, 1].imshow(cv2.cvtColor(mask_panel, cv2.COLOR_BGR2RGB))
        title1 = f"{name} Silhouette"
        if 'IoU' in metrics:
            title1 += f"\nIoU: {metrics['IoU']:.3f}"
        title1 += f"\n({100.0 * float(r_mask.mean()):.0f}% depth)"
        axes[row, 1].set_title(title1, fontsize=10)

        axes[row, 2].imshow(r_edges, cmap='magma', vmin=0.0, vmax=1.0)
        title2 = f"{name} Edges"
        if 'EDT' in metrics:
            title2 += f"\nEDT: {metrics['EDT']:.1f}px"
        if 'eRec' in metrics:
            title2 += f"  eRec: {metrics['eRec']:.3f}"
        axes[row, 2].set_title(title2, fontsize=10)

        axes[row, 3].imshow(nmap_to_rgb(r_normals))
        title3 = f"{name} Normals"
        if 'AngErr' in metrics:
            title3 += f"\nAngErr: {metrics['AngErr']:.1f}°"
        axes[row, 3].set_title(title3, fontsize=10)

        if _show_planes_column(p_info):
            planes_label_map = p_info.get('lod_label_map')
            if planes_label_map is None:
                planes_label_map = r_planes

            if 'plane_pairs' in p_info:
                r_planes_rgb, n_matched = render_lod_planes_from_pairs(
                    planes_label_map, p_info['plane_pairs'], pred_id_to_bgr,
                )
            else:
                lod_planes_e = enrich_planes(r_segments, r_normals, r_depth, K)
                matches = _match_lod_to_pred(
                    lod_planes_e, pred_planes_e, r_planes, pred_planes,
                )
                r_planes_rgb, n_matched = render_lod_planes_from_pairs(
                    planes_label_map, _matches_to_plane_pairs(matches), pred_id_to_bgr,
                )

            axes[row, 4].imshow(r_planes_rgb)
            title4 = f"{name} Planes\nmatched={n_matched}  score∈[0,1]"
            if 'PQ' in metrics:
                title4 += (
                    f"\nPQ/SQ/RQ/AP: {metrics['PQ']:.2f}/"
                    f"{metrics['SQ']:.2f}/{metrics['RQ']:.2f}/{metrics['AP']:.2f}"
                )
            axes[row, 4].set_title(title4, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
