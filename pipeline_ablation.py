#!/usr/bin/env python3
"""
pipeline.py
============
UAV-D4L LoD localization pipeline.

Fixed pipeline:
  GPS prior
    -> Cost Volume Selection (2-stage mask-IoU grid)
    -> multi-scale particle filter (IoU + edge)
    -> A+B iterative centroid PnP with IRLS + normal filter
    -> mask IoU candidate selection

Usage::

    PYTHONPATH=. python pipeline.py \
        --dataset swiss_epfl --split inPlace --data_root data/Swiss-EPFL \
        --fname urbanscape-2020-09-17-piloted_00015_DJI_0047_img.jpg \
        --out outputs/eval_swiss

    # Override hyper-parameters with a custom profile:
    PYTHONPATH=. python pipeline.py --config configs/pipeline/uavd4l_outTraj.yaml ...

Configs live under ``configs/pipeline/`` (``base.yaml`` + dataset profiles).
The profile is auto-selected from ``--dataset``, ``--data_root``, and ``--split``.
"""

import argparse
import os
import random
import sys
import time
from collections import Counter as _Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt as _edt
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from moge2.inference import load_model_base_normals, run_inference
from pose_solver.cost_volume import CostVolumeConfig, CostVolumeSelector, DEFAULT_BATCH_SIZE
from pose_solver.fine import _BatchGPUContext, multi_scale_particle_filter
from pose_solver.plane_pnp import image_centroid_ransac_pnp
from pose_solver.plane_pnp import pose_error as pnp_pose_error
from utils.lod import LoD
from utils.metrics import (
    DEFAULT_PLANE_IOU_THRESH as IOU_THRESH,
    compute_ap_at_iou,
    compute_panoptic_quality,
)
from utils.planes import enrich_planes, segment_normals_into_planes
from utils.pose import CameraPose
from utils.pose import decompose_pose as _dp
from utils.pose import euler_to_camera_pose, params_to_camera_pose
from utils.pose import pose_dict_from_matrix, pose_tensor_to_dict
from utils.visualization import plot_pipeline_results
from utils.io import scale_K
from utils.timing import GpuTimer, CpuTimer
from utils.pipeline_config import PipelineRunConfig
from datasets.uavd4l import UAVD4LDataset
from utils.normal_estimators import (
    load_normal_estimator, ALL_KEYS as NORMAL_MODEL_KEYS, METRIC3D_DEPTH_CONVENTION,
)
from utils.map_ablation import without_buildings
from utils.ablation_config import candidate_scoring_config

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "thirdparty"))

# Backward-compat grid exports (from configs/pipeline/*.yaml).
CV_GRID_COMPACT = PipelineRunConfig.load("uavd4l_inTraj").cv_grid
CV_GRID_WIDE = PipelineRunConfig.load("uavd4l_outTraj").cv_grid
CV_GRID_UAVD4L = CV_GRID_COMPACT
CV_GRID_SWISS = CV_GRID_WIDE


def select_cv_grid(dataset: str, data_root: str) -> tuple[dict, str]:
    """Deprecated: use ``PipelineRunConfig.resolve`` instead."""
    cfg = PipelineRunConfig.resolve(dataset, data_root)
    return cfg.cv_grid, cfg.profile


# ═══════════════════════════════════════════════════════════════════════════════
# Logging helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _stage_banner(title: str, timing_s: float | None = None) -> None:
    suffix = f" ({timing_s:.1f}s)" if timing_s is not None else ""
    line = f"── {title}{suffix} "
    print(f"\n{line}{'─' * max(0, 58 - len(line))}")


def _pred_mask_fraction(building_mask_np: np.ndarray) -> float:
    return float(building_mask_np.mean())


def _mask_overlap_score(inter: float, pred_px: float, union_px: float, mode: str) -> float:
    if mode == 'pred_recall':
        return float(inter) / float(pred_px) if pred_px > 0 else 0.0
    return float(inter) / float(union_px) if union_px > 0 else 0.0


def _centroid_dist_for_pairs(
    pred_label_map: np.ndarray,
    rend_label_map: np.ndarray | None,
    plane_pairs: list[tuple] | None,
) -> float | None:
    """Mean pixel centroid distance for matched (LoD plane, predicted plane) pairs."""
    if rend_label_map is None or not plane_pairs:
        return None
    dists = []
    for pair in plane_pairs:
        try:
            lpid = int(pair[0])
            ppid = int(pair[1])
        except (ValueError, TypeError, IndexError):
            continue
        rr_l, cc_l = np.where(rend_label_map == lpid)
        rr_p, cc_p = np.where(pred_label_map == ppid)
        if rr_l.size == 0 or rr_p.size == 0:
            continue
        c_l = np.array([cc_l.mean(), rr_l.mean()], dtype=np.float64)
        c_p = np.array([cc_p.mean(), rr_p.mean()], dtype=np.float64)
        dists.append(float(np.linalg.norm(c_l - c_p)))
    if not dists:
        return None
    return float(np.mean(dists))


def _centroid_dist_weighted_for_pairs(
    pred_label_map: np.ndarray,
    rend_label_map: np.ndarray | None,
    plane_pairs: list[tuple] | None,
) -> float | None:
    """Vote-fraction weighted centroid distance for matched plane pairs."""
    if rend_label_map is None or not plane_pairs:
        return None
    weighted_sum = 0.0
    weight_sum = 0.0
    for pair in plane_pairs:
        try:
            lpid = int(pair[0])
            ppid = int(pair[1])
        except (ValueError, TypeError, IndexError):
            continue
        vote_frac = 1.0
        if len(pair) >= 3:
            try:
                vote_frac = float(pair[2])
            except (ValueError, TypeError):
                vote_frac = 1.0
        w = max(vote_frac, 0.0)
        rr_l, cc_l = np.where(rend_label_map == lpid)
        rr_p, cc_p = np.where(pred_label_map == ppid)
        if rr_l.size == 0 or rr_p.size == 0:
            continue
        c_l = np.array([cc_l.mean(), rr_l.mean()], dtype=np.float64)
        c_p = np.array([cc_p.mean(), rr_p.mean()], dtype=np.float64)
        d = float(np.linalg.norm(c_l - c_p))
        weighted_sum += w * d
        weight_sum += w
    if weight_sum <= 1e-12:
        return None
    return float(weighted_sum / weight_sum)


def _camera_center_from_w2c(w2c: np.ndarray) -> np.ndarray:
    return -w2c[:3, :3].T @ w2c[:3, 3]


def _with_center_keep_rotation(w2c: np.ndarray, center_world: np.ndarray) -> np.ndarray:
    out = w2c.copy()
    out[:3, 3] = -out[:3, :3] @ center_world
    return out


def _damp_pose_step_by_center(
    w2c_prev: np.ndarray,
    w2c_new: np.ndarray,
    max_step_m: float,
) -> tuple[np.ndarray, float, bool]:
    """Limit camera-center step while keeping the candidate rotation."""
    c_prev = _camera_center_from_w2c(w2c_prev)
    c_new = _camera_center_from_w2c(w2c_new)
    delta = c_new - c_prev
    step = float(np.linalg.norm(delta))
    if step <= max_step_m or max_step_m <= 0:
        return w2c_new, step, False
    alpha = max_step_m / max(step, 1e-8)
    c_damped = c_prev + alpha * delta
    w2c_damped = _with_center_keep_rotation(w2c_new, c_damped)
    return w2c_damped, float(max_step_m), True


def _dash(val) -> str:
    return "  —  " if val is None else str(val)


def _print_pose_candidate_table(
    rows: list[dict],
    selected_lbl: str,
    *,
    show_gt: bool = True,
) -> None:
    """One row per candidate with selector and diagnostic metrics."""
    if show_gt:
        hdr = (f"  {'candidate':<10} {'t↓':>5} {'R↓':>5} "
               f"{'match↑':>6} {'inl↑':>5} {'pScore↑':>7} {'repr↓':>6} "
               f"{'rec↑':>5} {'IoU↑':>5} {'edge↑':>6} {'EDT↓':>6} {'eRec↑':>6} "
               f"{'dNCC↑':>6} {'nErr↓':>6} {'cDist↓':>7} {'cDW↓':>7} {'S↑':>6}")
    else:
        hdr = (f"  {'candidate':<10} {'match↑':>6} {'inl↑':>5} {'pScore↑':>7} "
               f"{'repr↓':>6} {'rec↑':>5} {'IoU↑':>5} {'edge↑':>6} {'EDT↓':>6} {'eRec↑':>6} "
               f"{'dNCC↑':>6} {'nErr↓':>6} {'cDist↓':>7} {'cDW↓':>7} {'S↑':>6}")
    print(hdr)
    for row in rows:
        lbl = row['lbl']
        marker = " *" if lbl == selected_lbl else "  "
        match_s = _dash(None if row.get('n_pts') is None else f"{row['n_pts']:d}")
        inl_s = _dash(None if row.get('n_inliers') is None else f"{row['n_inliers']:d}")
        reproj = row.get('reproj')
        reproj_s = _dash(None if reproj is None else f"{reproj:.1f}px")
        rec_s = f"{row.get('rec', 0.0):.3f}"
        iou_s = f"{row.get('iou', 0.0):.3f}"
        edge_s = f"{row.get('edge', 0.0):.3f}"
        score_s = f"{row.get('score', 0.0):.3f}"
        dncc_s = _dash(None if row.get('depth_ncc') is None else f"{row['depth_ncc']:.3f}")
        nerr_s = _dash(None if row.get('normal_err_deg') is None else f"{row['normal_err_deg']:.1f}")
        cdist_s = _dash(None if row.get('centroid_dist_px') is None else f"{row['centroid_dist_px']:.1f}")
        cdistw_s = _dash(None if row.get('centroid_dist_w_px') is None else f"{row['centroid_dist_w_px']:.1f}")
        edt = row['edt']
        edt_s = _dash(None if edt == float('inf') else f"{edt:.1f}")
        erec_s = f"{row.get('e_rec', 0.0):.3f}"
        pscore = row.get('p_score')
        pscore_s = _dash(None if pscore is None else f"{pscore:.3f}")
        
        if show_gt:
            te, re = row['t_err'], row['r_err']
            print(f"  {lbl:<10} {te:5.2f} {re:5.2f} {match_s:>6} {inl_s:>5} {pscore_s:>7} "
                  f"{reproj_s:>6} {rec_s:>5} {iou_s:>5} {edge_s:>6} {edt_s:>6} {erec_s:>6} "
                  f"{dncc_s:>6} {nerr_s:>6} {cdist_s:>7} {cdistw_s:>7} {score_s:>6}{marker}")
        else:
            print(f"  {lbl:<10} {match_s:>6} {inl_s:>5} {pscore_s:>7} "
                  f"{reproj_s:>6} {rec_s:>5} {iou_s:>5} {edge_s:>6} {edt_s:>6} {erec_s:>6} "
                  f"{dncc_s:>6} {nerr_s:>6} {cdist_s:>7} {cdistw_s:>7} {score_s:>6}{marker}")


def _safe_metrics(lst):
    return [x for x in lst if x is not None and not np.isnan(x)]


def _recall_pct(te_list, thresh_m: float) -> float:
    vals = _safe_metrics(te_list)
    if not vals:
        return float('nan')
    return 100.0 * sum(t < thresh_m for t in vals) / len(vals)


    # Save predicted poses to est_poses.txt
    est_poses_path = os.path.join(args.out, "est_poses.txt")
    with open(est_poses_path, "w") as f:
        for m in all_metrics:
            img_name = m["frame"]
            w2c = m.get("w2c_best_overall")
            if w2c is not None:
                import scipy.spatial.transform
                R_mat = w2c[:3, :3]
                t_vec = w2c[:3, 3]
                qw, qx, qy, qz = scipy.spatial.transform.Rotation.from_matrix(R_mat).as_quat()[[3, 0, 1, 2]]
                f.write(f"{img_name} {qw} {qx} {qy} {qz} {t_vec[0]} {t_vec[1]} {t_vec[2]}\n")
    print(f"Saved predicted poses to {est_poses_path}")
def _print_eval_summary_table(all_metrics: list[dict]) -> None:
    """Aggregate pose-error table: TE, RE, translation recall @1/2/3/5 m."""
    if not all_metrics:
        return

    stages = [
        ('GPS',                    'gps_prior_t_err',        'gps_prior_r_err'),
        ('Cost Volume Selection',  'cv_select_t_err',        'cv_select_r_err'),
        ('Particle Filter',        'pf_t_err',               'pf_r_err'),
        ('Final',                  'pnp_best_overall_t_err', 'pnp_best_overall_r_err'),
    ]

    print(f"\n{'=' * 82}")
    print(f"  POSE ERROR SUMMARY  ({len(all_metrics)} frames)")
    print(f"{'=' * 82}")
    hdr = (f"  {'method':<24} {'TE(m)':>7} {'RE(°)':>7} "
           f"{'R@1m':>6} {'R@2m':>6} {'R@3m':>6} {'R@5m':>6}")
    print(hdr)
    print(f"  {'-' * 78}")

    for name, te_key, re_key in stages:
        tes = _safe_metrics([m.get(te_key) for m in all_metrics])
        res = _safe_metrics([m.get(re_key) for m in all_metrics])
        if not tes:
            continue
        te_med = float(np.median(tes))
        re_med = float(np.median(res)) if res else float('nan')
        r1 = _recall_pct([m.get(te_key) for m in all_metrics], 1.0)
        r2 = _recall_pct([m.get(te_key) for m in all_metrics], 2.0)
        r3 = _recall_pct([m.get(te_key) for m in all_metrics], 3.0)
        r5 = _recall_pct([m.get(te_key) for m in all_metrics], 5.0)
        print(f"  {name:<24} {te_med:7.2f} {re_med:7.2f} "
              f"{r1:5.0f}% {r2:5.0f}% {r3:5.0f}% {r5:5.0f}%")

    if len(all_metrics) > 1:
        print(f"\n  Per-frame final TE / RE (m / °):")
        for m in all_metrics:
            stem = os.path.splitext(m.get('frame', '?'))[0]
            te = m.get('pnp_best_overall_t_err', float('nan'))
            re = m.get('pnp_best_overall_r_err', float('nan'))
            print(f"    {stem:<40} {te:6.2f}  {re:5.2f}")
            
    # Print timing summary
    print(f"\n{'=' * 82}")
    print(f"  COMPUTE TIMING SUMMARY (Avg over {len(all_metrics)} frames)")
    print(f"{'=' * 82}")
    
    time_keys = [
        ('Network Inference (No I/O)', 'time_moge'),
        ('Plane Segmentation',         'time_seg'),
        ('Edge Prep',                  'time_edge'),
        ('Cost Volume Selection',      'time_cv'),
        ('Particle Filter',            'time_pf'),
        ('Iterative Semantic PnP',     'time_ab'),
        ('Pose Selection',             'time_select'),
    ]
    
    total_avg_ms = 0.0
    for name, key in time_keys:
        times = _safe_metrics([m.get(key) for m in all_metrics])
        if not times:
            continue
        avg_ms = float(np.mean(times)) * 1000.0
        total_avg_ms += avg_ms
        print(f"  {name:<28} {avg_ms:8.2f} ms")
        
    print(f"  {'-' * 38}")
    print(f"  {'Total Pure Compute':<28} {total_avg_ms:8.2f} ms")
    
    print(f"{'=' * 82}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

def scale_K_matrix(K: np.ndarray, src_hw: tuple, dst_hw: tuple) -> np.ndarray:
    """Scale a (3, 3) intrinsic matrix from src (H, W) to dst (H, W)."""
    src_H, src_W = src_hw
    dst_H, dst_W = dst_hw
    sx, sy = dst_W / src_W, dst_H / src_H
    K2 = np.asarray(K, dtype=np.float64).copy()
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def warmup_gpu_rasterizer(lod, K, w2c_gps, rH, rW, cv_est, cfg: PipelineRunConfig):
    """Amortize nvdiffrast CUDA JIT and renderer init before the frame loop."""
    print("Warming up GPU rasterizer ...")
    t0 = time.perf_counter()

    cp = CameraPose.from_w2c(w2c_gps)
    cx, cy, cz, cyaw, cpitch, croll = _dp(cp)
    _warmup_mask = np.zeros(cfg.cv_mask_size, dtype=np.uint8)
    multi_scale_particle_filter(
        _warmup_mask, lod, K,
        cx, cy, cz, cyaw, cpitch, croll,
        iou_weight=1.0, edge_weight=0.0,
        image_size=(rH, rW), seed=0,
    )

    renderer = lod._ensure_renderer(K, (rH, rW), device='gpu')
    w2c_t = torch.tensor(np.asarray(w2c_gps, dtype=np.float32), device=renderer.dev).unsqueeze(0)
    K_t = torch.tensor(K, dtype=torch.float32, device=renderer.dev)
    with torch.no_grad():
        renderer.render_masks_batch(w2c_t, K_t)
        renderer.render_depth_normals_batch(w2c_t)
        renderer.render_edges_batch(w2c_t, sobel_scale=8.0)

    cv_est.lod = lod
    K_cv = scale_K_matrix(K, (rH, rW), cfg.cv_mask_size)
    lod._ensure_renderer(K_cv, cfg.cv_mask_size, device='gpu')
    renderer_c = lod.renderer
    renderer_c.minibatch_size = cv_est.conf.batch_size
    K_c = torch.tensor(K_cv, dtype=torch.float32, device=renderer_c.dev)
    with torch.no_grad():
        renderer_c.render_masks_batch(w2c_t, K_c)

    print(f"  GPU warm-up complete ({time.perf_counter() - t0:.1f}s)")


def crop_lod_for_batch(
    lod_full,
    dataset,
    rH,
    rW,
    cfg: PipelineRunConfig,
):
    """Crop once for the whole evaluation batch (single renderer for all frames).

    Centers on the GPS pose closest to the trajectory XY centroid and inflates
    ``footprint_extend_m`` by the batch XY spread plus cost-volume slack so every
    frame stays inside the crop without rebuilding the GPU mesh buffers.
    """
    first_intrinsics = dataset.get_first_intrinsics()
    K = scale_K(first_intrinsics['fx'], first_intrinsics['fy'], first_intrinsics['cx'], first_intrinsics['cy'], first_intrinsics['src_W'], first_intrinsics['src_H'], rW, rH)

    w2cs = dataset.get_w2c_gps_list()
    positions = np.array([CameraPose.from_w2c(w).position for w in w2cs])
    centroid_xy = positions[:, :2].mean(axis=0)
    spread_m = float(np.max(np.linalg.norm(positions[:, :2] - centroid_xy, axis=1)))
    ref_idx = int(np.argmin(np.linalg.norm(positions[:, :2] - centroid_xy, axis=1)))
    w2c_ref = w2cs[ref_idx]

    extend_total = cfg.crop_extend_m + spread_m + cfg.cv_xy_pad_m
    lod = lod_full.crop_to_camera_pose(
        w2c_ref, K, (rH, rW),
        margin=cfg.crop_margin,
        footprint_extend_m=extend_total,
    )
    return lod, K, w2c_ref, spread_m, extend_total

# ═══════════════════════════════════════════════════════════════════════════════
# Predicted-normals -> plane instance segmentation
# ═══════════════════════════════════════════════════════════════════════════════

def run_moge_inference_torch(moge, img_path, enc_res, device, target_hw):
    pil_img = PILImage.open(img_path).convert('RGB')
    if isinstance(moge, tuple):
        import cv2
        img_bgr = cv2.imread(img_path)
        rH, rW = target_hw
        img_bgr_resized = cv2.resize(img_bgr, (rW, rH))

    with GpuTimer() as gt:
        if isinstance(moge, tuple):
            moge_dn, segmentor, seg_type = moge
            
            if seg_type == 'sam3':
                pred_m_cv, _ = segmentor.predict_with_confidence(img_bgr_resized)
                pred_m_t = torch.from_numpy(pred_m_cv).to(device).float()
            elif seg_type == 'eomt':
                pred_m_cv = segmentor.predict(img_bgr_resized)
                pred_m_t = torch.from_numpy(pred_m_cv).to(device).float()
            torch.cuda.empty_cache()
            
            # Run moge2 vits for normals and depth
            # Note: We use moge2.inference.run_inference because it handles the base model too
            pred_n, _, _, pred_d, _ = run_inference(moge_dn, pil_img, enc_res=enc_res, device=device)
            rH, rW = target_hw
            pred_n_t = F.interpolate(pred_n.permute(2, 0, 1).unsqueeze(0), size=(rH, rW), mode='bilinear', align_corners=False)[0] # 3, H, W
            
            # Compute edges from normals using 3x3 Sobel
            kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
            kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
            
            grad_x = F.conv2d(pred_n_t.unsqueeze(0).view(-1, 1, rH, rW), kernel_x, padding=1)
            grad_y = F.conv2d(pred_n_t.unsqueeze(0).view(-1, 1, rH, rW), kernel_y, padding=1)
            edge_mag = torch.sqrt(grad_x**2 + grad_y**2).view(1, 3, rH, rW).mean(dim=1) # 1, H, W
            
            pred_m_t = F.interpolate(pred_m_t.unsqueeze(0).unsqueeze(0), size=(rH, rW), mode='nearest')[0, 0]
            
            # Dilate the SAM3 mask to fully encapsulate normal edges
            binary_mask = (pred_m_t > 0.5).float().unsqueeze(0).unsqueeze(0)
            dilated_mask = F.max_pool2d(binary_mask, kernel_size=5, stride=1, padding=2)[0, 0]
            
            # Mask out non-building edges using the dilated mask
            pred_e_t = edge_mag[0] * dilated_mask
            pred_e_t = torch.clamp(pred_e_t, 0.0, 1.0)
            
            if pred_d is not None:
                pred_d_t = F.interpolate(pred_d.unsqueeze(0).unsqueeze(0), size=(rH, rW), mode='bilinear', align_corners=False)[0, 0]
            else:
                pred_d_t = None
                
            pred_n_t = pred_n_t.permute(1, 2, 0)
            pred_n_t = F.normalize(pred_n_t, dim=-1)
        else:
            pred_n, pred_m, pred_e, pred_d, _ = run_inference(moge, pil_img, enc_res=enc_res, device=device)
            rH, rW = target_hw
            pred_n_t = F.interpolate(pred_n.permute(2, 0, 1).unsqueeze(0), size=(rH, rW), mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
            pred_n_t = F.normalize(pred_n_t, dim=-1)
            pred_m_t = F.interpolate(pred_m.unsqueeze(0).unsqueeze(0), size=(rH, rW), mode='bilinear', align_corners=False)[0, 0]
            if pred_e is not None:
                pred_e_t = F.interpolate(pred_e.unsqueeze(0).unsqueeze(0), size=(rH, rW), mode='bilinear', align_corners=False)[0, 0]
            else:
                pred_e_t = None
            if pred_d is not None:
                pred_d_t = F.interpolate(pred_d.unsqueeze(0).unsqueeze(0), size=(rH, rW), mode='bilinear', align_corners=False)[0, 0]
            else:
                pred_d_t = None

    return pred_n_t, pred_m_t, pred_e_t, pred_d_t, gt.elapsed_s


def prepare_pred_edge_maps(pred_m_t, pred_e_t, cfg: PipelineRunConfig):
    """Single GPU→CPU transfer; edge DT at full + cost-volume resolution."""
    pred_m_np = pred_m_t.detach().cpu().numpy()
    pred_e_np = pred_e_t.detach().cpu().numpy() if pred_e_t is not None else None
    building_mask_np = pred_m_np >= cfg.mask_thresh
    q_mask_u8 = building_mask_np.astype(np.uint8) * 255

    if pred_e_np is not None:
        edge_bin_full = pred_e_np > cfg.cv_edge_thresh
        pe_small = cv2.resize(
            pred_e_np, (cfg.cv_mask_size[1], cfg.cv_mask_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        edge_bin_cv = pe_small > cfg.cv_edge_thresh
    else:
        edge_bin_full = cv2.Canny(q_mask_u8, 50, 100).astype(bool)
        pe_small = cv2.resize(
            edge_bin_full.astype(np.float32),
            (cfg.cv_mask_size[1], cfg.cv_mask_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        edge_bin_cv = pe_small > 0.5

    edge_dt_full = _edt(~edge_bin_full).astype(np.float32)
    edge_dt_cv_np = _edt(~edge_bin_cv).astype(np.float32)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    edge_dt_cv_torch = torch.from_numpy(edge_dt_cv_np).to(dev)

    return {
        'q_mask_u8': q_mask_u8,
        'building_mask_np': building_mask_np,
        'pred_e_np': pred_e_np,
        'edge_dt_full': edge_dt_full,
        'edge_dt_cv_np': edge_dt_cv_np,
        'edge_dt_cv_torch': edge_dt_cv_torch,
    }


def _cv_grid_n_samples(conf: CostVolumeConfig) -> int:
    n = 1
    for ax in ('x', 'y', 'z', 'yaw_deg', 'pitch_deg', 'roll_deg'):
        if ax in conf.samples_xyz_deg:
            n *= conf.samples_xyz_deg[ax]
    return n


def _gps_pred_recall_cv(lod, K, w2c_gps, pred_mask_cv: np.ndarray, rH: int, rW: int) -> float:
    """Fraction of pred mask pixels hit by LoD render at GPS pose (CV resolution)."""
    K_cv = scale_K_matrix(K, (rH, rW), pred_mask_cv.shape)
    render_m = lod.render_mask(w2c_gps, K_cv, pred_mask_cv.shape) > 0
    pred_m = pred_mask_cv > 0
    pred_px = int(pred_m.sum())
    if pred_px == 0:
        return 0.0
    return float((pred_m & render_m).sum()) / pred_px


def _apply_cv_yaw_expand(cv_est: CostVolumeSelector, cfg: PipelineRunConfig, *, stage: str) -> dict:
    """Widen yaw grid; returns saved values to restore after selection."""
    conf = cv_est.conf
    saved = {
        'yaw_deg': conf.range_xyz_deg['yaw_deg'],
        'yaw_samples': conf.samples_xyz_deg['yaw_deg'],
    }
    if stage == 'large':
        conf.range_xyz_deg['yaw_deg'] = cfg.cv_yaw_deg_large_expanded
        conf.samples_xyz_deg['yaw_deg'] = cfg.cv_yaw_samples_large_expanded
    else:
        conf.range_xyz_deg['yaw_deg'] = cfg.cv_yaw_deg_small_expanded
        conf.samples_xyz_deg['yaw_deg'] = cfg.cv_yaw_samples_small_expanded
    return saved


def _restore_cv_yaw_grid(cv_est: CostVolumeSelector, saved: dict) -> None:
    cv_est.conf.range_xyz_deg['yaw_deg'] = saved['yaw_deg']
    cv_est.conf.samples_xyz_deg['yaw_deg'] = saved['yaw_samples']


def _cv_z_prescreen(
    lod,
    cv_ref: CostVolumeSelector,
    K_torch: torch.Tensor,
    mq_c: torch.Tensor,
    prior: dict,
    edge_dt_torch,
    pred_d_cv,
    cfg: PipelineRunConfig,
) -> dict:
    """Depth-heavy 1D z sweep at GPS xy/yaw to disambiguate altitude before 4D CV."""
    z_range = float(cv_ref.conf.range_xyz_deg['z'])
    if z_range < cfg.cv_z_prescreen_min_z_range or pred_d_cv is None:
        return prior
    z_conf = CostVolumeConfig(
        range_xyz_deg={'x': 0.0, 'y': 0.0, 'z': z_range, 'yaw_deg': 0.0},
        samples_xyz_deg={'x': 1, 'y': 1, 'z': cfg.cv_z_prescreen_samples, 'yaw_deg': 1},
        ori_image_size=cv_ref.conf.ori_image_size,
        mask_size=cv_ref.conf.mask_size,
        batch_size=cv_ref.conf.batch_size,
        mask_iou_mode=cv_ref.conf.mask_iou_mode,
        depth_gated=cv_ref.conf.depth_gated,
    )
    z_est = CostVolumeSelector(lod, z_conf)
    xi, z_score = z_est.select(
        K_torch, mq_c, prior, edge_dt=edge_dt_torch,
        iou_weight=0.15, edge_weight=0.15, depth_weight=0.70,
        pred_depth=pred_d_cv,
    )
    z_new = float(pose_tensor_to_dict(xi['params'])['z'])
    if abs(z_new - prior['z']) > 0.05:
        print(f"  CV z-prescreen: z {prior['z']:.1f} → {z_new:.1f}m  (score={z_score:.4f})")
    out = dict(prior)
    out['z'] = z_new
    return out


def run_cost_volume_selection(
    lod, K, building_mask, w2c_gps, rH, rW,
    cv_est_large, cv_est_small, cfg: PipelineRunConfig,
    pred_e_t=None, pred_d_t=None, gt_pose=None,
    edge_dt_torch=None, min_iou=0.0, max_edt=float('inf'),
):
    """2-stage cost-volume selection seeded from GPS prior (IoU + edge + depth)."""
    cv_est_large.lod = lod
    cv_est_small.lod = lod

    prior = pose_dict_from_matrix(np.linalg.inv(w2c_gps))
    K_torch = torch.tensor(K.astype(np.float32))

    mq_t = building_mask.float() if hasattr(building_mask, 'float') else torch.tensor(
        building_mask.astype(np.float32), device=cv_est_large.device)
    if mq_t.dim() == 2:
        mq_t = mq_t.unsqueeze(0).unsqueeze(0)
    mq_c = F.interpolate(mq_t, size=cfg.cv_mask_size, mode='nearest').squeeze() > 0.5

    pred_mask_cv = mq_c.detach().cpu().numpy()
    gps_recall = _gps_pred_recall_cv(lod, K, w2c_gps, pred_mask_cv.astype(bool), rH, rW)
    saved_large = saved_small = None
    if gps_recall < cfg.cv_expand_yaw_gps_mask_thresh:
        saved_large = _apply_cv_yaw_expand(cv_est_large, cfg, stage='large')
        saved_small = _apply_cv_yaw_expand(cv_est_small, cfg, stage='small')
        print(f"  CV yaw expanded (gps pred_recall={gps_recall:.3f} < "
              f"{cfg.cv_expand_yaw_gps_mask_thresh}, "
              f"stage1 yaw±{cfg.cv_yaw_deg_large_expanded / 2:.0f}°)")

    n_large = _cv_grid_n_samples(cv_est_large.conf)
    n_small = _cv_grid_n_samples(cv_est_small.conf)
    mb = cv_est_large.conf.batch_size
    n_chunk = (n_large + mb - 1) // mb + (n_small + mb - 1) // mb
    score_terms = ["mask"]
    if pred_e_t is not None:
        score_terms.append("edge")
    if pred_d_t is not None and cfg.cv_depth_weight > 0:
        score_terms.append("depth")
    print(f"  Cost volume: stage1={n_large}  stage2={n_small}  "
          f"renders @ {cfg.cv_mask_size[0]}x{cfg.cv_mask_size[1]}  minibatch={mb}  "
          f"(~{n_chunk} chunked passes/stage, {'+'.join(score_terms)})")

    if edge_dt_torch is None and pred_e_t is not None:
        pe = pred_e_t.unsqueeze(0).unsqueeze(0) if pred_e_t.dim() == 2 else pred_e_t
        pe_small = F.interpolate(pe.float(), size=cfg.cv_mask_size, mode='bilinear', align_corners=False).squeeze()
        edge_bin = (pe_small > cfg.cv_edge_thresh).cpu().numpy()
        edge_dt_torch = torch.from_numpy(_edt(~edge_bin).astype(np.float32)).to(cv_est_large.device)

    pred_d_cv = None
    if pred_d_t is not None and cfg.cv_depth_weight > 0:
        pd = pred_d_t.unsqueeze(0).unsqueeze(0) if pred_d_t.dim() == 2 else pred_d_t
        pred_d_cv = F.interpolate(
            pd.float(), size=cfg.cv_mask_size, mode='bilinear', align_corners=False,
        ).squeeze()

    prior = _cv_z_prescreen(
        lod, cv_est_large, K_torch, mq_c, prior, edge_dt_torch, pred_d_cv, cfg,
    )

    cv_kw = dict(
        iou_weight=cfg.cv_iou_weight,
        edge_weight=cfg.cv_edge_weight,
        depth_weight=cfg.cv_depth_weight,
        pred_depth=pred_d_cv,
        min_iou=min_iou,
        max_edt=max_edt,
    )

    with GpuTimer() as gt:
        xi_c, _ = cv_est_large.select(
            K_torch, mq_c, prior, edge_dt=edge_dt_torch, **cv_kw)
        xi_c, score = cv_est_small.select(
            K_torch, mq_c, pose_tensor_to_dict(xi_c['params']),
            edge_dt=edge_dt_torch, **cv_kw)
    t_cv = gt.elapsed_s

    if saved_large is not None:
        _restore_cv_yaw_grid(cv_est_large, saved_large)
    if saved_small is not None:
        _restore_cv_yaw_grid(cv_est_small, saved_small)

    cv_params = xi_c['params'].cpu().numpy()
    cv_cp = params_to_camera_pose(cv_params)
    w2c_cv = cv_cp.to_w2c()
    init_pose = _dp(cv_cp)

    te_cv, re_cv = float('nan'), float('nan')
    if gt_pose is not None:
        te_cv, re_cv = pnp_pose_error(cv_cp, gt_pose)
    return w2c_cv, init_pose, te_cv, re_cv, float(score), t_cv


def run_particle_filter_initialization(init_pose, lod, K, target_hw, cfg: PipelineRunConfig, *,
                                       gt_pose=None, q_mask_u8=None, edge_dt_cv_np=None,
                                       pred_depth_np=None, batch_size=None,
                                       pred_mask_frac: float = 1.0):
    """Multi-scale particle filter (IoU + edge + depth) from cost-volume prior."""
    rH, rW = target_hw
    _cx, _cy, _cz, _cyaw, _cpitch, _croll = init_pose
    _init_pose = init_pose

    if q_mask_u8 is None or edge_dt_cv_np is None:
        raise ValueError("q_mask_u8 and edge_dt_cv_np are required")

    depth_weight = cfg.pf_depth_weight if pred_depth_np is not None else 0.0

    with GpuTimer() as gt:
        _pf_result = multi_scale_particle_filter(
            q_mask_u8, lod, K,
            _cx, _cy, _cz, _cyaw, _cpitch, _croll,
            pred_depth=pred_depth_np,
            iou_weight=cfg.pf_iou_weight,
            edge_weight=cfg.pf_edge_weight,
            depth_weight=depth_weight,
            edge_dt=edge_dt_cv_np,
            image_size=(rH, rW), seed=cfg.pf_seed,
            batch_size=batch_size,
            stages=cfg.pf_stages,
            mask_iou_mode=cfg.mask_iou_mode,
        )
    t_pf = gt.elapsed_s
    _cx, _cy, _cz, _cyaw, _cpitch, _croll = _pf_result[:6]
    _pf_score = _pf_result[6]

    if pred_mask_frac < cfg.low_mask_frac:
        drift_m = float(np.linalg.norm([
            _cx - _init_pose[0], _cy - _init_pose[1], _cz - _init_pose[2],
        ]))
        if drift_m > cfg.pf_low_mask_max_drift_m:
            print(f"  PF low-mask guard: drift {drift_m:.2f}m > {cfg.pf_low_mask_max_drift_m}m "
                  f"(mask {pred_mask_frac:.1%}) — keeping CV init")
            _cx, _cy, _cz, _cyaw, _cpitch, _croll = _init_pose

    w2c_iter = euler_to_camera_pose(_cx, _cy, _cz, _cyaw, _cpitch, _croll).to_w2c()

    _te_c, _re_c = float('nan'), float('nan')
    if gt_pose is not None:
        _te_c, _re_c = pnp_pose_error(CameraPose.from_w2c(w2c_iter), gt_pose)

    return w2c_iter, _te_c, _re_c, _pf_score, t_pf


def run_iterative_semantic_pnp(
    w2c_init, lod, K, rH, rW, pred_label_map, pred_planes_e, cfg: PipelineRunConfig,
    n_pred_planes: int = 0, pred_mask_frac: float = 1.0,
):
    """Refine an estimated pose using query predictions and the map, never GT."""

    pred_e_by_pid = {p['plane_id']: p for p in pred_planes_e}
    _all_poses = [('Particle Filter', w2c_init.copy())]
    _pose_timings: dict[str, float] = {}
    _pose_pnp_meta: dict[str, dict] = {}

    if pred_mask_frac < cfg.pnp_skip_below_mask_frac:
        print(f"  PnP skipped (pred mask {pred_mask_frac:.1%} < "
              f"{cfg.pnp_skip_below_mask_frac:.0%}, LoD-noise mode)")
        return (_all_poses, None, None, _pose_timings, _pose_pnp_meta, 0.0)

    low_mask = pred_mask_frac < cfg.low_mask_frac
    # Count map planes at the available PF estimate, not at an evaluation pose.
    # Reuse this segmentation for the first PnP iteration.
    _gate_gpu_times: list[float] = []
    _gate_segmentation = lod.get_plane_segmentation(
        w2c_init, K, (rH, rW),
        normal_cos_thresh=cfg.gt_normal_cos,
        depth_rel_thresh=cfg.gt_depth_rel,
        min_pixels=cfg.min_pixels,
        gpu_render_time_s=_gate_gpu_times,
    )
    n_rendered_planes = len(_gate_segmentation[3])
    pred_rendered_ratio = n_pred_planes / max(n_rendered_planes, 1)
    strict_match = (
        n_rendered_planes > 0
        and pred_rendered_ratio > cfg.pnp_strict_pred_rendered_ratio
    )
    if strict_match:
        print(f"  PnP strict matching (pred/rendered={pred_rendered_ratio:.2f} > {cfg.pnp_strict_pred_rendered_ratio})")
        _min_overlap_px = cfg.pnp_strict_min_overlap_px
        _min_vote_frac = cfg.pnp_strict_min_vote_frac
        _min_normal_cos = cfg.pnp_strict_min_normal_cos
        _min_matches = cfg.pnp_strict_min_matches
        _reproj_px = cfg.pnp_strict_reproj_px
        _b_max_dist_px = cfg.pnp_strict_b_max_dist_px
        _b_lowe_ratio = cfg.pnp_strict_b_lowe_ratio
    else:
        _min_overlap_px = 5
        _min_vote_frac = 0.0
        _min_normal_cos = cfg.pnp_min_normal_cos
        _min_matches = 6
        _reproj_px = 20.0
        _b_max_dist_px = 120.0
        _b_lowe_ratio = 0.75

    def _merged_centroid_match_pnp(w2c_cur, lm_cur, lod_segs, lod_planes_e,
                                   min_overlap_px=_min_overlap_px,
                                   reprojection_error_px=_reproj_px,
                                   ransac_iter=5000,
                                   min_matches=_min_matches,
                                   min_normal_cos=_min_normal_cos,
                                   min_vote_frac=_min_vote_frac):
        R_ = w2c_cur[:3, :3]; t_ = w2c_cur[:3, 3]
        lod_e_dict = {p['plane_id']: p for p in lod_planes_e}
        pts_3d = []; pts_2d = []; pair_normal_cos = []; lod_pids = []
        plane_pairs = []
        for seg in lod_segs:
            lpid  = seg['plane_id']
            rows_l = seg['pixel_indices'][:, 0]
            cols_l = seg['pixel_indices'][:, 1]
            if len(rows_l) < min_overlap_px:
                continue
            pred_labels = pred_label_map[rows_l, cols_l]
            valid_labels = pred_labels[pred_labels >= 0]
            if not len(valid_labels):
                continue
            votes = _Counter(valid_labels.tolist())
            merged_r, merged_c, matched_ppids = [], [], []
            for ppid, cnt in votes.items():
                if cnt >= min_overlap_px:
                    inter_mask = (pred_labels == ppid)
                    merged_r.append(rows_l[inter_mask])
                    merged_c.append(cols_l[inter_mask])
                    matched_ppids.append(ppid)
            if not merged_r:
                continue
            primary_ppid, primary_votes = votes.most_common(1)[0]
            vote_frac = float(primary_votes) / float(len(valid_labels))
            if vote_frac < min_vote_frac:
                continue
            c2d = np.array([np.mean(np.concatenate(merged_c)),
                            np.mean(np.concatenate(merged_r))], dtype=np.float64)
            lp = lod_e_dict.get(lpid)
            if lp is None:
                continue
            c3d_cam = lp.get('centroid_3d')
            if c3d_cam is None:
                continue
            c3d_world = R_.T @ (np.array(c3d_cam) - t_)
            lod_n  = lp.get('normal_mean')
            pred_ns = [pred_e_by_pid[pp]['normal_mean']
                       for pp in matched_ppids
                       if pp in pred_e_by_pid
                       and pred_e_by_pid[pp].get('normal_mean') is not None]
            nc = None
            if lod_n is not None and pred_ns:
                avg_pn = np.mean(pred_ns, axis=0)
                avg_pn /= np.linalg.norm(avg_pn) + 1e-8
                nc = float(np.dot(lod_n, avg_pn))
                if nc < min_normal_cos:
                    continue
            pts_3d.append(c3d_world); pts_2d.append(c2d); lod_pids.append(lpid)
            pair_normal_cos.append(nc)
            plane_pairs.append((int(lpid), int(primary_ppid), vote_frac))

        if len(pts_3d) < min_matches:
            return {'success': False, 'n_pts': len(pts_3d), 'n_inliers': 0, 'inlier_ratio': 0.0, 'mean_normal_cos': 0.0, 'pose': None}
        pts_3d_arr = np.array(pts_3d, dtype=np.float64)
        pts_2d_arr = np.array(pts_2d, dtype=np.float64)
        K_cv = K.astype(np.float64)
        ret  = cv2.solvePnPRansac(
            pts_3d_arr, pts_2d_arr, K_cv, None,
            reprojectionError=reprojection_error_px,
            confidence=0.999, iterationsCount=ransac_iter,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if ret[0] is False or ret[3] is None or len(ret[3]) < 4:
            return {'success': False, 'n_pts': len(pts_3d), 'n_inliers': 0, 'inlier_ratio': 0.0, 'mean_normal_cos': 0.0, 'pose': None}
        _, rvec, tvec, inliers = ret
        inliers = inliers.ravel()
        if len(inliers) >= 4:
            rvec, tvec = cv2.solvePnPRefineLM(pts_3d_arr[inliers], pts_2d_arr[inliers], K_cv, None, rvec, tvec)
        R_est, _ = cv2.Rodrigues(rvec)
        w2c_est  = np.eye(4); w2c_est[:3, :3] = R_est; w2c_est[:3, 3] = tvec.ravel()
        pts_proj, _ = cv2.projectPoints(pts_3d_arr[inliers], rvec, tvec, K_cv, None)
        mre = float(np.mean(np.linalg.norm(pts_proj.reshape(-1, 2) - pts_2d_arr[inliers], axis=1)))
        inlier_nc = [pair_normal_cos[i] for i in inliers if i < len(pair_normal_cos) and pair_normal_cos[i] is not None]
        mnc = float(np.mean(inlier_nc)) if inlier_nc else 0.5
        pts_proj_all, _ = cv2.projectPoints(pts_3d_arr, rvec, tvec, K_cv, None)
        reproj_all = np.linalg.norm(pts_proj_all.reshape(-1, 2) - pts_2d_arr, axis=1).tolist()
        match_pairs = list(zip(lod_pids, pts_3d, pts_2d_arr.tolist(), reproj_all))
        return {'success': True, 'pose': CameraPose.from_w2c(w2c_est),
                'n_pts': len(pts_3d), 'n_inliers': len(inliers), 'inlier_ratio': len(inliers) / len(pts_3d),
                'mean_reproj_err': mre, 'mean_normal_cos': mnc, 'match_pairs': match_pairs,
                'plane_pairs': plane_pairs}

    np.random.seed(42)
    cv2.setRNGSeed(42)
    t_gpu_render = sum(_gate_gpu_times)
    t_cpu_compute = 0.0
    w2c_iter           = w2c_init.copy()
    _best_w2c_iter     = w2c_init.copy()
    _best_inlier_ratio = -1.0
    _best_iter_reproj  = float('inf')
    _pnp_ref_normals_vis   = None
    _pnp_ref_label_map_vis = None
    _irls_allowed_pids    = None
    _IRLS_THRESH_PX       = 30.0

    _all_poses = [('Particle Filter', w2c_init.copy())]
    _pose_timings = {}
    _pose_pnp_meta: dict[str, dict] = {}
    _plane_seg_cache: dict[bytes, tuple] = {
        np.ascontiguousarray(w2c_init).tobytes(): _gate_segmentation,
    }
    _pose_pnp_meta["_gate"] = {
        "n_predicted": n_pred_planes, "n_rendered": n_rendered_planes,
        "strict": bool(strict_match), "reference": "estimated_pose",
    }

    def _cached_plane_segmentation(w2c_arr, gpu_times: list[float]):
        key = np.ascontiguousarray(w2c_arr).tobytes()
        if key in _plane_seg_cache:
            return _plane_seg_cache[key]
        seg = lod.get_plane_segmentation(
            w2c_arr, K, (rH, rW),
            normal_cos_thresh=cfg.gt_normal_cos,
            depth_rel_thresh=cfg.gt_depth_rel,
            min_pixels=cfg.min_pixels,
            gpu_render_time_s=gpu_times,
        )
        _plane_seg_cache[key] = seg
        return seg

    for _itr in range(cfg.n_ab_iters):
        _gpu_t: list[float] = []
        _d_i, _n_i, _lm_i, _segs_i = _cached_plane_segmentation(w2c_iter, _gpu_t)
        t_gpu_render += sum(_gpu_t)
        if _itr == 0:
            _pnp_ref_normals_vis   = _n_i.copy()
            _pnp_ref_label_map_vis = _lm_i.copy()
        _segs_for_pnp = (_segs_i if _irls_allowed_pids is None else [s for s in _segs_i if s['plane_id'] in _irls_allowed_pids])
        if cfg.pnp_ab_mode != "b_only":
            with CpuTimer() as ct:
                _pe_i  = enrich_planes(_segs_i, _n_i, _d_i, K)
                pnp_a  = _merged_centroid_match_pnp(w2c_iter, _lm_i, _segs_for_pnp, _pe_i)
            t_cpu_compute += ct.elapsed_s
            t_iter_a = sum(_gpu_t) + ct.elapsed_s
            if not pnp_a['success']:
                print(f"  PnP A-iter{_itr+1}: aborted (n_pts={pnp_a.get('n_pts', 0)}, need ≥6 + RANSAC inliers)")
                break
            _a_reproj = pnp_a.get('mean_reproj_err', float('inf'))
            if pnp_a['inlier_ratio'] > _best_inlier_ratio:
                _best_inlier_ratio = pnp_a['inlier_ratio']
            if _a_reproj < _best_iter_reproj:
                _best_iter_reproj = _a_reproj
                _best_w2c_iter    = pnp_a['pose'].to_w2c().copy()
            _new_w2c = pnp_a['pose'].to_w2c()
            if strict_match or low_mask:
                _new_w2c, step_m, was_damped = _damp_pose_step_by_center(
                    w2c_iter, _new_w2c, cfg.pnp_strict_max_step_m
                )
                if was_damped:
                    print(f"  PnP A-iter{_itr+1}: damped step to {step_m:.2f}m "
                          f"(cap={cfg.pnp_strict_max_step_m:.2f}m)")
            else:
                _prev_pos = _camera_center_from_w2c(w2c_iter)
                _new_pos = _camera_center_from_w2c(_new_w2c)
                step_m = float(np.linalg.norm(_new_pos - _prev_pos))
            _mp = pnp_a.get('match_pairs', [])
            if cfg.pnp_retain_inliers and _mp:
                _irls_allowed_pids = {lpid for lpid, _, _, reproj in _mp if reproj <= _IRLS_THRESH_PX}
            lbl_a = f'A-iter{_itr+1}'
            _pose_pnp_meta[lbl_a] = {
                'n_pts': pnp_a['n_pts'],
                'n_inliers': pnp_a['n_inliers'],
                'reproj': pnp_a.get('mean_reproj_err'),
                'plane_pairs': pnp_a.get('plane_pairs', []),
                'lod_label_map': _lm_i.copy(),
            }
            _all_poses.append((lbl_a, _new_w2c.copy()))
            _pose_timings[lbl_a] = t_iter_a

            accept_a = _a_reproj <= _best_iter_reproj * 1.5
            if accept_a:
                w2c_iter = _new_w2c
                w2c_for_b = w2c_iter
            else:
                w2c_for_b = _best_w2c_iter.copy()
        else:
            w2c_for_b = w2c_iter

        if cfg.pnp_ab_mode != "a_only":
            _gpu_t_b: list[float] = []
            _d_b, _n_b, _lm_b, _segs_b = _cached_plane_segmentation(w2c_for_b, _gpu_t_b)
            t_gpu_render += sum(_gpu_t_b)
            with CpuTimer() as ct_b:
                _pe_b  = enrich_planes(_segs_b, _n_b, _d_b, K)
                pnp_b  = image_centroid_ransac_pnp(
                    _pe_b, pred_planes_e, w2c_for_b, K,
                    max_dist_px=_b_max_dist_px,
                    reprojection_error_px=_reproj_px,
                    min_matches=_min_matches,
                    lowe_ratio=_b_lowe_ratio,
                )
            t_cpu_compute += ct_b.elapsed_s
            t_iter_b = sum(_gpu_t_b) + ct_b.elapsed_s
            if pnp_b['success']:
                _mre_b = pnp_b.get('mean_reproj_err', float('inf'))
                w2c_b = pnp_b['pose'].to_w2c()
                if strict_match or low_mask:
                    w2c_b, b_step_m, damped_b = _damp_pose_step_by_center(
                        w2c_for_b, w2c_b, cfg.pnp_strict_max_step_m
                    )
                    if damped_b:
                        print(f"  PnP B-iter{_itr+1}: damped step to {b_step_m:.2f}m "
                              f"(cap={cfg.pnp_strict_max_step_m:.2f}m)")
                else:
                    _prev_b_pos = _camera_center_from_w2c(w2c_for_b)
                    _new_b_pos = _camera_center_from_w2c(w2c_b)
                    b_step_m = float(np.linalg.norm(_new_b_pos - _prev_b_pos))
                lbl_b = f'B-iter{_itr+1}'
                _pose_pnp_meta[lbl_b] = {
                    'n_pts': pnp_b.get('n_pts'),
                    'n_inliers': pnp_b.get('n_inliers'),
                    'reproj': _mre_b,
                    'step_m': b_step_m,
                    'plane_pairs': pnp_b.get('plane_pairs', []),
                    'lod_label_map': _lm_b.copy(),
                }
                _all_poses.append((lbl_b, w2c_b.copy()))
                _pose_timings[lbl_b] = t_iter_b
                w2c_iter = w2c_b

    t_ab = t_gpu_render + t_cpu_compute

    return (_all_poses, _pnp_ref_normals_vis, _pnp_ref_label_map_vis,
            _pose_timings, _pose_pnp_meta, t_ab)


def _pnp_round(lbl: str) -> int:
    if lbl == 'Particle Filter':
        return 0
    if 'iter' in lbl:
        try:
            return int(lbl.split('iter', 1)[1])
        except (IndexError, ValueError):
            pass
    return 0


def _pick_rank_aggregation(table_rows: list[dict]) -> str:
    """Fixed eight-feature rank aggregation; see pose_solver.selection."""
    from pose_solver.selection import pick_rank_aggregation
    return pick_rank_aggregation(table_rows)


def select_best_pose_hierarchical(
    _all_poses,
    lod,
    K,
    target_hw,
    frame_edges,
    gt_pose,
    stage_times,
    cfg: PipelineRunConfig,
    _pose_pnp_meta=None,
    t_ab=None,
    pred_depth_np=None,
    pred_normals_np=None,
    pred_label_map_np=None,
    candidate_records=None,
):

    _pose_pnp_meta = _pose_pnp_meta or {}
    show_gt = gt_pose is not None

    rH, rW = target_hw
    building_mask_np = frame_edges['building_mask_np']
    edge_dt_full = frame_edges['edge_dt_full']

    _all_w2cs = [w2c for _, w2c in _all_poses]
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    building_mask_t = torch.tensor(building_mask_np, dtype=torch.bool, device=dev)
    edge_dt_t = torch.tensor(edge_dt_full, dtype=torch.float32, device=dev)
    with GpuTimer() as gt:
        depth_list, normals_list = lod.render_depth_normals_batch(_all_w2cs, K, (rH, rW))
        depth_t = torch.from_numpy(np.stack(depth_list, axis=0)).to(dev)
        normals_t = torch.from_numpy(np.stack(normals_list, axis=0)).to(dev)
        dev = depth_t.device
        masks_t = depth_t > 0

        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32, device=dev).view(1, 1, 3, 3)
        ky = kx.transpose(2, 3)
        normals_bchw = normals_t.permute(0, 3, 1, 2).contiguous()
        gx = F.conv2d(normals_bchw, kx.expand(3, 1, 3, 3), groups=3, padding=1)
        gy = F.conv2d(normals_bchw, ky.expand(3, 1, 3, 3), groups=3, padding=1)
        lod_edges_t = (gx.pow(2) + gy.pow(2) + 1e-8).sqrt().sum(dim=1)
    t_select = gt.elapsed_s

    depth_scores = None
    if pred_depth_np is not None and cfg.cv_depth_weight > 0:
        depth_ctx = _BatchGPUContext(lod, device=str(dev), batch_size=256)
        depth_ctx.setup_stage(K, (rH, rW), frame_edges['q_mask_u8'], pred_depth=pred_depth_np)
        dm = depth_t
        dh, dw = dm.shape[-2], dm.shape[-1]
        if (dh, dw) != (depth_ctx.H, depth_ctx.W):
            padded = torch.zeros(
                dm.shape[0], depth_ctx.H, depth_ctx.W, device=dm.device, dtype=dm.dtype,
            )
            padded[:, :dh, :dw] = dm
            dm = padded
        depth_scores = depth_ctx._batch_depth_score(dm)

    pred_px_total = building_mask_t.sum().item()
    pred_depth_t = None
    if pred_depth_np is not None:
        pred_depth_t = torch.tensor(pred_depth_np, dtype=torch.float32, device=dev)
    pred_normals_t = None
    if pred_normals_np is not None:
        pred_normals_t = torch.tensor(pred_normals_np, dtype=torch.float32, device=dev)
    tau = 10.0
    _cand_metrics = []
    table_rows = []
    for i, (lbl, w2c_cand) in enumerate(_all_poses):
        _lm = masks_t[i]
        _i = (_lm & building_mask_t).sum().item()
        _u = (_lm | building_mask_t).sum().item()
        rec = float(_i) / float(pred_px_total) if pred_px_total > 0 else 0.0
        iou = float(_i) / float(_u) if _u > 0 else 0.0
        mask_sc = _mask_overlap_score(_i, pred_px_total, _u, cfg.mask_iou_mode)

        _edge_px = (lod_edges_t[i] > 0.5) & _lm
        pred_edges_t = edge_dt_t == 0
        e_rec = float((_edge_px & pred_edges_t).sum().item()) / max(float(pred_edges_t.sum().item()), 1.0)
        if _edge_px.any():
            edt_dist = float(edge_dt_t[_edge_px].mean().item())
        else:
            edt_dist = float('inf')
            
        raw_edge_sc = float(np.exp(-edt_dist / tau)) if edt_dist != float('inf') else 0.0
        n_edge_px = _edge_px.sum().item()
        
        # We can blend EDT with e_rec or just use the old econf
        # Since e_rec is very strict (exact pixel overlap), we can use a slightly relaxed overlap 
        pred_edges_relaxed = edge_dt_t < 2.0
        e_rec_relaxed = float((_edge_px & pred_edges_relaxed).sum().item()) / max(float(pred_edges_relaxed.sum().item()), 1.0)
        
        # A good pose typically has > 10% edge recall. Poses with tiny overlap (e.g. 1%) are heavily penalized.
        edge_conf = min(1.0, e_rec_relaxed / 0.10)
        
        edge_sc = raw_edge_sc * edge_conf
        depth_sc = float(depth_scores[i]) if depth_scores is not None else 0.0

        depth_ncc = None
        if pred_depth_t is not None:
            _valid_d = building_mask_t & _lm & (pred_depth_t > 0)
            if int(_valid_d.sum().item()) >= 50:
                _pd = pred_depth_t[_valid_d]
                _rd = depth_t[i][_valid_d]
                _pd = _pd - _pd.mean()
                _rd = _rd - _rd.mean()
                _den = (_pd.norm() * _rd.norm()).item()
                if _den > 1e-8:
                    depth_ncc = float((_pd * _rd).sum().item() / _den)

        normal_err_deg = None
        if pred_normals_t is not None:
            _valid_n = building_mask_t & _lm
            if int(_valid_n.sum().item()) >= 50:
                _pn = pred_normals_t[_valid_n]
                _rn = normals_t[i][_valid_n]
                _pn = F.normalize(_pn, dim=1, eps=1e-8)
                _rn = F.normalize(_rn, dim=1, eps=1e-8)
                _cos = (_pn * _rn).sum(dim=1).clamp(-1.0, 1.0)
                normal_err_deg = float(torch.rad2deg(torch.acos(_cos)).mean().item())

        combined = (
            cfg.cv_iou_weight * mask_sc
            + cfg.cv_edge_weight * edge_sc
            + cfg.cv_depth_weight * depth_sc
        )

        _te, _re = pnp_pose_error(CameraPose.from_w2c(w2c_cand), gt_pose) if show_gt else (0.0, 0.0)
        pnp_meta = _pose_pnp_meta.get(lbl, {})
        centroid_dist = _centroid_dist_for_pairs(
            pred_label_map_np,
            pnp_meta.get('lod_label_map'),
            pnp_meta.get('plane_pairs'),
        )
        centroid_dist_w = _centroid_dist_weighted_for_pairs(
            pred_label_map_np,
            pnp_meta.get('lod_label_map'),
            pnp_meta.get('plane_pairs'),
        )
        n_pts = pnp_meta.get('n_pts')
        n_inl = pnp_meta.get('n_inliers')
        inl_ratio = (float(n_inl) / float(n_pts)) if n_pts and n_inl else None
        
        plane_pairs = pnp_meta.get('plane_pairs', [])
        p_score = sum(s for _, _, s in plane_pairs) / len(plane_pairs) if plane_pairs else None
        
        table_rows.append({
            'lbl': lbl,
            't_err': _te,
            'r_err': _re,
            'n_pts': n_pts,
            'n_inliers': n_inl,
            'inl_ratio': inl_ratio,
            'p_score': p_score,
            'reproj': pnp_meta.get('reproj'),
            'rec': rec,
            'iou': iou,
            'edge': edge_sc,
            'edt': edt_dist,
            'e_rec': e_rec_relaxed,
            'depth_ncc': depth_ncc,
            'normal_err_deg': normal_err_deg,
            'centroid_dist_px': centroid_dist,
            'centroid_dist_w_px': centroid_dist_w,
            'score': combined,
        })
        _cand_metrics.append((lbl, w2c_cand, mask_sc, edt_dist, combined, centroid_dist))

    best_lbl = _pick_rank_aggregation(table_rows)
    if candidate_records is not None:
        from pose_solver.selection import rank_scores
        aggregate_scores = rank_scores(table_rows)
        candidate_records.extend(
            dict(row, w2c=pose, aggregate_rank=float(rank))
            for row, (_, pose), rank in zip(table_rows, _all_poses, aggregate_scores)
        )
    # Find the corresponding tuple in _cand_metrics
    best = next(m for m in _cand_metrics if m[0] == best_lbl)
    _best_lbl, _best_w2c_sel, _best_mask_sc, _best_edt, _best_combined, _best_cdist = best

    _stage_banner("A+B PnP refinement", t_ab)
    _print_pose_candidate_table(table_rows, selected_lbl=_best_lbl, show_gt=show_gt)

    _best_overall_w2c   = _best_w2c_sel.copy()
    _best_overall_label = _best_lbl
    _best_overall_score = _best_combined

    _best_overall_te, _best_overall_re = (
        pnp_pose_error(CameraPose.from_w2c(_best_overall_w2c), gt_pose)
        if show_gt else (float("nan"), float("nan"))
    )
    t_algo = sum(stage_times.values()) + t_select

    _stage_banner("Result")
    mask_lbl = 'rec' if cfg.mask_iou_mode == 'pred_recall' else 'IoU'
    print(f"  selected {_best_lbl}  →  t={_best_overall_te:.2f}m  R={_best_overall_re:.2f}deg  "
          f"(score={_best_combined:.3f}  {mask_lbl}={_best_mask_sc:.3f}  "
          f"EDT={_best_edt:.1f}px  cDist={_dash(None if _best_cdist is None else f'{_best_cdist:.1f}px')})")
    print(f"  algo compute: {t_algo:.1f}s  (select {t_select:.1f}s)")

    return _best_overall_w2c, _best_overall_label, _best_overall_score, _best_overall_te, _best_overall_re, t_select


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',     choices=['uavd4l', 'swiss_epfl'], default='uavd4l', help='Dataset to use')
    parser.add_argument('--split',       choices=['inPlace', 'outPlace'], default='inPlace', help='Split for Swiss-EPFL-v2')
    parser.add_argument('--config',      default=None,
                        help='Pipeline YAML under configs/pipeline/ (auto-selected from dataset if omitted)')
    parser.add_argument('--profile',     default=None,
                        help='Profile yaml stem, e.g. uavd4l_inTraj_slow (default: auto from dataset)')
    parser.add_argument('--mesh_path',   default=None, help='Path to LoD model (overrides default for dataset)')
    parser.add_argument('--ckpt',        default=None, help='MoGe checkpoint (default from config)')
    parser.add_argument('--n',           type=int, default=None)
    parser.add_argument(
        '--fname',
        nargs='+',
        default=None,
        help='One or more frame-name filters (substring match).',
    )
    parser.add_argument('--out',         default='outputs/eval_plane_seg')
    parser.add_argument('--data_root',   default='data/UAVD4L-LoD/inTraj')
    parser.add_argument('--render_size', type=int, nargs=2, default=None, metavar=('H', 'W'))
    parser.add_argument('--enc_res',     type=int, default=None, help='MoGe encoder resolution (default from config)')
    parser.add_argument('--device',      default='cuda')
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--vis',         action='store_true',
                        help='Save per-frame visualisation images and mosaics (slow).')
    parser.add_argument('--batchsize', type=int, default=None,
                        help='GPU batch size for render+score loops (default from config).')
    parser.add_argument('--no-crop', action='store_true',
                        help='Use full city mesh (paper default).')
    parser.add_argument('--crop', action='store_true', default=False,
                        help='Crop mesh to the evaluated trajectory (faster; not used in paper tables).')
    parser.add_argument('--crop_margin', type=float, default=None,
                        help='FOV footprint safety factor for trajectory crop.')
    parser.add_argument('--crop_extend_m', type=float, default=None,
                        help='Base crop radius slack (m) for trajectory crop.')
    parser.add_argument('--perturb_gt_t', type=float, default=0.0,
                        help='Translation noise to add to GT pose instead of using GPS (meters)')
    parser.add_argument('--perturb_gt_r', type=float, default=0.0,
                        help='Rotation noise to add to GT pose instead of using GPS (degrees)')
    parser.add_argument('--all_6D', action='store_true',
                        help='Prior-noise ablations only: add pitch/roll to stage-1 large CV '
                             'when --perturb_gt_r>0 (stage-2 stays 4D; PF refines attitude)')
    parser.add_argument('--normal_model', type=str, default=None,
                        choices=NORMAL_MODEL_KEYS,
                        help='Override normal/depth estimator backbone '
                             '(default: None). Options: '
                             + ', '.join(NORMAL_MODEL_KEYS))
    parser.add_argument('--stride', type=int, default=1,
                        help='Evaluate only every N-th frame.')
    parser.add_argument('--missing-buildings', action='store_true',
                        help='Also evaluate removal of small/medium/large visible buildings; default is the intact map only.')
    args = parser.parse_args()

    cfg = PipelineRunConfig.resolve(
        args.dataset, args.data_root, split=args.split,
        config_path=args.config, profile_name=args.profile,
    )
    # Coarse-cue ablations must not also change the final selector's composite
    # feature. Other variants keep their full resolved configuration unchanged.
    selection_cfg = candidate_scoring_config(cfg, PipelineRunConfig.resolve(
        args.dataset, args.data_root, split=args.split, profile_name=args.profile))
    if args.render_size is None:
        rH, rW = cfg.render_h, cfg.render_w
    else:
        rH, rW = args.render_size
    batchsize = args.batchsize if args.batchsize is not None else cfg.cv_batch_size
    ckpt = args.ckpt or cfg.moge_ckpt
    enc_res = args.enc_res if args.enc_res is not None else cfg.moge_enc_res
    if args.crop_margin is not None:
        cfg.crop_margin = args.crop_margin
    if args.crop_extend_m is not None:
        cfg.crop_extend_m = args.crop_extend_m

    use_crop = bool(args.crop) and not args.no_crop

    from pipeline import _initialize_result_files, _append_frame_result
    from utils.run_evidence import write_manifest, append_record
    scenario_names = (['Baseline', 'Missing_Small', 'Missing_Mid', 'Missing_Large']
                      if args.missing_buildings else ['Baseline'])
    scenario_outputs = {name: Path(args.out) / name for name in scenario_names}
    for folder in scenario_outputs.values():
        folder.mkdir(parents=True, exist_ok=True)
        _initialize_result_files(str(folder))
    os.makedirs(args.out, exist_ok=True)
    mesh_path = cfg.mesh_path_for(args.dataset, args.mesh_path)

    print(f"Pipeline config: {cfg.name}  (profile={cfg.profile})")

    # ── Load LoD mesh ─────────────────────────────────────────────────────────
    print(f"Loading mesh {mesh_path} ...")

    if str(mesh_path).endswith('.npz'):
        lod_full = LoD.from_npz(mesh_path)
    else:
        lod_full = LoD(mesh_path)
    lod_full.set_batch_size(batchsize)
    print(f"  {lod_full.num_vertices} vertices, {lod_full.num_faces} faces  "
          f"(batchsize={batchsize})")

    # ── Dataset ───────────────────────────────────────────────────────────────
    if args.dataset == 'uavd4l':
        from datasets.uavd4l import UAVD4LDataset
        dataset = UAVD4LDataset(
            data_root=args.data_root,
            render_size=(rH, rW),
            fname=args.fname,
            n=args.n,
            random_seed=args.random_seed
        )
    else:
        from datasets.swiss_epfl import SwissEPFLDataset
        dataset = SwissEPFLDataset(
            data_root=args.data_root,
            render_size=(rH, rW),
            split=args.split,
            fname=args.fname,
            n=args.n,
            random_seed=args.random_seed
        )
    if args.stride > 1:
        dataset.frames = dataset.frames[::args.stride]
    print(f"Evaluating {len(dataset)} frames at render size {rH}x{rW}\n")

    if args.perturb_gt_t > 0.0 or args.perturb_gt_r > 0.0:
        print(f"  [Robustness] Perturbing GT poses by t={args.perturb_gt_t}m, r={args.perturb_gt_r}deg")
        if args.random_seed is not None:
            np.random.seed(args.random_seed)
        
        for img_name in dataset.frames:
            w2c_gt = dataset.gt_poses.get(img_name)
            if w2c_gt is not None:
                # Add uniform noise in [-args.perturb_gt_t, args.perturb_gt_t]
                t_noise = np.random.uniform(-args.perturb_gt_t, args.perturb_gt_t, size=3)
                r_noise = np.eye(3)
                if args.perturb_gt_r > 0.0:
                    r_euler = np.random.uniform(-args.perturb_gt_r, args.perturb_gt_r, size=3)
                    
                    # Option A: Drones don't flip upside down. Cap Pitch/Roll noise to realistic ±5 deg.
                    # Yaw (r_euler[2]) still gets the full [-args.perturb_gt_r, args.perturb_gt_r] noise.
                    r_euler[0] = np.clip(r_euler[0], -5.0, 5.0)
                    r_euler[1] = np.clip(r_euler[1], -5.0, 5.0)
                    
                    r_noise = Rotation.from_euler('xyz', r_euler, degrees=True).as_matrix()
                
                # Apply noise to c2w
                c2w_gt = np.linalg.inv(w2c_gt)
                c2w_gt[:3, 3] += t_noise
                c2w_gt[:3, :3] = r_noise @ c2w_gt[:3, :3]
                w2c_pert = np.linalg.inv(c2w_gt)
                
                dataset.gps_poses[img_name] = w2c_pert
                stem = os.path.splitext(img_name)[0]
                dataset.gps_poses_stem[stem] = w2c_pert

        # Dynamically scale Cost Volume and PF hyper-parameters to cover the perturbation
        req_t = args.perturb_gt_t * 2.0 + 10.0
        req_r = args.perturb_gt_r * 2.0 + 5.0
        
        cv_large = cfg.cv_grid["large"]
        range_xyz = cv_large["range_xyz_deg"]
        samp_xyz = cv_large["samples_xyz_deg"]
        
        # Scale X/Y
        if args.perturb_gt_t > 0 and range_xyz["x"] < req_t:
            orig_step_x = range_xyz["x"] / max(1, samp_xyz["x"] - 1)
            range_xyz["x"] = req_t
            range_xyz["y"] = req_t
            # Cap at 9 samples to prevent grid explosion. The PF's sigma_trans bridges the gaps.
            samp_xyz["x"] = min(9, int(np.ceil(req_t / orig_step_x)) + 1)
            samp_xyz["y"] = min(9, int(np.ceil(req_t / orig_step_x)) + 1)
            if samp_xyz["x"] % 2 == 0: samp_xyz["x"] += 1
            if samp_xyz["y"] % 2 == 0: samp_xyz["y"] += 1
            
        # Scale Z
        if args.perturb_gt_t > 0 and range_xyz["z"] < req_t:
            orig_step_z = range_xyz["z"] / max(1, samp_xyz["z"] - 1)
            range_xyz["z"] = req_t
            samp_xyz["z"] = min(9, int(np.ceil(req_t / orig_step_z)) + 1)
            if samp_xyz["z"] % 2 == 0: samp_xyz["z"] += 1

        # Scale Yaw
        if args.perturb_gt_r > 0 and range_xyz["yaw_deg"] < req_r:
            orig_step_yaw = range_xyz["yaw_deg"] / max(1, samp_xyz["yaw_deg"] - 1)
            range_xyz["yaw_deg"] = req_r
            samp_xyz["yaw_deg"] = min(15, int(np.ceil(req_r / orig_step_yaw)) + 1)
            if samp_xyz["yaw_deg"] % 2 == 0: samp_xyz["yaw_deg"] += 1
            
            if args.all_6D:
                # With realistic limits, the maximum perturbation is 5 degrees. 
                # Setting the range to 10 degrees is plenty to capture it.
                pr_range = min(req_r, 10.0) 
                range_xyz["pitch_deg"] = pr_range
                range_xyz["roll_deg"] = pr_range
                
                # Cap pitch and roll samples to 3 to prevent exponential hypothesis explosion
                # The Particle Filter will refine them further from these coarse anchor points
                samp_xyz["pitch_deg"] = min(3, samp_xyz["yaw_deg"])
                samp_xyz["roll_deg"] = min(3, samp_xyz["yaw_deg"])
                
                # Make sure they are odd to include 0
                if samp_xyz["pitch_deg"] % 2 == 0: samp_xyz["pitch_deg"] += 1
                if samp_xyz["roll_deg"] % 2 == 0: samp_xyz["roll_deg"] += 1

        print(f"  [Robustness] Auto-scaled CV large grid: range={range_xyz}, samples={samp_xyz}")

        # Scale PF
        for stage in cfg.pf_stages:
            if args.perturb_gt_t > 0:
                stage["sigma_trans"] = max(stage["sigma_trans"], args.perturb_gt_t / 3.0)
            if args.perturb_gt_r > 0:
                stage["sigma_yaw"] = max(stage["sigma_yaw"], np.radians(args.perturb_gt_r / 2.0))
            if args.perturb_gt_t >= 100:
                stage["n_candidates"] = int(stage["n_candidates"] * 1.5)
                stage["n_beams"] = int(stage["n_beams"] * 1.5)
                
        # Scale Crop Margin
        if args.perturb_gt_t > 0:
            cfg.crop_extend_m = max(cfg.crop_extend_m, args.perturb_gt_t + 50.0)

    cv_grid = cfg.cv_grid
    cv_est_large = CostVolumeSelector(lod_full, CostVolumeConfig(
        range_xyz_deg=cv_grid["large"]["range_xyz_deg"],
        samples_xyz_deg=cv_grid["large"]["samples_xyz_deg"],
        ori_image_size=(rH, rW),
        mask_size=cfg.cv_mask_size,
        batch_size=batchsize,
        mask_iou_mode=cfg.cv_mask_iou_mode,
        depth_gated=cfg.cv_depth_gated,
    ))
    cv_est_small = CostVolumeSelector(lod_full, CostVolumeConfig(
        range_xyz_deg=cv_grid["small"]["range_xyz_deg"],
        samples_xyz_deg=cv_grid["small"]["samples_xyz_deg"],
        ori_image_size=(rH, rW),
        mask_size=cfg.cv_mask_size,
        batch_size=batchsize,
        mask_iou_mode=cfg.cv_mask_iou_mode,
        depth_gated=cfg.cv_depth_gated,
    ))
    _lg = cv_grid["large"]
    _n1 = cfg.cv_grid_n_samples("large")
    print(f"  Cost Volume Selection ready (profile={cfg.profile}, batchsize={batchsize}, "
          f"stage1={_n1} hyps, z±{_lg['range_xyz_deg']['z']/2:.0f}m × {_lg['samples_xyz_deg']['z']} samples).")

    lod = lod_full
    cv_est_large.lod = lod
    cv_est_small.lod = lod
    if len(dataset) > 0:
        first_intrinsics = dataset.get_first_intrinsics()
        K_warm = scale_K(first_intrinsics['fx'], first_intrinsics['fy'], first_intrinsics['cx'], first_intrinsics['cy'], first_intrinsics['src_W'], first_intrinsics['src_H'], rW, rH)
        w2c_warm = dataset.resolve_gps_w2c(dataset.frames[0])
        if use_crop:
            lod, K_warm, w2c_warm, _spread_m, _extend_total = crop_lod_for_batch(
                lod_full, dataset, rH, rW, cfg,
            )
            cv_est_large.lod = lod
            cv_est_small.lod = lod
            print(
                f"  Cropped mesh ({len(dataset)} frames): {lod.num_faces} faces "
                f"({lod.num_vertices} verts, margin={cfg.crop_margin}, "
                f"extend={_extend_total:.0f}m incl. {_spread_m:.0f}m traj spread)"
            )
        else:
            print(f"  Full mesh: {lod.num_faces} faces ({lod.num_vertices} verts)")
        warmup_gpu_rasterizer(lod, K_warm, w2c_warm, rH, rW, cv_est_large, cfg)

    # ── Load moge models ──────────────────────────────────────────────────────
    checkpoint_paths = []
    if cfg.moge_mode in ('sam3_moge2', 'eomt_moge2'):
        from huggingface_hub import hf_hub_download
        moge_checkpoint = hf_hub_download(repo_id=cfg.moge_vits_model,
            filename='model.pt', revision=cfg.moge_vits_revision or None)
        checkpoint_paths.append(moge_checkpoint)
    if cfg.moge_mode == 'sam3_moge2':
        print(f"Loading sam3fast mask model ({cfg.moge_sam3_ckpt}) and moge2-vits depth/normal model ({cfg.moge_vits_model})...")
        from utils.sam3fast import SAM3BuildingSegmentorFast
        from moge.model.v2 import MoGeModel
        
        moge_depth_normal = MoGeModel.from_pretrained(moge_checkpoint).to(args.device).eval()
        sam3_segmentor = SAM3BuildingSegmentorFast(checkpoint_path=cfg.moge_sam3_ckpt, device=args.device)
        checkpoint_paths.append(cfg.moge_sam3_ckpt)
        moge = (moge_depth_normal, sam3_segmentor, 'sam3')
    elif cfg.moge_mode == 'eomt_moge2':
        print(f"Loading eomt mask model and moge2-vits depth/normal model ({cfg.moge_vits_model})...")
        from utils.eomt_predictor import EoMTBuildingSegmentor
        from moge.model.v2 import MoGeModel
        
        moge_depth_normal = MoGeModel.from_pretrained(moge_checkpoint).to(args.device).eval()
        eomt_segmentor = EoMTBuildingSegmentor(device=args.device)
        checkpoint_paths.append(eomt_segmentor.checkpoint_path)
        moge = (moge_depth_normal, eomt_segmentor, 'eomt')
    else:
        print(f"Loading moge2ext checkpoint {ckpt} with base moge-2 normal_head ...")
        moge = load_model_base_normals(ckpt, device=args.device)
    print("  Model loaded.")
    
    normal_estimator = None
    if args.normal_model is not None:
        print(f"Loading overriding normal/depth estimator: {args.normal_model} ...")
        normal_estimator = load_normal_estimator(args.normal_model, device=args.device)
        checkpoint_paths.extend(getattr(normal_estimator, 'evidence_paths', []))

    for folder in scenario_outputs.values():
        write_manifest(folder, args, cfg, dataset, mesh_path, checkpoints=checkpoint_paths)
        if args.normal_model and args.normal_model.startswith('metric3dv2-'):
            append_record(folder / 'geometry_predictor_settings.json', METRIC3D_DEPTH_CONVENTION)
        append_record(folder / 'candidate_scoring_settings.json', {
            key: getattr(selection_cfg, key) for key in
            ('cv_iou_weight', 'cv_edge_weight', 'cv_depth_weight')})
        # Preserve the exact perturbed input priors, not just their generator seed.
        append_record(folder / 'effective_priors.jsonl', {
            'frames': dataset.frames,
            'w2c_priors': [dataset.resolve_gps_w2c(name) for name in dataset.frames],
        })

    base_lod = lod  # Immutable intact map for every query/scenario.
    # ── Per-frame loop ────────────────────────────────────────────────────────
    all_metrics  = []
    frame_thumbs = []
    n_frames = len(dataset)

    frame_pbar = tqdm(
        enumerate(dataset),
        total=n_frames,
        desc="Pipeline",
        unit="frame",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    for frame_idx, frame_data in frame_pbar:
        img_name = frame_data["img_name"]
        img_path = frame_data["img_path"]
        K = frame_data["K"]
        w2c = frame_data["w2c_gt"]
        gt_pose = frame_data["gt_pose"]
        w2c_gps = frame_data["w2c_gps"]
        
        stem = os.path.splitext(img_name)[0]
        frame_pbar.set_postfix_str(f"{frame_idx + 1}/{n_frames} {img_name}", refresh=False)
        tqdm.write(f"[{frame_idx + 1}/{n_frames}] {img_name}")

        # ── GPS prior ─────────────────────────────────────────────────────

        gps_prior_te, gps_prior_re = pnp_pose_error(CameraPose.from_w2c(w2c_gps), gt_pose)
        print(f"  GPS prior:        t={gps_prior_te:.2f}m  R={gps_prior_re:.2f}deg")

        # ── moge2ext inference (GPU) — image I/O outside GPU timer ────────
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _query_start = time.perf_counter()
        pred_n_r, pred_m_r, pred_e_r, pred_d_r, t_moge = run_moge_inference_torch(
            moge, img_path, enc_res=enc_res, device=args.device, target_hw=(rH, rW)
        )
        
        # Override depth and normals if requested
        if normal_estimator is not None:
            t0 = time.time()
            new_d_np, new_n_np = normal_estimator.infer_depth_and_normal(
                img_path, (rH, rW), intrinsics_target=K)
            pred_n_r = torch.from_numpy(new_n_np).to(args.device)
            pred_d_r = torch.from_numpy(new_d_np).to(args.device)
            
            # Recompute edges from the new normals
            kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=args.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=args.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            grad_x = F.conv2d(pred_n_r.permute(2, 0, 1).unsqueeze(0), kernel_x, padding=1, groups=3)
            grad_y = F.conv2d(pred_n_r.permute(2, 0, 1).unsqueeze(0), kernel_y, padding=1, groups=3)
            edge_mag = torch.sqrt(grad_x**2 + grad_y**2).squeeze(0).mean(dim=0)
            
            # Mask out non-building edges using the SAM3 mask dilated
            pred_m_t = pred_m_r
            binary_mask = (pred_m_t > 0.5).float().unsqueeze(0).unsqueeze(0)
            dilated_mask = F.max_pool2d(binary_mask, kernel_size=5, stride=1, padding=2)[0, 0]
            pred_e_r = torch.clamp(edge_mag * dilated_mask, 0.0, 1.0)
            
            t_moge += (time.time() - t0)

        building_mask_t = pred_m_r >= cfg.mask_thresh

        # ── Predicted plane segmentation (algorithm) ───────────────────────
        with CpuTimer() as t_seg_timer:
            pred_label_map, pred_segments = segment_normals_into_planes(
                pred_n_r, pred_d_r, building_mask_t,
                normal_cos_thresh=cfg.pred_normal_cos,
                depth_rel_thresh=cfg.pred_depth_rel,
                min_pixels=cfg.min_pixels,
            )
            for _s in pred_segments:
                _r, _c = _s['pixel_indices'][:, 0], _s['pixel_indices'][:, 1]
                pred_label_map[_r, _c] = _s['plane_id']

            pred_depth_for_e = pred_d_r if pred_d_r is not None else np.zeros((rH, rW), dtype=np.float32)
            pred_planes_e    = enrich_planes(pred_segments, pred_n_r, pred_depth_for_e, K)
        t_seg = t_seg_timer.elapsed_s

        _stage_banner("Inference", t_moge)
        _stage_banner("Plane Segmentation", t_seg)

        with GpuTimer() as t_edge_timer:
            frame_edges = prepare_pred_edge_maps(pred_m_r, pred_e_r, cfg)
        t_edge = t_edge_timer.elapsed_s


        # ── Ablation Selection ──────────────────────────────────────────
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _query_preprocessing_s = time.perf_counter() - _query_start
        eligible_buildings = None
        smallest_bid = mid_bid = largest_bid = None
        if args.missing_buildings:
            # Use this query's intrinsics and the intact map, never the previous
            # frame's dropped geometry or a renderer's cached camera settings.
            _, id_map = base_lod.render_segmentation(w2c_gps, K, (rH, rW), mode="instance")
            visible_bids, counts = np.unique(id_map, return_counts=True)
            valid_idx = visible_bids >= 0  # Zero is a valid mesh component ID.
            visible_bids = visible_bids[valid_idx]
            counts = counts[valid_idx]

            valid_size_idx = counts >= 100
            visible_bids = visible_bids[valid_size_idx]
            counts = counts[valid_size_idx]
            eligible_buildings = len(visible_bids)

            if len(visible_bids) >= 3:
                sorted_idx = np.argsort(counts, kind='stable')
                smallest_bid = int(visible_bids[sorted_idx[0]])
                mid_bid = int(visible_bids[sorted_idx[len(sorted_idx) // 2]])
                largest_bid = int(visible_bids[sorted_idx[-1]])

        ablation_scenarios = [
            ("Baseline", []),
            ("Missing_Small", [smallest_bid] if smallest_bid is not None else []),
            ("Missing_Mid", [mid_bid] if mid_bid is not None else []),
            ("Missing_Large", [largest_bid] if largest_bid is not None else []),
        ]
        if not args.missing_buildings:
            ablation_scenarios = ablation_scenarios[:1]

        for scenario_name, dropped_bids in ablation_scenarios:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _localization_start = time.perf_counter()
            lod = without_buildings(base_lod, dropped_bids)
            cv_est_large.lod = lod
            cv_est_small.lod = lod
            print(f"\n  === Scenario: {scenario_name} (Dropped: {dropped_bids}) ===")
            # ── Cost-volume 2-stage grid search ─────────────────────────────────
            gps_depth, _ = lod.render_depth_normals_batch([w2c_gps], K, (rH, rW))
            gps_edges = lod.render_edges_batch([w2c_gps], K, (rH, rW), sobel_scale=8.0)
            gps_mask = (gps_depth[0] > 0)

            # Calculate GPS IoU
            pred_mask_np = building_mask_t.cpu().numpy()
            _i = (gps_mask & pred_mask_np).sum()
            _u = (gps_mask | pred_mask_np).sum()
            gps_iou = float(_i) / float(_u) if _u > 0 else 0.0

            # Calculate GPS EDT
            gps_edge_px = (gps_edges[0] > 0.5) & gps_mask
            if gps_edge_px.any():
                gps_edt = float(frame_edges['edge_dt_full'][gps_edge_px].mean())
            else:
                gps_edt = float('inf')

            print(f"  GPS Prior Metrics: IoU={gps_iou:.3f}, EDT={gps_edt:.1f}px")

            w2c_cv, init_pose, te_cv, re_cv, _cv_score, t_cv = run_cost_volume_selection(
                lod, K, building_mask_t, w2c_gps, rH, rW,
                cv_est_large, cv_est_small, cfg, pred_e_r, pred_d_t=pred_d_r, gt_pose=gt_pose,
                edge_dt_torch=frame_edges['edge_dt_cv_torch'],
                min_iou=gps_iou, max_edt=gps_edt,
            )
            _stage_banner("Cost Volume Selection", t_cv)
            print(f"  score={_cv_score:.4f}  t={te_cv:.2f}m  R={re_cv:.2f}deg")

            pred_mask_frac = _pred_mask_fraction(frame_edges['building_mask_np'])
            if pred_mask_frac < cfg.low_mask_frac:
                print(f"  LoD-noise mode: pred mask {pred_mask_frac:.1%} < {cfg.low_mask_frac:.0%}")

            # ── Multi-scale particle filter from cost-volume ──────────────────
            pred_depth_np = pred_d_r.detach().cpu().numpy() if pred_d_r is not None else None
            w2c_iter, _te_c, _re_c, _pf_score, t_pf = run_particle_filter_initialization(
                init_pose, lod, K, (rH, rW), cfg, gt_pose=gt_pose,
                q_mask_u8=frame_edges['q_mask_u8'],
                edge_dt_cv_np=frame_edges['edge_dt_cv_np'],
                pred_depth_np=pred_depth_np,
                batch_size=batchsize,
                pred_mask_frac=pred_mask_frac,
            )
            _pf_init_w2c = w2c_iter.copy()
            _stage_banner("Particle Filter", t_pf)
            print(f"  t={_te_c:.2f}m  R={_re_c:.2f}deg  pf_score={_pf_score:.4f}")

            # ── A+B iterative PnP  (5 rounds) ─────────────────────────────────
            (
                _all_poses, _pnp_ref_normals_vis, _pnp_ref_label_map_vis,
                _pose_timings, _pose_pnp_meta, t_ab,
            ) = run_iterative_semantic_pnp(
                _pf_init_w2c, lod, K, rH, rW, pred_label_map, pred_planes_e, cfg,
                n_pred_planes=len(pred_segments),
                pred_mask_frac=pred_mask_frac,
            )

            candidate_records = []
            (
                _best_overall_w2c, _best_overall_label, _best_overall_score,
                _best_overall_te, _best_overall_re, t_select,
            ) = select_best_pose_hierarchical(
                _all_poses, lod, K, (rH, rW), frame_edges, gt_pose,
                stage_times={'moge': t_moge, 'cv': t_cv, 'pf': t_pf, 'ab': t_ab},
                cfg=selection_cfg, _pose_pnp_meta=_pose_pnp_meta, t_ab=t_ab,
                pred_depth_np=pred_depth_np,
                pred_normals_np=pred_n_r.detach().cpu().numpy(),
                pred_label_map_np=pred_label_map,
                candidate_records=candidate_records,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall_inference_s = _query_preprocessing_s + time.perf_counter() - _localization_start
            append_record(scenario_outputs[scenario_name] / 'candidates.jsonl', {
                'frame': img_name, 'candidates': candidate_records,
                'selected': _best_overall_label, 'gate': _pose_pnp_meta.get('_gate'),
                'w2c_cv': w2c_cv, 'w2c_pf': _pf_init_w2c,
                'scenario': scenario_name, 'dropped_buildings': dropped_bids,
                'eligible_building_count': eligible_buildings,
                'intact_face_count': base_lod.num_faces, 'scenario_face_count': lod.num_faces,
                'wall_inference_s': wall_inference_s,
            })
            frame_pbar.set_postfix(TE=f"{_best_overall_te:.1f}m", sel=_best_overall_label[:12], refresh=True)

            # ── Eval metrics (excluded from algo timing) ──────────────────────
            gt_depth, gt_normals, gt_label_map, gt_segments = lod.get_plane_segmentation(
                w2c, K, (rH, rW),
                normal_cos_thresh=cfg.gt_normal_cos,
                depth_rel_thresh=cfg.gt_depth_rel,
                min_pixels=cfg.min_pixels,
            )
            metrics = compute_panoptic_quality(gt_label_map, pred_label_map)
            pred_sizes = [s['pixel_count'] for s in pred_segments]
            ap = compute_ap_at_iou(gt_label_map, pred_label_map, pred_sizes)
            metrics['AP'] = ap
            metrics['frame'] = img_name
            metrics['scenario'] = scenario_name
            metrics['dropped_bids'] = dropped_bids
            all_metrics.append(metrics)
            print(f"  eval planes  GT={len(gt_segments)}  pred={len(pred_segments)}  "
                  f"PQ={metrics['PQ']:.3f}  SQ={metrics['SQ']:.3f}  "
                  f"RQ={metrics['RQ']:.3f}  AP={ap:.3f}")

            metrics.update({
                'cv_select_t_err':        te_cv,
                'cv_select_r_err':        re_cv,
                'pf_t_err':               _te_c,
                'pf_r_err':               _re_c,
                'gps_prior_t_err':        gps_prior_te,
                'gps_prior_r_err':        gps_prior_re,
                'pnp_best_overall_t_err': _best_overall_te,
                'pnp_best_overall_r_err': _best_overall_re,
                'pnp_best_overall_label': _best_overall_label,
                'w2c_best_overall':        _best_overall_w2c,
                'wall_inference_s':       wall_inference_s,
                'time_moge':              t_moge,
                'time_seg':               t_seg,
                'time_edge':              t_edge,
                'time_cv':                t_cv,
                'time_pf':                t_pf,
                'time_ab':                t_ab,
                'time_select':            t_select,
            })
            _append_frame_result(str(scenario_outputs[scenario_name]), metrics)

            # ── Visualisation (only when --vis) ───────────────────────────────
            if args.vis:

                def _pose_err(w2c_arr):
                    return pnp_pose_error(CameraPose.from_w2c(w2c_arr), gt_pose)

                poses_info = [
                    {
                        'name': 'GPS',
                        'pose': w2c_gps,
                        't_err': gps_prior_te,
                        'r_err': gps_prior_re,
                        'show_planes': False,
                    },
                    {
                        'name': 'Cost Volume Selection',
                        'pose': w2c_cv,
                        't_err': te_cv,
                        'r_err': re_cv,
                        'timing_s': t_cv,
                        'show_planes': False,
                    },
                    {
                        'name': 'Particle Filter',
                        'pose': _pf_init_w2c,
                        't_err': _te_c,
                        'r_err': _re_c,
                        'timing_s': t_pf,
                        'show_planes': False,
                    },
                ]

                for lbl, w2c_sub in _all_poses:
                    if lbl == 'Particle Filter':
                        continue
                    te_sub, re_sub = _pose_err(w2c_sub)
                    meta = _pose_pnp_meta.get(lbl, {})
                    poses_info.append({
                        'name': lbl,
                        'pose': w2c_sub,
                        't_err': te_sub,
                        'r_err': re_sub,
                        'timing_s': _pose_timings.get(lbl),
                        'show_planes': lbl.startswith(('A-iter', 'B-iter')),
                        'plane_pairs': meta.get('plane_pairs', []),
                        'lod_label_map': meta.get('lod_label_map'),
                    })

                poses_info.append({
                    'name': f'Final ({_best_overall_label})',
                    'pose': _best_overall_w2c,
                    't_err': _best_overall_te,
                    'r_err': _best_overall_re,
                    'timing_s': t_select,
                    'show_planes': False,
                })
                poses_info.append({
                    'name': 'GT',
                    'pose': w2c,
                    't_err': 0.0,
                    'r_err': 0.0,
                    'is_gt': True,
                    'show_planes': True,
                })

                uav_img = cv2.imread(img_path)
                if uav_img is not None:
                    uav_img = cv2.resize(uav_img, (rW, rH))

                out_path = os.path.join(args.out, f"{stem}_eval.png")
                pred_n_np = pred_n_r.cpu().numpy() if hasattr(pred_n_r, 'cpu') else pred_n_r
                pred_m_np = (pred_m_r.cpu().numpy() > 0.5) if hasattr(pred_m_r, 'cpu') else (pred_m_r > 0.5)
                pred_e_np = pred_e_r.cpu().numpy() if hasattr(pred_e_r, 'cpu') else pred_e_r

                np.savez_compressed(
                    os.path.join(args.out, f"{stem}_figure_assets.npz"),
                    w2c_gt=w2c,
                    w2c_gps=w2c_gps,
                    w2c_pf=_pf_init_w2c,
                    w2c_pnp=_best_overall_w2c,
                    pred_label_map=pred_label_map,
                    K=K,
                    gps_te=np.float32(gps_prior_te),
                    gps_re=np.float32(gps_prior_re),
                    pf_te=np.float32(_te_c),
                    pf_re=np.float32(_re_c),
                    pnp_te=np.float32(_best_overall_te),
                    pnp_re=np.float32(_best_overall_re),
                    pnp_label=np.array(_best_overall_label),
                )

                t_vis = time.perf_counter()
                plot_pipeline_results(
                    out_path, uav_img, pred_m_np, pred_e_np, pred_n_np, pred_label_map,
                    poses_info, lod, K, rH, rW,
                    pred_planes_e=pred_planes_e,
                    moge_timing_s=t_moge,
                    seg_normal_cos=cfg.gt_normal_cos,
                    seg_depth_rel=cfg.gt_depth_rel,
                    seg_min_pixels=cfg.min_pixels,
                )
                t_vis = time.perf_counter() - t_vis
                print(f"  -> Saved visualization: {out_path}  ({t_vis:.1f}s)")

        lod = base_lod
        cv_est_large.lod = base_lod
        cv_est_small.lod = base_lod

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    import csv
    print("=" * 60)
    print(f"PIPELINE COMPLETED  ({len(all_metrics)} evaluations)")
    print("=" * 60)
    
    ablation_csv_path = os.path.join(args.out, "ablation_results.csv")
    with open(ablation_csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "scenario", "dropped_bids", "gps_te", "gps_re", "cv_te", "cv_re", "pf_te", "pf_re", "pnp_te", "pnp_re", "PQ", "SQ", "RQ", "AP"])
        for m in all_metrics:
            writer.writerow([
                m.get("frame", ""),
                m.get("scenario", ""),
                str(m.get("dropped_bids", [])),
                f"{m.get('gps_prior_t_err', float('nan')):.2f}",
                f"{m.get('gps_prior_r_err', float('nan')):.2f}",
                f"{m.get('cv_select_t_err', float('nan')):.2f}",
                f"{m.get('cv_select_r_err', float('nan')):.2f}",
                f"{m.get('pf_t_err', float('nan')):.2f}",
                f"{m.get('pf_r_err', float('nan')):.2f}",
                f"{m.get('pnp_best_overall_t_err', float('nan')):.2f}",
                f"{m.get('pnp_best_overall_r_err', float('nan')):.2f}",
                f"{m.get('PQ', float('nan')):.3f}",
                f"{m.get('SQ', float('nan')):.3f}",
                f"{m.get('RQ', float('nan')):.3f}",
                f"{m.get('AP', float('nan')):.3f}",
            ])
            
    print(f"Saved ablation results to {ablation_csv_path}")


if __name__ == '__main__':
    main()
