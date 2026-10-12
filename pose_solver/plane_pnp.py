"""
pose_solver/plane_pnp.py
========================
Pose estimation from matched plane-instance correspondences.

Key idea
--------
After running plane instance segmentation on both the LoD render (GT) and the
query image (moge2ext pred), each matched pair (GT plane i ↔ pred plane j) with
IoU ≥ threshold identifies pixels where *both* segmentations agree on the plane
label.  For every such intersection pixel (u, v):

  * 3D world point: back-project using the GT depth rendered at the candidate
    pose, then transform to world coordinates via the inverse w2c.
  * 2D image point: (u, v) directly — the pixel in the query image.

Collecting these across all matched pairs yields a dense, noise-robust set of
2D–3D correspondences, which is fed to cv2.solvePnPRansac for 6-DoF pose
estimation.

Why this is better than centroid-PnP
--------------------------------------
* Centroid location is sensitive to segmentation boundary errors and
  partial occlusion.
* Using every intersection pixel averages out per-pixel noise; with ~130
  matched planes and O(100) pixels each we get ~13 000 correspondences —
  comparable to dense mask-agreement PnP (correspondence_pnp.py) but
  restricted to *high-confidence* agreed regions.

Relation to correspondence_pnp.py
----------------------------------
This module is a semantically filtered version of ``correspondence_pnp.py``.
Instead of accepting every pixel where ``pred_mask & rendered_mask``, we
accept only pixels where pred_plane_j and gt_plane_i *both* claim the same
physical plane.  Unmatched (FP / FN) pixels are excluded, reducing reprojection
noise from wrongly associated regions.

Usage
-----
    from pose_solver.plane_pnp import build_plane_correspondences, estimate_pose_pnp
    from pose_solver.correspondence_pnp import render_3d_coordinates

    xyz_map, depth, _ = render_3d_coordinates(lod, candidate_pose, K, (H, W))
    pts_2d, pts_3d = build_plane_correspondences(
        gt_label_map, pred_label_map, matched_pairs, gt_ids, pred_ids, xyz_map
    )
    pose_est, inliers = estimate_pose_pnp(pts_2d, pts_3d, K,
                                          pose_prior=candidate_pose)

"""
from __future__ import annotations

from utils.planes import plane_pair_score as _pps
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist as scipy_cdist

from typing import List, Optional, Tuple

import numpy as np
import cv2

from utils.pose import CameraPose


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Build pixel-level 2D–3D correspondences from matched planes
# ═══════════════════════════════════════════════════════════════════════════════

def build_plane_correspondences(
    gt_label_map: np.ndarray,           # (H, W) int  — GT plane ids (-1 = bg)
    pred_label_map: np.ndarray,         # (H, W) int  — pred plane ids (-1 = bg)
    matched_pairs: List[Tuple],         # [(gi, pi, iou), ...]  index-based
    gt_ids: List[int],                  # GT plane ids in order
    pred_ids: List[int],                # pred plane ids in order
    xyz_map: np.ndarray,                # (H, W, 3) float64  world coords (0 where invalid)
    max_per_pair: int = 500,            # max sampled pixels per matched pair
    min_intersection_px: int = 20,      # skip pair if intersection too small
    iou_weight: bool = True,            # weight sampling by IoU (more for better matches)
    pair_scores: Optional[dict] = None, # {(gi, pi): composite score} — overrides iou_weight
    rng_seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect 2D–3D correspondences from matched plane intersections.

    For each matched (GT plane i, pred plane j) pair:
        intersection = (gt_label_map == gt_id) & (pred_label_map == pred_id)
                     & (xyz_map has valid Z)

    All intersection pixels give:
        2D: (u, v) image coordinates
        3D: world XYZ from xyz_map

    Parameters
    ----------
    gt_label_map, pred_label_map : (H, W) int arrays of per-pixel plane ids
    matched_pairs : list of (gi, pi, iou) from compute_panoptic_quality or
                    match_planes — gi/pi are indices into gt_ids / pred_ids
    gt_ids, pred_ids : ordered lists of plane ids corresponding to gi/pi indices
    xyz_map : (H, W, 3) world coordinates rendered at the candidate pose
              (output of render_3d_coordinates). Zero where no mesh.
    max_per_pair : max pixels to sample per matched pair (for speed)
    min_intersection_px : pairs with fewer intersection pixels are skipped
    iou_weight : if True, pairs with higher IoU contribute up to max_per_pair;
                 lower-IoU pairs are sampled proportionally less
    rng_seed : NumPy random seed for reproducible subsampling

    Returns
    -------
    pts_2d : (N, 2) float64  — pixel (u, v) coordinates
    pts_3d : (N, 3) float64  — world XYZ
    """
    rng = np.random.default_rng(rng_seed)
    valid_3d = (xyz_map[:, :, 2] != 0).astype(bool)  # Z=0 → no mesh coverage

    pts_2d_list: List[np.ndarray] = []
    pts_3d_list: List[np.ndarray] = []

    for gi, pi, iou in matched_pairs:
        gt_id   = gt_ids[gi]
        pred_id = pred_ids[pi]

        inter = (gt_label_map == gt_id) & (pred_label_map == pred_id) & valid_3d
        vs, us = np.where(inter)
        if len(vs) < min_intersection_px:
            continue

        # Scale quota by composite pair score if provided, else fall back to IoU
        if pair_scores is not None:
            w = float(pair_scores.get((gi, pi), float(iou)))
        elif iou_weight:
            w = float(iou)
        else:
            w = 1.0
        quota = max(min_intersection_px, int(max_per_pair * w))

        if len(vs) > quota:
            idx = rng.choice(len(vs), quota, replace=False)
            vs, us = vs[idx], us[idx]

        pts_2d_list.append(np.stack([us, vs], axis=1).astype(np.float64))
        pts_3d_list.append(xyz_map[vs, us].astype(np.float64))

    if not pts_2d_list:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    return np.concatenate(pts_2d_list, axis=0), np.concatenate(pts_3d_list, axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PnP solver
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_pose_pnp(
    pts_2d: np.ndarray,              # (N, 2) float64 pixel coords (u, v)
    pts_3d: np.ndarray,              # (N, 3) float64 world coords
    K: np.ndarray,                   # (3, 3) camera intrinsics
    pose_prior: Optional[CameraPose] = None,   # initial guess for PnP
    reprojection_error_px: float = 8.0,
    confidence: float = 0.999,
    ransac_iter: int = 1000,
) -> Tuple[Optional[CameraPose], Optional[np.ndarray]]:
    """Solve for camera pose using PnP-RANSAC on plane correspondences.

    Parameters
    ----------
    pts_2d : (N, 2) image pixel coordinates (u, v)
    pts_3d : (N, 3) world coordinates
    K : (3, 3) camera intrinsics
    pose_prior : optional initial pose estimate (used as PnP starting guess)
    reprojection_error_px : RANSAC inlier threshold in pixels
    confidence : RANSAC confidence level
    ransac_iter : max RANSAC iterations

    Returns
    -------
    pose : CameraPose or None if PnP failed / too few inliers
    inliers : (M,) int array of inlier indices, or None
    """
    if len(pts_2d) < 6:
        return None, None

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Optional initial guess from prior pose
    use_init = pose_prior is not None
    if use_init:
        w2c = pose_prior.to_w2c()
        R_w2c = w2c[:3, :3]
        t_w2c = w2c[:3, 3].reshape(3, 1)
        rvec_init, _ = cv2.Rodrigues(R_w2c)
    else:
        rvec_init = None
        t_w2c     = None

    cv2.setRNGSeed(42)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d.astype(np.float64),
        pts_2d.astype(np.float64),
        K.astype(np.float64),
        dist_coeffs,
        rvec=rvec_init if use_init else None,
        tvec=t_w2c    if use_init else None,
        useExtrinsicGuess=use_init,
        iterationsCount=ransac_iter,
        reprojectionError=reprojection_error_px,
        confidence=confidence,
        flags=cv2.SOLVEPNP_SQPNP,
    )

    if not success or inliers is None or len(inliers) < 6:
        return None, None

    R_w2c, _ = cv2.Rodrigues(rvec)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R_w2c
    w2c[:3,  3] = tvec.ravel()

    return CameraPose.from_w2c(w2c), inliers.ravel()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. High-level convenience wrapper
# ═══════════════════════════════════════════════════════════════════════════════

def plane_pnp_refine(
    gt_label_map: np.ndarray,
    pred_label_map: np.ndarray,
    matched_pairs: List[Tuple],
    gt_ids: List[int],
    pred_ids: List[int],
    xyz_map: np.ndarray,
    K: np.ndarray,
    pose_prior: Optional[CameraPose] = None,
    max_per_pair: int = 500,
    reprojection_error_px: float = 8.0,
) -> dict:
    """Full plane-PnP pose estimation pipeline.

    Wraps ``build_plane_correspondences`` + ``estimate_pose_pnp`` and returns a
    diagnostic dict.

    Returns
    -------
    dict with keys:
        'pose'         : CameraPose or None
        'n_pairs'      : number of matched plane pairs used
        'n_pts'        : total 2D–3D correspondences
        'n_inliers'    : RANSAC inlier count (0 if failed)
        'inlier_ratio' : inliers / n_pts
        'success'      : bool
    """
    pts_2d, pts_3d = build_plane_correspondences(
        gt_label_map, pred_label_map, matched_pairs, gt_ids, pred_ids,
        xyz_map, max_per_pair=max_per_pair,
    )

    n_pairs  = len(matched_pairs)
    n_pts    = len(pts_2d)

    pose, inliers = estimate_pose_pnp(
        pts_2d, pts_3d, K,
        pose_prior=pose_prior,
        reprojection_error_px=reprojection_error_px,
    )

    n_inliers    = len(inliers) if inliers is not None else 0
    inlier_ratio = n_inliers / n_pts if n_pts > 0 else 0.0

    return {
        "pose":          pose,
        "n_pairs":       n_pairs,
        "n_pts":         n_pts,
        "n_inliers":     n_inliers,
        "inlier_ratio":  inlier_ratio,
        "success":       pose is not None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Centroid-based PnP from rel-normal matched planes
# ═══════════════════════════════════════════════════════════════════════════════

def plane_centroid_pnp(
    rn_matches: List[dict],       # output of match_planes_by_relative_normal
    w2c_prior: np.ndarray,        # (4, 4) world-to-camera matrix of the prior pose
    K: np.ndarray,                # (3, 3) camera intrinsics
    reprojection_error_px: float = 20.0,
    confidence: float = 0.999,
    ransac_iter: int = 5000,
    min_matches: int = 6,
) -> dict:
    """Estimate camera pose from plane centroid correspondences.

    For each matched pair:
      * 3D point: LoD plane centroid in **world** frame
                  (converted from GPS camera frame using w2c_prior).
      * 2D point: pred plane centroid in **image** frame.

    This gives one correspondence per matched pair (~100-180 pairs), which
    is sufficient for a robust PnP-RANSAC estimate.

    Parameters
    ----------
    rn_matches  : list of dicts from match_planes_by_relative_normal, each with
                  'lod' and 'pred' enriched plane dicts (must have centroid_3d,
                  centroid_2d keys from enrich_planes).
    w2c_prior   : (4, 4) prior world-to-camera matrix used when rendering the
                  LoD planes (centroid_3d is in THIS camera frame).
    K           : (3, 3) camera intrinsics.
    reprojection_error_px : RANSAC inlier threshold.
    confidence  : RANSAC confidence.
    ransac_iter : max RANSAC iterations.
    min_matches : minimum number of valid correspondences to attempt PnP.

    Returns
    -------
    dict with keys:
        pose     : CameraPose or None
        n_pts    : number of correspondences
        n_inliers: number of RANSAC inliers
        inlier_ratio: float
        success  : bool
    """
    R_prior = w2c_prior[:3, :3]   # world-to-camera rotation
    t_prior = w2c_prior[:3,  3]   # world-to-camera translation
    # camera-to-world: P_world = R^T @ (P_cam - t)

    pts_2d_list = []
    pts_3d_list = []
    for m in rn_matches:
        c3d_cam = m['lod'].get('centroid_3d')
        c2d     = m['pred'].get('centroid_2d')
        if c3d_cam is None or c2d is None:
            continue
        c3d_cam = np.asarray(c3d_cam, dtype=np.float64)
        # Skip if depth is zero / invalid
        if c3d_cam[2] < 1e-3:
            continue
        c3d_world = R_prior.T @ (c3d_cam - t_prior)
        pts_3d_list.append(c3d_world)
        pts_2d_list.append(np.asarray(c2d, dtype=np.float64))

    n_pts = len(pts_3d_list)
    base = {'n_pts': n_pts, 'n_inliers': 0, 'inlier_ratio': 0.0,
            'pose': None, 'success': False}
    if n_pts < min_matches:
        return base

    pts_3d = np.array(pts_3d_list, dtype=np.float64)   # (N, 3)
    pts_2d = np.array(pts_2d_list, dtype=np.float64)   # (N, 2)

    K_f64 = K.astype(np.float64)
    cv2.setRNGSeed(42)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, K_f64, None,
        iterationsCount=ransac_iter,
        reprojectionError=reprojection_error_px,
        confidence=confidence,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok or inliers is None or len(inliers) < min_matches:
        return {**base, 'n_pts': n_pts}

    n_inliers = int(len(inliers))
    # Refine on inliers only
    idx = inliers.ravel()
    ok2, rvec, tvec = cv2.solvePnP(
        pts_3d[idx], pts_2d[idx], K_f64, None,
        rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    R_mat, _ = cv2.Rodrigues(rvec)
    w2c_est = np.eye(4, dtype=np.float64)
    w2c_est[:3, :3] = R_mat
    w2c_est[:3,  3] = tvec.ravel()
    pose_est = CameraPose.from_w2c(w2c_est)

    return {
        'pose':         pose_est,
        'n_pts':        n_pts,
        'n_inliers':    n_inliers,
        'inlier_ratio': n_inliers / n_pts,
        'success':      True,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Sub-region centroid PnP (unbiased, richer correspondences)
# ═══════════════════════════════════════════════════════════════════════════════

def subregion_centroid_pnp(
    matches: List[dict],
    lod_label_map: np.ndarray,
    fid_map: np.ndarray,
    pred_label_map: np.ndarray,
    lod_e_by_pid: dict,
    w2c_prior: np.ndarray,
    K: np.ndarray,
    min_pixels: int = 20,
    reprojection_error_px: float = 20.0,
    confidence: float = 0.999,
    ransac_iter: int = 5000,
    min_pts: int = 6,
) -> dict:
    """PnP using per-(pred_plane, LoD_tile) sub-region centroids.

    For each matched (LoD plane lpid, pred plane ppid) pair:
      1. Find the set of LoD plane IDs visible within the pred plane region,
         using the face-ID map.  The matched LoD plane ``lpid`` is one of them.
      2. For each visible LoD tile ``ltid`` within pred plane ``ppid``:
           - Find intersection pixels: ``(pred_label_map == ppid) & (lod_label_map == ltid)``
           - 2D point: mean (u, v) of intersection pixels
           - 3D point: LoD plane ``ltid`` centroid in world frame (from ``lod_e_by_pid``)
      3. This produces one 2D–3D correspondence per sub-region, with
         unbiased centroids.  Compared to ``plane_centroid_pnp`` it provides
         2–5× more correspondences, each with smaller 2D bias.

    Parameters
    ----------
    matches : list of match dicts (same format as plane_centroid_pnp).
    lod_label_map : (H, W) LoD plane label map at prior pose.
    fid_map : (H, W) face-ID map at prior pose.
    pred_label_map : (H, W) predicted plane label map.
    lod_e_by_pid : dict mapping LoD plane_id → enriched plane dict (centroid_3d).
    w2c_prior : (4, 4) current world-to-camera matrix (to convert centroid_3d to world).
    K : (3, 3) camera intrinsics.
    min_pixels : minimum intersection pixels to use a sub-region.
    reprojection_error_px : RANSAC inlier threshold.
    confidence : RANSAC confidence.
    ransac_iter : max RANSAC iterations.
    min_pts : minimum total correspondences to attempt PnP.

    Returns
    -------
    dict with keys: pose, n_pairs, n_pts, n_inliers, inlier_ratio, success
    """
    R_prior = w2c_prior[:3, :3]
    t_prior = w2c_prior[:3, 3]

    pts_2d_list: List[np.ndarray] = []
    pts_3d_list: List[np.ndarray] = []
    n_pairs_used = 0

    for m in matches:
        ppid = int(m['pred']['plane_id'])

        # Find all LoD planes visible within this pred plane (via lod_label_map)
        pred_mask = (pred_label_map == ppid)
        lod_ids_in_pred = np.unique(lod_label_map[pred_mask])
        lod_ids_in_pred = lod_ids_in_pred[lod_ids_in_pred >= 0]

        for ltid in lod_ids_in_pred:
            lp = lod_e_by_pid.get(int(ltid))
            if lp is None:
                continue
            c3d_cam = lp.get('centroid_3d')
            if c3d_cam is None or float(c3d_cam[2]) < 1e-3:
                continue

            # Sub-region: pixels in pred plane ppid AND LoD plane ltid
            sub_mask = pred_mask & (lod_label_map == ltid)
            vs, us = np.where(sub_mask)
            if len(vs) < min_pixels:
                continue

            # 2D point: centroid of sub-region pixels
            u_c = float(us.mean())
            v_c = float(vs.mean())

            # 3D point: convert LoD centroid from camera frame to world frame
            c3d_cam_arr = np.asarray(c3d_cam, dtype=np.float64)
            c3d_world = R_prior.T @ (c3d_cam_arr - t_prior)

            pts_2d_list.append(np.array([[u_c, v_c]], dtype=np.float64))
            pts_3d_list.append(c3d_world.reshape(1, 3))
            n_pairs_used += 1

    base = {'n_pairs': n_pairs_used, 'n_pts': 0, 'n_inliers': 0,
            'inlier_ratio': 0.0, 'pose': None, 'success': False}

    if not pts_2d_list:
        return base

    pts_2d = np.concatenate(pts_2d_list, axis=0)
    pts_3d = np.concatenate(pts_3d_list, axis=0)
    n_pts = len(pts_2d)
    base['n_pts'] = n_pts
    base['n_pairs'] = n_pairs_used

    if n_pts < min_pts:
        return base

    K_f64 = K.astype(np.float64)
    cv2.setRNGSeed(42)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, K_f64, None,
        iterationsCount=ransac_iter,
        reprojectionError=reprojection_error_px,
        confidence=confidence,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok or inliers is None or len(inliers) < min_pts:
        return {**base, 'n_pts': n_pts}

    n_inliers = int(len(inliers))
    idx = inliers.ravel()
    ok2, rvec, tvec = cv2.solvePnP(
        pts_3d[idx], pts_2d[idx], K_f64, None,
        rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    R_mat, _ = cv2.Rodrigues(rvec)
    w2c_est = np.eye(4, dtype=np.float64)
    w2c_est[:3, :3] = R_mat
    w2c_est[:3, 3] = tvec.ravel()

    return {
        'pose':         CameraPose.from_w2c(w2c_est),
        'n_pairs':      n_pairs_used,
        'n_pts':        n_pts,
        'n_inliers':    n_inliers,
        'inlier_ratio': n_inliers / n_pts,
        'success':      True,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Pixel-level face-ID PnP (no centroid bias)
# ═══════════════════════════════════════════════════════════════════════════════

def pixel_face_pnp(
    matches: List[dict],
    lod_label_map: np.ndarray,
    fid_map: np.ndarray,
    pred_label_map: np.ndarray,
    face_centroids_world: np.ndarray,
    K: np.ndarray,
    max_per_pair: int = 300,
    reprojection_error_px: float = 8.0,
    confidence: float = 0.999,
    ransac_iter: int = 5000,
    min_pts: int = 10,
    rng_seed: int = 42,
    use_pair_score_weight: bool = True,  # scale per-pair pixel quota by composite score
) -> dict:
    """Pixel-level PnP using per-pixel face centroid correspondences.

    For each matched (LoD plane lpid, pred plane ppid) pair:
        - Find all pixels in the predicted plane: ``pred_label_map == ppid``
        - For each such pixel (u, v), look up the LoD face at that pixel
          from ``fid_map`` (rendered at the GPS/prior pose)
        - Only keep pixels whose face belongs to LoD plane ``lpid``
          (acts as an outlier filter: discards pixels where the GPS-rendered
          face-ID has slipped to a different LoD plane)
        - 2D point: ``(u, v)`` — actual pixel in the image
        - 3D point: ``face_centroids_world[fid_map[v, u]]`` — world centroid
          of the face visible at that pixel in the GPS-rendered view

    Because every pixel of the pred plane contributes one correspondence
    (not just the centroid), this method avoids the centroid-averaging bias
    that hurts ``plane_centroid_pnp`` when a predicted plane spans several
    LoD tiles.

    Parameters
    ----------
    matches : list of dicts, each with ``m['lod']['plane_id']`` and
              ``m['pred']['plane_id']`` (same format as plane_centroid_pnp).
    lod_label_map : (H, W) int — LoD plane label map rendered at prior pose.
    fid_map : (H, W) int — face-ID map rendered at prior pose (-1 = no face).
    pred_label_map : (H, W) int — predicted plane label map in image space.
    face_centroids_world : (F, 3) float — world-frame centroid of every face.
    K : (3, 3) camera intrinsics.
    max_per_pair : maximum pixels to sample per matched pair (for speed).
    reprojection_error_px : RANSAC inlier threshold.
    confidence : RANSAC confidence level.
    ransac_iter : max RANSAC iterations.
    min_pts : minimum correspondences required to attempt PnP.
    rng_seed : random seed for reproducible subsampling.

    Returns
    -------
    dict with keys: pose, n_pairs, n_pts, n_inliers, inlier_ratio, success
    """
    rng = np.random.default_rng(rng_seed)
    F = len(face_centroids_world)

    # Pre-build per-LoD-plane face-ID sets for fast filtering
    valid_fid = (fid_map >= 0) & (fid_map < F)
    lod_plane_ids = np.unique(lod_label_map[lod_label_map >= 0])
    lpid_to_face_mask: dict = {}
    for lpid in lod_plane_ids:
        fids_for_lp = fid_map[(lod_label_map == lpid) & valid_fid]
        if len(fids_for_lp):
            mask = np.zeros(F, dtype=bool)
            mask[fids_for_lp] = True
            lpid_to_face_mask[int(lpid)] = mask

    pts_2d_list: List[np.ndarray] = []
    pts_3d_list: List[np.ndarray] = []
    n_pairs_used = 0

    for m in matches:
        lpid = int(m['lod']['plane_id'])
        ppid = int(m['pred']['plane_id'])

        # All pixels in the predicted plane with a valid GPS face-ID
        pred_mask = (pred_label_map == ppid) & valid_fid
        vs, us = np.where(pred_mask)
        if len(vs) < 3:
            continue

        # Filter to faces that belong to LoD plane lpid
        face_mask = lpid_to_face_mask.get(lpid)
        if face_mask is not None:
            fids_here = fid_map[vs, us]
            keep = face_mask[fids_here]
            vs, us = vs[keep], us[keep]
        if len(vs) < 3:
            continue

        # Scale sampling quota by composite pair score (higher quality → more pixels)
        if use_pair_score_weight:
            _score = _pps(m)
            _w = _score if (not np.isnan(_score) and _score > 0) else float(
                m.get('vote_frac', m.get('iou', 1.0)))
        else:
            _w = 1.0
        _quota = max(3, int(max_per_pair * _w))
        if len(vs) > _quota:
            idx = rng.choice(len(vs), _quota, replace=False)
            vs, us = vs[idx], us[idx]

        fids = fid_map[vs, us]
        pt3d = face_centroids_world[fids].astype(np.float64)
        pt2d = np.stack([us, vs], axis=1).astype(np.float64)

        pts_2d_list.append(pt2d)
        pts_3d_list.append(pt3d)
        n_pairs_used += 1

    base = {'n_pairs': n_pairs_used, 'n_pts': 0, 'n_inliers': 0,
            'inlier_ratio': 0.0, 'pose': None, 'success': False}

    if not pts_2d_list:
        return base

    pts_2d = np.concatenate(pts_2d_list, axis=0)
    pts_3d = np.concatenate(pts_3d_list, axis=0)
    n_pts = len(pts_2d)
    base['n_pts'] = n_pts
    base['n_pairs'] = n_pairs_used

    if n_pts < min_pts:
        return base

    K_f64 = K.astype(np.float64)
    cv2.setRNGSeed(42)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, K_f64, None,
        iterationsCount=ransac_iter,
        reprojectionError=reprojection_error_px,
        confidence=confidence,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok or inliers is None or len(inliers) < min_pts:
        return {**base, 'n_pts': n_pts}

    n_inliers = int(len(inliers))
    idx = inliers.ravel()
    ok2, rvec, tvec = cv2.solvePnP(
        pts_3d[idx], pts_2d[idx], K_f64, None,
        rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    R_mat, _ = cv2.Rodrigues(rvec)
    w2c_est = np.eye(4, dtype=np.float64)
    w2c_est[:3, :3] = R_mat
    w2c_est[:3, 3] = tvec.ravel()

    return {
        'pose':         CameraPose.from_w2c(w2c_est),
        'n_pairs':      n_pairs_used,
        'n_pts':        n_pts,
        'n_inliers':    n_inliers,
        'inlier_ratio': n_inliers / n_pts,
        'success':      True,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Depth-calibrated ICP pose refinement
# ═══════════════════════════════════════════════════════════════════════════════

def depth_icp_refine(
    pred_depth: np.ndarray,           # (H, W) predicted depth (affine-relative)
    building_mask: np.ndarray,        # (H, W) bool — building pixels to use
    face_cents_world: np.ndarray,     # (F, 3) face centroids in world frame
    face_normals_world: np.ndarray,   # (F, 3) unit face normals in world frame (point outward)
    fid_map: np.ndarray,              # (H, W) int face-ID map rendered at w2c_init
    w2c_init: np.ndarray,             # (4, 4) initial w2c pose
    K: np.ndarray,                    # (3, 3) intrinsics
    n_iters: int = 5,
    min_calib_pts: int = 200,
    dist_thresh_m: float = 2.0,       # max point-to-face distance for ICP inlier
    stop_thresh_m: float = 0.05,      # convergence: camera centre movement < 5 cm
) -> dict:
    """Refine pose using calibrated predicted depth + point-to-plane ICP.

    Algorithm
    ---------
    For each iteration:
    1. Render face IDs at current pose.
    2. Calibrate predicted depth: fit scale s and shift t via least squares
       so that  d_lod[i] = s * pred_d[i] + t  for face-matched pixels.
    3. Lift all building pixels to 3D world points using calibrated depth.
    4. For each lifted point find the closest LoD face (nearest face centroid).
    5. Solve point-to-plane ICP update via SVD.
    6. Apply update; stop if camera centre moves < stop_thresh_m.

    Returns dict with keys: success, pose (CameraPose), w2c (4×4),
    n_calib_pts, scale, shift, n_icp_inliers, iters_run.
    """
    H, W = pred_depth.shape[:2]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # Pre-compute pixel ray directions (unit vectors in camera frame)
    vs_b, us_b = np.where(building_mask)              # building pixel indices
    if len(us_b) < min_calib_pts:
        return {'success': False, 'reason': 'too few building pixels'}

    # Pixel rays: (N, 3)  — direction in camera frame at unit depth
    rays = np.stack([
        (us_b - cx) / fx,
        (vs_b - cy) / fy,
        np.ones(len(us_b), dtype=np.float64),
    ], axis=1)   # (N, 3)

    w2c_cur = w2c_init.copy().astype(np.float64)
    n_calib, s_final, t_final = 0, 1.0, 0.0

    for _itr in range(n_iters):
        R_cur = w2c_cur[:3, :3]
        t_cur = w2c_cur[:3,  3]
        R_inv = R_cur.T              # world ← camera

        # ── Step 1: render face IDs at current pose ─────────────────────────
        # NOTE: fid_map is pre-rendered by the caller at the initial pose.
        # Inside the loop we re-use the caller-supplied fid_map only for the
        # FIRST iteration; for subsequent iterations the caller should pass
        # updated fid_maps.  Here we use the initial one throughout (cheap
        # approximation — works when the pose change is small).
        fids_b = fid_map[vs_b, us_b]         # face ID per building pixel
        valid_fid = (fids_b >= 0) & (fids_b < len(face_cents_world))
        if valid_fid.sum() < min_calib_pts:
            return {'success': False, 'reason': 'too few face-matched building pixels'}

        # ── Step 2: calibrate depth (scale + shift) ─────────────────────────
        # Expected camera-frame depth for face-matched pixels:
        P_w_matched = face_cents_world[fids_b[valid_fid]]   # (M, 3)
        P_cam = (R_cur @ P_w_matched.T).T + t_cur            # (M, 3)
        d_lod = P_cam[:, 2]                                  # (M,)

        d_pred_m = pred_depth[vs_b[valid_fid], us_b[valid_fid]].astype(np.float64)

        # Robust least squares: d_lod = s * d_pred + t
        A = np.column_stack([d_pred_m, np.ones(len(d_pred_m))])
        try:
            params, *_ = np.linalg.lstsq(A, d_lod, rcond=None)
        except np.linalg.LinAlgError:
            return {'success': False, 'reason': 'depth calibration failed'}
        s, t_shift = float(params[0]), float(params[1])

        # Sanity check: positive scale, reasonable range
        if s <= 0 or s > 1000 or not np.isfinite(s):
            return {'success': False, 'reason': f'bad depth scale s={s:.3g}'}

        n_calib = int(valid_fid.sum())
        s_final, t_final = s, t_shift

        # ── Step 3: lift all building pixels to world ───────────────────────
        d_cal = s * pred_depth[vs_b, us_b].astype(np.float64) + t_shift  # (N,)
        pos_depth = d_cal > 0.1
        if pos_depth.sum() < 10:
            return {'success': False, 'reason': 'calibrated depths mostly negative'}

        d_cal_valid = d_cal[pos_depth]
        rays_valid  = rays[pos_depth]            # (N', 3)

        P_cam_lifted = rays_valid * d_cal_valid[:, None]   # (N', 3)
        P_world = (R_inv @ (P_cam_lifted - t_cur).T).T     # (N', 3)

        # ── Step 4: find closest LoD face centroid for each lifted point ─────
        # Use KD-tree for memory-efficient nearest-neighbor (brute-force needs
        # C×F×3 floats which can be tens of GB for F=144k, C=50k).
        _kd = cKDTree(face_cents_world)
        nn_dists, nn_fids = _kd.query(P_world, workers=-1)

        inlier_mask = nn_dists < dist_thresh_m
        if inlier_mask.sum() < 10:
            return {'success': False, 'reason': 'too few ICP inliers'}

        P_src = P_world[inlier_mask]                         # (M, 3)
        P_tgt = face_cents_world[nn_fids[inlier_mask]]       # (M, 3)
        n_icp = int(inlier_mask.sum())

        # ── Step 5: SVD-based rigid transform (world ← world) ───────────────
        mu_src = P_src.mean(0)
        mu_tgt = P_tgt.mean(0)
        Ac = P_src - mu_src
        Bc = P_tgt - mu_tgt
        H_mat = Ac.T @ Bc                                    # (3, 3)
        U, _, Vt = np.linalg.svd(H_mat)
        R_delta = Vt.T @ U.T
        if np.linalg.det(R_delta) < 0:
            Vt[-1] *= -1
            R_delta = Vt.T @ U.T
        t_delta = mu_tgt - R_delta @ mu_src                  # world translation

        # Convert world-frame rigid update to w2c update:
        # new_world = R_delta @ old_world + t_delta
        # w2c maps: x_cam = R_cur @ x_world + t_cur
        # After update: x_cam = R_cur @ (R_delta^T @ (x_world_new - t_delta)) + t_cur
        #             = R_cur @ R_delta^T @ x_world_new + R_cur @ (-R_delta^T @ t_delta) + t_cur
        R_new = R_cur @ R_delta.T
        t_new = t_cur - R_new @ t_delta

        w2c_new = np.eye(4, dtype=np.float64)
        w2c_new[:3, :3] = R_new
        w2c_new[:3,  3] = t_new

        # ── Step 6: check convergence ────────────────────────────────────────
        cam_old = -R_cur.T @ t_cur
        cam_new = -R_new.T @ t_new
        delta_m = float(np.linalg.norm(cam_new - cam_old))

        w2c_cur = w2c_new

        if delta_m < stop_thresh_m:
            break

    return {
        'success': True,
        'pose':    CameraPose.from_w2c(w2c_cur),
        'w2c':     w2c_cur,
        'n_calib_pts': n_calib,
        'scale':   s_final,
        'shift':   t_final,
        'n_icp_inliers': n_icp,
        'iters_run': _itr + 1,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Depth-plane two-stage pose+scale solve (principal algorithm)
# ═══════════════════════════════════════════════════════════════════════════════

def depth_plane_pnp(
    matches: List[dict],
    pred_label_map: np.ndarray,      # (H, W) int  — predicted plane IDs
    pred_depth: np.ndarray,          # (H, W) float — MoGe depth (affine-relative)
    K: np.ndarray,                   # (3, 3) camera intrinsics
    w2c_init: np.ndarray,            # (4, 4) current render pose (LoD rendered at this)
    face_cents_world: np.ndarray,    # (F, 3) mesh face centroids in world frame
    lod_label_map: np.ndarray,       # (H, W) int  — LoD plane IDs rendered at w2c_init
    fid_map: np.ndarray,             # (H, W) int  — face IDs rendered at w2c_init
    n_outer: int = 5,
    max_pts_per_pair: int = 300,
    min_pts: int = 20,
    inlier_dist_m: float = 3.0,
    rng_seed: int = 42,
) -> dict:
    """Pose estimation by alternating two well-conditioned linear solves.

    **Why two stages?**
    The joint 5-DOF system [s, shift, tx, ty, tz] is rank-deficient for UAV
    scenes: most planes are horizontal (rooftops) → all `-n_cam` rows are ≈[0,0,-1]
    → tx and ty are barely constrained together with s/shift.

    **Stage 1 — calibrate depth** (2-DOF, fix t = t_cur):
        c_i * s * d_i + c_i * shift = d_w_i + n_cam_i · t_cur
        where c_i = n_cam_i · ray_i.

    **Stage 2 — solve translation** (3-DOF, fix s and shift):
        With X_cam_i = ray_i * (s*d_i + shift):
        n_cam_i · t = n_cam_i · X_cam_i - d_w_i

    Both stages use IRLS (Huber) for robustness.  R is held fixed
    (GPS rotation error ~1 deg is negligible vs 5-25 m translation error).

    NOTE: Disabled — horizontal plane degeneracy in UAV scenes makes this
    ill-conditioned (only tz is constrained, not tx/ty). Reverted to always
    return failure so the Edge-PF pipeline is not disrupted.
    """
    # Disabled: horizontal planes only constrain tz, not tx/ty → wild solutions
    return {'n_pairs': 0, 'n_pts': 0, 'n_inliers': 0,
            'inlier_ratio': 0.0, 'pose': None, 'success': False}

    rng = np.random.default_rng(rng_seed)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    F = len(face_cents_world)

    R_cur = w2c_init[:3, :3].astype(np.float64).copy()
    t_cur = w2c_init[:3,  3].astype(np.float64).copy()

    valid_fid = (fid_map >= 0) & (fid_map < F)

    def _irls(A: np.ndarray, b: np.ndarray, n_iter: int = 8,
              prior_mu: np.ndarray | None = None,
              prior_lam: float = 0.0) -> np.ndarray | None:
        """Weighted least-squares with iterative Huber re-weighting."""
        w = np.ones(len(b))
        x = None
        for _ in range(n_iter):
            Aw = A * w[:, None]
            AtA = Aw.T @ A
            Atb = Aw.T @ (b * w)
            if prior_mu is not None and prior_lam > 0.0:
                AtA += prior_lam * np.eye(len(prior_mu))
                Atb += prior_lam * prior_mu
            try:
                x = np.linalg.solve(AtA + 1e-12 * np.eye(AtA.shape[0]), Atb)
            except np.linalg.LinAlgError:
                return None
            resid = A @ x - b
            mad = max(float(np.median(np.abs(resid))), 1e-3)
            delta = 2.0 * mad
            w = np.where(np.abs(resid) <= delta, 1.0, delta / (np.abs(resid) + 1e-12))
        return x

    # ── Collect per-pixel constraints ────────────────────────────────────────
    rays_list  : List[np.ndarray] = []
    d_pred_list: List[np.ndarray] = []
    n_w_list   : List[np.ndarray] = []
    d_w_list   : List[np.ndarray] = []
    n_pairs = 0

    for m in matches:
        lpid = int(m['lod']['plane_id'])
        ppid = int(m['pred']['plane_id'])

        n_cam_lod = np.asarray(m['lod']['normal_mean'], dtype=np.float64)
        n_cam_lod /= np.linalg.norm(n_cam_lod) + 1e-12
        n_w = R_cur.T @ n_cam_lod
        n_w /= np.linalg.norm(n_w) + 1e-12

        lod_mask = (lod_label_map == lpid) & valid_fid
        if not lod_mask.any():
            continue
        fids_lp = fid_map[lod_mask]
        fids_lp = fids_lp[(fids_lp >= 0) & (fids_lp < F)]
        if len(fids_lp) == 0:
            continue
        d_w = float(n_w @ face_cents_world[fids_lp].mean(axis=0))

        vs, us = np.where(pred_label_map == ppid)
        if len(vs) < 3:
            continue
        d_pix = pred_depth[vs, us].astype(np.float64)
        ok = (d_pix > 0.0) & np.isfinite(d_pix)
        vs, us, d_pix = vs[ok], us[ok], d_pix[ok]
        if len(vs) < 3:
            continue
        if len(vs) > max_pts_per_pair:
            idx = rng.choice(len(vs), max_pts_per_pair, replace=False)
            vs, us, d_pix = vs[idx], us[idx], d_pix[idx]

        rays = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones(len(us))], axis=1)
        rays_list.append(rays)
        d_pred_list.append(d_pix)
        n_w_list.append(np.tile(n_w, (len(us), 1)))
        d_w_list.append(np.full(len(us), d_w))
        n_pairs += 1

    base = {'n_pairs': n_pairs, 'n_pts': 0, 'n_inliers': 0,
            'inlier_ratio': 0.0, 'pose': None, 'success': False}
    if not rays_list:
        return base

    rays_all   = np.concatenate(rays_list,   axis=0)   # (N, 3)
    d_pred_all = np.concatenate(d_pred_list, axis=0)   # (N,)
    n_w_all    = np.concatenate(n_w_list,    axis=0)   # (N, 3)
    d_w_all    = np.concatenate(d_w_list,    axis=0)   # (N,)
    N = len(d_pred_all)
    base['n_pts'] = N
    if N < min_pts:
        return base

    best_w2c       = w2c_init.copy()
    best_n_inliers = 0
    best_inlr_frac = 0.0
    s_best, shift_best = 1.0, 0.0

    for _outer in range(n_outer):
        n_cam_all = (R_cur @ n_w_all.T).T                         # (N, 3)
        c_all     = np.einsum('ij,ij->i', n_cam_all, rays_all)    # (N,)
        nc_t      = n_cam_all @ t_cur                              # (N,) n_cam · t_cur

        # ── Stage 1: calibrate (s, shift) fixing t = t_cur ───────────────────
        # c * s * d + c * shift = d_w + n_cam · t_cur
        A1 = np.stack([c_all * d_pred_all, c_all], axis=1)        # (N, 2)
        b1 = d_w_all + nc_t                                        # (N,)
        sol1 = _irls(A1, b1, prior_mu=np.array([1.0, 0.0]), prior_lam=1e-4)
        if sol1 is None or not np.all(np.isfinite(sol1)):
            break
        s_new, shift_new = float(sol1[0]), float(sol1[1])
        if not (0.01 <= s_new <= 500.0):
            break

        # ── Stage 2: solve (tx, ty, tz) fixing (s, shift) ────────────────────
        # X_cam = ray * (s*d + shift),  n_cam · t = n_cam · X_cam - d_w
        d_cal  = s_new * d_pred_all + shift_new                    # (N,)
        ok3d   = d_cal > 0.5
        if ok3d.sum() < min_pts:
            break
        X_cam_all  = rays_all * d_cal[:, None]                     # (N, 3)
        nc_X       = np.einsum('ij,ij->i', n_cam_all, X_cam_all)   # (N,)
        A2 = n_cam_all[ok3d]                                        # (M, 3)
        b2 = (nc_X - d_w_all)[ok3d]                                # (M,)
        sol2 = _irls(A2, b2, prior_mu=t_cur, prior_lam=1e-2)
        if sol2 is None or not np.all(np.isfinite(sol2)):
            break
        t_new = sol2

        # Update state
        s_best, shift_best = s_new, shift_new
        t_cur = t_new

        # Count inliers
        d_cal2 = s_best * d_pred_all + shift_best
        X_cam2 = rays_all * d_cal2[:, None]
        X_w2   = (R_cur.T @ (X_cam2 - t_cur[None, :]).T).T
        resid  = np.einsum('ij,ij->i', n_w_all, X_w2) - d_w_all
        inl    = np.abs(resid) < inlier_dist_m
        n_inl  = int(inl.sum())
        inlr   = n_inl / N

        if n_inl > best_n_inliers:
            best_n_inliers = n_inl
            best_inlr_frac = inlr
            best_w2c = np.eye(4, dtype=np.float64)
            best_w2c[:3, :3] = R_cur
            best_w2c[:3,  3] = t_cur

    print(f"    [depth_plane_pnp] n_pairs={n_pairs} N={N} "
          f"s={s_best:.3f} shift={shift_best:.2f} "
          f"best_inliers={best_n_inliers}/{N} ({best_inlr_frac*100:.1f}%)")

    if best_n_inliers < min_pts:
        return base

    return {
        'pose':         CameraPose.from_w2c(best_w2c),
        'n_pairs':      n_pairs,
        'n_pts':        N,
        'n_inliers':    best_n_inliers,
        'inlier_ratio': best_inlr_frac,
        'success':      True,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Image-space centroid RANSAC PnP (no pre-matching needed)
# ═══════════════════════════════════════════════════════════════════════════════

def image_centroid_ransac_pnp(
    lod_planes_e: List[dict],
    pred_planes_e: List[dict],
    w2c: np.ndarray,
    K: np.ndarray,
    max_dist_px: float = 120.0,
    reprojection_error_px: float = 20.0,
    confidence: float = 0.999,
    ransac_iter: int = 5000,
    min_matches: int = 6,
    iou_weight: float = 0.3,
    image_hw: Optional[Tuple[int, int]] = None,
    lowe_ratio: float = 0.75,
) -> dict:
    """Pose estimation by matching plane centroids (mutual NN) with RANSAC.

    Algorithm
    ---------
    For each LoD plane rendered at the current pose:
      1. Project its 3D centroid (camera frame at ``w2c``) into image space via K.
      2. Match to nearest pred plane centroid via **mutual nearest-neighbour** on
         pixel distance.  Keep only pairs within ``max_dist_px`` pixels.
      3. Convert LoD centroid to world frame: ``xyz_w = R^T @ (c3d_cam - t)``.
      4. Run ``solvePnPRansac`` on all collected correspondences.

    This differs from ``plane_centroid_pnp`` in that **no pre-matching is
    required**: correspondences are formed dynamically by nearest-centroid
    proximity in image space, making it applicable even when face-ID or
    normal-based matching is unavailable (e.g. with the base MoGe-2 model
    that lacks a fine-tuned mask head).

    Parameters
    ----------
    lod_planes_e  : enriched LoD planes (from ``enrich_planes``), each with
                    ``centroid_3d`` (camera frame at ``w2c``) and ``plane_id``.
    pred_planes_e : enriched pred planes, each with ``centroid_2d`` (image frame).
    w2c           : (4, 4) world-to-camera matrix used when rendering the LoD
                    planes (``centroid_3d`` is expressed in *this* camera frame).
    K             : (3, 3) camera intrinsics.
    max_dist_px   : centroid distance beyond which a pair is penalised but not
                    hard-rejected (the normalised distance saturates at 1.0).
    lowe_ratio    : Lowe ratio-test threshold (0 < ratio < 1).  A pair (i, j)
                    is kept only if ``d(i,j) < lowe_ratio × d(i, j2)`` where j2
                    is the second-nearest pred plane, AND symmetrically from the
                    pred side.  Planes whose nearest two candidates are similar
                    in distance are ambiguous and excluded.  Set to 1.0 to
                    disable.
    iou_weight    : weight in [0, 1] for the IoU term in the combined cost.
                    0 = pure centroid distance, 1 = pure (1 - IoU).
                    Keep at ≤ 0.3 so centroid proximity still dominates when
                    the pose has significant error (IoU at wrong pose is noisy).
    image_hw      : (H, W) for IoU computation.  Auto-detected from pixel_indices
                    if not provided.
    reprojection_error_px : RANSAC inlier threshold (pixels).
    confidence    : RANSAC confidence level.
    ransac_iter   : max RANSAC iterations.
    min_matches   : minimum correspondences needed to attempt PnP.

    Returns
    -------
    dict with keys:
        pose         : CameraPose or None
        n_pts        : total correspondences formed (before RANSAC)
        n_inliers    : RANSAC inlier count
        inlier_ratio : n_inliers / n_pts
        success      : bool
    """
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    base = {'n_pts': 0, 'n_inliers': 0, 'inlier_ratio': 0.0,
            'pose': None, 'success': False}

    # Collect valid pred centroids as (N_pred, 2) array
    pred_uvs      = []
    pred_pix_lists: List[Optional[np.ndarray]] = []
    pred_ids: List[int] = []
    for p in pred_planes_e:
        c2d = p.get('centroid_2d')
        if c2d is not None:
            pred_uvs.append(np.asarray(c2d, dtype=np.float64))
            pred_pix_lists.append(p.get('pixel_indices'))
            pred_ids.append(int(p['plane_id']))
    if not pred_uvs:
        return base
    pred_uvs = np.array(pred_uvs, dtype=np.float64)   # (M, 2)  col=u, row=v

    pts_3d_list: List[np.ndarray] = []
    pts_2d_list: List[np.ndarray] = []
    plane_pairs: List[tuple] = []

    # Collect LoD projected centroids and 3D world points (only valid/visible planes)
    lod_uv_proj:   List[np.ndarray] = []
    lod_xyz_world: List[np.ndarray] = []
    lod_pix_lists: List[Optional[np.ndarray]] = []
    lod_ids: List[int] = []

    for lp in lod_planes_e:
        c3d = lp.get('centroid_3d')
        if c3d is None:
            continue
        c3d = np.asarray(c3d, dtype=np.float64)
        if c3d[2] < 1e-3:   # behind camera
            continue
        u_proj = fx * c3d[0] / c3d[2] + cx
        v_proj = fy * c3d[1] / c3d[2] + cy
        lod_uv_proj.append(np.array([u_proj, v_proj]))
        lod_xyz_world.append(R.T @ (c3d - t))
        lod_pix_lists.append(lp.get('pixel_indices'))
        lod_ids.append(int(lp['plane_id']))

    if not lod_uv_proj:
        return base

    lod_uv_arr = np.array(lod_uv_proj, dtype=np.float64)   # (L, 2)
    lod_xyz_arr = np.array(lod_xyz_world, dtype=np.float64)  # (L, 3)

    L = len(lod_uv_arr)
    M = len(pred_uvs)

    # ── IoU matrix ───────────────────────────────────────────────────────────
    # Compute pixel-mask IoU between every (LoD i, pred j) pair efficiently
    # using a label-image approach: O(H*W) cost regardless of L*M.
    iou_mat = np.zeros((L, M), dtype=np.float64)
    # Only compute IoU if we have pixel data and image dimensions
    hw = image_hw
    if hw is None:
        # Auto-detect from max pixel index across all planes
        all_pix = [p for p in (lod_pix_lists + pred_pix_lists) if p is not None]
        if all_pix:
            all_pix_cat = np.concatenate(all_pix, axis=0)
            hw = (int(all_pix_cat[:, 0].max()) + 1, int(all_pix_cat[:, 1].max()) + 1)
    if hw is not None:
        H_img, W_img = hw
        # Build label maps: lod_map[k] = plane index i (or -1), pred_map[k] = j (or -1)
        lod_map  = np.full(H_img * W_img, -1, dtype=np.int32)
        pred_map = np.full(H_img * W_img, -1, dtype=np.int32)
        lod_areas  = np.zeros(L, dtype=np.int64)
        pred_areas = np.zeros(M, dtype=np.int64)
        for i, pix in enumerate(lod_pix_lists):
            if pix is not None and len(pix):
                flat = pix[:, 0] * W_img + pix[:, 1]
                lod_map[flat] = i
                lod_areas[i] = len(flat)
        for j, pix in enumerate(pred_pix_lists):
            if pix is not None and len(pix):
                flat = pix[:, 0] * W_img + pix[:, 1]
                pred_map[flat] = j
                pred_areas[j] = len(flat)
        # Find pixels covered by both a LoD plane and a pred plane
        both = (lod_map >= 0) & (pred_map >= 0)
        li = lod_map[both]   # (K,) LoD plane indices
        pj = pred_map[both]  # (K,) pred plane indices
        if len(li):
            # Count intersections for each (i,j) pair
            pair_idx = li.astype(np.int64) * M + pj.astype(np.int64)
            inter_flat = np.bincount(pair_idx, minlength=L * M).reshape(L, M)
            union_mat = (lod_areas[:, None] + pred_areas[None, :] - inter_flat).clip(min=1)
            iou_mat = inter_flat / union_mat

    # ── Mutual nearest-neighbour matching on centroid distance ───────────────
    # For each LoD plane i, find nearest pred plane j; keep only if pred j's
    # nearest LoD plane is also i (bidirectional consistency) and within
    # max_dist_px.  IoU is computed but reserved for future filtering once
    # the pose is near-correct; at multi-metre errors the masks don't overlap
    # even for correct pairs, making IoU unreliable as a match criterion here.
    #
    # Additionally apply Lowe's ratio test (both directions) to exclude
    # ambiguous pairs where two candidate planes are nearly equidistant.
    dist_mat = np.sqrt(
        ((lod_uv_arr[:, None, :] - pred_uvs[None, :, :]) ** 2).sum(-1)
    )  # (L, M)

    # Sort to get two nearest on each side
    lod_sorted = np.sort(dist_mat, axis=1)      # (L, M) sorted ascending per row
    pred_sorted = np.sort(dist_mat, axis=0)     # (L, M) sorted ascending per col

    # Nearest and second-nearest distances
    lod_d1  = lod_sorted[:, 0]                  # (L,) nearest pred distance per LoD
    lod_d2  = lod_sorted[:, 1] if M > 1 else lod_d1 + 1e-9   # (L,) 2nd nearest
    pred_d1 = pred_sorted[0, :]                 # (M,) nearest LoD distance per pred
    pred_d2 = pred_sorted[1, :] if L > 1 else pred_d1 + 1e-9  # (M,) 2nd nearest

    # Mutual nearest-neighbour on centroid distance
    lod_to_pred = np.argmin(dist_mat, axis=1)  # (L,)
    pred_to_lod = np.argmin(dist_mat, axis=0)  # (M,)

    for i in range(L):
        j = lod_to_pred[i]
        # Mutual check
        if pred_to_lod[j] != i:
            continue
        # Hard distance gate
        if dist_mat[i, j] > max_dist_px:
            continue
        # Lowe ratio test (LoD side): nearest must be clearly better than 2nd nearest
        if lowe_ratio < 1.0 and L > 1 and M > 1:
            if lod_d1[i] > lowe_ratio * lod_d2[i]:
                continue   # LoD plane i is ambiguous
            if pred_d1[j] > lowe_ratio * pred_d2[j]:
                continue   # pred plane j is ambiguous
        pts_3d_list.append(lod_xyz_arr[i])
        pts_2d_list.append(pred_uvs[j])
        plane_pairs.append((lod_ids[i], pred_ids[j], float(iou_mat[i, j])))

    n_pts = len(pts_3d_list)
    base['n_pts'] = n_pts
    base['plane_pairs'] = plane_pairs
    if n_pts < min_matches:
        return base

    pts_3d = np.array(pts_3d_list, dtype=np.float64)   # (N, 3)
    pts_2d = np.array(pts_2d_list, dtype=np.float64)   # (N, 2)

    K_f64 = K.astype(np.float64)
    cv2.setRNGSeed(42)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, K_f64, None,
        iterationsCount=ransac_iter,
        reprojectionError=reprojection_error_px,
        confidence=confidence,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok or inliers is None or len(inliers) < min_matches:
        return {**base, 'n_pts': n_pts, 'plane_pairs': plane_pairs}

    n_inliers = int(len(inliers))
    idx = inliers.ravel()
    # Refine on inliers only
    ok2, rvec, tvec = cv2.solvePnP(
        pts_3d[idx], pts_2d[idx], K_f64, None,
        rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    R_mat, _ = cv2.Rodrigues(rvec)
    w2c_est = np.eye(4, dtype=np.float64)
    w2c_est[:3, :3] = R_mat
    w2c_est[:3, 3] = tvec.ravel()

    # Mean reprojection error of inliers (tighter = better pose quality)
    pts3_inlier = pts_3d[idx].reshape(-1, 1, 3)
    proj_inlier, _ = cv2.projectPoints(pts3_inlier, rvec, tvec, K_f64, None)
    proj_inlier = proj_inlier.reshape(-1, 2)
    reproj_errs = np.linalg.norm(proj_inlier - pts_2d[idx], axis=1)
    mean_reproj_err = float(np.mean(reproj_errs))

    return {
        'pose':             CameraPose.from_w2c(w2c_est),
        'n_pts':            n_pts,
        'n_inliers':        n_inliers,
        'inlier_ratio':     n_inliers / n_pts,
        'mean_reproj_err':  mean_reproj_err,
        'success':          True,
        'plane_pairs':      plane_pairs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 8b. Image-space centroid LM (Levenberg-Marquardt, implicit matching)
# ═══════════════════════════════════════════════════════════════════════════════

def image_centroid_lm(
    lod_planes_e: List[dict],
    pred_planes_e: List[dict],
    w2c: np.ndarray,
    K: np.ndarray,
    max_dist_px: float = 200.0,
    max_lm_iters: int = 200,
    ftol: float = 1e-4,
    min_planes: int = 6,
) -> dict:
    """Pose refinement via Levenberg-Marquardt with implicit centroid matching.

    Minimises the sum-of-squared nearest-centroid residuals **without any
    explicit matching step**:

        r_i(T) = π(K, T, c_i^{3D})  −  c_{j*(T)}^{2D}
        j*(T)  = argmin_j ‖π(K, T, c_i^{3D}) − c_j^{2D}‖

    ``scipy.optimize.least_squares(method='lm')`` drives the optimisation.
    At every Jacobian evaluation the nearest-pred-centroid assignment is
    recomputed implicitly, so the pose and the matching co-evolve without
    alternation.  This avoids the "fix matching → solve PnP → repeat" loop
    that can diverge when the initial pose is off by several metres.

    Only LoD planes whose projected centroid falls within ``max_dist_px`` of
    *any* pred centroid are included in the residual (outlier rejection).

    Parameters
    ----------
    lod_planes_e  : enriched LoD planes (``centroid_3d`` in camera frame).
    pred_planes_e : enriched pred planes (``centroid_2d`` in image frame).
    w2c           : (4,4) initial world-to-camera matrix.
    K             : (3,3) camera intrinsics.
    max_dist_px   : maximum NN centroid distance; LoD planes further than this
                    from every pred plane are excluded from the residual.
    max_lm_iters  : maximum LM function evaluations.
    ftol          : relative cost change tolerance for convergence.
    min_planes    : minimum number of LoD planes required to attempt optimisation.

    Returns
    -------
    dict with keys  pose / success / cost_init / cost_final / n_residuals
    """

    base = {'pose': None, 'success': False, 'cost_init': float('nan'),
            'cost_final': float('nan'), 'n_residuals': 0}

    R0 = w2c[:3, :3]
    t0 = w2c[:3, 3]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # Collect LoD 3D world-frame centroids
    lod_xyz_world: List[np.ndarray] = []
    for lp in lod_planes_e:
        c3d = lp.get('centroid_3d')
        if c3d is None:
            continue
        c3d_f = np.asarray(c3d, dtype=np.float64)
        if c3d_f[2] < 1e-3:
            continue
        # centroid_3d is in camera frame at w2c; convert to world
        lod_xyz_world.append(R0.T @ (c3d_f - t0))
    if len(lod_xyz_world) < min_planes:
        return base
    pts3d = np.array(lod_xyz_world, dtype=np.float64)   # (L, 3)

    # Collect pred 2D centroids
    pred_uvs: List[np.ndarray] = []
    for pp in pred_planes_e:
        c2d = pp.get('centroid_2d')
        if c2d is not None:
            pred_uvs.append(np.asarray(c2d, dtype=np.float64))
    if not pred_uvs:
        return base
    pred_c = np.array(pred_uvs, dtype=np.float64)   # (M, 2)

    K64 = K.astype(np.float64)

    # Parametrisation: [rv0, rv1, rv2, tx, ty, tz]  (Rodrigues + translation)
    rv0, _ = cv2.Rodrigues(R0)
    x0 = np.concatenate([rv0.ravel(), t0])   # (6,)

    def _project(x: np.ndarray) -> np.ndarray:
        """Project pts3d to image; return (L, 2) array of (u, v)."""
        rv = x[:3]
        t  = x[3:]
        R_m, _ = cv2.Rodrigues(rv)
        # Camera-frame points
        p_cam = (R_m @ pts3d.T + t[:, None])  # (3, L)
        z = p_cam[2]
        # Only keep points in front of camera; clip to avoid div/0
        z = np.maximum(z, 1e-4)
        u = fx * p_cam[0] / z + cx
        v = fy * p_cam[1] / z + cy
        return np.stack([u, v], axis=1)   # (L, 2)

    def _residuals(x: np.ndarray) -> np.ndarray:
        """2-D residual per LoD plane: projection − nearest pred centroid."""
        proj = _project(x)                          # (L, 2)
        dists = scipy_cdist(proj, pred_c)           # (L, M)
        nn_j  = dists.argmin(axis=1)               # (L,)
        nn_d  = dists[np.arange(len(proj)), nn_j]  # (L,)
        # Exclude LoD planes too far from any pred centroid (outliers)
        mask = nn_d < max_dist_px
        if mask.sum() < min_planes:
            # Return large residuals so LM doesn't accept this step
            return np.full(2 * len(proj), max_dist_px)
        target = pred_c[nn_j]                       # (L, 2)
        res = proj - target                         # (L, 2)
        res[~mask] = 0.0                            # zero out outliers
        return res.ravel()                          # (2L,)

    r0 = _residuals(x0)
    cost_init = float(np.dot(r0, r0))
    base['cost_init'] = cost_init

    try:
        result = least_squares(
            _residuals, x0,
            method='lm',
            max_nfev=max_lm_iters,
            ftol=ftol,
            xtol=1e-6,
            gtol=1e-6,
        )
    except Exception:
        return base

    if not result.success and result.cost >= cost_init:
        return {**base, 'cost_init': cost_init, 'cost_final': float(result.cost)}

    # Build w2c from optimised parameters
    rv_opt = result.x[:3]
    t_opt  = result.x[3:]
    R_opt, _ = cv2.Rodrigues(rv_opt)
    w2c_opt = np.eye(4, dtype=np.float64)
    w2c_opt[:3, :3] = R_opt
    w2c_opt[:3, 3]  = t_opt

    return {
        'pose':        CameraPose.from_w2c(w2c_opt),
        'success':     True,
        'cost_init':   cost_init,
        'cost_final':  float(result.cost),
        'n_residuals': int(mask.sum()) if 'mask' in dir() else 0,
    }


# ───────────────────────────────────────────────────────────────────────────────

def image_centroid_ransac_lm(
    lod_planes_e: List[dict],
    pred_planes_e: List[dict],
    w2c: np.ndarray,
    K: np.ndarray,
    max_dist_px: float = 120.0,
    inlier_dist_px: float = 20.0,
    ransac_iter: int = 2000,
    min_matches: int = 6,
    lm_max_iters: int = 200,
    lm_inlier_gate: float = 60.0,
) -> dict:
    """RANSAC pose hypotheses with implicit-NN inlier counting, then LM refinement.

    Unlike ``image_centroid_ransac_pnp`` (which fixes correspondences before RANSAC),
    this function evaluates each RANSAC hypothesis using the **implicit nearest-
    neighbour at the candidate pose**, not the initial NN at the starting pose.

    Algorithm
    ---------
    1. From current ``w2c``, project LoD 3D centroids and form an initial pool of
       NN correspondences (same as ``image_centroid_ransac_pnp``).  This pool is
       used only for **hypothesis generation** (minimal PnP sample).
    2. For each RANSAC iteration:
       a. Sample 6 correspondences from the pool → solve minimal PnP → candidate T.
       b. Evaluate T by projecting ALL LoD 3D centroids at T and counting how many
          have a nearest pred centroid within ``inlier_dist_px`` pixels.
       c. The inlier count is with respect to the implicit NN at T, NOT the initial
          NN — so a hypothesis that happens to align the points correctly will score
          high even if the initial NN had wrong assignments.
    3. Keep the hypothesis T* with the most implicit-NN inliers.
    4. Run ``image_centroid_lm`` from T* for final LM refinement with a tighter
       distance gate (``lm_inlier_gate`` pixels).

    Parameters
    ----------
    lod_planes_e     : enriched LoD planes (``centroid_3d`` in camera frame at ``w2c``).
    pred_planes_e    : enriched pred planes (``centroid_2d`` in image frame).
    w2c              : (4,4) initial world-to-camera matrix.
    K                : (3,3) camera intrinsics.
    max_dist_px      : centroid proximity gate for the initial hypothesis pool.
    inlier_dist_px   : threshold for counting inliers at each RANSAC hypothesis.
    ransac_iter      : number of RANSAC hypotheses to evaluate.
    min_matches      : minimum correspondences in the hypothesis pool.
    lm_max_iters     : max LM function evaluations in the refinement step.
    lm_inlier_gate   : ``max_dist_px`` passed to ``image_centroid_lm`` for refinement.

    Returns
    -------
    dict with keys  pose / success / n_inliers / cost_final
    """

    base = {'pose': None, 'success': False, 'n_inliers': 0, 'cost_final': float('nan')}

    R0 = w2c[:3, :3]
    t0 = w2c[:3, 3]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    K64 = K.astype(np.float64)

    # ── Collect LoD 3D world-frame centroids ─────────────────────────────────
    lod_xyz_world: List[np.ndarray] = []
    for lp in lod_planes_e:
        c3d = lp.get('centroid_3d')
        if c3d is None:
            continue
        c3d_f = np.asarray(c3d, dtype=np.float64)
        if c3d_f[2] < 1e-3:
            continue
        lod_xyz_world.append(R0.T @ (c3d_f - t0))
    if len(lod_xyz_world) < min_matches:
        return base
    pts3d_all = np.array(lod_xyz_world, dtype=np.float64)   # (L, 3)

    # ── Collect pred 2D centroids ─────────────────────────────────────────────
    pred_uvs_list: List[np.ndarray] = []
    for pp in pred_planes_e:
        c2d = pp.get('centroid_2d')
        if c2d is not None:
            pred_uvs_list.append(np.asarray(c2d, dtype=np.float64))
    if not pred_uvs_list:
        return base
    pred_c = np.array(pred_uvs_list, dtype=np.float64)   # (M, 2)

    # ── Initial NN pool (hypothesis generation only) ─────────────────────────
    # Project LoD centroids at the starting pose to form initial correspondences.
    lod_uv0 = np.stack([
        fx * pts3d_all[:, 0] / (R0 @ pts3d_all.T + t0[:, None])[2] + cx,
        fy * pts3d_all[:, 1] / (R0 @ pts3d_all.T + t0[:, None])[2] + cy,
    ], axis=1)   # approximate; overridden by proper camera-space projection below

    # Proper projection at w2c
    cam0 = (R0 @ pts3d_all.T + t0[:, None])  # (3, L)
    lod_uv0 = np.stack([
        fx * cam0[0] / cam0[2] + cx,
        fy * cam0[1] / cam0[2] + cy,
    ], axis=1)  # (L, 2)

    dist0 = scipy_cdist(lod_uv0, pred_c)                 # (L, M)
    lod_nn = dist0.argmin(axis=1)                         # (L,)
    pred_nn = dist0.argmin(axis=0)                        # (M,)

    pool_3d: List[np.ndarray] = []
    pool_2d: List[np.ndarray] = []
    for i in range(len(pts3d_all)):
        j = lod_nn[i]
        if pred_nn[j] != i:
            continue
        if dist0[i, j] > max_dist_px:
            continue
        pool_3d.append(pts3d_all[i])
        pool_2d.append(pred_c[j])

    if len(pool_3d) < min_matches:
        return base

    pool3d = np.array(pool_3d, dtype=np.float64)   # (N, 3)
    pool2d = np.array(pool_2d, dtype=np.float64)   # (N, 2)
    N_pool = len(pool3d)

    # ── Helper: project pts3d_all at a candidate w2c ─────────────────────────
    def _project_all(rv: np.ndarray, t: np.ndarray) -> np.ndarray:
        R_m, _ = cv2.Rodrigues(rv)
        cam = R_m @ pts3d_all.T + t[:, None]   # (3, L)
        z = np.maximum(cam[2], 1e-4)
        return np.stack([fx * cam[0] / z + cx, fy * cam[1] / z + cy], axis=1)  # (L, 2)

    # ── RANSAC: sample from pool, evaluate with implicit NN at candidate T ───
    rng = np.random.default_rng(42)
    best_inliers = -1
    best_rv: Optional[np.ndarray] = None
    best_t:  Optional[np.ndarray] = None

    for _ in range(ransac_iter):
        idx = rng.choice(N_pool, size=min_matches, replace=False)
        ok, rv_c, tv_c = cv2.solvePnP(
            pool3d[idx], pool2d[idx], K64, None,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not ok:
            continue

        # Count inliers using implicit NN at this candidate pose
        proj_c = _project_all(rv_c.ravel(), tv_c.ravel())   # (L, 2)
        dist_c = scipy_cdist(proj_c, pred_c)                  # (L, M)
        nn_d   = dist_c.min(axis=1)                           # (L,)
        n_inl  = int((nn_d < inlier_dist_px).sum())

        if n_inl > best_inliers:
            best_inliers = n_inl
            best_rv = rv_c.ravel().copy()
            best_t  = tv_c.ravel().copy()

    if best_rv is None:
        return base

    # Build w2c from best RANSAC hypothesis
    R_best, _ = cv2.Rodrigues(best_rv)
    w2c_best = np.eye(4, dtype=np.float64)
    w2c_best[:3, :3] = R_best
    w2c_best[:3, 3]  = best_t

    # ── LM refinement from best hypothesis ───────────────────────────────────
    lm_res = image_centroid_lm(
        lod_planes_e, pred_planes_e, w2c_best, K,
        max_dist_px=lm_inlier_gate,
        max_lm_iters=lm_max_iters,
        min_planes=min_matches,
    )

    if lm_res['success']:
        return {
            'pose':        lm_res['pose'],
            'success':     True,
            'n_inliers':   best_inliers,
            'cost_final':  lm_res['cost_final'],
        }

    # Fall back to RANSAC hypothesis without LM
    return {
        'pose':        CameraPose.from_w2c(w2c_best),
        'success':     True,
        'n_inliers':   best_inliers,
        'cost_final':  float('nan'),
    }


# ───────────────────────────────────────────────────────────────────────────────

def image_centroid_trans_ransac(
    lod_planes_e: List[dict],
    pred_planes_e: List[dict],
    w2c: np.ndarray,
    K: np.ndarray,
    max_dist_px: float = 150.0,
    inlier_dist_px: float = 20.0,
    ransac_iter: int = 2000,
    min_matches: int = 3,
) -> dict:
    """Translation-only RANSAC pose refinement (rotation fixed from ``w2c``).

    Motivation
    ----------
    When the rotation estimate is already accurate (e.g. from a coarse stage)
    and only translation needs refinement, fixing rotation reduces the search
    from 6 DoF to 3 DoF.  A minimal translation hypothesis then needs only
    **2 correspondences** (instead of 6 for full PnP), making RANSAC far more
    likely to sample an all-inlier subset even when the correct correspondences
    are a small fraction of the pool.

    Algorithm
    ---------
    1. With rotation R fixed from ``w2c``, project LoD 3D world points to
       image via ``u = K (R X_w + t) / Z`` — now linear in translation t.
    2. Form an initial NN pool (same as centroid RANSAC, with mutual NN gate).
    3. RANSAC outer loop:
       a. Sample 2 correspondences → solve the 2-point linear system for t.
       b. Evaluate with implicit NN at candidate (R_fixed, t_cand): count
          LoD planes within ``inlier_dist_px`` of their nearest pred centroid.
       c. Keep hypothesis with most implicit-NN inliers.
    4. Refine on all inliers by least-squares (still translation only).

    Returns
    -------
    dict with keys  pose / success / n_inliers
    """

    base = {'pose': None, 'success': False, 'n_inliers': 0}

    R_fixed = w2c[:3, :3].copy()
    t0      = w2c[:3, 3].copy()
    fx, fy  = float(K[0, 0]), float(K[1, 1])
    cx, cy  = float(K[0, 2]), float(K[1, 2])
    K64     = K.astype(np.float64)

    # ── Collect LoD world-frame centroids ─────────────────────────────────────
    lod_world: List[np.ndarray] = []
    for lp in lod_planes_e:
        c3d = lp.get('centroid_3d')
        if c3d is None:
            continue
        c3d_f = np.asarray(c3d, dtype=np.float64)
        if c3d_f[2] < 1e-3:
            continue
        lod_world.append(R_fixed.T @ (c3d_f - t0))
    if len(lod_world) < min_matches:
        return base
    pts3d = np.array(lod_world, dtype=np.float64)   # (L, 3)

    # ── Collect pred 2D centroids ─────────────────────────────────────────────
    pred_c_list: List[np.ndarray] = []
    for pp in pred_planes_e:
        c2d = pp.get('centroid_2d')
        if c2d is not None:
            pred_c_list.append(np.asarray(c2d, dtype=np.float64))
    if not pred_c_list:
        return base
    pred_c = np.array(pred_c_list, dtype=np.float64)   # (M, 2)

    # ── Helper: project pts3d at (R_fixed, t) → (L, 2) ───────────────────────
    def _proj(t: np.ndarray) -> np.ndarray:
        cam = R_fixed @ pts3d.T + t[:, None]   # (3, L)
        z   = np.maximum(cam[2], 1e-4)
        return np.stack([fx * cam[0] / z + cx,
                         fy * cam[1] / z + cy], axis=1)   # (L, 2)

    # ── 2-point translation solver ────────────────────────────────────────────
    # Given correspondences (X_w_i → u_i), with R fixed:
    #   λ_i * [u_i, v_i, 1]^T = K (R X_w_i + t)
    # Eliminating λ_i:  u_i * (K R X_w + K t)_3 = (K R X_w + K t)_1
    #                   v_i * (K R X_w + K t)_3 = (K R X_w + K t)_2
    # Let b = K R X_w (precomputed constant), b = [bx, by, bz]:
    #   u_i * (bz + K33*tz + K31*tx + K32*ty) = bx + K11*tx + K13*tz
    # Since K is [fx 0 cx; 0 fy cy; 0 0 1]:
    #   u_i * (bz + tz) = bx + fx*tx + cx*(bz + tz)   ← fx*tx - (u_i-cx)*tz = (u_i-cx)*bz - bx
    #   v_i * (bz + tz) = by + fy*ty + cy*(bz + tz)   ← fy*ty - (v_i-cy)*tz = (v_i-cy)*bz - by
    # Linear in [tx, ty, tz].  2 correspondences → 4 equations, over-determined → lstsq.
    KR = K64 @ R_fixed   # (3, 3)

    def _solve_translation(idx3d: np.ndarray, uv2d: np.ndarray) -> Optional[np.ndarray]:
        """Solve for translation given 2+ correspondences (idx3d world pts, uv2d pixels)."""
        A_rows: List[np.ndarray] = []
        b_rows: List[float]     = []
        for xi, ui in zip(idx3d, uv2d):
            b_vec = KR @ xi           # (3,)
            u_px, v_px = ui[0], ui[1]
            # Row from u:  fx*tx + 0*ty + (cx - u_px)*tz = (u_px - cx)*b_vec[2] - b_vec[0]
            A_rows.append([fx,  0.0, cx - u_px])
            b_rows.append((u_px - cx) * b_vec[2] - b_vec[0])
            # Row from v:  0*tx + fy*ty + (cy - v_px)*tz = (v_px - cy)*b_vec[2] - b_vec[1]
            A_rows.append([0.0, fy,  cy - v_px])
            b_rows.append((v_px - cy) * b_vec[2] - b_vec[1])
        A = np.array(A_rows, dtype=np.float64)
        b = np.array(b_rows, dtype=np.float64)
        try:
            t_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        # Reject if camera is behind scene
        cam_test = R_fixed @ pts3d.T + t_sol[:, None]
        if (cam_test[2] < 0.1).mean() > 0.5:
            return None
        return t_sol

    # ── Initial NN pool ───────────────────────────────────────────────────────
    proj0   = _proj(t0)                              # (L, 2)
    dist0   = scipy_cdist(proj0, pred_c)             # (L, M)
    lod_nn  = dist0.argmin(axis=1)                   # (L,)
    pred_nn = dist0.argmin(axis=0)                   # (M,)

    pool_3d: List[np.ndarray] = []
    pool_2d: List[np.ndarray] = []
    for i in range(len(pts3d)):
        j = lod_nn[i]
        if pred_nn[j] != i:
            continue
        if dist0[i, j] > max_dist_px:
            continue
        pool_3d.append(pts3d[i])
        pool_2d.append(pred_c[j])

    if len(pool_3d) < min_matches:
        return base

    pool3d = np.array(pool_3d, dtype=np.float64)
    pool2d = np.array(pool_2d, dtype=np.float64)
    N_pool = len(pool3d)

    # ── RANSAC (2-point samples) ──────────────────────────────────────────────
    rng = np.random.default_rng(42)
    best_inliers = -1
    best_t: Optional[np.ndarray] = None

    for _ in range(ransac_iter):
        idx = rng.choice(N_pool, size=2, replace=False)
        t_cand = _solve_translation(pool3d[idx], pool2d[idx])
        if t_cand is None:
            continue

        proj_c  = _proj(t_cand)                      # (L, 2)
        dist_c  = scipy_cdist(proj_c, pred_c)        # (L, M)
        nn_d    = dist_c.min(axis=1)                 # (L,)
        n_inl   = int((nn_d < inlier_dist_px).sum())

        if n_inl > best_inliers:
            best_inliers = n_inl
            best_t = t_cand.copy()

    if best_t is None:
        return base

    # ── Refine on all inliers (least-squares translation) ────────────────────
    proj_best  = _proj(best_t)                       # (L, 2)
    dist_best  = scipy_cdist(proj_best, pred_c)      # (L, M)
    nn_j_best  = dist_best.argmin(axis=1)            # (L,)
    nn_d_best  = dist_best[np.arange(len(pts3d)), nn_j_best]
    inlier_mask = nn_d_best < inlier_dist_px

    if inlier_mask.sum() >= min_matches:
        inl_3d = pts3d[inlier_mask]
        inl_2d = pred_c[nn_j_best[inlier_mask]]
        t_refined = _solve_translation(inl_3d, inl_2d)
        if t_refined is not None:
            best_t = t_refined
            # Update inlier count
            proj_r = _proj(best_t)
            dist_r = scipy_cdist(proj_r, pred_c)
            nn_d_r = dist_r.min(axis=1)
            best_inliers = int((nn_d_r < inlier_dist_px).sum())

    w2c_out = np.eye(4, dtype=np.float64)
    w2c_out[:3, :3] = R_fixed
    w2c_out[:3, 3]  = best_t

    return {
        'pose':      CameraPose.from_w2c(w2c_out),
        'success':   True,
        'n_inliers': best_inliers,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Pose error metrics
# ═══════════════════════════════════════════════════════════════════════════════

def pose_error(est: CameraPose, gt: CameraPose) -> Tuple[float, float]:
    """Compute translation error (m) and rotation error (deg) between two poses.

    Returns
    -------
    t_err_m   : Euclidean distance between camera centres in world space (metres)
    R_err_deg : geodesic rotation angle (degrees)
    """
    t_err_m = float(np.linalg.norm(est.position - gt.position))

    # Geodesic angle: θ = arccos( (trace(R_err) - 1) / 2 )
    R_err    = est.R_c2w.T @ gt.R_c2w          # relative rotation
    cos_ang  = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    R_err_deg = float(np.degrees(np.arccos(cos_ang)))

    return t_err_m, R_err_deg
