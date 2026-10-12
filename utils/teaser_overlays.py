"""Teaser figure overlays: SAM3 mask and LoD-guided plane merge on dimmed UAV images."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt


def nearest_label_fill(label_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Fill -1 pixels within valid_mask using nearest assigned neighbor."""
    empty = (label_map < 0) & valid_mask
    if not empty.any():
        return label_map
    assigned = label_map >= 0
    _, nearest_idx = distance_transform_edt(~assigned, return_indices=True)
    filled = label_map.copy()
    filled[empty] = label_map[nearest_idx[0][empty], nearest_idx[1][empty]]
    return filled


def lod_guided_merge_planes(
    pred_label_map: np.ndarray,
    lod_label_map: np.ndarray,
    building_mask: np.ndarray,
    *,
    min_overlap_px: int = 10,
    min_frag_px: int = 500,
) -> np.ndarray:
    """Merge predicted sub-planes into LoD plane footprints for teaser visualization."""
    merged = np.full_like(pred_label_map, -1)
    n_lod = int(lod_label_map.max()) + 1 if lod_label_map.max() >= 0 else 0
    lod_footprint = lod_label_map >= 0

    for lod_pid in range(n_lod):
        lod_mask = lod_label_map == lod_pid
        if not lod_mask.any():
            continue
        pred_labels_here = pred_label_map[lod_mask]
        valid_preds = pred_labels_here[pred_labels_here >= 0]
        if len(valid_preds) == 0:
            continue
        unique, counts = np.unique(valid_preds, return_counts=True)
        matched = unique[counts >= min_overlap_px]
        for ppid in matched:
            merged[(pred_label_map == ppid) & lod_mask] = lod_pid

    for lod_pid in range(n_lod):
        region = merged == lod_pid
        if region.sum() < min_frag_px:
            merged[region] = -1

    for lod_pid in range(n_lod):
        lod_mask = lod_label_map == lod_pid
        assigned = (merged == lod_pid) & lod_mask
        empty = (merged < 0) & lod_mask
        if assigned.any() and empty.any():
            merged[empty] = lod_pid

    merged[~lod_footprint] = -1
    merged = nearest_label_fill(merged, lod_footprint & building_mask)
    merged[~building_mask] = -1
    return merged


def label_map_to_rgb(label_map: np.ndarray, black_mask: np.ndarray | None = None) -> np.ndarray:
    """Colorize plane label map with tab20."""
    colored = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    valid = label_map >= 0
    if not valid.any():
        return colored
    cmap = plt.get_cmap("tab20")
    labels = label_map[valid]
    colors = (cmap((labels % 20) / 20.0)[:, :3] * 255).astype(np.uint8)
    colored[valid] = colors
    if black_mask is not None:
        colored[black_mask] = 0
    return colored


def make_dimmed_uav(uav_bgr: np.ndarray, dim_factor: float, black_mask: np.ndarray) -> np.ndarray:
    dimmed = (uav_bgr.astype(np.float32) * dim_factor).clip(0, 255).astype(np.uint8)
    dimmed[black_mask] = 0
    return dimmed


def make_sam3_overlay(
    uav_bgr: np.ndarray,
    building_mask: np.ndarray,
    black_mask: np.ndarray,
    *,
    dim_factor: float = 0.35,
) -> np.ndarray:
    """White SAM3 mask on dimmed UAV background."""
    overlay = make_dimmed_uav(uav_bgr, dim_factor, black_mask)
    overlay[building_mask] = [255, 255, 255]
    return overlay


def make_planes_overlay(
    uav_bgr: np.ndarray,
    label_map: np.ndarray,
    black_mask: np.ndarray,
    *,
    dim_factor: float = 0.35,
) -> np.ndarray:
    """Colored planes with white contours on dimmed UAV background."""
    overlay = make_dimmed_uav(uav_bgr, dim_factor, black_mask)
    valid = (label_map >= 0) & (~black_mask)
    if valid.any():
        cmap = plt.get_cmap("tab20")
        labels = label_map[valid]
        colors_bgr = (cmap((labels % 20) / 20.0)[:, :3][:, ::-1] * 255).astype(np.uint8)
        overlay[valid] = colors_bgr

    contour_layer = np.zeros(label_map.shape, dtype=np.uint8)
    for pid in np.unique(label_map[label_map >= 0]):
        mask_i = (label_map == pid).astype(np.uint8)
        contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_layer, contours, -1, 255, 1)
    overlay[(contour_layer > 0) & (~black_mask)] = [255, 255, 255]
    overlay[black_mask] = 0
    return overlay
