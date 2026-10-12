"""
method/scoring.py
=================
Batch alignment scorer for the fine pose estimation stage.

Combines silhouette IoU, depth NCC, and normal consistency into a
joint score. Pre-computes per-image data once and scores batches of
rendered depth maps efficiently.
"""

import numpy as np
from utils.metrics import normals_from_depth


class BatchAlignmentScorer:
    """Pre-computes per-image data and scores batches of rendered depth maps.

    Faster than per-candidate scoring because:
      - Predicted depth/normal values under the building mask are extracted once
      - Pixel grid for depth unprojection is pre-allocated
      - Two-phase strategy: normals only computed for top-k candidates
    """

    def __init__(
        self,
        pred_depth: np.ndarray,
        pred_normals: np.ndarray,
        query_mask: np.ndarray,
        K: np.ndarray,
        min_pixels: int = 50,
    ):
        """
        Args:
            pred_depth: (H, W) float32 predicted metric depth
            pred_normals: (H, W, 3) float32 camera-frame unit normals
            query_mask: (H, W) uint8 building mask (>0 = building)
            K: (3, 3) intrinsics at the rendering resolution
            min_pixels: minimum valid pixels for meaningful scoring
        """
        H, W = pred_depth.shape
        self.H, self.W = H, W
        self.min_pixels = min_pixels

        mask_bool = query_mask > 0

        # Pre-index predicted depth under building region
        self._pred_valid = (pred_depth > 0.1) & mask_bool
        self._pred_flat = pred_depth[self._pred_valid].astype(np.float64)

        # Pre-index predicted normals under building region
        pn_norm = np.linalg.norm(pred_normals, axis=-1)
        self._pn_valid = mask_bool & (pred_depth > 0.1) & (pn_norm > 0.5)
        self._pred_normals_flat = pred_normals[self._pn_valid].astype(np.float32)

        # Pre-compute pixel grid for depth unprojection
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                           np.arange(H, dtype=np.float32))
        self._ux = ((u - cx) / fx).astype(np.float32)
        self._vy = ((v - cy) / fy).astype(np.float32)

    def score_depth_ncc_batch(self, rendered_depths: list) -> np.ndarray:
        """Compute depth NCC for a batch of rendered depth maps.

        Returns:
            (N,) float32 NCC values (clamped ≥ 0)
        """
        N = len(rendered_depths)
        nccs = np.zeros(N, dtype=np.float32)
        pred_valid = self._pred_valid
        pred_flat = self._pred_flat
        mp = self.min_pixels

        if len(pred_flat) < mp:
            return nccs

        for i, rd in enumerate(rendered_depths):
            sub = rd[pred_valid] > 0.1
            if sub.sum() < mp:
                continue

            o = pred_flat[sub]
            r = rd[pred_valid][sub].astype(np.float64)

            o_m, r_m = o.mean(), r.mean()
            o_s, r_s = o.std(), r.std()
            if o_s < 1e-6 or r_s < 1e-6:
                continue

            ncc = ((o - o_m) * (r - r_m)).mean() / (o_s * r_s)
            nccs[i] = max(float(ncc), 0.0)

        return nccs

    def _normals_from_rendered(self, rd: np.ndarray):
        """Compute normals from a rendered depth map (fast path with pre-allocated grid)."""
        z = rd.astype(np.float32)
        x = self._ux * z
        y = self._vy * z

        dx = np.empty((self.H, self.W, 3), dtype=np.float32)
        dy = np.empty((self.H, self.W, 3), dtype=np.float32)
        dx[:, :-1, 0] = x[:, 1:] - x[:, :-1]
        dx[:, :-1, 1] = y[:, 1:] - y[:, :-1]
        dx[:, :-1, 2] = z[:, 1:] - z[:, :-1]
        dx[:, -1, :] = 0
        dy[:-1, :, 0] = x[1:, :] - x[:-1, :]
        dy[:-1, :, 1] = y[1:, :] - y[:-1, :]
        dy[:-1, :, 2] = z[1:, :] - z[:-1, :]
        dy[-1, :, :] = 0

        n = np.cross(dx, dy)
        nlen = np.linalg.norm(n, axis=-1, keepdims=True)
        nlen = np.where(nlen > 1e-8, nlen, 1.0)
        n /= nlen

        rn_valid = (rd > 0.1) & (nlen[..., 0] > 1e-8)
        return n, rn_valid

    def score_normal_batch(self, rendered_depths: list) -> np.ndarray:
        """Compute normal consistency for a batch of rendered depth maps.

        Returns:
            (N,) float32 cosine similarity (clamped ≥ 0)
        """
        N = len(rendered_depths)
        ncos = np.zeros(N, dtype=np.float32)
        pn_valid = self._pn_valid
        pn_flat = self._pred_normals_flat
        mp = self.min_pixels

        if len(pn_flat) < mp:
            return ncos

        for i, rd in enumerate(rendered_depths):
            rn, rn_valid = self._normals_from_rendered(rd)
            sub = rn_valid[pn_valid]
            if sub.sum() < mp:
                continue

            pn_sub = pn_flat[sub]
            rn_sub = rn[pn_valid][sub]

            cos_sim = (pn_sub * rn_sub).sum(axis=-1)
            ncos[i] = max(float(np.clip(cos_sim, -1.0, 1.0).mean()), 0.0)

        return ncos

    def score_batch(
        self,
        ious: np.ndarray,
        rendered_depths: list,
        iou_weight: float = 0.3,
        depth_weight: float = 0.4,
        normal_weight: float = 0.3,
    ) -> np.ndarray:
        """Combined batch scoring.

        Phase 1: IoU + depth NCC for all candidates
        Phase 2: normal consistency only for top-k candidates (expensive)

        Returns:
            (N,) float32 combined scores
        """
        N = len(ious)
        nccs = self.score_depth_ncc_batch(rendered_depths)
        scores = iou_weight * ious + depth_weight * nccs

        if normal_weight > 0:
            TOP_K = min(N, max(10, N // 4))
            top_idx = np.argsort(scores)[-TOP_K:]
            top_depths = [rendered_depths[j] for j in top_idx]
            top_ncos = self.score_normal_batch(top_depths)
            for j, idx in enumerate(top_idx):
                scores[idx] += normal_weight * top_ncos[j]

        return scores
