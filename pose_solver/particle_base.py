"""Shared base logic for particle filtering and coarse pose matching.

CoarsePoseEstimator, ParticleFilterRefiner, and ParticleFilterRefinerDepth
each used to duplicate _params_to_w2c_batch / _sample_candidates / _update_center.
This mixin consolidates that code for all three classes.
"""
from torch.quasirandom import SobolEngine

import numpy as np
import torch

from utils.pose import (
    wrap_angle_deg_torch,
    circular_mean_deg_torch,
    euler_xyz_extrinsic_to_matrix_torch,
)
from utils.metrics import iou_binary_batch, iou_multiclass_batch


class PoseHypothesisBase:
    """Subclasses must provide self.renderer (with .dev) and self.conf
    (with beams / num_candidates / update_temperature).

    Not every subclass uses _sample_candidates / _update_center
    (CoarsePoseEstimator uses a grid instead of Gaussian sampling),
    but keeping them together aids readability.
    """

    # Subclasses should assign these in __init__
    renderer = None
    conf = None
    device = None

    # ------------------------------
    # (N, 6) parameter tensor -> (N, 4, 4) w2c matrix
    # ------------------------------
    def _params_to_w2c_batch(self, params_batch: torch.Tensor) -> torch.Tensor:
        """
        Convert (N, 6) params [x,y,z,pitch,roll,yaw] to (N, 4, 4) w2c matrices.

        Performance: pure-torch euler→matrix replaces scipy to avoid GPU↔CPU
        round-trips. Bit-exact vs scipy in float64; ~1e-7 error in float32.
        """
        n = params_batch.shape[0]
        xyz = params_batch[:, :3]

        # Euler order: PITCH, ROLL, YAW (ZXY intrinsic)
        pry_deg = params_batch[:, 3:6]                                    # stay on GPU

        R_c2w = euler_xyz_extrinsic_to_matrix_torch(pry_deg)              # (N, 3, 3) GPU
        R_w2c = R_c2w.transpose(1, 2)
        t_w2c = -torch.bmm(R_w2c, xyz.unsqueeze(-1)).squeeze(-1)

        w2c = torch.zeros(n, 4, 4, device=self.device, dtype=torch.float32)
        w2c[:, :3, :3] = R_w2c.to(dtype=torch.float32)
        w2c[:, :3, 3] = t_w2c
        w2c[:, 3, 3] = 1.0

        return w2c

    # ------------------------------
    # Batch IoU scoring
    # ------------------------------
    def _iou_binary_batch(self, mask_gt: torch.Tensor, masks_pred: torch.Tensor) -> torch.Tensor:
        """Binary IoU (legacy path). Kept for backward-compatible subclass calls."""
        return iou_binary_batch(mask_gt, masks_pred)

    def _iou_score_batch(self, target: torch.Tensor, hyps: torch.Tensor) -> torch.Tensor:
        """Unified scoring entry point; dispatches by target.dtype.

        - target.dtype == bool / float (binary)     → iou_binary_batch
        - target.dtype == long / int (class map)    → iou_multiclass_batch;
                                                       requires renderer.num_classes

        Background uses ignore_index=0.
        """
        is_class_map = target.dtype in (torch.long, torch.int32, torch.int64, torch.int16)
        if not is_class_map:
            return iou_binary_batch(target, hyps)

        num_classes = getattr(self.renderer, 'num_classes', None)
        if num_classes is None:
            raise RuntimeError(
                "multiclass IoU requires renderer.num_classes, but the current renderer has none."
            )
        return iou_multiclass_batch(target, hyps, num_classes=num_classes, ignore_index=0)

    # ------------------------------
    # Normal cosine-similarity scoring
    # ------------------------------
    def _normal_score_batch(self, query_normals: torch.Tensor,
                             hyp_normals: torch.Tensor) -> torch.Tensor:
        """Mean cosine similarity between query and hypothesis camera-space normals.

        Args:
            query_normals : (H, W, 3) float32 — camera-space unit normals at GT pose.
                            Background pixels are zero.
            hyp_normals   : (B, H, W, 3) float32 — rendered normals at each hypothesis.
                            Background pixels are zero.

        Returns:
            scores : (B,) float — mean cosine similarity on pixels where BOTH hit mesh.
                     Higher = better alignment.
        """
        # valid mask: pixels where both query and hypothesis hit the mesh
        q_hit = (query_normals.norm(dim=-1) > 0.1)          # (H, W)
        h_hit = (hyp_normals.norm(dim=-1) > 0.1)             # (B, H, W)
        both  = q_hit.unsqueeze(0) & h_hit                   # (B, H, W)

        # dot product of unit normals → cosine similarity
        cos = (hyp_normals * query_normals.unsqueeze(0)).sum(dim=-1)  # (B, H, W)
        cos = cos.clamp(-1.0, 1.0)

        # mean over valid pixels (avoid div-by-zero)
        n_valid = both.float().sum(dim=(-1, -2)).clamp(min=1)         # (B,)
        scores  = (cos * both.float()).sum(dim=(-1, -2)) / n_valid     # (B,)
        return scores

    # ------------------------------
    # Gaussian sampling: (B*k, 6) particles near beam centers (particle filter)
    # ------------------------------
    def _sample_candidates(self,
                           center_poses: torch.Tensor,
                           yaw_std: float,
                           pitch_roll_std: float,
                           trans_std: float,
                           k: int,
                           sobol_seed: int = 0) -> torch.Tensor:
        """
        Batch Gaussian perturbations around (x,y,z) and yaw.

        Uses Sobol quasi-random sequences (deterministic particle filter) instead of
        torch.randn for fully reproducible runs and more uniform coverage.
        Maps uniform (0,1)^6 → standard normal via inverse CDF (erfinv).

        center_poses: (B, 6) tensor [x, y, z, pitch, roll, yaw]
        k: number of candidates per beam
        sobol_seed: 0 = unscrambled deterministic (default); >0 = scrambled with
                    that seed for multi-restart diversity.
        Returns: (B * k, 6) particle tensor
        """

        B = center_poses.shape[0]
        N = B * k

        # Repeat each center k times: (B, 6) -> (B, 1, 6) -> (B, k, 6) -> (N, 6)
        expanded_centers = center_poses.unsqueeze(1).repeat(1, k, 1).reshape(N, 6)

        # Sobol quasi-random samples in (0,1)^6, mapped to standard normal.
        # sobol_seed=0 → unscrambled deterministic sequence (default).
        # sobol_seed>0 → scrambled with that seed (for multi-restart diversity).
        sobol = SobolEngine(dimension=6, scramble=(sobol_seed != 0),
                            seed=sobol_seed if sobol_seed != 0 else None)
        u = sobol.draw(N).clamp(1e-6, 1.0 - 1e-6).to(self.device)  # (N, 6) uniform
        z = (2.0 * u - 1.0).erfinv() * (2.0 ** 0.5)                # (N, 6) standard normal

        # Scale each dimension by its std: [x, y, z, pitch, roll, yaw]
        stds = torch.tensor(
            [trans_std, trans_std, trans_std, pitch_roll_std, pitch_roll_std, yaw_std],
            device=self.device, dtype=torch.float32,
        )
        noise = z * stds.unsqueeze(0)  # (N, 6)

        # Apply noise
        new_poses = expanded_centers.clone() + noise

        # Wrap yaw angle
        new_poses[:, 5] = wrap_angle_deg_torch(new_poses[:, 5])

        return new_poses  # (N, 6)

    # ------------------------------
    # Weighted beam-center update
    # ------------------------------
    def _update_center(self, poses: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Batch-update B beam centers.
        poses: (N, 6) tensor, N = B * k
        scores: (N,) tensor
        Returns: (B, 6) new center poses
        """
        B = self.conf.beams
        k = self.conf.num_candidates
        T = self.conf.update_temperature

        # Reshape for per-beam operations
        poses_b = poses.reshape(B, k, 6)   # (B, k, 6)
        scores_b = scores.reshape(B, k)    # (B, k)

        # 1. Compute per-beam weights
        w = torch.softmax(scores_b / T, dim=1)  # (B, k)

        # 2. Update translation (weighted mean)
        translations = poses_b[:, :, :3]   # (B, k, 3)
        w_expanded = w.unsqueeze(-1)        # (B, k, 1)
        new_translations = (translations * w_expanded).sum(dim=1)  # (B, 3)

        # 3. Update yaw (weighted circular mean)
        yaws_deg = poses_b[:, :, 5]                     # (B, k)
        new_yaws_deg = circular_mean_deg_torch(yaws_deg, w)  # (B,)

        # 4. Pitch and roll: weighted circular mean per beam
        pitch_deg = poses_b[:, :, 3]                    # (B, k)
        new_pitch_deg = circular_mean_deg_torch(pitch_deg, w)  # (B,)
        roll_deg = poses_b[:, :, 4]                     # (B, k)
        new_roll_deg = circular_mean_deg_torch(roll_deg, w)    # (B,)

        # 5. Assemble new centers
        new_centers = torch.stack([
            new_translations[:, 0], new_translations[:, 1], new_translations[:, 2],
            new_pitch_deg,
            new_roll_deg,
            new_yaws_deg
        ], dim=1)  # (B, 6)

        return new_centers
