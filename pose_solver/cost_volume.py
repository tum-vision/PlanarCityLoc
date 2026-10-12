"""Cost-volume pose selection: 4-DoF grid search scored by mask IoU (+ edge DT + depth)."""
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import torch

from utils.lod import LoD
from utils.metrics import mask_overlap_batch
from utils.pose import wrap_angle_deg_torch
from utils.pose import scale_intrinsics_torch
from .edge_scoring import score_mask_edges_vs_dt
from .fine import _BatchGPUContext
from .particle_base import PoseHypothesisBase

# Default pipeline batch size (Renderer default is 52).
DEFAULT_BATCH_SIZE = 256


@dataclass
class CostVolumeConfig:
    range_xyz_deg: Dict[str, float]
    samples_xyz_deg: Dict[str, int]
    ori_image_size: Tuple[int, int]
    mask_size: Tuple[int, int]
    batch_size: int = DEFAULT_BATCH_SIZE
    mask_iou_mode: str = "union"
    depth_gated: bool = True


class CostVolumeSelector(PoseHypothesisBase):
    def __init__(self, lod: LoD, conf: CostVolumeConfig):
        self.lod = lod
        self.conf = conf
        self.mask_iou_mode = conf.mask_iou_mode
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._depth_ctx: _BatchGPUContext | None = None

    @property
    def renderer(self):
        return self.lod.renderer

    def _iou_score_batch(self, target: torch.Tensor, hyps: torch.Tensor) -> torch.Tensor:
        is_class_map = target.dtype in (torch.long, torch.int32, torch.int64, torch.int16)
        if is_class_map:
            return super()._iou_score_batch(target, hyps)
        return mask_overlap_batch(self.mask_iou_mode, target, hyps)

    def _ensure_cv_renderer(self, K_torch: torch.Tensor) -> None:
        K_np = K_torch.detach().cpu().numpy()
        self.lod._ensure_renderer(K_np, self.conf.mask_size, device='gpu')
        self.lod.renderer.minibatch_size = self.conf.batch_size

    def _setup_depth_scorer(
        self,
        K_torch: torch.Tensor,
        mq_mask_torch: torch.Tensor,
        pred_depth: torch.Tensor,
    ) -> None:
        """Prepare affine-invariant MoGe-vs-LoD depth scoring at CV resolution."""
        H, W = self.conf.mask_size
        K_np = K_torch.detach().cpu().numpy()
        q_mask_u8 = (mq_mask_torch.detach().cpu().numpy().astype(np.uint8) * 255)
        pred_d_np = pred_depth.detach().float().cpu().numpy()
        if pred_d_np.shape != (H, W):
            raise ValueError(f"Expected pred_depth size {(H, W)}, got {pred_d_np.shape}")
        self._depth_ctx = _BatchGPUContext(
            self.lod, device=str(self.device), batch_size=self.conf.batch_size,
        )
        self._depth_ctx.setup_stage(K_np, (H, W), q_mask_u8, pred_depth=pred_d_np)

    def _combine_scores(
        self,
        iou_scores: torch.Tensor,
        edge_scores: torch.Tensor | None,
        depth_scores: torch.Tensor | None,
        *,
        iou_weight: float,
        edge_weight: float,
        depth_weight: float,
    ) -> torch.Tensor:
        appearance = iou_weight * iou_scores
        if edge_scores is not None and edge_weight > 0:
            appearance = appearance + edge_weight * edge_scores
        if depth_scores is not None and depth_weight > 0:
            if self.conf.depth_gated:
                # Multiplicative gate: high mask overlap at wrong altitude still fails depth.
                scores = appearance * (0.15 + 0.85 * depth_scores)
            else:
                scores = appearance + depth_weight * depth_scores
        else:
            scores = appearance
        return scores

    def _score_render_chunk(
        self,
        w2c_chunk: torch.Tensor,
        K_torch: torch.Tensor,
        mq_mask_torch: torch.Tensor,
        edge_dt: torch.Tensor | None,
        *,
        iou_weight: float,
        edge_weight: float,
        depth_weight: float,
        H: int,
        min_iou: float = 0.0,
        max_edt: float = float('inf'),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Render and score up to ``batch_size`` hypotheses."""
        mhyps, depth_maps = self.lod.renderer.render_masks_batch(w2c_chunk, K_torch)

        iou_scores = self._iou_score_batch(mq_mask_torch, mhyps)
        edge_scores = (
            score_mask_edges_vs_dt(mhyps, edge_dt, max_edt=max_edt)
            if edge_dt is not None else None
        )
        depth_scores = None
        if depth_weight > 0 and self._depth_ctx is not None:
            if depth_maps is None:
                depth_maps = self.lod.renderer.render_depth_batch(w2c_chunk)[:, 0]
            dh, dw = depth_maps.shape[-2], depth_maps.shape[-1]
            if (dh, dw) != (self._depth_ctx.H, self._depth_ctx.W):
                padded = torch.zeros(
                    depth_maps.shape[0], self._depth_ctx.H, self._depth_ctx.W,
                    device=depth_maps.device, dtype=depth_maps.dtype,
                )
                padded[:, :dh, :dw] = depth_maps
                depth_maps = padded
            depth_scores = self._depth_ctx._batch_depth_score(depth_maps)

        scores = self._combine_scores(
            iou_scores, edge_scores, depth_scores,
            iou_weight=iou_weight, edge_weight=edge_weight, depth_weight=depth_weight,
        )
        
        # Hard constraint: restrict candidates with worse IoU than the GPS prior
        scores = torch.where(iou_scores >= min_iou, scores, torch.zeros_like(scores))
        
        return scores, iou_scores, edge_scores

    def select(
        self,
        K: torch.Tensor,
        mq_mask: torch.Tensor,
        prior: Dict,
        edge_dt: torch.Tensor = None,
        *,
        iou_weight: float = 0.7,
        edge_weight: float = 0.3,
        depth_weight: float = 0.0,
        pred_depth: torch.Tensor = None,
        min_iou: float = 0.0,
        max_edt: float = float('inf'),
    ) -> Tuple[Dict, float]:
        """Grid-search cost-volume selection (mask IoU + optional edge DT + depth).

        When ``pred_depth`` is set and ``depth_weight > 0``, rendered LoD depth is
        compared to the MoGe prediction via per-hypothesis affine alignment (see
        ``_BatchGPUContext._batch_depth_score``).
        """
        H, W = self.conf.mask_size
        mq_mask_torch = mq_mask.to(self.device)
        if mq_mask.shape != (H, W):
            raise ValueError(f"Expected Mq size {self.conf.mask_size}, got {mq_mask.shape}")

        K_torch = scale_intrinsics_torch(
            K=K,
            old_size=self.conf.ori_image_size,
            new_size=self.conf.mask_size,
        ).to(self.device)

        self._depth_ctx = None
        if pred_depth is not None and depth_weight > 0:
            pred_depth_t = pred_depth.to(self.device).float()
            if pred_depth_t.shape != (H, W):
                raise ValueError(f"Expected pred_depth size {(H, W)}, got {pred_depth_t.shape}")
            self._setup_depth_scorer(K_torch, mq_mask_torch, pred_depth_t)

        params_batch = self.sample_grid(prior)
        if params_batch.shape[0] == 0:
            return prior, -1.0

        w2c_batch = self._params_to_w2c_batch(params_batch)
        self._ensure_cv_renderer(K_torch)

        mb = self.conf.batch_size
        best_idx = 0
        best_combined = float('-inf')
        best_iou = float('-inf')

        for s in range(0, w2c_batch.shape[0], mb):
            e = min(s + mb, w2c_batch.shape[0])
            scores, iou_scores, _ = self._score_render_chunk(
                w2c_batch[s:e], K_torch, mq_mask_torch, edge_dt,
                iou_weight=iou_weight, edge_weight=edge_weight, depth_weight=depth_weight, H=H,
                min_iou=min_iou, max_edt=max_edt)
            
            # Penalize by distance from the prior (GPS) to mitigate perceptual aliasing
            hyp_xyz = params_batch[s:e, :3]
            prior_xyz = torch.tensor([prior['x'], prior['y'], prior['z']], device=self.device)
            dist_m = torch.norm(hyp_xyz - prior_xyz, dim=-1)
            scores = scores - 0.002 * dist_m

            local_idx = int(scores.argmax().item())
            local_score = float(scores[local_idx].item())
            if local_score > best_combined:
                best_combined = local_score
                best_idx = s + local_idx
                best_iou = float(iou_scores[local_idx].item())

        best_pose = {'params': params_batch[best_idx], 'w2c': w2c_batch[best_idx]}
        return best_pose, best_combined

    def sample_grid(self, prior: Dict) -> torch.Tensor:
        axes = ['x', 'y', 'z', 'yaw_deg']
        ranges = [self.conf.range_xyz_deg[ax] for ax in axes]
        counts = [self.conf.samples_xyz_deg[ax] for ax in axes]

        grid_tensors = []
        for ax, r, n in zip(axes, ranges, counts):
            if n <= 1:
                grid_tensors.append(torch.tensor([prior[ax]], device=self.device, dtype=torch.float32))
            else:
                start = prior[ax] - r / 2.0
                end = prior[ax] + r / 2.0
                grid_tensors.append(torch.linspace(start, end, n, device=self.device, dtype=torch.float32))

        grid_x, grid_y, grid_z, grid_yaw = torch.meshgrid(*grid_tensors, indexing='ij')
        xyz_params = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], dim=-1)
        yaw_params = wrap_angle_deg_torch(grid_yaw.reshape(-1, 1))
        roll_params = torch.full_like(yaw_params, prior['roll_deg'])
        pitch_params = torch.full_like(yaw_params, prior['pitch_deg'])
        return torch.cat([xyz_params, pitch_params, roll_params, yaw_params], dim=-1)

    def select_topk(self, K: torch.Tensor, mq_mask: torch.Tensor, prior: Dict,
                    k: int = 9) -> Tuple[List[Dict], List[float]]:
        H, W = self.conf.mask_size
        mq_mask_torch = mq_mask.to(self.device)
        if mq_mask.shape != (H, W):
            raise ValueError(f"Expected Mq size {self.conf.mask_size}, got {mq_mask.shape}")

        K_torch = scale_intrinsics_torch(
            K=K,
            old_size=self.conf.ori_image_size,
            new_size=self.conf.mask_size,
        ).to(self.device)

        params_batch = self.sample_grid(prior)
        if params_batch.shape[0] == 0:
            return [prior], [-1.0]

        w2c_batch = self._params_to_w2c_batch(params_batch)
        self._ensure_cv_renderer(K_torch)
        mhyps, _ = self.lod.renderer.render_masks_batch(w2c_batch, K_torch)
        scores = self._iou_score_batch(mq_mask_torch, mhyps)

        k_actual = min(k, params_batch.shape[0])
        topk_scores, topk_indices = torch.topk(scores, k_actual)

        poses_out, scores_out = [], []
        for idx, sc in zip(topk_indices, topk_scores):
            poses_out.append({'params': params_batch[idx], 'w2c': w2c_batch[idx]})
            scores_out.append(sc.item())
        return poses_out, scores_out

    def select_normals(self, K: torch.Tensor, query_normals: torch.Tensor,
                       prior: Dict) -> Tuple[Dict, float]:
        H, W = self.conf.mask_size
        query_normals = query_normals.to(self.device)

        K_torch = scale_intrinsics_torch(
            K=K,
            old_size=self.conf.ori_image_size,
            new_size=self.conf.mask_size,
        ).to(self.device)

        params_batch = self.sample_grid(prior)
        if params_batch.shape[0] == 0:
            return prior, -1.0

        w2c_batch = self._params_to_w2c_batch(params_batch)
        self._ensure_cv_renderer(K_torch)
        hyp_normals, _ = self.lod.renderer.render_masks_batch(w2c_batch, K_torch)
        scores = self._normal_score_batch(query_normals, hyp_normals)

        best_idx = scores.argmax()
        return {'params': params_batch[best_idx], 'w2c': w2c_batch[best_idx]}, scores[best_idx].item()
