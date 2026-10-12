"""Load and resolve YAML pipeline configs (base + dataset profile)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

import numpy as np

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "pipeline"


def _parse_pf_stages(raw_stages: list) -> list[dict]:
    """Convert YAML PF stage specs to runtime dicts (yaw in radians)."""
    stages = []
    for s in raw_stages:
        stages.append({
            "name": str(s["name"]),
            "mask_size": (int(s["mask_size"][0]), int(s["mask_size"][1])),
            "sigma_trans": float(s["sigma_trans"]),
            "sigma_yaw": np.radians(float(s["sigma_yaw_deg"])),
            "n_iter": int(s["n_iter"]),
            "n_beams": int(s["n_beams"]),
            "n_candidates": int(s["n_candidates"]),
            "gamma": float(s["gamma"]),
        })
    return stages


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def resolve_profile_name(dataset: str, data_root: str, split: str | None = None) -> str:
    """Map CLI dataset args to a profile yaml stem under ``configs/pipeline/``."""
    if dataset == "swiss_epfl":
        sp = split or "inPlace"
        return f"swiss_epfl_{sp}"
    if "outtraj" in data_root.replace("\\", "/").lower():
        return "uavd4l_outTraj"
    return "uavd4l_inTraj"


def load_yaml_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


@dataclass
class PipelineRunConfig:
    """Runtime hyper-parameters for ``pipeline.py``."""

    name: str
    profile: str
    render_h: int
    render_w: int
    cv_mask_size: tuple[int, int]
    cv_grid: dict
    cv_iou_weight: float
    cv_edge_weight: float
    cv_depth_weight: float
    cv_edge_thresh: float
    cv_xy_pad_m: float
    cv_batch_size: int
    cv_mask_iou_mode: str
    cv_depth_gated: bool
    cv_z_prescreen_min_z_range: float
    cv_z_prescreen_samples: int
    cv_expand_yaw_gps_mask_thresh: float
    cv_yaw_deg_large_expanded: float
    cv_yaw_deg_small_expanded: float
    cv_yaw_samples_large_expanded: int
    cv_yaw_samples_small_expanded: int
    gt_normal_cos: float
    gt_depth_rel: float
    pred_normal_cos: float
    pred_depth_rel: float
    min_pixels: int
    mask_thresh: float
    iou_thresh: float
    mask_iou_mode: str
    low_mask_frac: float
    n_ab_iters: int
    stop_thresh_m: float
    reproj_scale: float
    pnp_min_normal_cos: float
    pnp_ab_mode: str
    pnp_retain_inliers: bool
    pnp_strict_pred_rendered_ratio: float
    pnp_strict_min_overlap_px: int
    pnp_strict_min_vote_frac: float
    pnp_strict_min_normal_cos: float
    pnp_strict_min_matches: int
    pnp_strict_reproj_px: float
    pnp_strict_b_max_dist_px: float
    pnp_strict_b_lowe_ratio: float
    pnp_strict_max_step_m: float
    pnp_skip_below_mask_frac: float
    pf_iou_weight: float
    pf_edge_weight: float
    pf_depth_weight: float
    pf_low_mask_max_drift_m: float
    pf_seed: int
    pf_stages: list
    select_edt_eps_px: float
    select_reproj_eps_px: float
    select_reproj_abs_max_px: float
    select_b_top_iou_margin: float
    select_iou_margin: float
    select_b_score_uptick: float
    select_b_score_cliff: float
    select_b_score_flat: float
    select_b_reproj_plateau_px: float
    select_pnp_trust_margin: float
    select_a_over_b_margin: float
    select_cdist_lambda: float
    select_cdist_missing_penalty: float
    select_tie_eps: float
    moge_mode: str
    moge_ckpt: str
    moge_sam3_ckpt: str
    moge_vits_model: str
    moge_vits_revision: str
    moge_enc_res: int
    mesh_uavd4l: str
    mesh_swiss_epfl: str
    crop_margin: float
    crop_extend_m: float

    @classmethod
    def from_merged_dict(cls, raw: dict) -> PipelineRunConfig:
        cv = raw["cv"]
        seg = raw["segmentation"]
        pnp = raw["pnp"]
        pf = raw["particle_filter"]
        sel = raw["pose_selection"]
        moge = raw["moge"]
        mesh = raw["mesh"]
        crop = raw["crop"]
        render = raw["render"]
        mask_h, mask_w = cv["mask_size"]
        return cls(
            name=raw.get("name", "unnamed"),
            profile=raw.get("profile", "default"),
            render_h=int(render["h"]),
            render_w=int(render["w"]),
            cv_mask_size=(int(mask_h), int(mask_w)),
            cv_grid=raw["cv_grid"],
            cv_iou_weight=float(cv["iou_weight"]),
            cv_edge_weight=float(cv["edge_weight"]),
            cv_depth_weight=float(cv["depth_weight"]),
            cv_edge_thresh=float(cv["edge_thresh"]),
            cv_xy_pad_m=float(cv["xy_pad_m"]),
            cv_batch_size=int(cv["batch_size"]),
            cv_mask_iou_mode=str(cv.get("mask_iou_mode", "union")),
            cv_depth_gated=bool(cv.get("depth_gated", True)),
            cv_z_prescreen_min_z_range=float(cv.get("z_prescreen_min_z_range", 40.0)),
            cv_z_prescreen_samples=int(cv.get("z_prescreen_samples", 15)),
            cv_expand_yaw_gps_mask_thresh=float(cv["expand_yaw_gps_mask_thresh"]),
            cv_yaw_deg_large_expanded=float(cv["yaw_deg_large_expanded"]),
            cv_yaw_deg_small_expanded=float(cv["yaw_deg_small_expanded"]),
            cv_yaw_samples_large_expanded=int(cv["yaw_samples_large_expanded"]),
            cv_yaw_samples_small_expanded=int(cv["yaw_samples_small_expanded"]),
            gt_normal_cos=float(seg["gt_normal_cos"]),
            gt_depth_rel=float(seg["gt_depth_rel"]),
            pred_normal_cos=float(seg["pred_normal_cos"]),
            pred_depth_rel=float(seg["pred_depth_rel"]),
            min_pixels=int(seg["min_pixels"]),
            mask_thresh=float(seg["mask_thresh"]),
            iou_thresh=float(seg["iou_thresh"]),
            mask_iou_mode=str(seg.get("mask_iou_mode", "union")),
            low_mask_frac=float(seg.get("low_mask_frac", 0.08)),
            n_ab_iters=int(pnp["n_ab_iters"]),
            stop_thresh_m=float(pnp["stop_thresh_m"]),
            reproj_scale=float(pnp["reproj_scale"]),
            pnp_min_normal_cos=float(pnp["min_normal_cos"]),
            pnp_ab_mode=str(pnp.get("ab_mode", "both")),
            pnp_retain_inliers=bool(pnp.get("retain_inliers", False)),
            pnp_strict_pred_rendered_ratio=float(pnp["strict_pred_rendered_ratio"]),
            pnp_strict_min_overlap_px=int(pnp["strict_min_overlap_px"]),
            pnp_strict_min_vote_frac=float(pnp["strict_min_vote_frac"]),
            pnp_strict_min_normal_cos=float(pnp["strict_min_normal_cos"]),
            pnp_strict_min_matches=int(pnp["strict_min_matches"]),
            pnp_strict_reproj_px=float(pnp["strict_reproj_px"]),
            pnp_strict_b_max_dist_px=float(pnp["strict_b_max_dist_px"]),
            pnp_strict_b_lowe_ratio=float(pnp["strict_b_lowe_ratio"]),
            pnp_strict_max_step_m=float(pnp["strict_max_step_m"]),
            pnp_skip_below_mask_frac=float(pnp.get("skip_below_mask_frac", 0.08)),
            pf_iou_weight=float(pf["iou_weight"]),
            pf_edge_weight=float(pf["edge_weight"]),
            pf_depth_weight=float(pf["depth_weight"]),
            pf_low_mask_max_drift_m=float(pf.get("low_mask_max_drift_m", 2.0)),
            pf_seed=int(pf["seed"]),
            pf_stages=_parse_pf_stages(pf["stages"]),
            select_edt_eps_px=float(sel["edt_eps_px"]),
            select_reproj_eps_px=float(sel["reproj_eps_px"]),
            select_reproj_abs_max_px=float(sel["reproj_abs_max_px"]),
            select_b_top_iou_margin=float(sel["b_top_iou_margin"]),
            select_iou_margin=float(sel["iou_margin"]),
            select_b_score_uptick=float(sel.get("b_score_uptick", 0.001)),
            select_b_score_cliff=float(sel.get("b_score_cliff", 0.01)),
            select_b_score_flat=float(sel.get("b_score_flat", 0.005)),
            select_b_reproj_plateau_px=float(sel.get("b_reproj_plateau_px", 0.2)),
            select_pnp_trust_margin=float(sel.get("pnp_trust_margin", 0.01)),
            select_a_over_b_margin=float(sel.get("a_over_b_margin", 0.01)),
            select_cdist_lambda=float(sel.get("cdist_lambda", 0.05)),
            select_cdist_missing_penalty=float(sel.get("cdist_missing_penalty", 1.0)),
            select_tie_eps=float(sel.get("tie_eps", 0.007)),
            moge_mode=str(moge.get("mode", "sam3_moge2")),
            moge_ckpt=str(moge.get("ckpt", "")),
            moge_sam3_ckpt=str(moge.get("sam3_ckpt", "")),
            moge_vits_model=str(moge.get("moge_vits_model", "")),
            moge_vits_revision=str(moge.get("moge_vits_revision", "")),
            moge_enc_res=int(moge["enc_res"]),
            mesh_uavd4l=str(mesh["uavd4l"]),
            mesh_swiss_epfl=str(mesh["swiss_epfl"]),
            crop_margin=float(crop["margin"]),
            crop_extend_m=float(crop["extend_m"]),
        )

    @classmethod
    def load(cls, profile: str | Path, overrides: list[Path] | None = None, config_dir: Path | None = None) -> PipelineRunConfig:
        config_dir = config_dir or CONFIG_DIR
        base = load_yaml_config(config_dir / "base.yaml")
        profile_path = Path(profile)
        if not profile_path.suffix:
            profile_path = config_dir / f"{profile}.yaml"
        elif not profile_path.is_absolute():
            profile_path = config_dir / profile_path
        profile_cfg = load_yaml_config(profile_path)
        merged = _deep_merge(base, profile_cfg)
        
        if overrides:
            for override_path in overrides:
                if not override_path.is_absolute() and config_dir:
                    # try resolving relative to config_dir first, or keep as is
                    if (config_dir / override_path).exists():
                        override_path = config_dir / override_path
                override_cfg = load_yaml_config(override_path)
                merged = _deep_merge(merged, override_cfg)
                
        return cls.from_merged_dict(merged)

    @classmethod
    def resolve(
        cls,
        dataset: str,
        data_root: str,
        split: str | None = None,
        config_path: str | Path | None = None,
        profile_name: str | None = None,
        config_dir: Path | None = None,
    ) -> PipelineRunConfig:
        profile = profile_name or resolve_profile_name(dataset, data_root, split=split)
        overrides = [Path(config_path)] if config_path is not None else None
        return cls.load(profile, overrides=overrides, config_dir=config_dir)

    def mesh_path_for(self, dataset: str, mesh_override: str | None = None) -> str:
        if mesh_override:
            return mesh_override
        if dataset == "swiss_epfl":
            return self.mesh_swiss_epfl
        return self.mesh_uavd4l

    def cv_grid_n_samples(self, stage: str = "large") -> int:
        block = self.cv_grid[stage]
        n = 1
        for ax in ("x", "y", "z", "yaw_deg"):
            n *= int(block["samples_xyz_deg"][ax])
        return n
