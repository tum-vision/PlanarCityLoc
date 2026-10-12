"""Keep candidate scoring fixed when ablating coarse-search scoring cues."""
from dataclasses import replace

SCORE_VARIANTS = {'score_iou_only', 'score_iou_edge', 'score_iou_depth'}


def candidate_scoring_config(cfg, reference):
    if cfg.name not in SCORE_VARIANTS:
        return cfg
    return replace(cfg, **{key: getattr(reference, key) for key in
                          ('cv_iou_weight', 'cv_edge_weight', 'cv_depth_weight')})
