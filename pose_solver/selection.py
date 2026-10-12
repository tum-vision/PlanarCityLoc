"""Untuned candidate selection, independent of evaluation labels.

Only the six render-based features evaluated for every candidate contribute.
PnP-fit residuals are diagnostics, not comparable scores for the unfitted PF
candidate. Missing/nonfinite render-based features still receive the worst
rank (e.g. insufficient visible overlap); exact ties receive average ranks.
An aggregate tie keeps the earliest candidate (PF first, then A1, B1, ...).
No ground-truth/error fields are read. Frozen v2 selection remains available
only by explicit protocol for saved-candidate evaluation.
"""

import numpy as np

FEATURES = (
    ("score", 1), ("rec", 1), ("iou", 1), ("edge", 1),
    ("depth_ncc", 1), ("normal_err_deg", -1),
)
LEGACY_FEATURES = FEATURES + (
    ("reproj", -1), ("centroid_dist_px", -1),
)
LEGACY_PROTOCOL = "camera-ready-v2-equal-rank-estimated-plane-gate-no-cross-render-index-retention"
PROTOCOL = "camera-ready-v3-shared-six-rank-estimated-plane-gate-no-cross-render-index-retention"
PROTOCOL_FILES = {
    LEGACY_PROTOCOL: "configs/evaluation_camera_ready_v2.json",
    PROTOCOL: "configs/evaluation_camera_ready_v3.json",
}


def features_for_protocol(protocol):
    if protocol == PROTOCOL:
        return FEATURES
    if protocol == LEGACY_PROTOCOL:
        return LEGACY_FEATURES
    raise ValueError(f"Unknown pose-selection protocol: {protocol!r}")


def average_ranks(values):
    """Zero-based ascending average ranks; ties do not depend on input order."""
    values = np.asarray(values, dtype=np.float64)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    means = ends - (counts + 1) / 2
    return means[inverse]


def rank_scores(rows, *, protocol=PROTOCOL):
    features = features_for_protocol(protocol)
    if not rows:
        raise ValueError("At least one pose candidate is required")
    scores = np.zeros(len(rows), dtype=np.float64)
    for key, direction in features:
        values = []
        for row in rows:
            value = row.get(key)
            values.append(direction * float(value)
                          if value is not None and np.isfinite(value) else -np.inf)
        # Ranks are integers/half-integers, so sum exactly before normalizing.
        # Adding six separately rounded 1/6 contributions can otherwise turn a
        # mathematical aggregate tie into an order-dependent winner.
        scores += average_ranks(values)
    return scores / len(features)


def pick_rank_aggregation(rows, *, protocol=PROTOCOL):
    return rows[int(np.argmax(rank_scores(rows, protocol=protocol)))]["lbl"]
