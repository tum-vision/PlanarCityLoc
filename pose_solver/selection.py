"""Untuned candidate selection, independent of evaluation labels.

Each of eight features contributes equally, as in the reported evaluations.
Missing or nonfinite values receive the worst rank, and exact feature ties
receive average ranks. An aggregate tie keeps the earliest candidate
(PF first, then A1, B1, ...). No ground-truth/error fields are read.
"""

import numpy as np

SHARED_FEATURES = (
    ("score", 1), ("rec", 1), ("iou", 1), ("edge", 1),
    ("depth_ncc", 1), ("normal_err_deg", -1),
)
FEATURES = SHARED_FEATURES + (
    ("reproj", -1), ("centroid_dist_px", -1),
)
PROTOCOL = "camera-ready-v2-equal-rank-estimated-plane-gate-no-cross-render-index-retention"
# Preserve identifiers when verifying existing run manifests. The shared-only
# variant is supported for reading saved runs, not used by pipeline defaults.
SHARED_PROTOCOL = "camera-ready-v3-shared-six-rank-estimated-plane-gate-no-cross-render-index-retention"
PROTOCOL_FILES = {
    PROTOCOL: "configs/evaluation.json",
}


def features_for_protocol(protocol):
    if protocol == PROTOCOL:
        return FEATURES
    if protocol == SHARED_PROTOCOL:
        return SHARED_FEATURES
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
