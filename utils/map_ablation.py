"""Explicit, non-mutating map variants for missing-building experiments."""
import numpy as np


def without_buildings(lod, building_ids):
    """Remove faces, including occlusion geometry, while preserving the input map.

    Masking rendered pixels would leave missing buildings acting as occluders;
    a fresh face-filtered mesh models actual absence and avoids cross-frame state.
    """
    if not building_ids:
        return lod
    from utils.lod import LoD
    original_ids = lod.get_building_ids()
    keep = ~np.isin(original_ids, building_ids)
    if keep.all() or not keep.any():
        raise ValueError('Requested building removal is absent or empties the map')
    result = LoD.from_arrays(
        lod.vertices, lod.faces[keep], labels=lod.labels[keep],
        utm_offset=lod.utm_offset,
        polygon_ids=lod.polygon_ids[keep] if lod.polygon_ids is not None else None,
        source_path=lod.path, renderer_minibatch=lod._batch_size,
    )
    result._building_ids = original_ids[keep].copy()
    return result
