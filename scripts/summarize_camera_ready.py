#!/usr/bin/env python3
"""Verify saved poses against GT and report joint, not translation-only, recall."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.run_evidence import clean_json, sha256
from pose_solver.selection import PROTOCOL, features_for_protocol, pick_rank_aggregation, rank_scores


def pose_errors(w2c, gt_w2c):
    est, gt = np.asarray(w2c, dtype=float), np.asarray(gt_w2c, dtype=float)
    if est.shape != (4, 4) or not np.isfinite(est).all():
        return float('inf'), float('inf')
    np.testing.assert_allclose(est[3], [0, 0, 0, 1], rtol=0, atol=1e-6)
    np.testing.assert_allclose(est[:3, :3] @ est[:3, :3].T, np.eye(3), rtol=0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.det(est[:3, :3]), 1, rtol=0, atol=1e-5)
    te = np.linalg.norm(-est[:3, :3].T @ est[:3, 3] + gt[:3, :3].T @ gt[:3, 3])
    cosine = (np.trace(est[:3, :3] @ gt[:3, :3].T) - 1) / 2
    return float(te), float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1))))


def statistics(errors):
    """All requested queries form the denominator; invalid/missing are failures."""
    arr = np.asarray(errors, dtype=float).reshape(-1, 2)
    if not len(arr):
        raise ValueError('Empty evaluation')
    arr[~np.isfinite(arr)] = np.inf
    return {
        'n_queries': len(arr), 'n_failed': int((~np.isfinite(arr).all(axis=1)).sum()),
        'median_translation_m': float(np.median(arr[:, 0])),
        'median_rotation_deg': float(np.median(arr[:, 1])),
        'joint_recall_percent': {str(t): float(100 * np.mean((arr[:, 0] < t) & (arr[:, 1] < t))) for t in (1, 2, 3, 5)},
        'translation_only_recall_percent': {str(t): float(100 * np.mean(arr[:, 0] < t)) for t in (1, 2, 3, 5)},
    }


def verify_summary_equal(saved, computed, path='summary'):
    """Exact structure/hashes/counts; allow only numerical CPU roundoff in floats.

    arccos/radians conversion differs by a few ulps across cluster CPUs. The
    1e-10 absolute tolerance is far below reported precision and cannot hide a
    changed query count, source hash, or recall-threshold crossing.
    """
    if isinstance(saved, dict) and isinstance(computed, dict):
        if saved.keys() != computed.keys():
            raise ValueError(f'{path}: different fields')
        for key in saved:
            verify_summary_equal(saved[key], computed[key], path+'/'+key)
    elif isinstance(saved, list) and isinstance(computed, list):
        if len(saved) != len(computed):
            raise ValueError(f'{path}: different lengths')
        for i, (left, right) in enumerate(zip(saved, computed)):
            verify_summary_equal(left, right, path+'/'+str(i))
    elif isinstance(saved, float) and isinstance(computed, float):
        if not np.isclose(saved, computed, rtol=0, atol=1e-10):
            raise ValueError(f'{path}: numerical result changed: {saved} != {computed}')
    elif type(saved) != type(computed) or saved != computed:
        raise ValueError(f'{path}: evidence changed: {saved!r} != {computed!r}')


def save_or_verify_summary(path, result):
    """Preserve existing evidence byte-for-byte; never overwrite a summary."""
    path = Path(path)
    try:
        stream = path.open('x')
    except FileExistsError:
        verify_summary_equal(json.loads(path.read_text()), result, str(path))
        return 'Verified unchanged'
    with stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return 'Wrote verified'


def verify_candidate_selection(record, *, protocol=PROTOCOL):
    """Check the declared fixed selector, not merely final-pose consistency."""
    candidates = record['candidates']
    labels = [candidate['lbl'] for candidate in candidates]
    if not labels or labels[0] != 'Particle Filter' or len(labels) != len(set(labels)):
        raise ValueError('Missing initial PF candidate or duplicate candidate labels')
    np.testing.assert_array_equal(candidates[0]['w2c'], record['w2c_pf'])
    if pick_rank_aggregation(candidates, protocol=protocol) != record['selected']:
        raise ValueError('Saved candidate selection disagrees with fixed equal-weight ranks')
    np.testing.assert_allclose(
        [candidate['aggregate_rank'] for candidate in candidates],
        rank_scores(candidates, protocol=protocol), rtol=0, atol=1e-12,
    )


def summarize(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'run_manifest.json').read_text())
    protocol = manifest['protocol']['protocol']
    features_for_protocol(protocol)  # Fail closed rather than reinterpret unknown evidence.
    if manifest['arguments'].get('withhold_evaluation_labels'):
        raise ValueError('Do not evaluate a GT-withholding diagnostic as a benchmark')
    records = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
    candidates = [json.loads(line) for line in (folder / 'candidates.jsonl').read_text().splitlines()]
    expected = manifest['frames']
    names = [r['frame'] for r in records]
    if len(expected) != len(set(expected)) or Counter(names) != Counter(expected):
        raise ValueError(f'Incomplete/duplicate query coverage: {len(names)} records, {len(expected)} expected')
    if Counter(r['frame'] for r in candidates) != Counter(expected):
        raise ValueError('Candidate coverage differs from metrics')
    by_name = {r['frame']: r for r in candidates}
    args = manifest['arguments']
    if args['dataset'] == 'uavd4l':
        from datasets.uavd4l import UAVD4LDataset
        data = UAVD4LDataset(args['data_root'], render_size=(480, 640))
    else:
        from datasets.swiss_epfl import SwissEPFLDataset
        data = SwissEPFLDataset(args['data_root'], render_size=(480, 640), split=args['split'])
    stages = {'cv': [], 'pf': [], 'final': []}
    per_scene = {}
    per_frame = []
    for row in records:
        frame = row['frame']
        poses = by_name[frame]
        verify_candidate_selection(poses, protocol=protocol)
        errors = {}
        for stage, pose, prefix in (
            ('cv', poses['w2c_cv'], 'cv_select'), ('pf', poses['w2c_pf'], 'pf'),
            ('final', row['w2c'], 'pnp_best_overall')):
            pair = pose_errors(pose, data.gt_poses[frame])
            # A small tolerance accommodates acos near zero, not rounded logs.
            np.testing.assert_allclose(pair, [row[prefix+'_t_err'], row[prefix+'_r_err']], rtol=0, atol=1e-5)
            stages[stage].append(pair)
            errors[stage] = pair
        selected = next(c for c in poses['candidates'] if c['lbl'] == poses['selected'])
        np.testing.assert_array_equal(selected['w2c'], row['w2c'])
        if args['dataset'] == 'swiss_epfl':
            scene = frame.split('_', 1)[0]
            per_scene.setdefault(scene, []).append(errors['final'])
        per_frame.append({'frame': frame, 'errors': errors})
    wall = np.asarray([r['wall_inference_s'] for r in records], dtype=float)
    if not np.isfinite(wall).all() or np.any(wall <= 0):
        raise ValueError('Invalid wall-clock query timing')
    return clean_json({
        'protocol': manifest['protocol']['protocol'],
        'verified_query_coverage': True, 'verified_saved_poses_against_gt': True,
        'recall_definition': 'translation_m < threshold AND rotation_deg < threshold',
        'manifest_sha256': sha256(folder / 'run_manifest.json'),
        'metrics_sha256': sha256(folder / 'metrics.jsonl'),
        'candidates_sha256': sha256(folder / 'candidates.jsonl'),
        'stages': {s: statistics(e) for s, e in stages.items()},
        'scenes': {s: statistics(e) for s, e in per_scene.items()},
        'wall_inference_s': {'mean': float(wall.mean()), 'median': float(np.median(wall)),
                             'p95': float(np.percentile(wall, 95)), 'n': len(wall)},
        'frames': per_frame,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_folder')
    args = parser.parse_args()
    result = summarize(args.run_folder)
    destination = Path(args.run_folder, 'verified_summary.json')
    action = save_or_verify_summary(destination, result)
    print(f'{action} summary: {destination}', file=sys.stderr)
    print(json.dumps({k: v for k, v in result.items() if k != 'frames'}, indent=2))
