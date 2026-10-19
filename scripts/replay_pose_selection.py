#!/usr/bin/env python3
"""Audit recorded runs and replay the current selector without changing evidence.

This is post-hoc validation on saved candidate pools, not fresh GPU inference
or an independent held-out evaluation. The report is separate from paper results.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pose_solver.selection import FEATURES, PROTOCOL, pick_rank_aggregation
from scripts.summarize_results import (
    pose_errors, statistics, summarize, verify_summary_equal,
)
from utils.run_evidence import current_protocol, sha256


def replay(folder):
    folder = Path(folder)
    files = ['run_manifest.json', 'metrics.jsonl', 'candidates.jsonl', 'verified_summary.json']
    hashes = {name: sha256(folder / name) for name in files}
    saved = json.loads((folder / 'verified_summary.json').read_text())
    audited = summarize(folder)
    verify_summary_equal(saved, audited)
    manifest = json.loads((folder / 'run_manifest.json').read_text())
    args = manifest['arguments']
    if args['dataset'] == 'uavd4l':
        from datasets.uavd4l import UAVD4LDataset
        data = UAVD4LDataset(args['data_root'], render_size=(480, 640))
    else:
        from datasets.swiss_epfl import SwissEPFLDataset
        data = SwissEPFLDataset(args['data_root'], render_size=(480, 640), split=args['split'])
    frames = {r['frame']: r for r in audited['frames']}
    errors = []
    counts = dict(changed_selection=0, retains_pf=0, joint2_gains_vs_pf=0,
                  joint2_losses_vs_pf=0, joint2_both=0, joint2_neither=0)
    selected_labels = {}
    for line in (folder / 'candidates.jsonl').read_text().splitlines():
        record = json.loads(line)
        candidates = record['candidates']
        selected = pick_rank_aggregation(candidates)
        # Selection must be identical when evaluation errors and undeclared
        # diagnostics are absent. Only the declared selection features remain.
        stripped = [{k: c[k] for k in ('lbl', *(f[0] for f in FEATURES)) if k in c}
                    for c in candidates]
        if selected != pick_rank_aggregation(stripped):
            raise ValueError('Selection depends on undeclared candidate fields')
        candidate = next(c for c in candidates if c['lbl'] == selected)
        pair = pose_errors(candidate['w2c'], data.gt_poses[record['frame']])
        np.testing.assert_allclose(pair, [candidate['t_err'], candidate['r_err']], rtol=0, atol=1e-5)
        errors.append(pair)
        selected_labels[record['frame']] = selected
        counts['changed_selection'] += selected != record['selected']
        counts['retains_pf'] += selected == 'Particle Filter'
        before = all(e < 2 for e in frames[record['frame']]['errors']['pf'])
        after = all(e < 2 for e in pair)
        key = ('joint2_both' if before and after else 'joint2_losses_vs_pf' if before
               else 'joint2_gains_vs_pf' if after else 'joint2_neither')
        counts[key] += 1
    if hashes != {name: sha256(folder / name) for name in files}:
        raise ValueError('Source evidence changed during replay')
    return {
        'source_folder': str(folder.resolve()),
        'source_protocol': manifest['protocol']['protocol'],
        'source_sha256': hashes,
        'source_saved_pose_and_selection_audit_passed': True,
        'source_evidence_unchanged': True,
        'all_dataset_queries_present': set(frames) == set(data.frames),
        'n_queries': len(errors),
        'particle_filter': audited['stages']['pf'],
        'recorded_final': audited['stages']['final'],
        'reselected_final': statistics(errors),
        'transitions': counts,
        'selected_labels': selected_labels,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_folders', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True, help='New JSON report; never overwrite')
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f'Refusing to overwrite {args.out}')
    results = [replay(folder) for folder in args.run_folders]
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'kind': 'Post-hoc saved-candidate reselection; not fresh inference or held-out validation',
        'selection_protocol': current_protocol(),
        'code_sha256': {p: sha256(ROOT / p) for p in (
            'pose_solver/selection.py', 'scripts/replay_pose_selection.py',
            'scripts/summarize_results.py', 'utils/run_evidence.py')},
        'runs': results,
        'published_results_modified': False,
    }
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for result in results:
        recalls = [result[k]['joint_recall_percent']['2'] for k in
                   ('particle_filter', 'recorded_final', 'reselected_final')]
        print(f"{result['source_folder']}: n={result['n_queries']}; "
              f"PF / recorded / {PROTOCOL.split('-')[2]}: " + ' / '.join(f'{r:.4f}%' for r in recalls))
    print(f'Saved separate replay report: {args.out}')


if __name__ == '__main__':
    main()
