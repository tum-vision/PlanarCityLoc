#!/usr/bin/env python3
"""
aggregate_results.py
====================
Parse pipeline.py log files and aggregate per-frame TE/RE metrics
across all sequences to produce a paper-ready summary table.

Usage:
    PYTHONPATH=. python utils/aggregate_results.py
"""
import re
import os
import sys
import json
import glob
import numpy as np


def parse_pipeline_log(log_path):
    """
    Parse pipeline.py log. Returns dict: {seq_name: {'te': [...], 're': [...]}}
    Handles both per-seq and single-seq logs.
    """
    results = {}
    try:
        txt = open(log_path).read()
    except FileNotFoundError:
        return results

    current_seq = None
    te_list, re_list = [], []
    in_table = False

    for line in txt.split('\n'):
        # Detect sequence name from "Processing <seq>..." line
        m = re.match(r'^\s*Processing\s+(\S+)\.\.\.', line)
        if m:
            if current_seq is not None and te_list:
                results.setdefault(current_seq, {'te': [], 're': []})
                results[current_seq]['te'] += te_list
                results[current_seq]['re'] += re_list
            current_seq = m.group(1)
            te_list, re_list = [], []
            in_table = False
            continue

        # Summary table delimiter
        if '===================' in line:
            in_table = not in_table
            continue

        # Parse table rows: "    frame_name   te   re"
        parts = line.split()
        if len(parts) == 3:
            try:
                te = float(parts[1])
                re_val = float(parts[2])
                if current_seq is None:
                    current_seq = "all"
                te_list.append(te)
                re_list.append(re_val)
            except ValueError:
                pass

    # Save last sequence
    if current_seq is not None and te_list:
        results.setdefault(current_seq, {'te': [], 're': []})
        results[current_seq]['te'] += te_list
        results[current_seq]['re'] += re_list

    return results


def parse_matching_metrics_json(metrics_path):
    """Parse a pipeline_matching.py metrics.json into {seq: {te, re}} format."""
    try:
        data = json.load(open(metrics_path))
    except Exception:
        return {}

    te_list = [m.get('pnp_best_overall_t_err', float('nan')) for m in data]
    re_list = [m.get('pnp_best_overall_r_err', float('nan')) for m in data]
    te_list = [x for x in te_list if not np.isnan(x)]
    re_list = [x for x in re_list if not np.isnan(x)]

    # Extract seq from frame names
    seq_results = {}
    for m in data:
        fname = m.get('frame', '')
        # Sequence prefix before the last underscore in the frame name.
        parts = fname.replace('.jpg', '').rsplit('_', 1)
        seq = parts[0] if len(parts) == 2 else 'unknown'
        te = m.get('pnp_best_overall_t_err', float('nan'))
        re_v = m.get('pnp_best_overall_r_err', float('nan'))
        if not np.isnan(te):
            seq_results.setdefault(seq, {'te': [], 're': []})
            seq_results[seq]['te'].append(te)
            seq_results[seq]['re'].append(re_v)
    return seq_results


def recall_pct(values, thresh):
    arr = [x for x in values if not np.isnan(x)]
    if not arr:
        return float('nan')
    return np.mean(np.array(arr) <= thresh) * 100


def print_summary(all_results, label, header=True):
    """Print a formatted summary table."""
    if header:
        print(f"\n{'='*90}")
        print(f"  {label}")
        print(f"{'='*90}")
        print(f"  {'Sequence':<35} {'TE(m)':>7} {'RE(°)':>7} {'R@1m':>6} {'R@2m':>6} {'R@3m':>6} {'R@5m':>6} {'N':>5}")
        print(f"  {'-'*83}")

    all_te, all_re = [], []
    for seq, v in sorted(all_results.items()):
        te = np.median(v['te']) if v['te'] else float('nan')
        re = np.median(v['re']) if v['re'] else float('nan')
        r1 = recall_pct(v['te'], 1.0)
        r2 = recall_pct(v['te'], 2.0)
        r3 = recall_pct(v['te'], 3.0)
        r5 = recall_pct(v['te'], 5.0)
        n = len(v['te'])
        print(f"  {seq:<35} {te:7.2f} {re:7.2f} {r1:5.1f}% {r2:5.1f}% {r3:5.1f}% {r5:5.1f}% {n:>5}")
        all_te += v['te']
        all_re += v['re']

    if all_te:
        te_m = np.median(all_te)
        re_m = np.median(all_re)
        r1 = recall_pct(all_te, 1.0)
        r2 = recall_pct(all_te, 2.0)
        r3 = recall_pct(all_te, 3.0)
        r5 = recall_pct(all_te, 5.0)
        n = len(all_te)
        print(f"  {'-'*83}")
        print(f"  {'OVERALL':<35} {te_m:7.2f} {re_m:7.2f} {r1:5.1f}% {r2:5.1f}% {r3:5.1f}% {r5:5.1f}% {n:>5}")

    return all_te, all_re


def _latest(pattern: str) -> str:
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else pattern


def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs = os.path.join(ROOT, 'slurm', 'logs')

    configs = [
        {
            'label': 'UAVD4L inTraj LoD3',
            'type': 'pipeline_log',
            'log': _latest(os.path.join(logs, 'eval_full_intraj_*.log')),
        },
        {
            'label': 'UAVD4L outTraj LoD3',
            'type': 'pipeline_log',
            'log': _latest(os.path.join(logs, 'eval_full_outtraj_*.log')),
        },
        {
            'label': 'Swiss-EPFL inPlace LoD2',
            'type': 'pipeline_log',
            'log': _latest(os.path.join(logs, 'eval_full_swiss_inplace_*.log')),
        },
        {
            'label': 'Swiss-EPFL outPlace LoD2',
            'type': 'pipeline_log',
            'log': _latest(os.path.join(logs, 'eval_full_swiss_outplace_*.log')),
        },
    ]

    for cfg in configs:
        t = cfg['type']
        if t == 'pipeline_log':
            results = parse_pipeline_log(cfg['log'])
            if not results:
                print(f"\n[MISSING] {cfg['label']}  ({cfg['log']})")
                continue
        elif t == 'matching_json':
            jpath = cfg['json']
            if not os.path.exists(jpath):
                print(f"\n[MISSING] {cfg['label']}  ({jpath})")
                continue
            data = json.load(open(jpath))
            te = [m.get('pnp_best_overall_t_err', float('nan')) for m in data]
            re = [m.get('pnp_best_overall_r_err', float('nan')) for m in data]
            results = {'all': {'te': [x for x in te if not np.isnan(x)],
                               're': [x for x in re if not np.isnan(x)]}}
        elif t == 'matching_dir':
            d = cfg['dir']
            if not os.path.isdir(d):
                print(f"\n[MISSING] {cfg['label']}  ({d})")
                continue
            results = {}
            for mj in glob.glob(os.path.join(d, '*/metrics.json')):
                seq = os.path.basename(os.path.dirname(mj))
                r = parse_matching_metrics_json(mj)
                for k, v in r.items():
                    results.setdefault(seq, {'te': [], 're': []})
                    results[seq]['te'] += v['te']
                    results[seq]['re'] += v['re']
            if not results:
                print(f"\n[MISSING] {cfg['label']}  (no metrics.json found in {d})")
                continue
        else:
            continue

        print_summary(results, cfg['label'])


if __name__ == '__main__':
    main()
