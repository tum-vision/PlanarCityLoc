#!/usr/bin/env python3
"""Compare paired runs with/without evaluation labels, independently of accuracy."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(first, second):
    def read(folder):
        return [json.loads(line) for line in Path(folder, 'candidates.jsonl').read_text().splitlines()]
    a, b = read(first), read(second)
    if not a or len(a) != len(b):
        raise AssertionError('Missing or unequal number of completed queries')
    for left, right in zip(a, b):
        for key in ('frame', 'selected', 'gate'):
            if left[key] != right[key]:
                raise AssertionError(f'{left["frame"]}: changed {key}')
        for key in ('w2c_cv', 'w2c_pf'):
            np.testing.assert_allclose(left[key], right[key], rtol=0, atol=1e-7)
        if len(left['candidates']) != len(right['candidates']):
            raise AssertionError(f'{left["frame"]}: candidate count changed')
        for x, y in zip(left['candidates'], right['candidates']):
            if x['lbl'] != y['lbl']:
                raise AssertionError('Candidate order changed')
            np.testing.assert_allclose(x['w2c'], y['w2c'], rtol=0, atol=1e-7)
            for key in x:
                if key in ('t_err', 'r_err', 'w2c', 'lbl'):
                    continue
                if x[key] is None or y[key] is None:
                    if x[key] != y[key]:
                        raise AssertionError(f'Feature availability changed: {key}')
                else:
                    np.testing.assert_allclose(x[key], y[key], rtol=0, atol=1e-7)
    return {'passed': True, 'n_frames': len(a), 'matrix_absolute_tolerance': 1e-7,
            'compared': 'CV, PF, every PnP pose, ranking features, gate and selected label',
            'with_labels': str(first), 'without_labels': str(second)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('with_labels')
    parser.add_argument('without_labels')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = compare(args.with_labels, args.without_labels)
    with Path(args.out).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))
