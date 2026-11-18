#!/usr/bin/env python3
"""
smoke_test.py — Quick sanity check for the extended 30-dataset benchmark.

Run from project root:
    python scripts/smoke_test.py              # test all 30 datasets
    python scripts/smoke_test.py --new-only   # test only the 20 new datasets
    python scripts/smoke_test.py --fast       # 1 fraction, 1 seed, original only

Exit code 0 = all passed. Non-zero = some failed (see output for details).
"""

import os
import sys
import argparse
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import (
    load_and_prepare, AVAILABLE_DATASETS, ORIGINAL_DATASETS, NEW_DATASETS
)

def smoke_one(name: str, fraction: float = 0.10) -> dict:
    t0 = time.time()
    try:
        d = load_and_prepare(name, train_fraction=fraction)
        cfg = d['config']
        n   = d['n_train']
        # Basic sanity checks
        assert n > 0,                        f"n_train=0"
        assert len(d['y_test']) > 0,         f"y_test empty"
        assert 0 < len(d['y_test']) <= cfg.horizon, \
            f"y_test len={{len(d['y_test'])}} out of range (0, {{cfg.horizon}}]"
        import numpy as np
        assert np.all(np.isfinite(d['y_train'])), "NaN/Inf in y_train"
        assert np.all(np.isfinite(d['y_test'])),  "NaN/Inf in y_test"
        return {
            'name':    name,
            'status':  'OK',
            'n_train': n,
            'n_full':  d['n_train_full'],
            'horizon': cfg.horizon,
            'freq':    cfg.freq,
            'elapsed': time.time() - t0,
            'error':   '',
        }
    except Exception as e:
        return {
            'name':    name,
            'status':  'FAIL',
            'n_train': 0,
            'n_full':  0,
            'horizon': 0,
            'freq':    '',
            'elapsed': time.time() - t0,
            'error':   str(e),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--new-only',   action='store_true')
    parser.add_argument('--fast',       action='store_true',
                        help='Original datasets only, frac=0.05')
    parser.add_argument('--fraction',   type=float, default=0.10)
    parser.add_argument('datasets',     nargs='*',
                        help='Specific datasets (default: all)')
    args = parser.parse_args()

    if args.datasets:
        names = args.datasets
    elif args.fast:
        names = ORIGINAL_DATASETS
    elif args.new_only:
        names = NEW_DATASETS
    else:
        names = AVAILABLE_DATASETS

    frac = 0.05 if args.fast else args.fraction

    print(f"\nSmoke test — {len(names)} datasets @ frac={frac}")
    print(f"Data dir: {os.environ.get('FM_DATA_DIR', 'src/../data')}\n")

    results = []
    for name in names:
        r = smoke_one(name, frac)
        results.append(r)
        flag = '✓' if r['status'] == 'OK' else '✗'
        err  = f"  → {r['error']}" if r['error'] else ''
        print(f"  {flag} {r['name']:<22}  n_train={r['n_train']:>6,}  "
              f"n_full={r['n_full']:>7,}  horizon={r['horizon']:>4}  "
              f"freq={r['freq']:<7}  {r['elapsed']:.1f}s{err}")

    ok   = [r for r in results if r['status'] == 'OK']
    fail = [r for r in results if r['status'] == 'FAIL']

    print(f"\n{'='*60}")
    print(f"Passed: {len(ok)}/{len(results)}")
    if fail:
        print(f"Failed: {[r['name'] for r in fail]}")
        print()
        print("To fix failed datasets, check that data files are present in data/")
        print("Re-run: ./run_all.sh download")
        sys.exit(len(fail))
    else:
        print("All OK — ready to run experiments.")
        sys.exit(0)


if __name__ == "__main__":
    main()
