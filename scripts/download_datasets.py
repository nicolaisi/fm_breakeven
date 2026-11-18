"""
Pre-fetch all datasets before running experiments.

Run this ONCE before launching run_all.sh on the new extended benchmark:

    cd ~/Development/fm_breakeven
    python scripts/download_datasets.py [--new-only] [--test]

It will download and validate all datasets, printing a summary table.
Existing cached files are skipped automatically.
"""

import sys
import os
import argparse
import time

# Make sure src/ is on the path regardless of where script is invoked
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_loader import (
    DATASETS, ORIGINAL_DATASETS, NEW_DATASETS, ALL_DATASETS,
    load_and_prepare, DATA_DIR
)


def check_all(names, verbose=True, test_fraction=0.05):
    """Try loading each dataset and report status."""
    results = []
    for name in names:
        t0 = time.time()
        try:
            d = load_and_prepare(name, train_fraction=test_fraction)
            cfg = d['config']
            elapsed = time.time() - t0
            results.append({
                'name':    name,
                'status':  'OK',
                'n_full':  d['n_train_full'],
                'horizon': cfg.horizon,
                'freq':    cfg.freq,
                'elapsed': elapsed,
                'error':   '',
            })
        except Exception as e:
            elapsed = time.time() - t0
            results.append({
                'name':    name,
                'status':  'FAIL',
                'n_full':  0,
                'horizon': 0,
                'freq':    '',
                'elapsed': elapsed,
                'error':   str(e)[:80],
            })
    return results


def print_summary(results):
    ok   = [r for r in results if r['status'] == 'OK']
    fail = [r for r in results if r['status'] == 'FAIL']

    print()
    print("=" * 75)
    print(f"{'Dataset':<22} {'Status':<6} {'n_train':>8} "
          f"{'horizon':>8} {'freq':<8} {'time(s)':>7}")
    print("-" * 75)
    for r in results:
        flag = '✓' if r['status'] == 'OK' else '✗'
        err  = f"  → {r['error']}" if r['error'] else ''
        print(f"{flag} {r['name']:<20} {r['status']:<6} "
              f"{r['n_full']:>8,} {r['horizon']:>8} "
              f"{r['freq']:<8} {r['elapsed']:>6.1f}s{err}")
    print("=" * 75)
    print(f"Passed: {len(ok)} / {len(results)}")
    if fail:
        print(f"Failed: {[r['name'] for r in fail]}")
        print()
        print("To retry failed downloads, delete the cached zip/tsf file and re-run:")
        for r in fail:
            print(f"  rm -f data/{r['name']}*")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Pre-fetch and validate all datasets for fm_breakeven"
    )
    parser.add_argument('--new-only', action='store_true',
                        help="Only download the 20 new datasets")
    parser.add_argument('--original-only', action='store_true',
                        help="Only re-validate the original 10 datasets")
    parser.add_argument('--test', action='store_true',
                        help="Just test that loading works (no full download)")
    parser.add_argument('datasets', nargs='*',
                        help="Specific datasets to check (default: all)")
    args = parser.parse_args()

    if args.datasets:
        names = args.datasets
    elif args.new_only:
        names = NEW_DATASETS
    elif args.original_only:
        names = ORIGINAL_DATASETS
    else:
        names = ALL_DATASETS

    print(f"\nfm_breakeven dataset downloader")
    print(f"Data directory: {DATA_DIR}")
    print(f"Datasets to process: {len(names)}")
    print(f"  Original: {sum(1 for n in names if n in ORIGINAL_DATASETS)}")
    print(f"  New:      {sum(1 for n in names if n not in ORIGINAL_DATASETS)}")
    print()

    if args.test:
        print("TEST MODE: using 5% fraction for speed\n")
        fraction = 0.05
    else:
        fraction = 0.10

    results = check_all(names, verbose=True, test_fraction=fraction)
    print_summary(results)

    # Exit with error code if any failed
    n_fail = sum(1 for r in results if r['status'] == 'FAIL')
    sys.exit(n_fail)


if __name__ == "__main__":
    main()
