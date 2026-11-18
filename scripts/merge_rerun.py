"""Replace one dataset's rows in the rerun CSVs with the results of a new run.

The old rows are archived under results/archive/<archive-name>/ first.

Usage:
  python scripts/merge_rerun.py --dataset Wind-Farms --archive-name wind_farms_hourly \
      --classical results/classical_results_<ts>.csv \
      --foundation results/foundation_raw_<ts>.csv [results/foundation_raw_<ts2>.csv ...]
"""
import argparse
import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RESULTS = os.path.join(ROOT, 'results')

# rerun file -> rows of a foundation_raw file that belong in it
FM_FILES = {
    'foundation_rerun_zeroshot.csv': lambda d: (d.setting == 'zero-shot') & (d.model != 'Lag-Llama'),
    'foundation_rerun_lagllama.csv': lambda d: (d.setting == 'zero-shot') & (d.model == 'Lag-Llama'),
    'foundation_rerun_lora.csv': lambda d: d.setting == 'fine-tuned-lora',
    'foundation_rerun_fullft.csv': lambda d: d.setting == 'fine-tuned-full',
}


def replace(target, new, dataset, archive_dir):
    path = os.path.join(RESULTS, target)
    old = pd.read_csv(path)
    new = new[new.dataset == dataset].reindex(columns=old.columns)
    if new.empty:
        print(f'  {target}: no new rows for {dataset}, left unchanged')
        return
    n_old = int((old.dataset == dataset).sum())
    if n_old != len(new):
        print(f'  WARNING {target}: replacing {n_old} rows with {len(new)}')
    old[old.dataset == dataset].to_csv(os.path.join(archive_dir, target), index=False)
    out = pd.concat([old[old.dataset != dataset], new], ignore_index=True)
    out.to_csv(path, index=False)
    print(f'  {target}: replaced {n_old} rows with {len(new)}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--archive-name', required=True)
    p.add_argument('--classical')
    p.add_argument('--foundation', nargs='*', default=[])
    args = p.parse_args()

    archive_dir = os.path.join(RESULTS, 'archive', args.archive_name)
    os.makedirs(archive_dir, exist_ok=True)
    if args.classical:
        replace('classical_rerun.csv', pd.read_csv(args.classical), args.dataset, archive_dir)
    if args.foundation:
        fm = pd.concat([pd.read_csv(f) for f in args.foundation], ignore_index=True)
        for target, select in FM_FILES.items():
            replace(target, fm[select(fm)], args.dataset, archive_dir)


if __name__ == '__main__':
    main()
