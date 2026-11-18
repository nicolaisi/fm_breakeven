"""Load the experiment results used by the figures and the summary.

Classical results: results/classical_rerun.csv
FM results:        every results/foundation_rerun_*.csv (zero-shot, Lag-Llama, LoRA, ...)

These are explicit files (copied/merged from the timestamped run outputs) so that
test runs in results/ cannot leak into the analysis.

All MASE values used for break-even and plotting are on a *fixed* denominator:
the in-sample seasonal-naive MAE of the full training pool, recovered per dataset
as median(mae / mase) over classical runs at train_fraction 1.0.
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), '..')
FRACS = [0.02, 0.05, 0.10, 0.20, 0.50, 1.00]


def load_results(results_dir=None):
    d = results_dir or os.environ.get('FM_RESULTS_DIR') or os.path.join(ROOT, 'results')
    cl = pd.read_csv(os.path.join(d, 'classical_rerun.csv'))
    fm_files = sorted(glob.glob(os.path.join(d, 'foundation_rerun_*.csv')))
    if not fm_files:
        raise FileNotFoundError(f"no foundation_rerun_*.csv in {d}")
    fm = pd.concat([pd.read_csv(f) for f in fm_files], ignore_index=True)
    fm = fm.drop_duplicates(subset=['dataset', 'model', 'setting', 'train_fraction', 'seed'])

    # Fixed denominator per dataset, applied to every model and setting
    full = cl[(cl.train_fraction == 1.0) & cl.mae.notna() & (cl.mase > 0)]
    denom = (full.mae / full.mase).groupby(full.dataset).median()
    cl['mase_fixed'] = cl.mae / cl.dataset.map(denom)
    fm['mase_fixed'] = fm.mae / fm.dataset.map(denom)
    return cl, fm


def best_zero_shot(fm, ds):
    """(model, MASE) of the best zero-shot FM on `ds`: mean over seeds per model,
    then the best model. Moirai and Lag-Llama sample their forecasts, so the
    best single seed would partly reflect luck."""
    z = fm[(fm.dataset == ds) & (fm.setting == 'zero-shot')]
    per_model = z.groupby('model').mase_fixed.mean()
    return per_model.idxmin(), per_model.min()


def detect_breakeven(cl, fm, ds):
    """(pattern, n_star, frac) using fixed-denominator MASE: the smallest training
    fraction at which any classical method matches or beats the best zero-shot FM
    (seed mean; classical runs are identical across seeds)."""
    zs = best_zero_shot(fm, ds)[1]
    c = cl[(cl.dataset == ds) & cl.mase_fixed.notna()]
    for frac in FRACS:
        sub = c[c.train_fraction == frac]
        if len(sub) and sub.mase_fixed.min() <= zs:
            pat = 'Early' if frac <= 0.05 else ('Mid' if frac <= 0.15 else 'Late')
            return pat, int(sub.n_train.median()), frac
    return 'FM-dominant', None, None
