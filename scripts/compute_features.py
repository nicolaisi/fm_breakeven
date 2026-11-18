#!/usr/bin/env python3
"""
compute_features.py — Extract dataset features for all 30 benchmark datasets.

Features extracted:
  tau_acf      : lag where |ACF| < 0.5 (temporal memory)
  hurst        : Hurst exponent via R/S analysis
  noise_ratio  : std(diff(x)) / std(x)
  seas_strength: seasonal strength from STL
  spec_entropy : spectral entropy of normalised periodogram
  n_train      : training set size

Usage:
    python scripts/compute_features.py
    python scripts/compute_features.py --datasets ETTh1 ETTh2
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ── Feature functions ──────────────────────────────────────────────────────

def acf_decay_lag(y, threshold=0.5, max_lag=200):
    n = len(y)
    y = np.asarray(y, dtype=float)
    y = y - np.mean(y)
    acf = np.correlate(y, y, mode='full')[n-1:]
    if acf[0] == 0: return max_lag
    acf = acf / acf[0]
    for lag in range(1, min(max_lag, n)):
        if abs(acf[lag]) < threshold:
            return lag
    return max_lag

def hurst_exponent(y, max_lag=100):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 20: return 0.5
    lags = range(2, min(max_lag, n // 2))
    rs_vals = []
    for lag in lags:
        segments = [y[i:i+lag] for i in range(0, n - lag, lag)]
        if not segments: continue
        rs_seg = []
        for seg in segments:
            mean = np.mean(seg)
            devs = np.cumsum(seg - mean)
            r = devs.max() - devs.min()
            s = np.std(seg, ddof=1)
            if s > 0: rs_seg.append(r / s)
        if rs_seg: rs_vals.append(np.mean(rs_seg))
    if len(rs_vals) < 4: return 0.5
    log_lags = np.log(list(lags)[:len(rs_vals)])
    log_rs   = np.log(rs_vals)
    try:
        h = np.polyfit(log_lags, log_rs, 1)[0]
        return float(np.clip(h, 0, 1))
    except:
        return 0.5

def noise_ratio(y):
    y = np.asarray(y, dtype=float)
    s = np.std(y)
    return float(np.std(np.diff(y)) / s) if s > 0 else 0.0

def seasonal_strength(y, period):
    try:
        from statsmodels.tsa.seasonal import STL
        y = np.asarray(y, dtype=float)
        if len(y) < 2 * period + 1 or period < 2: return 0.0
        res = STL(y, period=period, robust=True).fit()
        var_resid  = np.var(res.resid)
        var_season = np.var(y - res.trend)
        if var_season < 1e-10: return 0.0
        return float(max(0, 1 - var_resid / var_season))
    except:
        return 0.0

def spectral_entropy(y):
    y = np.asarray(y, dtype=float)
    n = len(y)
    ps = np.abs(np.fft.rfft(y - np.mean(y)))**2
    ps_sum = ps.sum()
    if ps_sum < 1e-10: return 0.0
    ps_norm = ps / ps_sum
    ps_norm = ps_norm[ps_norm > 0]
    max_ent = np.log(len(ps_norm)) if len(ps_norm) > 1 else 1.0
    if max_ent < 1e-10: return 0.0
    return float(-np.sum(ps_norm * np.log(ps_norm)) / max_ent)

def acf_mean_abs(y, max_lag=50):
    """Mean |ACF| over lags 1..max_lag. Does not saturate like tau_acf."""
    y = np.asarray(y, dtype=float) - np.mean(y)
    n = len(y)
    acf = np.correlate(y, y, mode='full')[n-1:]
    if acf[0] == 0: return 0.0
    acf = acf / acf[0]
    lags = min(max_lag, n - 1)
    return float(np.mean(np.abs(acf[1:lags+1])))

def dfa_hurst(y):
    """Detrended Fluctuation Analysis — robust Hurst estimate, does not saturate at 1.0."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 20: return 0.5
    y_cum = np.cumsum(y - np.mean(y))
    scales = np.unique(np.logspace(np.log10(4), np.log10(max(8, n // 4)), 20).astype(int))
    scales = scales[scales >= 4]
    flucts, valid = [], []
    for s in scales:
        n_segs = n // s
        if n_segs < 2: continue
        rms = []
        for k in range(n_segs):
            seg = y_cum[k*s:(k+1)*s]
            x = np.arange(s, dtype=float)
            p = np.polyfit(x, seg, 1)
            rms.append(np.sqrt(np.mean((seg - np.polyval(p, x))**2)))
        flucts.append(np.mean(rms)); valid.append(s)
    if len(valid) < 4: return 0.5
    try:
        h = np.polyfit(np.log(valid), np.log(flucts), 1)[0]
        return float(np.clip(h, 0.0, 1.5))
    except:
        return 0.5

def periodogram_sharpness(y, period):
    """Peak power / mean power near the dominant frequency. High = sharp periodic signal."""
    y = np.asarray(y, dtype=float) - np.mean(y)
    n = len(y)
    ps = np.abs(np.fft.rfft(y))**2
    mean_power = ps.mean()
    if mean_power < 1e-10: return 0.0
    if period > 1 and n > period * 2:
        freqs = np.fft.rfftfreq(n)
        target = 1.0 / period
        width = target * 0.15 + 1.0 / n
        mask = np.abs(freqs - target) <= width
        peak_power = ps[mask].max() if mask.any() else ps.max()
    else:
        peak_power = ps.max()
    return float(peak_power / mean_power)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--results-dir', default='results')
    args = parser.parse_args()

    from data_loader import load_and_prepare, ALL_DATASETS, DATASETS as DATASET_CONFIGS

    datasets = args.datasets or ALL_DATASETS
    rows = []

    for ds_name in datasets:
        try:
            config = DATASET_CONFIGS[ds_name]
            result = load_and_prepare(ds_name)
            y_train = result['y_train']
            y = y_train.astype(float)
            n = len(y)
            sp = config.seasonal_period

            tau   = acf_decay_lag(y)
            H     = hurst_exponent(y)
            nr    = noise_ratio(y)
            S     = seasonal_strength(y, sp)
            phi   = spectral_entropy(y)
            acf_m = acf_mean_abs(y)
            dfa_h = dfa_hurst(y)
            sharp = periodogram_sharpness(y, sp)

            print(f"  {ds_name:<22} n={n:6,}  tau={tau:3d}  H={H:.2f}  "
                  f"noise={nr:.2f}  seas={S:.2f}  ent={phi:.2f}  "
                  f"acf_m={acf_m:.2f}  dfa_h={dfa_h:.2f}  sharp={sharp:.1f}")

            rows.append({'dataset': ds_name, 'tau_acf': tau, 'hurst': H,
                         'noise_ratio': nr, 'seas_strength': S,
                         'spec_entropy': phi, 'n_train': n,
                         'acf_mean_50': acf_m, 'dfa_hurst': dfa_h,
                         'periodogram_sharpness': sharp})
        except Exception as e:
            print(f"  {ds_name:<22} ERROR: {e}")

    df = pd.DataFrame(rows)
    out = os.path.join(args.results_dir, 'dataset_features.csv')
    if args.datasets and os.path.exists(out):
        # update only the requested datasets, keep the other rows
        old = pd.read_csv(out)
        df = pd.concat([old[~old.dataset.isin(df.dataset)], df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} datasets to {out}")
    print(df[['dataset','tau_acf','hurst','noise_ratio','seas_strength','spec_entropy','n_train']].to_string(index=False))

if __name__ == '__main__':
    main()
