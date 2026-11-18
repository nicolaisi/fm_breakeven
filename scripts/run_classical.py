#!/usr/bin/env python3
"""
Run classical model experiments (CPU only).

Usage:
    python scripts/run_classical.py                    # Full run
    python scripts/run_classical.py --quick            # Quick test
    python scripts/run_classical.py --datasets ETTh1   # Specific dataset
    python scripts/run_classical.py --models Naive XGBoost  # Specific models
"""

import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_loader import load_and_prepare, AVAILABLE_DATASETS
from models import get_model, CLASSICAL_MODELS
from features import extract_all_features


# Default configuration
TRAIN_FRACTIONS = [0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
SEEDS = [0, 1, 2]


def compute_mase(y_true: np.ndarray, y_pred: np.ndarray, 
                 y_train: np.ndarray, seasonal_period: int) -> float:
    """Compute Mean Absolute Scaled Error."""
    mae = np.mean(np.abs(y_true - y_pred))
    
    if len(y_train) > seasonal_period:
        naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
    else:
        naive_errors = np.abs(np.diff(y_train))
    
    naive_mae = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
    return mae / naive_mae if naive_mae > 0 else mae


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def run_single_experiment(
    dataset_name: str,
    model_name: str,
    train_fraction: float,
    seed: int,
    xgb_features: str = "paper",
    stat_max_train: int = 200,
) -> dict:
    """Run a single experiment."""
    np.random.seed(seed)
    
    # Load data
    data = load_and_prepare(dataset_name, train_fraction=train_fraction)
    y_train = data['y_train']
    y_test = data['y_test']
    config = data['config']
    
    result = {
        'dataset': dataset_name,
        'model': model_name,
        'train_fraction': train_fraction,
        'n_train': len(y_train),
        'seed': seed,
        'mase': None,
        'mae': None,
        'n_windows': None,
        'xgb_features': xgb_features if model_name == 'XGBoost' else None,
        'stat_max_train': (stat_max_train or None) if model_name in ('ETS', 'ARIMA') else None,
        'error': None,
    }
    
    # Check minimum samples
    min_samples = max(config.seasonal_period, 15)  # lowered to handle short competition series
    if len(y_train) < min_samples:
        result['error'] = f"Insufficient data: {len(y_train)} < {min_samples}"
        return result
    
    try:
        extra_kwargs, fit_kwargs = {}, {}
        if model_name == 'XGBoost':
            extra_kwargs.update(feature_set=xgb_features, freq=config.freq)
            if xgb_features == 'paper':
                fit_kwargs['t0'] = data['train_start']   # calendar features
            elif config.freq in ('15min', '10min'):
                # legacy: the original runs used 192 lags on 10/15-min data
                extra_kwargs['n_lags'] = 192
        elif model_name in ('ETS', 'ARIMA'):
            extra_kwargs['max_train_samples'] = stat_max_train or None

        # Fit model
        model = get_model(
            model_name,
            horizon=config.horizon,
            seasonal_period=config.seasonal_period,
            **extra_kwargs
        )
        model.fit(y_train, **fit_kwargs)

        # Forecast every test window from its own origin, conditioning on
        # all history before it (training pool + buffer + earlier test points).
        maes = []
        for context, target in data['windows']:
            y_pred = model.predict_from(context)
            n = min(len(target), len(y_pred))
            maes.append(compute_mae(target[:n], y_pred[:n]))
        mae = float(np.mean(maes))

        # Per-run MASE (denominator from this run's training data). The
        # analysis scripts rescale to a fixed full-pool denominator via
        # mae / denom_full, so `mae` is what matters downstream.
        result['mase'] = compute_mase(np.zeros(1), np.full(1, mae), y_train,
                                      config.seasonal_period)
        result['mae'] = mae
        result['n_windows'] = len(maes)

    except Exception as e:
        result['error'] = str(e)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Run classical experiments")
    parser.add_argument('--datasets', nargs='+', default=None,
                        help=f'Datasets (default: all). Available: {AVAILABLE_DATASETS}')
    parser.add_argument('--models', nargs='+', default=None,
                        help=f'Models (default: all). Available: {list(CLASSICAL_MODELS.keys())}')
    parser.add_argument('--fractions', nargs='+', type=float, default=None,
                        help='Training fractions (default: 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Random seeds (default: 0, 1, 2)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 2 datasets, 2 models, 3 fractions, 1 seed')
    parser.add_argument('--xgb-features', choices=['paper', 'legacy'], default='paper',
                        help="XGBoost features: 'paper' = lags up to min(2m,48), rolling "
                             "mean/std over {m/2,m,2m}, calendar indicators (default); "
                             "'legacy' = the original 24/192 lags + one mean/std")
    parser.add_argument('--stat-max-train', type=int, default=200,
                        help='ETS/ARIMA fit on at most this many recent samples '
                             '(default 200, as in the original runs; 0 = no cap)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set configuration
    if args.quick:
        datasets = ['ETTh1', 'Exchange']
        models = ['Naive', 'XGBoost']
        fractions = [0.1, 0.5, 1.0]
        seeds = [0]
    else:
        datasets = args.datasets or AVAILABLE_DATASETS
        models = args.models or list(CLASSICAL_MODELS.keys())
        fractions = args.fractions or TRAIN_FRACTIONS
        seeds = args.seeds or SEEDS
    
    # Setup output directory
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), '..', 'results'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Print configuration
    total = len(datasets) * len(models) * len(fractions) * len(seeds)
    print("=" * 60)
    print("Classical Model Experiments")
    print("=" * 60)
    print(f"Datasets:  {datasets}")
    print(f"Models:    {models}")
    print(f"Fractions: {fractions}")
    print(f"Seeds:     {seeds}")
    print(f"XGBoost features: {args.xgb_features}")
    print(f"ETS/ARIMA max train samples: {args.stat_max_train or 'no cap'}")
    print(f"Total:     {total} experiments")
    print("=" * 60)
    
    # Run experiments
    results = []
    pbar = tqdm(total=total, desc="Running")
    
    partial_file = os.path.join(
        output_dir, f"classical_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}_partial.csv")
    for dataset in datasets:
        if results:   # checkpoint after each dataset
            pd.DataFrame(results).to_csv(partial_file, index=False)
        for model in models:
            for frac in fractions:
                for seed in seeds:
                    pbar.set_description(f"{dataset[:6]}/{model[:6]}/{frac:.0%}")
                    
                    result = run_single_experiment(
                        dataset, model, frac, seed,
                        xgb_features=args.xgb_features,
                        stat_max_train=args.stat_max_train)
                    results.append(result)
                    
                    pbar.update(1)
    
    pbar.close()
    
    # Save results
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_file = os.path.join(output_dir, f"classical_results_{timestamp}.csv")
    df.to_csv(output_file, index=False)
    if os.path.exists(partial_file):
        os.remove(partial_file)
    
    # Also save as latest
    df.to_csv(os.path.join(output_dir, "classical_results_latest.csv"), index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    valid = df[df['mase'].notna()]
    print(f"Successful: {len(valid)}/{len(df)}")
    
    if len(valid) > 0:
        full_data = valid[valid['train_fraction'] == 1.0]
        if len(full_data) > 0:
            # MASE table
            mase_pivot = full_data.groupby(['dataset', 'model'])['mase'].mean().unstack()
            print("\nMASE at 100% training data:")
            print(mase_pivot.round(3).to_string())
            print("  * Exchange MASE is expected to be high (near-random-walk at h=30); see MAE below.")

            # MAE table
            mae_pivot = full_data.groupby(['dataset', 'model'])['mae'].mean().unstack()
            print("\nMAE at 100% training data:")
            print(mae_pivot.round(4).to_string())

            # Best model per dataset (MASE, except Exchange uses MAE)
            print("\nBest model per dataset:")
            for ds in mase_pivot.index:
                if ds == 'Exchange':
                    metric_pivot = mae_pivot
                    metric_name = 'MAE'
                else:
                    metric_pivot = mase_pivot
                    metric_name = 'MASE'
                best = metric_pivot.loc[ds].idxmin()
                print(f"  {ds}: {best} ({metric_pivot.loc[ds, best]:.4f} {metric_name})")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
