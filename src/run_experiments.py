#!/usr/bin/env python3
"""
Main run script for Foundation Model Break-Even Experiments.

Usage:
    python run_experiments.py --quick          # Quick test (subset of configs)
    python run_experiments.py --full           # Full experiment grid
    python run_experiments.py --analyze RESULTS_FILE  # Analyze existing results
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, RESULTS_DIR, FIGURES_DIR, TRAIN_FRACTIONS, SEEDS
from synthetic_data import generate_all_datasets, load_synthetic_dataset, GENERATORS
from features import extract_all_features
from models import get_model
from analysis import (
    aggregate_results, fit_all_scaling_curves, 
    compute_all_breakevens, print_summary
)
from visualization import generate_all_figures


def compute_mase(y_true, y_pred, y_train, seasonal_period):
    """Compute Mean Absolute Scaled Error."""
    mae = np.mean(np.abs(y_true - y_pred))
    
    if len(y_train) <= seasonal_period:
        naive_errors = np.abs(np.diff(y_train))
    else:
        naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
    
    naive_mae = np.mean(naive_errors)
    
    if naive_mae == 0:
        return mae
    
    return mae / naive_mae


def run_single_experiment(dataset_name, model_name, train_fraction, seed, verbose=False):
    """Run a single experiment configuration."""
    result = {
        'dataset': dataset_name,
        'model': model_name,
        'train_fraction': train_fraction,
        'seed': seed,
        'n_train': None,
        'mase': None,
        'training_time_sec': None,
        'error': None,
    }
    
    try:
        import time
        
        # Load synthetic dataset
        df, config = load_synthetic_dataset(dataset_name)
        
        # Get first column as target
        y_full = df.iloc[:, 0].values
        
        # Handle config (dict or object)
        if isinstance(config, dict):
            horizon = config['horizon']
            seasonal_period = config['seasonal_period']
        else:
            horizon = config.horizon
            seasonal_period = config.seasonal_period
        
        # Split: 60% train, 20% val, 20% test
        n_total = len(y_full)
        train_end = int(n_total * 0.6)
        test_start = int(n_total * 0.8)
        
        y_train_full = y_full[:train_end]
        y_test = y_full[test_start:test_start + horizon]
        
        # Subsample training data
        n_subsample = max(1, int(len(y_train_full) * train_fraction))
        y_train = y_train_full[-n_subsample:]
        
        result['n_train'] = len(y_train)
        
        # Check minimum samples
        min_samples = max(seasonal_period * 2, 50)
        if len(y_train) < min_samples:
            result['error'] = f"Insufficient data: {len(y_train)} < {min_samples}"
            return result
        
        # Set seed
        np.random.seed(seed)
        
        # Get and fit model
        model = get_model(model_name, horizon=horizon, seasonal_period=seasonal_period)
        
        start_time = time.time()
        model.fit(y_train)
        training_time = time.time() - start_time
        result['training_time_sec'] = training_time
        
        # Predict
        y_pred = model.predict()
        
        # Ensure matching lengths
        min_len = min(len(y_test), len(y_pred))
        y_test_trim = y_test[:min_len]
        y_pred_trim = y_pred[:min_len]
        
        # Compute MASE
        mase = compute_mase(y_test_trim, y_pred_trim, y_train, seasonal_period)
        result['mase'] = mase
        
        if verbose:
            print(f"  {dataset_name}/{model_name}/{train_fraction*100:.0f}%/s{seed}: MASE={mase:.4f}")
        
    except Exception as e:
        result['error'] = str(e)
        if verbose:
            print(f"  ERROR: {e}")
    
    return result


def run_experiments(datasets, models, fractions, seeds, verbose=True):
    """Run full experiment grid."""
    from tqdm import tqdm
    
    total = len(datasets) * len(models) * len(fractions) * len(seeds)
    
    print(f"\nRunning {total} experiments...")
    print(f"  Datasets: {datasets}")
    print(f"  Models: {models}")
    print(f"  Fractions: {fractions}")
    print(f"  Seeds: {seeds}")
    print()
    
    results = []
    
    pbar = tqdm(total=total)
    
    for dataset in datasets:
        for model in models:
            for frac in fractions:
                for seed in seeds:
                    pbar.set_description(f"{dataset[:8]}/{model[:6]}/{frac*100:.0f}%")
                    
                    result = run_single_experiment(
                        dataset, model, frac, seed, verbose=False
                    )
                    results.append(result)
                    pbar.update(1)
    
    pbar.close()
    
    return pd.DataFrame(results)


def extract_features_for_datasets(datasets):
    """Extract features for all datasets."""
    print("\nExtracting dataset features...")
    
    results = []
    
    for name in datasets:
        print(f"  {name}...")
        try:
            df, config = load_synthetic_dataset(name)
            y = df.iloc[:, 0].values
            
            # Use training portion
            train_end = int(len(y) * 0.6)
            y_train = y[:train_end]
            
            if isinstance(config, dict):
                sp = config['seasonal_period']
            else:
                sp = config.seasonal_period
            
            features = extract_all_features(y_train, sp)
            features['dataset'] = name
            features['n_train'] = len(y_train)
            
            results.append(features)
            
            print(f"    τ_acf={features['tau_acf']:.1f}, H={features['hurst_H']:.2f}, "
                  f"S={features['seasonal_strength']:.2f}, Φ={features['spectral_entropy']:.2f}")
            
        except Exception as e:
            print(f"    Error: {e}")
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Run break-even experiments")
    parser.add_argument('--quick', action='store_true', help='Quick test with subset')
    parser.add_argument('--full', action='store_true', help='Full experiment grid')
    parser.add_argument('--analyze', type=str, help='Analyze existing results file')
    parser.add_argument('--features-only', action='store_true', help='Extract features only')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Available datasets and models
    datasets = list(GENERATORS.keys())
    models = ['Naive', 'ETS', 'ARIMA', 'XGBoost']
    
    if args.analyze:
        # Analyze existing results
        print(f"Loading results from {args.analyze}...")
        raw_df = pd.read_csv(args.analyze)
        
        print(f"Loaded {len(raw_df)} results")
        
        # Generate analysis
        agg_df = aggregate_results(raw_df)
        params_df = fit_all_scaling_curves(agg_df)
        breakeven_df = compute_all_breakevens(params_df)
        
        # Save
        agg_df.to_csv(os.path.join(RESULTS_DIR, "aggregated_results.csv"), index=False)
        params_df.to_csv(os.path.join(RESULTS_DIR, "fitted_parameters.csv"), index=False)
        breakeven_df.to_csv(os.path.join(RESULTS_DIR, "breakeven_analysis.csv"), index=False)
        
        # Print summary
        print_summary(params_df, breakeven_df)
        
        # Generate figures
        generate_all_figures(agg_df, params_df, breakeven_df)
        
        return
    
    if args.features_only:
        features_df = extract_features_for_datasets(datasets)
        features_df.to_csv(os.path.join(RESULTS_DIR, "dataset_features.csv"), index=False)
        print(f"\nFeatures saved to {RESULTS_DIR}/dataset_features.csv")
        return
    
    # Generate synthetic data first
    print("=" * 60)
    print("Step 1: Generating Synthetic Datasets")
    print("=" * 60)
    generate_all_datasets(seed=42, save=True)
    
    # Extract features
    print("\n" + "=" * 60)
    print("Step 2: Extracting Dataset Features")
    print("=" * 60)
    features_df = extract_features_for_datasets(datasets)
    features_df.to_csv(os.path.join(RESULTS_DIR, "dataset_features.csv"), index=False)
    
    # Set experiment configuration
    if args.quick:
        fractions = [0.05, 0.2, 1.0]
        seeds = [0]
        print("\n[Quick mode: reduced configuration]")
    else:
        fractions = TRAIN_FRACTIONS
        seeds = SEEDS
    
    # Run experiments
    print("\n" + "=" * 60)
    print("Step 3: Running Experiments")
    print("=" * 60)
    
    results_df = run_experiments(datasets, models, fractions, seeds)
    
    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"raw_results_{timestamp}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\nRaw results saved to {results_file}")
    
    # Analyze
    print("\n" + "=" * 60)
    print("Step 4: Analyzing Results")
    print("=" * 60)
    
    # Filter successful experiments
    valid_df = results_df[results_df['mase'].notna()]
    print(f"Successful experiments: {len(valid_df)}/{len(results_df)}")
    
    # Aggregate
    agg_df = aggregate_results(valid_df)
    agg_df.to_csv(os.path.join(RESULTS_DIR, "aggregated_results.csv"), index=False)
    
    # Fit scaling curves
    params_df = fit_all_scaling_curves(agg_df)
    params_df.to_csv(os.path.join(RESULTS_DIR, "fitted_parameters.csv"), index=False)
    
    # Compute break-even
    breakeven_df = compute_all_breakevens(params_df)
    breakeven_df.to_csv(os.path.join(RESULTS_DIR, "breakeven_analysis.csv"), index=False)
    
    # Print summary
    print_summary(params_df, breakeven_df)
    
    # Generate figures
    print("\n" + "=" * 60)
    print("Step 5: Generating Figures")
    print("=" * 60)
    generate_all_figures(agg_df, params_df, breakeven_df)
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
