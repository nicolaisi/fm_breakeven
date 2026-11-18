"""
Main experiment runner for break-even analysis.

Runs experiments across datasets, models, training sizes, and seeds.
Outputs results to CSV for analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import time
import os
from tqdm import tqdm
from datetime import datetime

# Add src to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TRAIN_FRACTIONS, SEEDS, DATASETS, 
    CLASSICAL_MODELS, RESULTS_DIR
)
from data_loader import (
    load_dataset, prepare_train_test_split, 
    subsample_train, get_series_for_forecasting
)
from models import get_model
from features import extract_all_features


def compute_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonal_period: int
) -> float:
    """Compute Mean Absolute Scaled Error.
    
    MASE = MAE / naive_MAE
    where naive_MAE is the in-sample seasonal naive MAE.
    """
    # Forecast MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # In-sample seasonal naive MAE
    n_train = len(y_train)
    if n_train <= seasonal_period:
        # Fall back to simple diff if not enough data
        naive_errors = np.abs(np.diff(y_train))
    else:
        naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
    
    naive_mae = np.mean(naive_errors)
    
    if naive_mae == 0:
        return mae  # Avoid division by zero
    
    return mae / naive_mae


def run_single_experiment(
    dataset_name: str,
    model_name: str,
    train_fraction: float,
    seed: int,
    verbose: bool = False
) -> Dict:
    """Run a single experiment configuration.
    
    Returns:
        Dictionary with experiment results
    """
    result = {
        'dataset': dataset_name,
        'model': model_name,
        'train_fraction': train_fraction,
        'seed': seed,
        'n_train': None,
        'mase': None,
        'training_time_sec': None,
        'inference_time_sec': None,
        'error': None,
    }
    
    try:
        # Load data
        df, config = load_dataset(dataset_name)
        
        # Get target series
        y_full = get_series_for_forecasting(df)
        
        # Split data
        train_end = int(len(y_full) * 0.6)
        test_start = int(len(y_full) * 0.8)
        
        y_train_full = y_full[:train_end]
        y_test = y_full[test_start:test_start + config.horizon]
        
        # Subsample training data
        n_subsample = max(1, int(len(y_train_full) * train_fraction))
        y_train = y_train_full[-n_subsample:]  # Take most recent
        
        result['n_train'] = len(y_train)
        
        if verbose:
            print(f"  Training samples: {len(y_train)}")
        
        # Check minimum data requirements
        min_samples = max(config.seasonal_period * 2, 50)
        if len(y_train) < min_samples:
            result['error'] = f"Insufficient data: {len(y_train)} < {min_samples}"
            return result
        
        # Set random seed
        np.random.seed(seed)
        
        # Get model
        model = get_model(
            model_name,
            horizon=config.horizon,
            seasonal_period=config.seasonal_period
        )
        
        # Fit model
        start_time = time.time()
        model.fit(y_train)
        training_time = time.time() - start_time
        result['training_time_sec'] = training_time
        
        # Generate predictions
        start_time = time.time()
        y_pred = model.predict()
        inference_time = time.time() - start_time
        result['inference_time_sec'] = inference_time
        
        # Ensure predictions match test length
        y_pred = y_pred[:len(y_test)]
        y_test_trimmed = y_test[:len(y_pred)]
        
        # Compute MASE
        mase = compute_mase(
            y_test_trimmed, 
            y_pred, 
            y_train,
            config.seasonal_period
        )
        result['mase'] = mase
        
        if verbose:
            print(f"  MASE: {mase:.4f}, Train time: {training_time:.2f}s")
        
    except Exception as e:
        result['error'] = str(e)
        if verbose:
            print(f"  Error: {e}")
    
    return result


def run_experiments(
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    fractions: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    output_file: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Run full experiment grid.
    
    Args:
        datasets: List of dataset names (default: all)
        models: List of model names (default: classical only)
        fractions: List of training fractions (default: config)
        seeds: List of random seeds (default: config)
        output_file: Path to save results CSV
        verbose: Print progress
    
    Returns:
        DataFrame with all results
    """
    # Set defaults
    if datasets is None:
        datasets = list(DATASETS.keys())
    if models is None:
        models = list(CLASSICAL_MODELS.keys())
    if fractions is None:
        fractions = TRAIN_FRACTIONS
    if seeds is None:
        seeds = SEEDS
    
    # Calculate total experiments
    total = len(datasets) * len(models) * len(fractions) * len(seeds)
    
    if verbose:
        print(f"Running {total} experiments...")
        print(f"  Datasets: {datasets}")
        print(f"  Models: {models}")
        print(f"  Fractions: {fractions}")
        print(f"  Seeds: {seeds}")
        print()
    
    results = []
    
    # Create progress bar
    pbar = tqdm(total=total, disable=not verbose)
    
    for dataset_name in datasets:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*60}")
        
        for model_name in models:
            for fraction in fractions:
                for seed in seeds:
                    pbar.set_description(
                        f"{dataset_name}/{model_name}/{fraction*100:.0f}%/s{seed}"
                    )
                    
                    result = run_single_experiment(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        train_fraction=fraction,
                        seed=seed,
                        verbose=False
                    )
                    results.append(result)
                    pbar.update(1)
    
    pbar.close()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(RESULTS_DIR, f"results_{timestamp}.csv")
    
    df.to_csv(output_file, index=False)
    if verbose:
        print(f"\nResults saved to: {output_file}")
    
    # Print summary
    if verbose:
        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        
        # Success rate
        n_success = df['mase'].notna().sum()
        print(f"Successful experiments: {n_success}/{len(df)}")
        
        # Average MASE by model
        print("\nAverage MASE by model (at 100% data):")
        full_data = df[df['train_fraction'] == 1.0]
        mase_by_model = full_data.groupby('model')['mase'].mean().sort_values()
        for model, mase in mase_by_model.items():
            print(f"  {model}: {mase:.4f}")
    
    return df


def extract_dataset_features(
    datasets: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Extract features for all datasets.
    
    Returns:
        DataFrame with dataset features
    """
    if datasets is None:
        datasets = list(DATASETS.keys())
    
    results = []
    
    for dataset_name in datasets:
        if verbose:
            print(f"Extracting features for {dataset_name}...")
        
        try:
            df, config = load_dataset(dataset_name)
            y = get_series_for_forecasting(df)
            
            # Use training portion only
            train_end = int(len(y) * 0.6)
            y_train = y[:train_end]
            
            features = extract_all_features(y_train, config.seasonal_period)
            features['dataset'] = dataset_name
            features['n_total'] = len(y)
            features['n_train'] = len(y_train)
            features['seasonal_period'] = config.seasonal_period
            
            results.append(features)
            
            if verbose:
                for k, v in features.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.3f}")
                print()
                
        except Exception as e:
            if verbose:
                print(f"  Error: {e}\n")
    
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['dataset', 'n_total', 'n_train', 'seasonal_period',
            'tau_acf', 'hurst_H', 'noise_ratio', 'seasonal_strength', 'spectral_entropy']
    df = df[[c for c in cols if c in df.columns]]
    
    if output_file is None:
        output_file = os.path.join(RESULTS_DIR, "dataset_features.csv")
    
    df.to_csv(output_file, index=False)
    if verbose:
        print(f"Features saved to: {output_file}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run break-even experiments")
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to run (default: all available)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Models to run (default: classical only)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with subset of configs')
    parser.add_argument('--features-only', action='store_true',
                        help='Only extract dataset features')
    
    args = parser.parse_args()
    
    if args.features_only:
        extract_dataset_features(datasets=args.datasets)
    else:
        if args.quick:
            # Quick test configuration
            fractions = [0.1, 0.5, 1.0]
            seeds = [0]
        else:
            fractions = None
            seeds = None
        
        run_experiments(
            datasets=args.datasets,
            models=args.models,
            fractions=fractions,
            seeds=seeds
        )
