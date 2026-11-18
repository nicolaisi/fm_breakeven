#!/usr/bin/env python3
"""
Analyze experiment results and generate summary tables/figures.

Usage:
    python scripts/analyze_results.py
    python scripts/analyze_results.py --results-dir custom/results/
    python scripts/analyze_results.py --classical-only
    python scripts/analyze_results.py --combine  # Combine classical + foundation results
"""

import os
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from scipy import stats

from config import RESULTS_DIR, FIGURES_DIR
from analysis import (
    aggregate_results, fit_all_scaling_curves, 
    compute_all_breakevens, print_summary
)
from visualization import generate_all_figures


def load_results(results_dir: str, pattern: str = "*.csv"):
    """Load all result files matching pattern."""
    import glob
    files = glob.glob(os.path.join(results_dir, pattern))
    
    dfs = []
    for f in files:
        if 'raw' in f.lower():
            print(f"Loading {f}...")
            dfs.append(pd.read_csv(f))
    
    if not dfs:
        print(f"No result files found in {results_dir}")
        return None
    
    return pd.concat(dfs, ignore_index=True)


def compute_breakeven_regression(features_df: pd.DataFrame, breakeven_df: pd.DataFrame):
    """Fit regression to predict break-even from features."""
    
    # Merge features with break-even
    merged = breakeven_df.merge(features_df, on='dataset')
    
    # Filter to valid break-even points
    valid = merged[merged['n_star'].notna() & (merged['n_star'] < 1e6)]
    
    if len(valid) < 3:
        print("Not enough valid break-even points for regression")
        return None
    
    # Features
    X = valid[['tau_acf', 'hurst_H', 'noise_ratio', 'seasonal_strength', 'spectral_entropy']].values
    y = np.log(valid['n_star'].values)
    
    # Add intercept
    X = np.column_stack([np.ones(len(X)), X])
    
    # OLS regression
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        
        # Predictions
        y_pred = X @ coeffs
        
        # R-squared
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        feature_names = ['intercept', 'tau_acf', 'hurst_H', 'noise_ratio', 'seasonal_strength', 'spectral_entropy']
        
        print("\nBreak-even Prediction Regression:")
        print("-" * 40)
        for name, coef in zip(feature_names, coeffs):
            print(f"  {name}: {coef:.4f}")
        print(f"  R-squared: {r_squared:.4f}")
        
        return {
            'coefficients': dict(zip(feature_names, coeffs)),
            'r_squared': r_squared,
            'n_points': len(valid)
        }
        
    except Exception as e:
        print(f"Regression failed: {e}")
        return None


def generate_latex_tables(params_df: pd.DataFrame, breakeven_df: pd.DataFrame, output_dir: str):
    """Generate LaTeX tables for paper."""
    
    # Table: Scaling parameters
    latex_params = params_df.pivot_table(
        index='dataset',
        columns='model',
        values=['A', 'alpha', 'B'],
        aggfunc='mean'
    ).round(3)
    
    with open(os.path.join(output_dir, "table_parameters.tex"), 'w') as f:
        f.write(latex_params.to_latex())
    
    # Table: Break-even points
    if len(breakeven_df) > 0:
        latex_breakeven = breakeven_df[['dataset', 'model', 'best_classical', 'n_star']].copy()
        latex_breakeven['n_star'] = latex_breakeven['n_star'].apply(
            lambda x: f"{x:.0f}" if pd.notna(x) else "∞"
        )
        
        with open(os.path.join(output_dir, "table_breakeven.tex"), 'w') as f:
            f.write(latex_breakeven.to_latex(index=False))
    
    print(f"LaTeX tables saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument('--results-dir', type=str, default=RESULTS_DIR,
                        help='Directory containing result files')
    parser.add_argument('--classical-only', action='store_true',
                        help='Only analyze classical model results')
    parser.add_argument('--combine', action='store_true',
                        help='Combine classical and foundation results')
    parser.add_argument('--latex', action='store_true',
                        help='Generate LaTeX tables')
    
    args = parser.parse_args()
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("=" * 60)
    print("Analyzing Results")
    print("=" * 60)
    
    # Load results
    if os.path.exists(os.path.join(args.results_dir, "scaling_raw_results.csv")):
        raw_df = pd.read_csv(os.path.join(args.results_dir, "scaling_raw_results.csv"))
    else:
        raw_df = load_results(args.results_dir)
    
    if raw_df is None or len(raw_df) == 0:
        print("No results to analyze")
        return
    
    print(f"Loaded {len(raw_df)} experiments")
    
    # Filter to valid results
    valid_df = raw_df[raw_df['mase'].notna()]
    print(f"Valid experiments: {len(valid_df)}")
    
    # Aggregate
    print("\nAggregating results...")
    agg_df = valid_df.groupby(['dataset', 'model', 'train_fraction', 'n_train']).agg({
        'mase': ['mean', 'std', 'count']
    }).reset_index()
    agg_df.columns = ['dataset', 'model', 'train_fraction', 'n_train', 'mase_mean', 'mase_std', 'n_seeds']
    agg_df.to_csv(os.path.join(args.results_dir, "aggregated_results.csv"), index=False)
    
    # Fit scaling curves
    print("Fitting scaling curves...")
    params_df = fit_all_scaling_curves(agg_df)
    params_df.to_csv(os.path.join(args.results_dir, "fitted_parameters.csv"), index=False)
    
    # Get model list for break-even
    models = valid_df['model'].unique().tolist()
    classical = [m for m in models if m in ['Naive', 'ETS', 'ARIMA', 'XGBoost']]
    
    # Compute break-even
    print("Computing break-even points...")
    breakeven_df = compute_all_breakevens(params_df, classical_models=classical)
    breakeven_df.to_csv(os.path.join(args.results_dir, "breakeven_analysis.csv"), index=False)
    
    # Print summary
    print_summary(params_df, breakeven_df)
    
    # Load features if available
    features_file = os.path.join(args.results_dir, "dataset_features.csv")
    if os.path.exists(features_file):
        features_df = pd.read_csv(features_file)
        regression_results = compute_breakeven_regression(features_df, breakeven_df)
    
    # Generate figures
    print("\nGenerating figures...")
    generate_all_figures(agg_df, params_df, breakeven_df)
    
    # Generate LaTeX tables if requested
    if args.latex:
        generate_latex_tables(params_df, breakeven_df, args.results_dir)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print(f"Results: {args.results_dir}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
