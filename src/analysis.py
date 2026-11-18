"""
Analysis module for scaling curves and break-even computation.

Functions:
- Fit power-law scaling curves: Error(n) = A * n^(-alpha) + B
- Compute break-even points between model pairs
- Aggregate results and generate summary tables
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Optional
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULTS_DIR, FIGURES_DIR


def power_law(n: np.ndarray, A: float, alpha: float, B: float) -> np.ndarray:
    """Power-law scaling function: Error(n) = A * n^(-alpha) + B"""
    return A * np.power(n, -alpha) + B


def fit_power_law(
    n_values: np.ndarray,
    error_values: np.ndarray,
    bounds: Tuple = ([0.1, 0.01, 0], [100, 2.0, 10])
) -> Tuple[Dict[str, float], float]:
    """Fit power-law to scaling data.
    
    Args:
        n_values: Training set sizes
        error_values: Corresponding error values (e.g., MASE)
        bounds: Parameter bounds for (A, alpha, B)
    
    Returns:
        Tuple of (parameters dict, R-squared)
    """
    try:
        # Initial guess
        p0 = [1.0, 0.3, np.min(error_values) * 0.9]
        
        # Fit in log space for better numerical stability
        popt, pcov = curve_fit(
            power_law,
            n_values,
            error_values,
            p0=p0,
            bounds=bounds,
            maxfev=5000
        )
        
        A, alpha, B = popt
        
        # Compute R-squared
        y_pred = power_law(n_values, A, alpha, B)
        ss_res = np.sum((error_values - y_pred) ** 2)
        ss_tot = np.sum((error_values - np.mean(error_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        params = {
            'A': A,
            'alpha': alpha,
            'B': B,
        }
        
        return params, r_squared
        
    except Exception as e:
        # Return default values if fitting fails
        return {'A': np.nan, 'alpha': np.nan, 'B': np.nan}, 0.0


def compute_breakeven(
    params1: Dict[str, float],
    params2: Dict[str, float],
    n_range: Tuple[int, int] = (10, 100000)
) -> Optional[float]:
    """Compute break-even point between two models.
    
    Finds n* where Error1(n*) = Error2(n*)
    
    Args:
        params1: Power-law parameters for model 1 (e.g., foundation model)
        params2: Power-law parameters for model 2 (e.g., classical model)
        n_range: Range to search for break-even
    
    Returns:
        Break-even point n*, or None if no crossover in range
    """
    if any(np.isnan(v) for v in params1.values()) or \
       any(np.isnan(v) for v in params2.values()):
        return None
    
    # Sample points in log space
    n_values = np.logspace(
        np.log10(n_range[0]),
        np.log10(n_range[1]),
        1000
    )
    
    error1 = power_law(n_values, params1['A'], params1['alpha'], params1['B'])
    error2 = power_law(n_values, params2['A'], params2['alpha'], params2['B'])
    
    # Find sign changes in difference
    diff = error1 - error2
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    
    if len(sign_changes) == 0:
        # No crossover
        return None
    
    # Return first crossover point
    idx = sign_changes[0]
    n_star = n_values[idx]
    
    return n_star


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results across seeds.
    
    Returns:
        DataFrame with mean and std MASE per (dataset, model, train_fraction)
    """
    # Filter out errors
    df_valid = df[df['mase'].notna()].copy()
    
    # Aggregate
    agg = df_valid.groupby(['dataset', 'model', 'train_fraction', 'n_train']).agg({
        'mase': ['mean', 'std', 'count'],
        'training_time_sec': 'mean',
    }).reset_index()
    
    # Flatten column names
    agg.columns = ['dataset', 'model', 'train_fraction', 'n_train',
                   'mase_mean', 'mase_std', 'n_seeds', 'train_time_mean']
    
    return agg


def fit_all_scaling_curves(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Fit scaling curves for all (dataset, model) combinations.
    
    Returns:
        DataFrame with fitted parameters
    """
    results = []
    
    for (dataset, model), group in agg_df.groupby(['dataset', 'model']):
        # Sort by n_train
        group = group.sort_values('n_train')
        
        n_values = group['n_train'].values
        error_values = group['mase_mean'].values
        
        # Need at least 3 points for fitting
        if len(n_values) < 3:
            continue
        
        params, r_squared = fit_power_law(n_values, error_values)
        
        results.append({
            'dataset': dataset,
            'model': model,
            'A': params['A'],
            'alpha': params['alpha'],
            'B': params['B'],
            'R_squared': r_squared,
            'n_points': len(n_values),
        })
    
    return pd.DataFrame(results)


def compute_all_breakevens(
    params_df: pd.DataFrame,
    classical_models: List[str] = ['Naive', 'ETS', 'ARIMA', 'XGBoost'],
    foundation_models: List[str] = []  # Will be populated when FM results available
) -> pd.DataFrame:
    """Compute break-even points for all dataset-model combinations.
    
    Returns:
        DataFrame with break-even analysis
    """
    results = []
    
    for dataset in params_df['dataset'].unique():
        dataset_params = params_df[params_df['dataset'] == dataset]
        
        # Get best classical model
        classical = dataset_params[dataset_params['model'].isin(classical_models)]
        if len(classical) == 0:
            continue
        
        # Find best classical (lowest B = best asymptotic performance)
        best_classical_idx = classical['B'].idxmin()
        best_classical = classical.loc[best_classical_idx]
        best_classical_name = best_classical['model']
        best_classical_params = {
            'A': best_classical['A'],
            'alpha': best_classical['alpha'],
            'B': best_classical['B']
        }
        
        # Compare each model to best classical
        for _, row in dataset_params.iterrows():
            model = row['model']
            model_params = {
                'A': row['A'],
                'alpha': row['alpha'],
                'B': row['B']
            }
            
            # Compute break-even
            n_star = compute_breakeven(model_params, best_classical_params)
            
            results.append({
                'dataset': dataset,
                'model': model,
                'best_classical': best_classical_name,
                'n_star': n_star,
                'model_A': row['A'],
                'model_alpha': row['alpha'],
                'model_B': row['B'],
                'classical_A': best_classical['A'],
                'classical_alpha': best_classical['alpha'],
                'classical_B': best_classical['B'],
            })
    
    return pd.DataFrame(results)


def generate_summary_tables(
    raw_df: pd.DataFrame,
    output_dir: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """Generate all summary tables for the paper.
    
    Returns:
        Dictionary of table name -> DataFrame
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    
    tables = {}
    
    # Table 1: Aggregated results
    agg_df = aggregate_results(raw_df)
    agg_df.to_csv(os.path.join(output_dir, "aggregated_results.csv"), index=False)
    tables['aggregated'] = agg_df
    
    # Table 2: Fitted parameters
    params_df = fit_all_scaling_curves(agg_df)
    params_df.to_csv(os.path.join(output_dir, "fitted_parameters.csv"), index=False)
    tables['parameters'] = params_df
    
    # Table 3: Break-even analysis
    breakeven_df = compute_all_breakevens(params_df)
    breakeven_df.to_csv(os.path.join(output_dir, "breakeven_analysis.csv"), index=False)
    tables['breakeven'] = breakeven_df
    
    # Table 4: Model comparison at full data
    full_data = agg_df[agg_df['train_fraction'] == 1.0].copy()
    comparison = full_data.pivot_table(
        index='dataset',
        columns='model',
        values='mase_mean'
    )
    comparison.to_csv(os.path.join(output_dir, "model_comparison.csv"))
    tables['comparison'] = comparison
    
    print(f"Summary tables saved to {output_dir}")
    
    return tables


def print_summary(params_df: pd.DataFrame, breakeven_df: pd.DataFrame):
    """Print human-readable summary of results."""
    
    print("\n" + "="*70)
    print("SCALING CURVE ANALYSIS SUMMARY")
    print("="*70)
    
    # Average parameters by model
    print("\nAverage Scaling Parameters by Model:")
    print("-" * 50)
    model_avg = params_df.groupby('model')[['A', 'alpha', 'B', 'R_squared']].mean()
    print(model_avg.round(3).to_string())
    
    # Break-even summary
    print("\n\nBreak-Even Analysis:")
    print("-" * 50)
    
    for dataset in breakeven_df['dataset'].unique():
        ds_data = breakeven_df[breakeven_df['dataset'] == dataset]
        best_classical = ds_data['best_classical'].iloc[0]
        
        print(f"\n{dataset} (vs {best_classical}):")
        
        for _, row in ds_data.iterrows():
            model = row['model']
            n_star = row['n_star']
            
            if model == best_classical:
                continue
            
            if n_star is None or np.isnan(n_star):
                print(f"  {model}: No crossover (classical always wins)")
            else:
                print(f"  {model}: n* = {n_star:.0f}")
    
    # Overall statistics
    print("\n\nOverall Statistics:")
    print("-" * 50)
    
    valid_breakevens = breakeven_df['n_star'].dropna()
    if len(valid_breakevens) > 0:
        print(f"Break-even range: {valid_breakevens.min():.0f} - {valid_breakevens.max():.0f}")
        print(f"Median break-even: {valid_breakevens.median():.0f}")
    
    no_crossover = breakeven_df['n_star'].isna().sum()
    total = len(breakeven_df)
    print(f"No crossover cases: {no_crossover}/{total}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument('--results-file', type=str, required=True,
                        help='Path to raw results CSV')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_file}...")
    raw_df = pd.read_csv(args.results_file)
    print(f"Loaded {len(raw_df)} experiment results")
    
    # Generate tables
    tables = generate_summary_tables(raw_df)
    
    # Print summary
    print_summary(tables['parameters'], tables['breakeven'])
