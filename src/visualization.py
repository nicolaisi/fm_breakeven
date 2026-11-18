"""
Visualization module for scaling curves and break-even analysis.

Generates publication-ready figures for the paper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Optional, Tuple
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FIGURES_DIR
from analysis import power_law

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color schemes
CLASSICAL_COLORS = {
    'Naive': '#1f77b4',
    'ETS': '#2ca02c',
    'ARIMA': '#17becf',
    'XGBoost': '#9467bd',
}

FOUNDATION_COLORS = {
    'Chronos-Bolt-S': '#ff7f0e',
    'Chronos-Bolt-B': '#d62728',
    'Chronos-Bolt-L': '#8c564b',
    'Moirai-S': '#e377c2',
    'Moirai-B': '#bcbd22',
    'Moirai-L': '#ff9896',
    'Lag-Llama': '#c5b0d5',
}

ALL_COLORS = {**CLASSICAL_COLORS, **FOUNDATION_COLORS}


def plot_scaling_curves_single(
    agg_df: pd.DataFrame,
    params_df: pd.DataFrame,
    dataset: str,
    ax: Optional[plt.Axes] = None,
    show_fitted: bool = True,
    show_breakeven: bool = True
) -> plt.Axes:
    """Plot scaling curves for a single dataset.
    
    Args:
        agg_df: Aggregated results DataFrame
        params_df: Fitted parameters DataFrame
        dataset: Dataset name
        ax: Matplotlib axes (creates new if None)
        show_fitted: Show fitted power-law curves
        show_breakeven: Show vertical line at break-even point
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    # Filter data for this dataset
    ds_data = agg_df[agg_df['dataset'] == dataset]
    ds_params = params_df[params_df['dataset'] == dataset]
    
    if len(ds_data) == 0:
        ax.text(0.5, 0.5, f"No data for {dataset}", 
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Plot each model
    for model in ds_data['model'].unique():
        model_data = ds_data[ds_data['model'] == model].sort_values('n_train')
        
        n_values = model_data['n_train'].values
        mase_mean = model_data['mase_mean'].values
        mase_std = model_data['mase_std'].values
        
        color = ALL_COLORS.get(model, '#333333')
        
        # Plot data points with error bars
        ax.errorbar(
            n_values, mase_mean, yerr=mase_std,
            fmt='o-', color=color, label=model,
            capsize=3, markersize=5, linewidth=1.5
        )
        
        # Plot fitted curve
        if show_fitted:
            model_params = ds_params[ds_params['model'] == model]
            if len(model_params) > 0:
                p = model_params.iloc[0]
                if not np.isnan(p['A']):
                    n_fit = np.logspace(np.log10(n_values.min()), 
                                       np.log10(n_values.max()), 100)
                    mase_fit = power_law(n_fit, p['A'], p['alpha'], p['B'])
                    ax.plot(n_fit, mase_fit, '--', color=color, alpha=0.5, linewidth=1)
    
    # Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Training Samples (n)')
    ax.set_ylabel('MASE')
    ax.set_title(dataset)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_scaling_curves_grid(
    agg_df: pd.DataFrame,
    params_df: pd.DataFrame,
    datasets: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot scaling curves for multiple datasets in a grid.
    
    Args:
        agg_df: Aggregated results DataFrame
        params_df: Fitted parameters DataFrame
        datasets: List of datasets to plot (default: all)
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    if datasets is None:
        datasets = agg_df['dataset'].unique().tolist()
    
    n_datasets = len(datasets)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes)
    
    for idx, dataset in enumerate(datasets):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        plot_scaling_curves_single(agg_df, params_df, dataset, ax=ax)
    
    # Hide empty subplots
    for idx in range(n_datasets, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_parameter_comparison(
    params_df: pd.DataFrame,
    parameter: str = 'alpha',
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot parameter comparison across datasets and models.
    
    Args:
        params_df: Fitted parameters DataFrame
        parameter: Which parameter to plot ('A', 'alpha', or 'B')
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Pivot data
    pivot = params_df.pivot(index='dataset', columns='model', values=parameter)
    
    # Plot grouped bar chart
    x = np.arange(len(pivot.index))
    width = 0.8 / len(pivot.columns)
    
    for i, model in enumerate(pivot.columns):
        offset = (i - len(pivot.columns)/2 + 0.5) * width
        color = ALL_COLORS.get(model, '#333333')
        ax.bar(x + offset, pivot[model], width, label=model, color=color)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel(f'Scaling Parameter {parameter}')
    ax.set_title(f'Scaling Exponent Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_breakeven_summary(
    breakeven_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot break-even points summary.
    
    Args:
        breakeven_df: Break-even analysis DataFrame
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter to models with valid break-even
    valid = breakeven_df[breakeven_df['n_star'].notna()].copy()
    
    if len(valid) == 0:
        ax.text(0.5, 0.5, "No break-even points found",
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Plot break-even by dataset
    datasets = valid['dataset'].unique()
    x = np.arange(len(datasets))
    
    for i, dataset in enumerate(datasets):
        ds_data = valid[valid['dataset'] == dataset]
        for _, row in ds_data.iterrows():
            model = row['model']
            n_star = row['n_star']
            color = ALL_COLORS.get(model, '#333333')
            ax.scatter(i, n_star, c=color, s=100, label=model if i == 0 else "")
    
    ax.set_yscale('log')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Break-even Point (n*)')
    ax.set_title('Break-even Points Across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
              loc='upper left', bbox_to_anchor=(1, 1))
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_prediction_scatter(
    features_df: pd.DataFrame,
    breakeven_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot predicted vs observed break-even points.
    
    Args:
        features_df: Dataset features DataFrame
        breakeven_df: Break-even analysis DataFrame with predictions
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # This would use actual prediction results
    # For now, create placeholder
    ax.text(0.5, 0.5, "Prediction scatter\n(requires regression results)",
            ha='center', va='center', transform=ax.transAxes)
    
    ax.set_xlabel('Predicted log(n*)')
    ax.set_ylabel('Observed log(n*)')
    ax.set_title('Break-even Prediction Accuracy')
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def generate_all_figures(
    agg_df: pd.DataFrame,
    params_df: pd.DataFrame,
    breakeven_df: pd.DataFrame,
    output_dir: Optional[str] = None
):
    """Generate all figures for the paper.
    
    Args:
        agg_df: Aggregated results
        params_df: Fitted parameters
        breakeven_df: Break-even analysis
        output_dir: Output directory for figures
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating figures...")
    
    # Figure 1: Scaling curves grid
    plot_scaling_curves_grid(
        agg_df, params_df,
        save_path=os.path.join(output_dir, "scaling_curves_all.pdf")
    )
    
    # Figure 2: Alpha comparison
    plot_parameter_comparison(
        params_df, parameter='alpha',
        save_path=os.path.join(output_dir, "alpha_comparison.pdf")
    )
    
    # Figure 3: Break-even summary
    plot_breakeven_summary(
        breakeven_df,
        save_path=os.path.join(output_dir, "breakeven_summary.pdf")
    )
    
    print(f"All figures saved to {output_dir}")


if __name__ == "__main__":
    # Test with sample data
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Directory with analysis results')
    args = parser.parse_args()
    
    if args.results_dir:
        from config import RESULTS_DIR
        results_dir = args.results_dir or RESULTS_DIR
        
        agg_df = pd.read_csv(os.path.join(results_dir, "aggregated_results.csv"))
        params_df = pd.read_csv(os.path.join(results_dir, "fitted_parameters.csv"))
        breakeven_df = pd.read_csv(os.path.join(results_dir, "breakeven_analysis.csv"))
        
        generate_all_figures(agg_df, params_df, breakeven_df)
    else:
        print("Run with --results-dir to generate figures from results")
