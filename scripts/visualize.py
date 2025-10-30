"""
Author: Wenyu Ouyang
Date: 2025-10-30
LastEditTime: 2025-10-30
LastEditors: Wenyu Ouyang
Description: Visualization script for unified evaluation results
FilePath: \\hydromodel\\scripts\\visualize_unified.py
Copyright (c) 2023-2025 Wenyu Ouyang. All rights reserved.

This script provides comprehensive visualization for model evaluation results
from the unified architecture, including:
- Time series comparison (observed vs simulated)
- Scatter plots
- Flow duration curves
- Monthly/Annual statistics
- Multi-basin comparison
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings('ignore')

# Set publication-quality plot defaults
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def load_evaluation_results(eval_dir):
    """
    Load evaluation results from directory.

    Parameters
    ----------
    eval_dir : str
        Path to evaluation directory (e.g., results/exp_name/evaluation_test/)

    Returns
    -------
    ds : xarray.Dataset
        Simulation results
    metrics_df : pandas.DataFrame
        Performance metrics
    """
    eval_path = Path(eval_dir)

    # Find NetCDF file
    nc_files = list(eval_path.glob("*_evaluation_results.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No evaluation results (.nc) found in {eval_dir}")

    nc_file = nc_files[0]
    print(f"Loading results from: {nc_file}")

    # Load NetCDF
    ds = xr.open_dataset(nc_file)

    # Load metrics if available
    metrics_file = eval_path / "basins_metrics.csv"
    if metrics_file.exists():
        metrics_df = pd.read_csv(metrics_file, index_col=0)
    else:
        metrics_df = None
        print("Warning: No metrics file found")

    return ds, metrics_df


def plot_timeseries_with_rain(ds, basin_id, save_path, title_suffix=""):
    """
    Plot time series comparison with precipitation.

    Parameters
    ----------
    ds : xarray.Dataset
        Simulation results
    basin_id : str
        Basin ID to plot
    save_path : str
        Output file path
    title_suffix : str
        Additional text for title (e.g., "Training Period")
    """
    # Find basin index
    basin_ids = [str(b) for b in ds['basin'].values]
    if basin_id not in basin_ids:
        print(f"Warning: Basin {basin_id} not found in results")
        return

    basin_idx = basin_ids.index(basin_id)

    # Extract data
    time = pd.to_datetime(ds['time'].values)
    qobs = ds['qobs'].values[:, basin_idx]
    qsim = ds['qsim'].values[:, basin_idx]

    # Get precipitation if available
    if 'prcp' in ds.variables:
        prcp = ds['prcp'].values[:, basin_idx]
        has_prcp = True
    else:
        has_prcp = False

    # Create figure
    if has_prcp:
        fig = plt.figure(figsize=(14, 6))
        gs = GridSpec(2, 1, height_ratios=[1, 3], hspace=0.05)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
    else:
        fig, ax2 = plt.subplots(figsize=(14, 5))
        ax1 = None

    # Plot precipitation
    if has_prcp:
        ax1.bar(time, prcp, width=0.8, color='steelblue', alpha=0.7, label='Precipitation')
        ax1.set_ylabel('Precipitation (mm/day)')
        ax1.invert_yaxis()  # Invert to show rain from top
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(labelbottom=False)

    # Plot streamflow
    ax2.plot(time, qobs, label='Observed', color='#2E86AB', linewidth=1.5, alpha=0.9)
    ax2.plot(time, qsim, label='Simulated', color='#E63946', linewidth=1.2, alpha=0.9, linestyle='--')

    # Calculate metrics
    valid_mask = ~np.isnan(qobs) & ~np.isnan(qsim)
    if np.sum(valid_mask) > 0:
        nse = 1 - np.sum((qobs[valid_mask] - qsim[valid_mask])**2) / \
              np.sum((qobs[valid_mask] - np.mean(qobs[valid_mask]))**2)
        rmse = np.sqrt(np.mean((qobs[valid_mask] - qsim[valid_mask])**2))
        pbias = np.sum(qsim[valid_mask] - qobs[valid_mask]) / np.sum(qobs[valid_mask]) * 100

        # Add metrics to legend
        metrics_text = f'NSE={nse:.3f}, RMSE={rmse:.2f}, PBIAS={pbias:.1f}%'
        ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.set_xlabel('Date')
    ax2.set_ylabel('Streamflow (m³/s)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Title
    title = f'Streamflow Simulation - Basin {basin_id}'
    if title_suffix:
        title += f' ({title_suffix})'
    if ax1:
        ax1.set_title(title, fontsize=13, fontweight='bold')
    else:
        ax2.set_title(title, fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")


def plot_scatter(ds, basin_id, save_path, title_suffix=""):
    """
    Plot scatter plot of observed vs simulated.

    Parameters
    ----------
    ds : xarray.Dataset
        Simulation results
    basin_id : str
        Basin ID to plot
    save_path : str
        Output file path
    title_suffix : str
        Additional text for title
    """
    # Find basin index
    basin_ids = [str(b) for b in ds['basin'].values]
    if basin_id not in basin_ids:
        print(f"Warning: Basin {basin_id} not found in results")
        return

    basin_idx = basin_ids.index(basin_id)

    # Extract data
    qobs = ds['qobs'].values[:, basin_idx]
    qsim = ds['qsim'].values[:, basin_idx]

    # Remove NaN
    valid_mask = ~np.isnan(qobs) & ~np.isnan(qsim)
    qobs = qobs[valid_mask]
    qsim = qsim[valid_mask]

    if len(qobs) == 0:
        print(f"Warning: No valid data for basin {basin_id}")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 7))

    # Scatter plot with density coloring
    from matplotlib.colors import LogNorm
    h = ax.hist2d(qobs, qsim, bins=50, cmap='YlOrRd', norm=LogNorm(), alpha=0.7)
    plt.colorbar(h[3], ax=ax, label='Count (log scale)')

    # 1:1 line
    max_val = max(qobs.max(), qsim.max())
    min_val = min(qobs.min(), qsim.min())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.7, label='1:1 line')

    # Calculate metrics
    nse = 1 - np.sum((qobs - qsim)**2) / np.sum((qobs - np.mean(qobs))**2)
    r2 = np.corrcoef(qobs, qsim)[0, 1]**2
    rmse = np.sqrt(np.mean((qobs - qsim)**2))
    pbias = np.sum(qsim - qobs) / np.sum(qobs) * 100

    # Add metrics text
    metrics_text = f'NSE = {nse:.3f}\n$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f}\nPBIAS = {pbias:.1f}%'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Observed Streamflow (m³/s)')
    ax.set_ylabel('Simulated Streamflow (m³/s)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right')

    # Title
    title = f'Observed vs Simulated - Basin {basin_id}'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")


def plot_flow_duration_curve(ds, basin_id, save_path, title_suffix=""):
    """
    Plot flow duration curve.

    Parameters
    ----------
    ds : xarray.Dataset
        Simulation results
    basin_id : str
        Basin ID to plot
    save_path : str
        Output file path
    title_suffix : str
        Additional text for title
    """
    # Find basin index
    basin_ids = [str(b) for b in ds['basin'].values]
    if basin_id not in basin_ids:
        print(f"Warning: Basin {basin_id} not found in results")
        return

    basin_idx = basin_ids.index(basin_id)

    # Extract data
    qobs = ds['qobs'].values[:, basin_idx]
    qsim = ds['qsim'].values[:, basin_idx]

    # Remove NaN
    qobs = qobs[~np.isnan(qobs)]
    qsim = qsim[~np.isnan(qsim)]

    if len(qobs) == 0 or len(qsim) == 0:
        print(f"Warning: No valid data for basin {basin_id}")
        return

    # Calculate FDC
    qobs_sorted = np.sort(qobs)[::-1]
    qsim_sorted = np.sort(qsim)[::-1]

    exceedance_obs = np.arange(1, len(qobs_sorted) + 1) / len(qobs_sorted) * 100
    exceedance_sim = np.arange(1, len(qsim_sorted) + 1) / len(qsim_sorted) * 100

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(exceedance_obs, qobs_sorted, label='Observed', color='#2E86AB', linewidth=2, alpha=0.9)
    ax.plot(exceedance_sim, qsim_sorted, label='Simulated', color='#E63946', linewidth=2, alpha=0.9, linestyle='--')

    ax.set_xlabel('Exceedance Probability (%)')
    ax.set_ylabel('Streamflow (m³/s)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.legend()

    # Title
    title = f'Flow Duration Curve - Basin {basin_id}'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")


def plot_monthly_comparison(ds, basin_id, save_path, title_suffix=""):
    """
    Plot monthly average comparison.

    Parameters
    ----------
    ds : xarray.Dataset
        Simulation results
    basin_id : str
        Basin ID to plot
    save_path : str
        Output file path
    title_suffix : str
        Additional text for title
    """
    # Find basin index
    basin_ids = [str(b) for b in ds['basin'].values]
    if basin_id not in basin_ids:
        print(f"Warning: Basin {basin_id} not found in results")
        return

    basin_idx = basin_ids.index(basin_id)

    # Extract data
    time = pd.to_datetime(ds['time'].values)
    qobs = ds['qobs'].values[:, basin_idx]
    qsim = ds['qsim'].values[:, basin_idx]

    # Create DataFrame
    df = pd.DataFrame({'time': time, 'qobs': qobs, 'qsim': qsim})
    df['month'] = df['time'].dt.month

    # Calculate monthly average
    monthly_obs = df.groupby('month')['qobs'].mean()
    monthly_sim = df.groupby('month')['qsim'].mean()
    monthly_obs_std = df.groupby('month')['qobs'].std()
    monthly_sim_std = df.groupby('month')['qsim'].std()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    months = np.arange(1, 13)
    width = 0.35

    ax.bar(months - width/2, monthly_obs, width, label='Observed',
           color='#2E86AB', alpha=0.8, yerr=monthly_obs_std, capsize=3)
    ax.bar(months + width/2, monthly_sim, width, label='Simulated',
           color='#E63946', alpha=0.8, yerr=monthly_sim_std, capsize=3)

    ax.set_xlabel('Month')
    ax.set_ylabel('Average Streamflow (m³/s)')
    ax.set_xticks(months)
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Title
    title = f'Monthly Average Streamflow - Basin {basin_id}'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")


def plot_metrics_comparison(metrics_df, save_path):
    """
    Plot metrics comparison for multiple basins.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Metrics for all basins
    save_path : str
        Output file path
    """
    if metrics_df is None or len(metrics_df) == 0:
        print("Warning: No metrics to plot")
        return

    # Select key metrics
    metrics_to_plot = ['NSE', 'KGE', 'RMSE', 'PBIAS']
    available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]

    if len(available_metrics) == 0:
        print("Warning: No standard metrics found")
        return

    # Create figure
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    basins = metrics_df.index
    x = np.arange(len(basins))

    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        values = metrics_df[metric].values

        # Color based on performance
        if metric in ['NSE', 'KGE']:
            colors = ['green' if v >= 0.7 else 'orange' if v >= 0.5 else 'red' for v in values]
        elif metric == 'RMSE':
            colors = ['green' if v <= np.percentile(values, 33) else
                     'orange' if v <= np.percentile(values, 67) else 'red' for v in values]
        else:  # PBIAS
            colors = ['green' if abs(v) <= 10 else 'orange' if abs(v) <= 25 else 'red' for v in values]

        ax.bar(x, values, color=colors, alpha=0.7)
        ax.set_xlabel('Basin')
        ax.set_ylabel(metric)
        ax.set_title(metric, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(basins, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Add reference lines
        if metric == 'NSE':
            ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Satisfactory')
            ax.axhline(y=0.7, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Good')
            ax.legend(fontsize=8)
        elif metric == 'KGE':
            ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(y=0.7, color='green', linestyle='--', linewidth=1, alpha=0.5)

    plt.suptitle('Model Performance Metrics Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {save_path}")


def visualize_evaluation(eval_dir, output_dir=None, plot_types='all', basins=None):
    """
    Main visualization function.

    Parameters
    ----------
    eval_dir : str
        Path to evaluation directory
    output_dir : str, optional
        Output directory for plots (default: eval_dir/figures)
    plot_types : str or list
        Types of plots to generate: 'all', 'timeseries', 'scatter', 'fdc', 'monthly', 'metrics'
    basins : list, optional
        List of basins to plot (default: all basins)
    """
    # Load results
    ds, metrics_df = load_evaluation_results(eval_dir)

    # Setup output directory
    if output_dir is None:
        output_dir = Path(eval_dir) / "figures"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving figures to: {output_dir}")

    # Get basin list
    all_basins = [str(b) for b in ds['basin'].values]
    if basins is None:
        basins = all_basins
    else:
        basins = [b for b in basins if b in all_basins]

    if len(basins) == 0:
        print("Error: No valid basins to plot")
        return

    print(f"Plotting for {len(basins)} basin(s): {basins}")

    # Determine plot types
    if isinstance(plot_types, str):
        plot_types = [plot_types]

    if 'all' in plot_types or plot_types == ['all']:
        plot_types = ['timeseries', 'scatter', 'fdc', 'monthly', 'metrics']

    # Determine title suffix from path
    title_suffix = ""
    if "train" in str(eval_dir).lower():
        title_suffix = "Training Period"
    elif "test" in str(eval_dir).lower():
        title_suffix = "Test Period"
    elif "valid" in str(eval_dir).lower():
        title_suffix = "Validation Period"

    # Generate plots
    print("\nGenerating plots...")

    for basin_id in basins:
        print(f"\n  Basin {basin_id}:")

        if 'timeseries' in plot_types:
            save_path = output_dir / f"timeseries_{basin_id}.png"
            plot_timeseries_with_rain(ds, basin_id, save_path, title_suffix)

        if 'scatter' in plot_types:
            save_path = output_dir / f"scatter_{basin_id}.png"
            plot_scatter(ds, basin_id, save_path, title_suffix)

        if 'fdc' in plot_types:
            save_path = output_dir / f"fdc_{basin_id}.png"
            plot_flow_duration_curve(ds, basin_id, save_path, title_suffix)

        if 'monthly' in plot_types:
            save_path = output_dir / f"monthly_{basin_id}.png"
            plot_monthly_comparison(ds, basin_id, save_path, title_suffix)

    # Multi-basin plots
    if 'metrics' in plot_types and metrics_df is not None and len(basins) > 1:
        print("\n  Multi-basin comparison:")
        save_path = output_dir / "metrics_comparison.png"
        # Filter metrics for selected basins
        metrics_subset = metrics_df.loc[basins]
        plot_metrics_comparison(metrics_subset, save_path)

    print(f"\n[OK] All figures saved to: {output_dir}")
    print(f"[OK] Visualization complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize model evaluation results from unified architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
  # Visualize all plots for test period
  python visualize_unified.py --eval-dir results/my_exp/evaluation_test

  # Visualize specific plot types
  python visualize_unified.py --eval-dir results/my_exp/evaluation_test --plot-types timeseries scatter

  # Visualize specific basins
  python visualize_unified.py --eval-dir results/my_exp/evaluation_test --basins 01013500 01022500

  # Custom output directory
  python visualize_unified.py --eval-dir results/my_exp/evaluation_test --output-dir my_figures

Available plot types:
  - timeseries: Time series with precipitation
  - scatter: Observed vs simulated scatter plot
  - fdc: Flow duration curve
  - monthly: Monthly average comparison
  - metrics: Multi-basin metrics comparison
  - all: Generate all plot types (default)
        """
    )

    parser.add_argument(
        "--eval-dir",
        type=str,
        required=True,
        help="Path to evaluation directory (e.g., results/exp_name/evaluation_test/)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figures (default: eval_dir/figures)"
    )

    parser.add_argument(
        "--plot-types",
        type=str,
        nargs='+',
        default='all',
        choices=['all', 'timeseries', 'scatter', 'fdc', 'monthly', 'metrics'],
        help="Types of plots to generate (default: all)"
    )

    parser.add_argument(
        "--basins",
        type=str,
        nargs='+',
        default=None,
        help="Basin IDs to plot (default: all basins)"
    )

    args = parser.parse_args()

    try:
        visualize_evaluation(
            eval_dir=args.eval_dir,
            output_dir=args.output_dir,
            plot_types=args.plot_types if isinstance(args.plot_types, list) else [args.plot_types],
            basins=args.basins
        )
        return 0
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
