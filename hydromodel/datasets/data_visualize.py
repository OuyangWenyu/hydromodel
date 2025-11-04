r"""
Author: Wenyu Ouyang
Date: 2025-01-22
LastEditTime: 2025-11-04 14:41:16
LastEditors: zhuanglaihong
Description: Show results of calibration and validation
FilePath: /hydromodel/hydromodel/datasets/unified_data_loader.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import xarray as xr
from pathlib import Path
from hydroutils import hydro_file, hydro_stat, hydro_plot
from hydrodatasource.reader.data_source import SelfMadeHydroDataset


def plot_precipitation(precipitation, ax=None):
    """
    Plots precipitation data from an xarray.DataArray.

    Parameters
    ----------
    precipitation : xarray.DataArray
        The precipitation data with time as the coordinate.
    ax : matplotlib.axes._axes.Axes, optional
        The matplotlib axis on which to plot. If None, a new figure and axis are created.

    Returns
    -------
    ax : matplotlib.axes._axes.Axes
        The axis with the plotted data.
    """
    # If no axis is provided, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 4))

    # Extract time and precipitation values from the xarray.DataArray
    time = precipitation.time.values
    values = precipitation.values

    # Plot the precipitation data as a bar chart
    ax.bar(
        time,  # Use time as the x-axis
        values,  # Use precipitation values as the y-axis
        color="blue",
        label="Precipitation",
        width=0.8,
    )

    # Set the x and y axis labels
    ax.set_xlabel("Date")
    ax.set_ylabel("Precipitation (mm/d)", color="black")

    # Invert the y-axis
    ax.invert_yaxis()

    # Format the x-axis to display year and month
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))

    # Rotate the x-axis labels to avoid overlap
    plt.xticks(rotation=45)

    # Add a legend
    ax.legend(loc="lower right")

    return ax


def plot_sim_and_obs_streamflow(
    date,
    sim,
    obs,
    ax=None,
    xlabel="Date",
    ylabel="Streamflow (m³/s)",
    basin_id="",
    title_suffix="",
):
    # If no external subplot is provided, create a new one
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 4))

    # Plot with better styling
    ax.plot(
        date,
        obs,
        color="#2E86AB",
        linestyle="solid",
        linewidth=1.5,
        alpha=0.9,
        label="Observed",
    )
    ax.plot(
        date,
        sim,
        color="#E63946",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
        label="Simulated",
    )

    # Calculate metrics
    valid = ~np.isnan(obs) & ~np.isnan(sim)
    if np.sum(valid) > 0:
        nse = 1 - np.sum((obs[valid] - sim[valid]) ** 2) / np.sum(
            (obs[valid] - obs[valid].mean()) ** 2
        )
        rmse = np.sqrt(np.mean((obs[valid] - sim[valid]) ** 2))
        pbias = np.sum(sim[valid] - obs[valid]) / np.sum(obs[valid]) * 100

        metrics_text = f"NSE={nse:.3f}, RMSE={rmse:.2f}, PBIAS={pbias:.1f}%"
        ax.text(
            0.02,
            0.98,
            metrics_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--")

    return ax


def plot_sim_and_obs(
    date,
    prcp,
    sim,
    obs,
    save_fig,
    xlabel="Date",
    ylabel=None,
    basin_id="",
    title_suffix="",
):
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(2, 1, height_ratios=[1, 3], hspace=0.05)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Plot precipitation data on the upper subplot
    plot_precipitation(prcp, ax=ax1)

    # Plot streamflow
    ax2.plot(
        date,
        obs,
        color="#2E86AB",
        linestyle="solid",
        linewidth=1.5,
        alpha=0.9,
        label="Observed",
    )
    ax2.plot(
        date,
        sim,
        color="#E63946",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
        label="Simulated",
    )

    # Calculate metrics
    valid = ~np.isnan(obs) & ~np.isnan(sim)
    if np.sum(valid) > 0:
        nse = 1 - np.sum((obs[valid] - sim[valid]) ** 2) / np.sum(
            (obs[valid] - obs[valid].mean()) ** 2
        )
        rmse = np.sqrt(np.mean((obs[valid] - sim[valid]) ** 2))
        pbias = np.sum(sim[valid] - obs[valid]) / np.sum(obs[valid]) * 100

        metrics_text = f"NSE={nse:.3f}, RMSE={rmse:.2f}, PBIAS={pbias:.1f}%"
        ax2.text(
            0.02,
            0.98,
            metrics_text,
            transform=ax2.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))
    import matplotlib.dates as mdates

    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Title
    title = (
        f"Streamflow Simulation - Basin {basin_id}"
        if basin_id
        else "Streamflow Simulation"
    )
    if title_suffix:
        title += f" ({title_suffix})"
    ax1.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_fig, dpi=300, bbox_inches="tight")
    plt.close()


def plot_train_iteration(likelihood, save_fig):
    # matplotlib.use("Agg")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.subplots()
    ax.plot(likelihood)
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Iteration")
    plt.savefig(save_fig, bbox_inches="tight")
    # plt.cla()
    plt.close()


# ==================================================================================
# Visualization for unified evaluation architecture
# ==================================================================================


def plot_scatter(qobs, qsim, save_path, basin_id="", title_suffix=""):
    """Plot scatter: observed vs simulated with density heatmap."""
    valid = ~np.isnan(qobs) & ~np.isnan(qsim)
    qobs, qsim = qobs[valid], qsim[valid]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Density heatmap
    from matplotlib.colors import LogNorm

    h = ax.hist2d(
        qobs, qsim, bins=50, cmap="YlOrRd", norm=LogNorm(), alpha=0.7
    )
    plt.colorbar(h[3], ax=ax, label="Count (log scale)")

    # 1:1 line
    lims = [min(qobs.min(), qsim.min()), max(qobs.max(), qsim.max())]
    ax.plot(lims, lims, "k--", linewidth=2, alpha=0.7, label="1:1 line")

    # Metrics
    nse = 1 - np.sum((qobs - qsim) ** 2) / np.sum((qobs - qobs.mean()) ** 2)
    r2 = np.corrcoef(qobs, qsim)[0, 1] ** 2
    rmse = np.sqrt(np.mean((qobs - qsim) ** 2))
    pbias = np.sum(qsim - qobs) / np.sum(qobs) * 100

    metrics_text = f"NSE = {nse:.3f}\n$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f}\nPBIAS = {pbias:.1f}%"
    ax.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Observed Streamflow (m³/s)")
    ax.set_ylabel("Simulated Streamflow (m³/s)")
    title = f"Observed vs Simulated - Basin {basin_id}"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_fdc(qobs, qsim, save_path, basin_id="", title_suffix=""):
    """Plot flow duration curve."""
    qobs_sorted = np.sort(qobs[~np.isnan(qobs)])[::-1]
    qsim_sorted = np.sort(qsim[~np.isnan(qsim)])[::-1]

    exc_obs = np.arange(1, len(qobs_sorted) + 1) / len(qobs_sorted) * 100
    exc_sim = np.arange(1, len(qsim_sorted) + 1) / len(qsim_sorted) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        exc_obs,
        qobs_sorted,
        label="Observed",
        color="#2E86AB",
        linewidth=2.5,
        alpha=0.9,
    )
    ax.plot(
        exc_sim,
        qsim_sorted,
        label="Simulated",
        color="#E63946",
        linewidth=2.5,
        alpha=0.9,
        linestyle="--",
    )

    ax.set_xlabel("Exceedance Probability (%)")
    ax.set_ylabel("Streamflow (m³/s)")
    ax.set_yscale("log")
    title = f"Flow Duration Curve - Basin {basin_id}"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_monthly(time, qobs, qsim, save_path, basin_id="", title_suffix=""):
    """Plot monthly average comparison with error bars."""
    df = pd.DataFrame({"time": time, "qobs": qobs, "qsim": qsim})
    df["month"] = df["time"].dt.month

    monthly_obs = df.groupby("month")["qobs"].mean()
    monthly_sim = df.groupby("month")["qsim"].mean()
    monthly_obs_std = df.groupby("month")["qobs"].std()
    monthly_sim_std = df.groupby("month")["qsim"].std()

    fig, ax = plt.subplots(figsize=(10, 6))
    months = np.arange(1, 13)
    width = 0.35

    ax.bar(
        months - width / 2,
        monthly_obs,
        width,
        label="Observed",
        color="#5A9FB0",
        alpha=0.8,
        yerr=monthly_obs_std,
        capsize=3,
        error_kw={"linewidth": 1.5, "ecolor": "black"},
    )
    ax.bar(
        months + width / 2,
        monthly_sim,
        width,
        label="Simulated",
        color="#E17F7F",
        alpha=0.8,
        yerr=monthly_sim_std,
        capsize=3,
        error_kw={"linewidth": 1.5, "ecolor": "black"},
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Average Streamflow (m³/s)")
    ax.set_xticks(months)
    ax.set_xticklabels(
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    )
    title = f"Monthly Average Streamflow - Basin {basin_id}"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def visualize_evaluation(
    eval_dir, output_dir=None, plot_types="all", basins=None
):
    """Visualize unified evaluation results using existing plotting functions.

    Parameters
    ----------
    eval_dir : str
        Path to evaluation directory (e.g., results/exp_name/evaluation_test/)
    output_dir : str, optional
        Output directory for figures (default: eval_dir/figures)
    plot_types : str or list
        'all', 'timeseries', 'scatter', 'fdc', 'monthly'
    basins : list, optional
        Basin IDs to plot (default: all basins)
    """
    eval_path = Path(eval_dir)

    # Load NetCDF results
    nc_files = list(eval_path.glob("*_evaluation_results.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No .nc file found in {eval_dir}")

    ds = xr.open_dataset(nc_files[0])
    print(f"Loaded: {nc_files[0].name}")

    # Setup output
    output_dir = Path(output_dir) if output_dir else eval_path / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get basins
    all_basins = [str(b) for b in ds["basin"].values]
    basins = basins if basins else all_basins
    basins = [b for b in basins if b in all_basins]

    if not basins:
        print("No valid basins to plot")
        return

    # Determine plot types
    if isinstance(plot_types, str):
        plot_types = [plot_types]
    if "all" in plot_types:
        plot_types = ["timeseries", "scatter", "fdc", "monthly"]

    # Determine title suffix from path
    title_suffix = ""
    eval_dir_lower = str(eval_dir).lower()
    if "train" in eval_dir_lower or "calibration" in eval_dir_lower:
        title_suffix = "Training Period"
    elif "test" in eval_dir_lower:
        title_suffix = "Test Period"
    elif "valid" in eval_dir_lower:
        title_suffix = "Validation Period"

    print(f"Plotting {len(basins)} basin(s): {basins}")
    print(f"Plot types: {', '.join(plot_types)}")

    # Plot each basin
    for basin_id in basins:
        basin_idx = all_basins.index(basin_id)
        print(f"\n  Basin {basin_id}:")

        # Extract data for this basin
        time = pd.to_datetime(ds["time"].values)
        qobs = ds["qobs"].values[:, basin_idx]
        qsim = ds["qsim"].values[:, basin_idx]

        # Timeseries
        if "timeseries" in plot_types:
            save_path = output_dir / f"{basin_id}_timeseries.png"
            if "prcp" in ds.variables:
                prcp_data = ds["prcp"][:, basin_idx]
                plot_sim_and_obs(
                    time,
                    prcp_data,
                    qsim,
                    qobs,
                    save_path,
                    ylabel="Streamflow (m³/s)",
                    basin_id=basin_id,
                    title_suffix=title_suffix,
                )
            else:
                fig, ax = plt.subplots(figsize=(20, 8))
                plot_sim_and_obs_streamflow(
                    time,
                    qsim,
                    qobs,
                    ax=ax,
                    ylabel="Streamflow (m³/s)",
                    basin_id=basin_id,
                    title_suffix=title_suffix,
                )
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()
            print(f"    Saved: {save_path.name}")

        # Scatter
        if "scatter" in plot_types:
            save_path = output_dir / f"{basin_id}_scatter.png"
            plot_scatter(qobs, qsim, save_path, basin_id, title_suffix)
            print(f"    Saved: {save_path.name}")

        # Flow duration curve
        if "fdc" in plot_types:
            save_path = output_dir / f"{basin_id}_fdc.png"
            plot_fdc(qobs, qsim, save_path, basin_id, title_suffix)
            print(f"    Saved: {save_path.name}")

        # Monthly
        if "monthly" in plot_types:
            save_path = output_dir / f"{basin_id}_monthly.png"
            plot_monthly(time, qobs, qsim, save_path, basin_id, title_suffix)
            print(f"    Saved: {save_path.name}")

    print(f"\n[OK] All figures saved to: {output_dir}")
