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
    ylabel="Streamflow (m^3/s)",
    basin_id="",
    title_suffix="",
    time_unit=None,
    is_flood_event=False,
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

    # Configure x-axis format based on data type and time resolution
    import matplotlib.dates as mdates

    if is_flood_event and time_unit:
        # Flood event data: use finer time resolution
        if "h" in time_unit.lower() or "H" in time_unit:
            # Hourly data (1h, 3h, 6h, etc.)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            # Set appropriate interval based on time unit
            if "3h" in time_unit or "3H" in time_unit:
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            elif "1h" in time_unit or "1H" in time_unit:
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
            else:
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        else:
            # Daily data (1D, 1d, etc.)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    else:
        # Continuous long-term data: use monthly interval (original behavior)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

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
    time_unit=None,
    is_flood_event=False,
    qobs_unit="m^3/s",
    prcp_unit="mm/day",
):
    import matplotlib.dates as mdates

    # Create single subplot with twin axes
    fig, ax_flow = plt.subplots(figsize=(14, 6))
    ax_prcp = ax_flow.twinx()  # Create second y-axis for precipitation

    # Plot streamflow on left y-axis (ax_flow)
    ax_flow.plot(
        date,
        obs,
        color="#2E86AB",
        linestyle="solid",
        linewidth=1.5,
        alpha=0.9,
        label="Observed",
        zorder=3,
    )
    ax_flow.plot(
        date,
        sim,
        color="#E63946",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
        label="Simulated",
        zorder=3,
    )

    # Plot precipitation on right y-axis (ax_prcp) - inverted at top
    # Support both xarray.DataArray and numpy.ndarray for backward compatibility
    if hasattr(prcp, "time") and hasattr(prcp, "values"):
        # xarray.DataArray - extract values
        prcp_values = prcp.values
    elif isinstance(prcp, np.ndarray) or isinstance(prcp, (list, tuple)):
        # numpy.ndarray or array-like
        prcp_values = np.array(prcp)
    else:
        # Unknown type - raise error with helpful message
        raise TypeError(
            f"prcp must be xarray.DataArray or numpy.ndarray, got {type(prcp)}"
        )

    # Calculate bar width based on date range
    if len(date) > 1:
        time_diff = (date[1] - date[0]).total_seconds() / 86400  # days
        width = time_diff * 0.8
    else:
        width = 0.8

    ax_prcp.bar(
        date,
        prcp_values,
        color="skyblue",
        alpha=0.6,
        label="Precipitation",
        width=width,
        zorder=1,
    )

    # Set precipitation y-axis range so max bar only reaches halfway
    # After invert_yaxis(), the range [0, max*2] will be inverted to [max*2, 0]
    max_prcp = np.nanmax(prcp_values)
    if max_prcp > 0:
        ax_prcp.set_ylim(
            0, max_prcp * 2
        )  # Max bar will be at 50% of plot height

    ax_prcp.invert_yaxis()  # Invert so precipitation bars hang from top

    # Calculate metrics
    valid = ~np.isnan(obs) & ~np.isnan(sim)
    if np.sum(valid) > 0:
        nse = 1 - np.sum((obs[valid] - sim[valid]) ** 2) / np.sum(
            (obs[valid] - obs[valid].mean()) ** 2
        )
        rmse = np.sqrt(np.mean((obs[valid] - sim[valid]) ** 2))
        pbias = np.sum(sim[valid] - obs[valid]) / np.sum(obs[valid]) * 100

        # Display metrics on right side, vertically stacked
        metrics_text = f"NSE={nse:.3f}\nRMSE={rmse:.2f}\nPBIAS={pbias:.1f}%"
        ax_flow.text(
            0.98,
            0.70,  # Moved down from 0.97 to avoid overlap with legend
            metrics_text,
            transform=ax_flow.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            zorder=5,
        )

    # Configure axes labels and styling
    ax_flow.set_xlabel(xlabel)
    ax_flow.set_ylabel(
        ylabel if ylabel else f"Streamflow ({qobs_unit})", color="black"
    )
    ax_prcp.set_ylabel(f"Precipitation ({prcp_unit})", color="blue")

    # Set tick colors to match axis labels
    ax_flow.tick_params(axis="y", labelcolor="black")
    ax_prcp.tick_params(axis="y", labelcolor="blue")

    # Add legends
    lines_flow, labels_flow = ax_flow.get_legend_handles_labels()
    lines_prcp, labels_prcp = ax_prcp.get_legend_handles_labels()
    ax_flow.legend(
        lines_flow + lines_prcp,
        labels_flow + labels_prcp,
        loc="upper right",
        framealpha=0.9,
    )

    ax_flow.grid(True, alpha=0.3, linestyle="--", zorder=0)

    # Configure x-axis format based on data type and time resolution
    if is_flood_event and time_unit:
        # Flood event data: use finer time resolution
        if "h" in time_unit.lower() or "H" in time_unit:
            # Hourly data (1h, 3h, 6h, etc.)
            ax_flow.xaxis.set_major_formatter(
                mdates.DateFormatter("%m-%d %H:%M")
            )
            # Set appropriate interval based on time unit
            if "3h" in time_unit or "3H" in time_unit:
                ax_flow.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            elif "1h" in time_unit or "1H" in time_unit:
                ax_flow.xaxis.set_major_locator(mdates.HourLocator(interval=3))
            else:
                ax_flow.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        else:
            # Daily data (1D, 1d, etc.)
            ax_flow.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax_flow.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    else:
        # Continuous long-term data: use monthly interval (original behavior)
        ax_flow.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax_flow.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.setp(ax_flow.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Title
    title = (
        f"Streamflow Simulation - Basin {basin_id}"
        if basin_id
        else "Streamflow Simulation"
    )
    if title_suffix:
        title += f" ({title_suffix})"
    ax_flow.set_title(title, fontsize=13, fontweight="bold")

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

    ax.set_xlabel("Observed Streamflow (m^3/s)")
    ax.set_ylabel("Simulated Streamflow (m^3/s)")
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
    ax.set_ylabel("Streamflow (m^3/s)")
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
    ax.set_ylabel("Average Streamflow (m^3/s)")
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


def _identify_flood_events(flood_markers, event_ids=None):
    """
    Identify individual flood events from flood_event_marker array.

    Parameters
    ----------
    flood_markers : np.ndarray
        Array of flood event markers (1=flood period, 0=gap, NaN=warmup)
    event_ids : np.ndarray, optional
        Array of event IDs. If provided, events are grouped by unique event_id.
        This keeps events with multiple peaks together.

    Returns
    -------
    list of dict
        List of flood events, each containing:
        - start_idx: start index of the event
        - end_idx: end index of the event (exclusive)
        - length: length of the event
        - event_id: event ID (if provided)
    """
    events = []

    if event_ids is not None:
        # Group by event_id (keeps multi-peak events together)
        # IMPORTANT: Only select flood period (marker==1), exclude warmup period (marker==NaN)
        unique_ids = np.unique(event_ids[event_ids > 0])  # Exclude 0 (gaps)
        for eid in unique_ids:
            # Find all timesteps belonging to this event AND in flood period (marker==1)
            event_mask = (event_ids == eid) & (flood_markers == 1)
            event_indices = np.where(event_mask)[0]

            if len(event_indices) > 0:
                events.append(
                    {
                        "start_idx": event_indices[0],
                        "end_idx": event_indices[-1] + 1,
                        "length": event_indices[-1] - event_indices[0] + 1,
                        "event_id": int(eid),
                    }
                )
    else:
        # Fall back to marker-based identification (splits multi-peak events)
        in_event = False
        event_start = None

        for i, marker in enumerate(flood_markers):
            if marker == 1:
                if not in_event:
                    # Start of a new event
                    in_event = True
                    event_start = i
            else:
                if in_event:
                    # End of current event
                    events.append(
                        {
                            "start_idx": event_start,
                            "end_idx": i,
                            "length": i - event_start,
                        }
                    )
                    in_event = False
                    event_start = None

        # Handle case where last event extends to the end
        if in_event and event_start is not None:
            events.append(
                {
                    "start_idx": event_start,
                    "end_idx": len(flood_markers),
                    "length": len(flood_markers) - event_start,
                }
            )

    return events


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
        'all', 'timeseries', 'scatter', 'fdc', 'monthly', 'events'
        For flood event data: 'events' will plot individual flood events
        For continuous data: standard plots (timeseries, scatter, fdc, monthly)
    basins : list, optional
        Basin IDs to plot (default: all basins)
    """
    import yaml

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

    # Check if this is flood event data
    is_flood_event_data = "flood_event_marker" in ds.variables

    # Load time_unit from calibration_config.yaml if exists
    time_unit = None
    config_file = eval_path.parent / "calibration_config.yaml"
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                time_unit_list = config.get("data_cfgs", {}).get(
                    "time_unit", None
                )
                if time_unit_list and len(time_unit_list) > 0:
                    time_unit = time_unit_list[0]  # Use first time unit
        except Exception as e:
            print(f"Warning: Failed to load time_unit from config: {e}")

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
        if is_flood_event_data:
            plot_types = ["events", "scatter", "fdc"]
        else:
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
    print(
        f"Data type: {'Flood Event' if is_flood_event_data else 'Continuous'}"
    )
    print(f"Plot types: {', '.join(plot_types)}")

    # Plot each basin
    for basin_id in basins:
        basin_idx = all_basins.index(basin_id)

        # Create basin-specific output directory
        basin_output_dir = output_dir / basin_id
        basin_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Basin {basin_id}:")

        # Extract data for this basin
        time = pd.to_datetime(ds["time"].values)
        qobs = ds["qobs"].values[:, basin_idx]
        qsim = ds["qsim"].values[:, basin_idx]

        # Extract units from NetCDF attributes
        qobs_unit = ds["qobs"].attrs.get("units", "m^3/s")
        prcp_unit = (
            ds["prcp"].attrs.get("units", "mm/day")
            if "prcp" in ds.variables
            else "mm/day"
        )

        if "prcp" in ds.variables:
            prcp = ds["prcp"].values[:, basin_idx]
        else:
            prcp = None

        # Extract flood markers and event_ids first (for filtering)
        flood_markers = None
        event_ids = None
        if is_flood_event_data:
            if "flood_event_marker" in ds.variables:
                flood_markers = ds["flood_event_marker"].values[:, basin_idx]
            if "event_id" in ds.variables:
                event_ids = ds["event_id"].values[:, basin_idx]

        # For multi-basin flood event data, filter out invalid time steps
        # (where event_id==0, which are padding gaps from other basins)
        if is_flood_event_data and event_ids is not None:
            valid_mask = (
                event_ids > 0
            )  # Only keep timesteps with valid event_id

            if valid_mask.any() and valid_mask.sum() < len(event_ids):
                # Need to filter - there are some padding timesteps
                print(
                    f"    Filtering padding: {len(event_ids)} -> {valid_mask.sum()} timesteps"
                )
                time = time[valid_mask]
                qobs = qobs[valid_mask]
                qsim = qsim[valid_mask]
                if prcp is not None:
                    prcp = prcp[valid_mask]
                if flood_markers is not None:
                    flood_markers = flood_markers[valid_mask]
                event_ids = event_ids[valid_mask]

        # ==================== FLOOD EVENT DATA ====================
        if is_flood_event_data and "events" in plot_types:

            # Identify individual flood events
            events = _identify_flood_events(flood_markers, event_ids)

            if event_ids is not None:
                print(
                    f"    Found {len(events)} flood events (grouped by event_id)"
                )
            else:
                print(f"    Found {len(events)} flood peaks")

            # Create events subdirectory
            events_dir = basin_output_dir / "flood_events"
            events_dir.mkdir(parents=True, exist_ok=True)

            # Plot each flood event
            for event_idx, event in enumerate(events):
                start = event["start_idx"]
                end = event["end_idx"]

                # Extract event data
                event_time = time[start:end]
                event_qobs = qobs[start:end]
                event_qsim = qsim[start:end]
                event_prcp = prcp[start:end] if prcp is not None else None

                # IMPORTANT: Only plot flood period (marker==1), exclude gaps and warmup
                # Extract flood markers for this event range
                event_markers = flood_markers[start:end]

                # Create mask: only keep data where marker==1 (flood period)
                flood_mask = event_markers == 1

                # Filter data: keep only flood period, set others to NaN
                event_qobs_filtered = np.where(flood_mask, event_qobs, np.nan)
                event_qsim_filtered = np.where(flood_mask, event_qsim, np.nan)
                event_prcp_filtered = (
                    np.where(flood_mask, event_prcp, np.nan)
                    if event_prcp is not None
                    else None
                )

                # Also filter time to only flood period for better x-axis display
                event_time_filtered = event_time[flood_mask]
                event_qobs_final = event_qobs_filtered[flood_mask]
                event_qsim_final = event_qsim_filtered[flood_mask]
                event_prcp_final = (
                    event_prcp_filtered[flood_mask]
                    if event_prcp_filtered is not None
                    else None
                )

                # Skip if no valid data
                valid_obs = ~np.isnan(event_qobs_final)
                if not valid_obs.any():
                    continue

                # Generate filename with start and end time
                # Use first and last valid timestamp from filtered data
                time_start = pd.Timestamp(event_time_filtered[0])
                time_end = pd.Timestamp(event_time_filtered[-1])

                # Format based on time resolution
                if time_unit and (
                    "h" in time_unit.lower() or "H" in time_unit
                ):
                    # Hourly data: include hour and minute
                    start_str = time_start.strftime("%Y%m%d")
                    end_str = time_end.strftime("%Y%m%d")
                else:
                    # Daily or longer: only date
                    start_str = time_start.strftime("%Y%m%d")
                    end_str = time_end.strftime("%Y%m%d")

                save_path = events_dir / f"event_{start_str}_{end_str}.png"

                # Create title with time range
                if time_unit and (
                    "h" in time_unit.lower() or "H" in time_unit
                ):
                    # Hourly data: show date and hour
                    title_time = f"{time_start.strftime('%Y-%m-%d')} to {time_end.strftime('%Y-%m-%d')}"
                else:
                    # Daily data: show only date
                    title_time = f"{time_start.strftime('%Y-%m-%d')} to {time_end.strftime('%Y-%m-%d')}"

                if event_prcp_final is not None:
                    # Plot with precipitation (using filtered data)
                    plot_sim_and_obs(
                        event_time_filtered,
                        event_prcp_final,
                        event_qsim_final,
                        event_qobs_final,
                        save_path,
                        basin_id=basin_id,
                        title_suffix=title_time,
                        time_unit=time_unit,
                        is_flood_event=True,
                        qobs_unit=qobs_unit,
                        prcp_unit=prcp_unit,
                    )
                else:
                    # Plot without precipitation (using filtered data)
                    fig, ax = plt.subplots(figsize=(14, 6))
                    plot_sim_and_obs_streamflow(
                        event_time_filtered,
                        event_qsim_final,
                        event_qobs_final,
                        ax=ax,
                        ylabel=f"Streamflow ({qobs_unit})",
                        basin_id=basin_id,
                        title_suffix=title_time,
                        time_unit=time_unit,
                        is_flood_event=True,
                    )
                    plt.tight_layout()
                    plt.savefig(save_path, dpi=300, bbox_inches="tight")
                    plt.close()

                if (event_idx + 1) % 10 == 0:
                    print(f"    Plotted {event_idx+1}/{len(events)} events")

            print(
                f"    Saved {len(events)} flood events to: {events_dir.name}/"
            )

        # ==================== CONTINUOUS DATA OR LEGACY PLOTS ====================
        # Timeseries (for continuous data)
        if "timeseries" in plot_types:
            save_path = basin_output_dir / f"{basin_id}_timeseries.png"
            if prcp is not None:
                plot_sim_and_obs(
                    time,
                    prcp,
                    qsim,
                    qobs,
                    save_path,
                    basin_id=basin_id,
                    title_suffix=title_suffix,
                    qobs_unit=qobs_unit,
                    prcp_unit=prcp_unit,
                )
            else:
                fig, ax = plt.subplots(figsize=(20, 8))
                plot_sim_and_obs_streamflow(
                    time,
                    qsim,
                    qobs,
                    ax=ax,
                    ylabel=f"Streamflow ({qobs_unit})",
                    basin_id=basin_id,
                    title_suffix=title_suffix,
                )
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()
            print(f"    Saved: {save_path.name}")

        # Scatter (for both flood event and continuous data)
        if "scatter" in plot_types:
            save_path = basin_output_dir / f"{basin_id}_scatter.png"
            # For flood event data, only use flood periods
            if is_flood_event_data and flood_markers is not None:
                # Use already filtered flood_markers
                flood_mask = flood_markers == 1
                qobs_plot = qobs[flood_mask]
                qsim_plot = qsim[flood_mask]
            else:
                qobs_plot = qobs
                qsim_plot = qsim
            plot_scatter(
                qobs_plot, qsim_plot, save_path, basin_id, title_suffix
            )
            print(f"    Saved: {save_path.name}")

        # Flow duration curve (for both flood event and continuous data)
        if "fdc" in plot_types:
            save_path = basin_output_dir / f"{basin_id}_fdc.png"
            # For flood event data, only use flood periods
            if is_flood_event_data and flood_markers is not None:
                # Use already filtered flood_markers
                flood_mask = flood_markers == 1
                qobs_plot = qobs[flood_mask]
                qsim_plot = qsim[flood_mask]
            else:
                qobs_plot = qobs
                qsim_plot = qsim
            plot_fdc(qobs_plot, qsim_plot, save_path, basin_id, title_suffix)
            print(f"    Saved: {save_path.name}")

        # Monthly (only for continuous data, not meaningful for flood events)
        if "monthly" in plot_types and not is_flood_event_data:
            save_path = basin_output_dir / f"{basin_id}_monthly.png"
            plot_monthly(time, qobs, qsim, save_path, basin_id, title_suffix)
            print(f"    Saved: {save_path.name}")

    print(f"\n[OK] All figures saved to: {output_dir}")
