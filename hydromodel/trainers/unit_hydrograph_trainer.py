"""
Author: Wenyu Ouyang
Date: 2025-08-07
LastEditTime: 2025-08-07 14:51:12
LastEditors: Wenyu Ouyang
Description: Unit hydrograph training, optimization and evaluation functions
FilePath: /hydromodel/hydromodel/trainers/unit_hydrograph_trainer.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import itertools
import os
from typing import Optional, List
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from hydroutils.hydro_stat import (
    nse,
    flood_peak_error,
    flood_volume_error,
)


# =============================================================================
# OBJECTIVE FUNCTIONS FOR OPTIMIZATION
# =============================================================================


def objective_function_multi_event(
    uh_params,
    all_event_data,
    smoothing_factor=0.1,
    peak_violation_weight=10000.0,
    apply_peak_penalty=True,
    net_rain_name="P_eff",
    obs_flow_name="Q_obs_eff",
):
    """
    Multi-event objective function for unit hydrograph optimization.

    Parameters
    ----------
    uh_params : np.ndarray
        Unit hydrograph parameters to optimize
    all_event_data : list
        List of event dictionaries containing time series data
    smoothing_factor : float, optional
        Weight for smoothness penalty (default: 0.1)
    peak_violation_weight : float, optional
        Weight for peak flow violation penalty (default: 10000.0)
    apply_peak_penalty : bool, optional
        Whether to apply peak flow penalty (default: True)
    net_rain_name : str, optional
        Key name for net rainfall data (default: "P_eff")
    obs_flow_name : str, optional
        Key name for observed flow data (default: "Q_obs_eff")

    Returns
    -------
    float
        Total objective function value (lower is better)
    """
    from hydromodel.models.unit_hydrograph import uh_conv

    # Normalize UH parameters
    uh_normalized = np.array(uh_params) / np.sum(uh_params)

    total_rmse = 0
    total_events = 0
    total_peak_penalty = 0

    for event_data in all_event_data:
        try:
            net_rain = np.array(event_data[net_rain_name])
            obs_flow = np.array(event_data[obs_flow_name])

            if len(net_rain) == 0 or len(obs_flow) == 0:
                continue

            # Simulate flow using unit hydrograph convolution
            sim_flow = uh_conv(net_rain, uh_normalized, truncate=True)

            # Ensure same length
            min_len = min(len(sim_flow), len(obs_flow))
            sim_flow = sim_flow[:min_len]
            obs_flow = obs_flow[:min_len]

            if min_len == 0:
                continue

            # Calculate RMSE
            rmse = np.sqrt(np.mean((sim_flow - obs_flow) ** 2))
            total_rmse += rmse

            # Peak flow penalty
            if apply_peak_penalty:
                obs_peak = np.max(obs_flow)
                sim_peak = np.max(sim_flow)
                if obs_peak > 0:
                    peak_error = abs(sim_peak - obs_peak) / obs_peak
                    if peak_error > 0.5:  # More than 50% error
                        total_peak_penalty += (
                            peak_error * peak_violation_weight
                        )

            total_events += 1

        except (KeyError, TypeError, ValueError) as e:
            # Skip problematic events
            continue

    if total_events == 0:
        return 1e6  # Return large penalty for no valid events

    # Average RMSE across events
    avg_rmse = total_rmse / total_events

    # Smoothness penalty
    if len(uh_params) > 1:
        smoothness_penalty = smoothing_factor * np.sum(np.diff(uh_params) ** 2)
    else:
        smoothness_penalty = 0

    # Peak penalty
    avg_peak_penalty = (
        total_peak_penalty / total_events if total_events > 0 else 0
    )

    total_objective = avg_rmse + smoothness_penalty + avg_peak_penalty

    return total_objective


def optimize_shared_unit_hydrograph(
    all_event_data,
    n_uh=24,
    smoothing_factor=0.1,
    peak_violation_weight=10000.0,
    apply_peak_penalty=True,
    net_rain_name="P_eff",
    obs_flow_name="Q_obs_eff",
    method="SLSQP",
    max_iterations=500,
):
    """
    Optimize shared unit hydrograph parameters for multiple flood events.

    Parameters
    ----------
    all_event_data : list
        List of event dictionaries containing time series data
    n_uh : int, optional
        Length of unit hydrograph (default: 24)
    smoothing_factor : float, optional
        Weight for smoothness penalty (default: 0.1)
    peak_violation_weight : float, optional
        Weight for peak flow violation penalty (default: 10000.0)
    apply_peak_penalty : bool, optional
        Whether to apply peak flow penalty (default: True)
    net_rain_name : str, optional
        Key name for net rainfall data (default: "P_eff")
    obs_flow_name : str, optional
        Key name for observed flow data (default: "Q_obs_eff")
    method : str, optional
        Optimization method (default: "SLSQP")
    max_iterations : int, optional
        Maximum optimization iterations (default: 500)

    Returns
    -------
    dict
        Optimization results containing:
        - success: bool, whether optimization succeeded
        - x: optimized parameters
        - fun: final objective value
        - nit: number of iterations
        - message: optimization message
    """
    from hydromodel.models.unit_hydrograph import init_unit_hydrograph

    # Initialize unit hydrograph
    initial_uh = init_unit_hydrograph(
        n_uh, method="gamma", shape=2.0, scale=2.0
    )

    # Set bounds and constraints
    bounds = [(0.001, 1.0) for _ in range(n_uh)]  # Positive bounds

    # Constraint: parameters must sum to 1.0
    def constraint_sum_to_one(x):
        return np.sum(x) - 1.0

    constraints = {"type": "eq", "fun": constraint_sum_to_one}

    # Optimization
    result = minimize(
        objective_function_multi_event,
        initial_uh,
        args=(
            all_event_data,
            smoothing_factor,
            peak_violation_weight,
            apply_peak_penalty,
            net_rain_name,
            obs_flow_name,
        ),
        method=method,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iterations},
    )

    return result


def optimize_uh_for_group(
    events_in_group,
    group_name,
    weights,
    n_uh_group,
    net_rain_name="P_eff",
    obs_flow_name="Q_obs_eff",
    method="SLSQP",
    max_iterations=500,
):
    """
    Optimize unit hydrograph for a specific flood category group.

    Parameters
    ----------
    events_in_group : list
        List of events in this category
    group_name : str
        Name of the flood category (e.g., "small", "medium", "large")
    weights : dict
        Weight configuration for this group
    n_uh_group : int
        Length of unit hydrograph for this group
    net_rain_name : str, optional
        Key name for net rainfall data (default: "P_eff")
    obs_flow_name : str, optional
        Key name for observed flow data (default: "Q_obs_eff")
    method : str, optional
        Optimization method (default: "SLSQP")
    max_iterations : int, optional
        Maximum optimization iterations (default: 500)

    Returns
    -------
    dict
        Optimization results for this group
    """
    if len(events_in_group) == 0:
        return {
            "success": False,
            "message": f"No events in group {group_name}",
            "x": np.ones(n_uh_group) / n_uh_group,
            "fun": float("inf"),
            "nit": 0,
        }

    # Get optimization weights for this group
    smoothing_factor = weights.get("smoothing_factor", 0.1)
    peak_violation_weight = weights.get("peak_violation_weight", 10000.0)

    # Optimize unit hydrograph for this group
    result = optimize_shared_unit_hydrograph(
        events_in_group,
        n_uh=n_uh_group,
        smoothing_factor=smoothing_factor,
        peak_violation_weight=peak_violation_weight,
        apply_peak_penalty=True,
        net_rain_name=net_rain_name,
        obs_flow_name=obs_flow_name,
        method=method,
        max_iterations=max_iterations,
    )

    return result


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================


def evaluate_single_event_from_uh(
    event_data,
    U_optimized,
    category_name=None,
    net_rain_key="P_eff",
    obs_flow_key="Q_obs_eff",
):
    """
    Evaluate performance of unit hydrograph on a single flood event.

    Parameters
    ----------
    event_data : dict
        Event data dictionary containing time series
    U_optimized : np.ndarray
        Optimized unit hydrograph parameters
    category_name : str, optional
        Category name for this event (e.g., "small", "medium", "large")
    net_rain_key : str, optional
        Key for net rainfall data (default: "P_eff")
    obs_flow_key : str, optional
        Key for observed flow data (default: "Q_obs_eff")

    Returns
    -------
    dict or None
        Dictionary containing evaluation metrics, or None if evaluation failed
    """
    from hydromodel.models.unit_hydrograph import uh_conv

    try:
        # Extract data
        net_rain = np.array(event_data[net_rain_key])
        obs_flow = np.array(event_data[obs_flow_key])

        if len(net_rain) == 0 or len(obs_flow) == 0:
            return None

        # Ensure UH is normalized
        U_normalized = np.array(U_optimized) / np.sum(U_optimized)

        # Simulate flow
        sim_flow = uh_conv(net_rain, U_normalized, truncate=True)

        # Ensure same length
        min_len = min(len(sim_flow), len(obs_flow))
        sim_flow = sim_flow[:min_len]
        obs_flow = obs_flow[:min_len]

        if min_len == 0:
            return None

        # Calculate performance metrics
        nse_value = nse(obs_flow, sim_flow)
        rmse_value = np.sqrt(np.mean((sim_flow - obs_flow) ** 2))
        peak_error = flood_peak_error(obs_flow, sim_flow)
        volume_error = flood_volume_error(obs_flow, sim_flow)

        # Prepare result dictionary
        result = {
            "NSE": nse_value,
            "RMSE": rmse_value,
            "洪峰误差": peak_error,
            "洪量误差": volume_error,
            "事件编号": event_data.get("event_id", "unknown"),
            "降水量": np.sum(net_rain),
            "径流量": np.sum(obs_flow),
            "洪峰流量": np.max(obs_flow),
        }

        # Add category information if provided
        if category_name is not None:
            result["所属类别"] = category_name

        return result

    except (KeyError, TypeError, ValueError, IndexError) as e:
        # Return None for failed evaluations
        return None


def categorize_floods_by_peak(
    all_events_data,
    net_rain_key="P_eff",
    obs_flow_key="Q_obs_eff",
    small_medium_threshold=10.0,
    medium_large_threshold=25.0,
):
    """
    Categorize flood events based on peak flow magnitude.

    Parameters
    ----------
    all_events_data : list
        List of event dictionaries
    net_rain_key : str, optional
        Key for net rainfall data (default: "P_eff")
    obs_flow_key : str, optional
        Key for observed flow data (default: "Q_obs_eff")
    small_medium_threshold : float, optional
        Threshold between small and medium floods (default: 10.0)
    medium_large_threshold : float, optional
        Threshold between medium and large floods (default: 25.0)

    Returns
    -------
    tuple
        (categories, thresholds) where:
        - categories: list of category names for each event
        - thresholds: dict with threshold values
    """
    categories = []
    peak_flows = []

    for event_data in all_events_data:
        try:
            obs_flow = np.array(event_data[obs_flow_key])
            if len(obs_flow) > 0:
                peak_flow = np.max(obs_flow)
                peak_flows.append(peak_flow)

                # Categorize based on peak flow
                if peak_flow < small_medium_threshold:
                    categories.append("small")
                elif peak_flow < medium_large_threshold:
                    categories.append("medium")
                else:
                    categories.append("large")
            else:
                categories.append("small")  # Default for empty data
                peak_flows.append(0.0)

        except (KeyError, TypeError, ValueError):
            categories.append("small")  # Default for problematic data
            peak_flows.append(0.0)

    thresholds = {
        "small_medium": small_medium_threshold,
        "medium_large": medium_large_threshold,
    }

    return categories, thresholds


# =============================================================================
# RESULT PROCESSING AND VISUALIZATION
# =============================================================================


def save_results_to_csv(
    report_data,
    output_filename,
    title="Evaluation Report",
    sort_columns=None,
    float_format="%.6f",
    encoding="utf-8-sig",
):
    """
    Save evaluation results to CSV file with metadata.

    Parameters
    ----------
    report_data : list or pd.DataFrame
        Evaluation results data
    output_filename : str
        Output CSV file path
    title : str, optional
        Title for the report (default: "Evaluation Report")
    sort_columns : list, optional
        Columns to sort by (default: None)
    float_format : str, optional
        Float formatting (default: "%.6f")
    encoding : str, optional
        File encoding (default: "utf-8-sig")

    Returns
    -------
    pd.DataFrame
        Processed and sorted DataFrame
    """
    # Convert to DataFrame if necessary
    if isinstance(report_data, list):
        df = pd.DataFrame(report_data)
    else:
        df = report_data.copy()

    # Sort if requested
    if sort_columns:
        # Check which columns exist
        existing_columns = [col for col in sort_columns if col in df.columns]
        if existing_columns:
            ascending = [
                False if col == "NSE" else True for col in existing_columns
            ]
            df = df.sort_values(existing_columns, ascending=ascending)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # Save to CSV with metadata
    metadata_lines = [
        f"# {title}",
        f"# Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Total events: {len(df)}",
        "#",
    ]

    save_dataframe_to_csv(
        df,
        output_filename,
        metadata_lines=metadata_lines,
        float_format=float_format,
        encoding=encoding,
    )

    return df


def print_report_preview(
    report_df_sorted, title="Evaluation Report", top_n=None
):
    """
    Print preview of evaluation results.

    Parameters
    ----------
    report_df_sorted : pd.DataFrame
        Sorted evaluation results DataFrame
    title : str, optional
        Title for the report (default: "Evaluation Report")
    top_n : int, optional
        Number of top results to show (default: None shows all)
    """
    print(f"\n📊 {title}")
    print("=" * 80)

    if len(report_df_sorted) == 0:
        print("No evaluation results to display.")
        return

    # Display configuration
    display_df = report_df_sorted.head(top_n) if top_n else report_df_sorted

    # Key columns to display
    key_columns = []
    for col in [
        "事件编号",
        "所属类别",
        "NSE",
        "RMSE",
        "洪峰误差",
        "洪量误差",
        "洪峰流量",
    ]:
        if col in display_df.columns:
            key_columns.append(col)

    if key_columns:
        print(
            display_df[key_columns].to_string(index=False, float_format="%.4f")
        )
    else:
        print(display_df.to_string(index=False, float_format="%.4f"))

    # Summary statistics
    if "NSE" in report_df_sorted.columns:
        print(f"\n📈 Performance Summary:")
        print(f"   Average NSE: {report_df_sorted['NSE'].mean():.4f}")
        print(f"   Median NSE: {report_df_sorted['NSE'].median():.4f}")
        print(f"   Min NSE: {report_df_sorted['NSE'].min():.4f}")
        print(f"   Max NSE: {report_df_sorted['NSE'].max():.4f}")

    if top_n and len(report_df_sorted) > top_n:
        print(f"\n... showing top {top_n} of {len(report_df_sorted)} results")


def print_category_statistics(report_df_sorted):
    """
    Print statistics by flood category.

    Parameters
    ----------
    report_df_sorted : pd.DataFrame
        Evaluation results DataFrame with category information
    """
    if "所属类别" not in report_df_sorted.columns:
        return

    print(f"\n📊 Category Performance Statistics:")
    print("-" * 50)

    for category in ["small", "medium", "large"]:
        cat_data = report_df_sorted[report_df_sorted["所属类别"] == category]
        if len(cat_data) > 0:
            print(
                f"\n🏷️ {category.capitalize()} floods ({len(cat_data)} events):"
            )
            if "NSE" in cat_data.columns:
                print(
                    f"   NSE: {cat_data['NSE'].mean():.4f} ± {cat_data['NSE'].std():.4f}"
                )
            if "RMSE" in cat_data.columns:
                print(
                    f"   RMSE: {cat_data['RMSE'].mean():.4f} ± {cat_data['RMSE'].std():.4f}"
                )
            if "洪峰流量" in cat_data.columns:
                print(
                    f"   Peak flow range: {cat_data['洪峰流量'].min():.2f} - {cat_data['洪峰流量'].max():.2f}"
                )


def save_dataframe_to_csv(
    df,
    filepath,
    metadata_lines=None,
    float_format="%.6f",
    encoding="utf-8-sig",
    **kwargs,
):
    """
    Save DataFrame to CSV with optional metadata header.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    filepath : str
        Output file path
    metadata_lines : list, optional
        Lines to write as header comments (default: None)
    float_format : str, optional
        Float formatting (default: "%.6f")
    encoding : str, optional
        File encoding (default: "utf-8-sig")
    **kwargs
        Additional arguments for pd.DataFrame.to_csv()
    """
    # Default CSV parameters
    csv_kwargs = {
        "index": False,
        "encoding": encoding,
        "float_format": float_format,
        "header": True,
    }
    csv_kwargs.update(kwargs)

    if metadata_lines:
        with open(filepath, "w", encoding=encoding, newline="") as f:
            f.write("\n".join(metadata_lines) + "\n")
            df.to_csv(f, **csv_kwargs)
    else:
        df.to_csv(filepath, **csv_kwargs)
