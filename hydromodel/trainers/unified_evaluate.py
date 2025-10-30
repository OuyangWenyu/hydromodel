r"""
Author: Wenyu Ouyang
Date: 2025-08-11
LastEditTime: 2025-08-11 20:00:00
LastEditors: Wenyu Ouyang
Description: Unified evaluation interface for all hydrological models
FilePath: /hydromodel/hydromodel/trainers/unified_evaluate.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import json
import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List, Optional, Union, Any
from collections import OrderedDict

from hydroutils import hydro_stat
from hydromodel.models.model_config import read_model_param_dict
from hydromodel.datasets.unified_data_loader import UnifiedDataLoader
from hydromodel.trainers.unified_simulate import UnifiedSimulator
from hydrodatasource.utils.utils import streamflow_unit_conv
from hydromodel.datasets.data_preprocess import get_basin_area


def evaluate(config: Dict[str, Any], param_dir: str = None, **kwargs) -> Dict[str, Any]:
    """
    Unified evaluation interface for all hydrological models.

    Parameters
    ----------
    config : Dict
        Configuration dictionary containing all settings.
        Must contain 'data_cfgs', 'model_cfgs', 'training_cfgs' keys
    param_dir : str, optional
        Directory where calibrated parameters are stored.
        If None, will use output_dir from config
    **kwargs
        Additional arguments

    Returns
    -------
    Dict[str, Any]
        Dictionary containing evaluation results
    """
    # Validate config structure
    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary")

    if (
        "data_cfgs" not in config
        or "model_cfgs" not in config
        or "training_cfgs" not in config
    ):
        raise ValueError(
            "Config dictionary must contain 'data_cfgs', 'model_cfgs', and 'training_cfgs' keys"
        )

    data_config = config["data_cfgs"]
    model_cfgs = config["model_cfgs"]
    model_config = {
        "name": model_cfgs.get("model_name"),
        **model_cfgs.get("model_params", {}),
    }
    training_config = config["training_cfgs"]

    # Get evaluation period from config
    eval_period = kwargs.get("eval_period", data_config.get("test_period"))
    if eval_period is None:
        raise ValueError("Evaluation period not specified in config or kwargs")

    # Update data_config with evaluation period
    eval_data_config = data_config.copy()
    eval_data_config["train_period"] = eval_period
    eval_data_config["test_period"] = eval_period

    # Determine parameter directory
    if param_dir is None:
        output_dir = os.path.join(
            training_config.get("output_dir", "results"),
            training_config.get("experiment_name", "experiment"),
        )
        param_dir = output_dir

    # Load evaluation data
    warmup_length = eval_data_config.get("warmup_length", 365)
    data_loader = UnifiedDataLoader(eval_data_config)
    p_and_e, qobs = data_loader.load_data()

    # Store observation data
    is_event_data = eval_data_config.get("is_event_data", False)
    if is_event_data:
        true_obs = qobs  # Keep complete time series for event data
    else:
        true_obs = qobs[warmup_length:, :, :]  # Remove warmup period

    # Load calibrated parameters
    basin_ids = eval_data_config.get("basin_ids", [])
    basin_ids = [str(bid) for bid in basin_ids]
    model_name = model_config["name"]

    # Get parameter range file
    # Priority: 1) from config, 2) from param_dir, 3) use default (None -> MODEL_PARAM_DICT)
    param_range_file = training_config.get("param_range_file")
    if param_range_file is None or not os.path.exists(param_range_file):
        # Try to find param_range.yaml in param_dir
        candidate_file = os.path.join(param_dir, "param_range.yaml")
        if os.path.exists(candidate_file):
            param_range_file = candidate_file
        else:
            # Use None to trigger default MODEL_PARAM_DICT loading
            param_range_file = None
            print("Note: param_range.yaml not found. Using default parameter ranges from MODEL_PARAM_DICT.")

    # Load param_range (will use default MODEL_PARAM_DICT if file_path is None)
    try:
        param_range = read_model_param_dict(param_range_file)
        parameter_names = param_range[model_name]["param_name"]
        has_param_range = True
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: Could not load parameter ranges for model '{model_name}': {e}")
        print("Will proceed without parameter denormalization.")
        param_range = None
        has_param_range = False
        parameter_names = None

    # Create base model config
    base_model_config = {
        "type": "lumped",
        "model_name": model_name,
        "model_params": model_config.copy(),
        "parameters": {},
    }

    # Get basin configurations
    basin_configs = data_loader.get_basin_configs()

    # Evaluate each basin
    results = {}
    all_qsim = []
    all_qobs = []

    for i, basin_id in enumerate(basin_ids):
        # Load calibrated parameters for this basin
        params = _load_basin_parameters(
            basin_id, param_dir, parameter_names, model_name
        )

        # If we don't have parameter_names yet, get them from the first loaded params
        if parameter_names is None and params is not None:
            parameter_names = list(params.keys())

        # Update model config with parameters
        model_cfg = base_model_config.copy()
        model_cfg["parameters"] = params

        # Get basin config
        basin_config = basin_configs.get(basin_id, {"basin_area": 1000.0})

        # Create simulator
        simulator = UnifiedSimulator(model_cfg, basin_config)

        # Run simulation
        sim_results = simulator.simulate(
            inputs=p_and_e[:, i : i + 1, :],
            warmup_length=warmup_length,
            is_event_data=is_event_data,
        )

        # Extract qsim
        if isinstance(sim_results, dict) and "qsim" in sim_results:
            qsim = sim_results["qsim"]
        elif isinstance(sim_results, np.ndarray):
            qsim = sim_results
        else:
            qsim = (
                list(sim_results.values())[0]
                if isinstance(sim_results, dict)
                else sim_results
            )

        # Store simulation results
        all_qsim.append(qsim)
        all_qobs.append(true_obs[:, i : i + 1, :])

        # Calculate metrics for this basin
        # hydro_stat.stat_error expects (ngrid, nt) shape
        # We have (time, 1, 1) for single basin, need to reshape to (1, time)
        qobs_basin = true_obs[:, i : i + 1, :].squeeze()  # (time,)
        qsim_basin = qsim.squeeze()  # (time,)

        # Reshape to (1, time) for hydro_stat.stat_error
        if qobs_basin.ndim == 1:
            qobs_basin = qobs_basin.reshape(1, -1)
        if qsim_basin.ndim == 1:
            qsim_basin = qsim_basin.reshape(1, -1)

        basin_metrics = hydro_stat.stat_error(
            qobs_basin,
            qsim_basin,
        )

        results[basin_id] = {
            "metrics": basin_metrics,
            "parameters": params,
        }

    # Stack results for all basins
    all_qsim = np.concatenate(all_qsim, axis=1)
    all_qobs = np.concatenate(all_qobs, axis=1)

    # Save results
    eval_output_dir = kwargs.get("eval_output_dir", param_dir)
    _save_evaluation_results(
        eval_output_dir,
        model_name,
        all_qsim,
        all_qobs,
        basin_ids,
        p_and_e,
        data_loader.ds,
        warmup_length,
        is_event_data,
        eval_data_config,
    )

    # Save metrics summary
    _save_metrics_summary(eval_output_dir, results, basin_ids)

    # Save parameters summary
    _save_parameters_summary(
        eval_output_dir, results, basin_ids, parameter_names, param_range, model_name
    )

    return results


def _load_basin_parameters(
    basin_id: str, param_dir: str, parameter_names: List[str] = None, model_name: str = None
) -> OrderedDict:
    """
    Load calibrated parameters for a basin.

    Parameters
    ----------
    basin_id : str
        Basin ID
    param_dir : str
        Directory where parameters are stored
    parameter_names : List[str], optional
        List of parameter names. If None, will try to infer from data.
    model_name : str, optional
        Model name

    Returns
    -------
    OrderedDict
        Ordered dictionary of parameters
    """
    # Try loading from different possible sources
    # 1. Try loading from basin-specific sceua results
    sceua_file = os.path.join(param_dir, f"{basin_id}_sceua.csv")
    if os.path.exists(sceua_file):
        df = pd.read_csv(sceua_file)
        if "like1" in df.columns and len(df) > 0:
            best_run = df.loc[df["like1"].idxmin()]
            params = OrderedDict()

            # If parameter_names is provided, use it
            if parameter_names is not None:
                # Try parx1, parx2, ... format
                for j, name in enumerate(parameter_names):
                    col = f"parx{j+1}"
                    if col in df.columns:
                        params[name] = float(best_run[col])
                    else:
                        # Try par{name} format
                        col = f"par{name}"
                        if col in df.columns:
                            params[name] = float(best_run[col])
                if len(params) == len(parameter_names):
                    return params
            else:
                # Try to infer parameters from columns
                # Look for parx1, parx2, ... columns
                parx_cols = [col for col in df.columns if col.startswith("parx")]
                if parx_cols:
                    parx_cols = sorted(parx_cols, key=lambda x: int(x[4:]))
                    for col in parx_cols:
                        param_name = col  # Use column name as parameter name
                        params[param_name] = float(best_run[col])
                    return params
                # Look for par{name} columns
                par_cols = [col for col in df.columns if col.startswith("par") and col != "pareto_front"]
                if par_cols:
                    for col in par_cols:
                        param_name = col[3:]  # Remove "par" prefix
                        params[param_name] = float(best_run[col])
                    return params

    # 2. Try loading from calibrate_params.txt
    params_file = os.path.join(param_dir, f"{basin_id}_calibrate_params.txt")
    if os.path.exists(params_file):
        params_array = np.loadtxt(params_file)
        if len(params_array.shape) > 1:
            params_array = params_array[1:].flatten()

        if parameter_names is not None:
            params = OrderedDict(zip(parameter_names, params_array))
        else:
            # Use generic parameter names if no parameter_names provided
            params = OrderedDict((f"param_{i}", val) for i, val in enumerate(params_array))
        return params

    # 3. Try loading from unified results
    results_file = os.path.join(param_dir, "calibration_results.json")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            calib_results = json.load(f)
        if basin_id in calib_results:
            basin_result = calib_results[basin_id]
            if "best_params" in basin_result and model_name in basin_result["best_params"]:
                params_dict = basin_result["best_params"][model_name]
                return OrderedDict(params_dict)

    raise FileNotFoundError(
        f"Could not find calibrated parameters for basin {basin_id} in {param_dir}"
    )


def _save_evaluation_results(
    output_dir: str,
    model_name: str,
    qsim: np.ndarray,
    qobs: np.ndarray,
    basin_ids: List[str],
    p_and_e: np.ndarray,
    ds_original: xr.Dataset,
    warmup_length: int,
    is_event_data: bool,
    data_config: Dict,
):
    """Save evaluation results to NetCDF file."""
    os.makedirs(output_dir, exist_ok=True)

    # Get time and basin coordinates
    time_start = warmup_length if not is_event_data else 0
    times = ds_original["time"].data[time_start:]
    basins = np.array([str(bid) for bid in basin_ids])

    # Ensure correct dimensions
    if qsim.ndim == 3:
        qsim = qsim.squeeze(axis=2)
    if qobs.ndim == 3:
        qobs = qobs.squeeze(axis=2)

    # Create xarray Dataset
    ds = xr.Dataset(
        {
            "qsim": (["time", "basin"], qsim),
            "qobs": (["time", "basin"], qobs),
            "prcp": (["time", "basin"], p_and_e[time_start:, :, 0]),
            "pet": (["time", "basin"], p_and_e[time_start:, :, 1]),
        },
        coords={"time": times, "basin": basins},
    )

    # Add attributes
    ds["qsim"].attrs["units"] = "mm/day"
    ds["qsim"].attrs["long_name"] = "Simulated streamflow"
    ds["qobs"].attrs["units"] = "mm/day"
    ds["qobs"].attrs["long_name"] = "Observed streamflow"
    ds["prcp"].attrs["units"] = "mm/day"
    ds["prcp"].attrs["long_name"] = "Precipitation"
    ds["pet"].attrs["units"] = "mm/day"
    ds["pet"].attrs["long_name"] = "Potential evapotranspiration"

    # Convert units if needed
    if "data_source_type" in data_config:
        data_type = data_config["data_source_type"]
        data_dir = data_config.get("data_source_path", "")
        basin_area = get_basin_area(basins, data_type, data_dir)

        # Convert to m³/s
        target_unit = "m^3/s"
        ds_qsim = streamflow_unit_conv(
            ds[["qsim"]], basin_area, target_unit=target_unit, inverse=True
        )
        ds_qobs = streamflow_unit_conv(
            ds[["qobs"]], basin_area, target_unit=target_unit, inverse=True
        )

        # Update dataset
        ds["qsim"] = ds_qsim["qsim"]
        ds["qobs"] = ds_qobs["qobs"]

    # Save to NetCDF
    output_file = os.path.join(output_dir, f"{model_name}_evaluation_results.nc")
    ds.to_netcdf(output_file)
    print(f"Evaluation results saved to: {output_file}")


def _save_metrics_summary(
    output_dir: str, results: Dict, basin_ids: List[str]
):
    """Save metrics summary to CSV file."""
    metrics_list = []
    for basin_id in basin_ids:
        if basin_id in results:
            metrics = results[basin_id]["metrics"]
            # Handle case where metrics values are arrays (extract first element)
            processed_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    processed_metrics[key] = value.flatten()[0] if value.size > 0 else value
                else:
                    processed_metrics[key] = value
            metrics_list.append(processed_metrics)

    if not metrics_list:
        print("Warning: No metrics to save!")
        return

    metrics_df = pd.DataFrame(metrics_list, index=basin_ids)
    metrics_file = os.path.join(output_dir, "basins_metrics.csv")
    metrics_df.to_csv(metrics_file, sep=",", index=True, header=True)

    print("\n" + "=" * 80)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 80)
    print(f"Metrics saved to: {metrics_file}")
    print("\nMetrics for each basin:")
    print("-" * 80)

    # Print with better formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(metrics_df)

    # Also print summary statistics
    if len(basin_ids) > 1:
        print("\n" + "-" * 80)
        print("Summary Statistics Across All Basins:")
        print("-" * 80)
        print(metrics_df.describe().loc[['mean', 'std', 'min', 'max']])

    print("=" * 80 + "\n")


def _save_parameters_summary(
    output_dir: str,
    results: Dict,
    basin_ids: List[str],
    parameter_names: List[str],
    param_range: Dict = None,
    model_name: str = None,
):
    """Save parameters summary to CSV file."""
    # Check if we have parameter information
    if parameter_names is None:
        print("Warning: No parameter names available. Skipping parameter summary.")
        return

    # Normalized parameters
    norm_params_list = []
    denorm_params_list = []
    has_denorm = param_range is not None and model_name is not None

    for basin_id in basin_ids:
        if basin_id in results:
            params = results[basin_id]["parameters"]
            norm_params = [params[name] for name in parameter_names]
            norm_params_list.append(norm_params)

            # Denormalize parameters if param_range is available
            if has_denorm:
                try:
                    param_ranges = param_range[model_name]["param_range"]
                    denorm_params = [
                        (param_ranges[name][1] - param_ranges[name][0]) * params[name]
                        + param_ranges[name][0]
                        for name in parameter_names
                    ]
                    denorm_params_list.append(denorm_params)
                except (KeyError, TypeError):
                    has_denorm = False
                    print(f"Warning: Could not denormalize parameters for basin {basin_id}")

    # Save normalized parameters
    norm_params_df = pd.DataFrame(norm_params_list, columns=parameter_names, index=basin_ids)
    norm_file = os.path.join(output_dir, "basins_norm_params.csv")
    norm_params_df.to_csv(norm_file, sep=",", index=True, header=True)
    print(f"Parameters summary saved to: {norm_file}")

    # Save denormalized parameters if available
    if has_denorm and denorm_params_list:
        denorm_params_df = pd.DataFrame(denorm_params_list, columns=parameter_names, index=basin_ids)
        denorm_file = os.path.join(output_dir, "basins_denorm_params.csv")
        denorm_params_df.to_csv(denorm_file, sep=",", index=True, header=True)
        print(f"Denormalized parameters saved to: {denorm_file}")

        print("-" * 50)
        print("Normalized Parameters:")
        print(norm_params_df)
        print("-" * 50)
        print("Denormalized Parameters:")
        print(denorm_params_df)
    else:
        print("Note: Parameter denormalization skipped (param_range.yaml not available)")
        print("-" * 50)
        print("Parameters (normalized [0,1] or as-is):")
        print(norm_params_df)
