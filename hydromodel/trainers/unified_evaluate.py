r"""
Author: zhuanglaihong
Date: 2025-10-31
LastEditTime: 2025-11-01 20:00:00
LastEditors: zhuanglaihong
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


class UnifiedEvaluator:
    """
    Unified evaluator for all hydrological models.

    This class provides a unified interface for evaluating calibrated models,
    similar to UnifiedModelSetup for calibration.

    Key features:
    - Loads calibrated parameters from multiple sources
    - Evaluates models on specified periods
    - Calculates performance metrics
    - Saves results in unified NetCDF format
    """

    def __init__(
        self,
        data_config: Dict[str, Any],
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        eval_period: Optional[List[str]] = None,
        param_dir: Optional[str] = None,
    ):
        """
        Initialize UnifiedEvaluator.

        Parameters
        ----------
        data_config : Dict
            Data configuration
        model_config : Dict
            Model configuration
        training_config : Dict
            Training configuration
        eval_period : List[str], optional
            Evaluation period [start, end]
        param_dir : str, optional
            Directory containing calibrated parameters
        """
        self.data_config = data_config.copy()
        self.model_config = model_config
        self.training_config = training_config

        # Set evaluation period
        if eval_period is None:
            eval_period = data_config.get("test_period")
        if eval_period is None:
            raise ValueError(
                "Evaluation period not specified in config or init"
            )

        # Update data_config with evaluation period
        self.data_config["train_period"] = eval_period
        self.data_config["test_period"] = eval_period

        # Set parameter directory
        if param_dir is None:
            output_dir = os.path.join(
                training_config.get("output_dir", "results"),
                training_config.get("experiment_name", "experiment"),
            )
            self.param_dir = output_dir
        else:
            self.param_dir = param_dir

        # Basic settings
        self.warmup_length = self.data_config.get("warmup_length", 365)
        self.is_event_data = self.data_config.get("is_event_data", False)
        self.model_name = model_config["name"]

        # Load data
        self._load_data()

        # Load parameter ranges
        self._load_parameter_ranges()

        # Get basin configurations
        self.basin_configs = self.data_loader.get_basin_configs()
        self.basin_ids = [
            str(bid) for bid in self.data_config.get("basin_ids", [])
        ]

        # Store full data for multi-basin support (already saved in _load_data)
        self.p_and_e_full = self.p_and_e

    def _load_data(self):
        """Load evaluation data using UnifiedDataLoader."""
        self.data_loader = UnifiedDataLoader(self.data_config)
        self.p_and_e, self.qobs_full = self.data_loader.load_data()

        # Store observation data
        if self.is_event_data:
            self.true_obs = (
                self.qobs_full
            )  # Keep complete time series for event data
        else:
            self.true_obs = self.qobs_full[
                self.warmup_length :, :, :
            ]  # Remove warmup period

    def _load_parameter_ranges(self):
        """Load parameter ranges for denormalization."""
        # Get parameter range file
        param_range_file = self.training_config.get("param_range_file")
        if param_range_file is None or not os.path.exists(param_range_file):
            candidate_file = os.path.join(self.param_dir, "param_range.yaml")
            if os.path.exists(candidate_file):
                param_range_file = candidate_file
            else:
                param_range_file = None
                print(
                    "Note: param_range.yaml not found. Using default parameter ranges."
                )

        # Load param_range
        try:
            self.param_range = read_model_param_dict(param_range_file)
            self.parameter_names = self.param_range[self.model_name][
                "param_name"
            ]
            self.has_param_range = True
        except (FileNotFoundError, KeyError) as e:
            print(
                f"Warning: Could not load parameter ranges for model '{self.model_name}': {e}"
            )
            print("Will proceed without parameter denormalization.")
            self.param_range = None
            self.parameter_names = None
            self.has_param_range = False

    def evaluate_basin(self, basin_id: str) -> Dict[str, Any]:
        """
        Evaluate model for a single basin.

        Parameters
        ----------
        basin_id : str
            Basin ID

        Returns
        -------
        Dict
            Evaluation results including metrics, qsim, qobs
        """
        i = self.basin_ids.index(basin_id)

        # Load calibrated parameters
        params = _load_basin_parameters(
            basin_id, self.param_dir, self.parameter_names, self.model_name
        )

        # Update parameter_names if first basin
        if self.parameter_names is None and params is not None:
            self.parameter_names = list(params.keys())

        # Create model config
        base_model_config = {
            "type": "lumped",
            "model_name": self.model_name,
            "model_params": self.model_config.copy(),
            "parameters": params,
        }

        # Get basin config
        basin_config = self.basin_configs.get(basin_id, {"basin_area": 1000.0})

        # Create simulator
        simulator = UnifiedSimulator(base_model_config, basin_config)

        # Run simulation
        sim_results = simulator.simulate(
            inputs=self.p_and_e[:, i : i + 1, :],
            warmup_length=self.warmup_length,
            is_event_data=self.is_event_data,
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

        # Get observation
        qobs_basin = self.true_obs[:, i : i + 1, :]

        # Calculate metrics
        qobs_reshaped = qobs_basin.squeeze()
        qsim_reshaped = qsim.squeeze()

        if qobs_reshaped.ndim == 1:
            qobs_reshaped = qobs_reshaped.reshape(1, -1)
        if qsim_reshaped.ndim == 1:
            qsim_reshaped = qsim_reshaped.reshape(1, -1)

        basin_metrics = hydro_stat.stat_error(
            qobs_reshaped,
            qsim_reshaped,
        )

        return {
            "metrics": basin_metrics,
            "parameters": params,
            "qsim": qsim,
            "qobs": qobs_basin,
        }

    def evaluate_all(
        self, save_results: bool = True, eval_output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model for all basins.

        Parameters
        ----------
        save_results : bool
            Whether to save evaluation results
        eval_output_dir : str, optional
            Output directory for results

        Returns
        -------
        Dict
            Evaluation results for all basins
        """
        results = {}
        all_qsim = []
        all_qobs = []

        total_basins = len(self.basin_ids)
        print(f"\n📊 {'='*60}")
        print(f"📊 Starting evaluation for {total_basins} basin(s)")
        print(f"📊 {'='*60}\n")

        for i, basin_id in enumerate(self.basin_ids):
            print(f"▶️  Basin {i+1}/{total_basins}: {basin_id}")

            basin_result = self.evaluate_basin(basin_id)
            results[basin_id] = {
                "metrics": basin_result["metrics"],
                "parameters": basin_result["parameters"],
            }
            all_qsim.append(basin_result["qsim"])
            all_qobs.append(basin_result["qobs"])

            # Print key metrics
            metrics = basin_result["metrics"]
            nse = metrics.get("NSE", [np.nan])[0]
            print(f"  NSE: {nse:.4f}")
            print(f"✅ Basin {i+1}/{total_basins} completed: {basin_id}\n")

        # Stack results
        all_qsim = np.concatenate(all_qsim, axis=1)
        all_qobs = np.concatenate(all_qobs, axis=1)

        # Save results if requested
        if save_results:
            print(f"💾 Saving evaluation results...")
            output_dir = eval_output_dir or self.param_dir
            self._save_all_results(output_dir, results, all_qsim, all_qobs)

        print(f"\n🎉 {'='*60}")
        print(f"🎉 Evaluation completed successfully!")
        print(f"🎉 Total basins evaluated: {total_basins}")
        print(f"🎉 {'='*60}\n")

        return results

    def _save_all_results(
        self,
        output_dir: str,
        results: Dict,
        all_qsim: np.ndarray,
        all_qobs: np.ndarray,
    ):
        """Save all evaluation results."""
        _save_evaluation_results(
            output_dir,
            self.model_name,
            all_qsim,
            all_qobs,
            self.basin_ids,
            self.p_and_e,
            self.data_loader.ds,
            self.warmup_length,
            self.is_event_data,
            self.data_config,
            data_path=self.data_loader.data_path,
        )

        _save_metrics_summary(output_dir, results, self.basin_ids)

        _save_parameters_summary(
            output_dir,
            results,
            self.basin_ids,
            self.parameter_names,
            self.param_range,
            self.model_name,
        )


def evaluate(
    config: Dict[str, Any], param_dir: str = None, **kwargs
) -> Dict[str, Any]:
    """
    Unified evaluation interface for all hydrological models.

    This is a convenience function that wraps UnifiedEvaluator class for backward compatibility.

    Parameters
    ----------
    config : Dict
        Configuration dictionary containing all settings.
        Must contain 'data_cfgs', 'model_cfgs', 'training_cfgs' keys
    param_dir : str, optional
        Directory where calibrated parameters are stored.
        If None, will use output_dir from config
    **kwargs
        Additional arguments including:
        - eval_period: Evaluation period
        - eval_output_dir: Output directory for results

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

    # Create UnifiedEvaluator instance
    evaluator = UnifiedEvaluator(
        data_config=data_config,
        model_config=model_config,
        training_config=training_config,
        eval_period=eval_period,
        param_dir=param_dir,
    )

    # Run evaluation
    eval_output_dir = kwargs.get("eval_output_dir", None)
    results = evaluator.evaluate_all(
        save_results=True, eval_output_dir=eval_output_dir
    )

    return results


# ============================================================================
# Helper Functions
# ============================================================================


def _load_basin_parameters(
    basin_id: str,
    param_dir: str,
    parameter_names: List[str] = None,
    model_name: str = None,
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

    # 0. PRIORITY: Try loading from unified calibration_results.json first
    # This is the most reliable source and works for all algorithms (SCE-UA, GA, scipy)
    results_file = os.path.join(param_dir, "calibration_results.json")
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                calib_results = json.load(f)
            if basin_id in calib_results:
                basin_result = calib_results[basin_id]
                if (
                    "best_params" in basin_result
                    and model_name in basin_result["best_params"]
                ):
                    params_dict = basin_result["best_params"][model_name]
                    print(
                        f"Loaded parameters for basin {basin_id} from calibration_results.json"
                    )
                    return OrderedDict(params_dict)
        except Exception as e:
            print(
                f"Warning: Failed to load from calibration_results.json: {e}"
            )

    # 1. Try loading from GA results
    ga_file = os.path.join(param_dir, f"{basin_id}_ga.csv")
    if os.path.exists(ga_file):
        try:
            df = pd.read_csv(ga_file)
            if "objective_value" in df.columns and len(df) > 0:
                # Find row with minimum objective value (best generation)
                best_run = df.loc[df["objective_value"].idxmin()]
                params = OrderedDict()

                # Extract parameters from param_{name} columns
                param_cols = [
                    col for col in df.columns if col.startswith("param_")
                ]
                if param_cols and parameter_names is not None:
                    for name in parameter_names:
                        col = f"param_{name}"
                        if col in df.columns:
                            params[name] = float(best_run[col])
                    if len(params) == len(parameter_names):
                        print(
                            f"Loaded parameters for basin {basin_id} from GA results"
                        )
                        return params
        except Exception as e:
            print(f"Warning: Failed to load from GA results: {e}")

    # 2. Try loading from scipy results
    scipy_file = os.path.join(param_dir, f"{basin_id}_scipy.csv")
    if os.path.exists(scipy_file):
        try:
            df = pd.read_csv(scipy_file)
            if "objective_value" in df.columns and len(df) > 0:
                # Find row with minimum objective value
                best_run = df.loc[df["objective_value"].idxmin()]
                params = OrderedDict()

                # Extract parameters from param_{name} columns
                param_cols = [
                    col for col in df.columns if col.startswith("param_")
                ]
                if param_cols and parameter_names is not None:
                    for name in parameter_names:
                        col = f"param_{name}"
                        if col in df.columns:
                            params[name] = float(best_run[col])
                    if len(params) == len(parameter_names):
                        print(
                            f"Loaded parameters for basin {basin_id} from scipy results"
                        )
                        return params
        except Exception as e:
            print(f"Warning: Failed to load from scipy results: {e}")

    # 3. Try loading from basin-specific sceua results
    sceua_file = os.path.join(param_dir, f"{basin_id}_sceua.csv")
    if os.path.exists(sceua_file):
        try:
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
                        print(
                            f"Loaded parameters for basin {basin_id} from SCE-UA results"
                        )
                        return params
                else:
                    # Try to infer parameters from columns
                    # Look for parx1, parx2, ... columns
                    parx_cols = [
                        col for col in df.columns if col.startswith("parx")
                    ]
                    if parx_cols:
                        parx_cols = sorted(parx_cols, key=lambda x: int(x[4:]))
                        for col in parx_cols:
                            param_name = (
                                col  # Use column name as parameter name
                            )
                            params[param_name] = float(best_run[col])
                        print(
                            f"Loaded parameters for basin {basin_id} from SCE-UA results"
                        )
                        return params
                    # Look for par{name} columns
                    par_cols = [
                        col
                        for col in df.columns
                        if col.startswith("par") and col != "pareto_front"
                    ]
                    if par_cols:
                        for col in par_cols:
                            param_name = col[3:]  # Remove "par" prefix
                            params[param_name] = float(best_run[col])
                        print(
                            f"Loaded parameters for basin {basin_id} from SCE-UA results"
                        )
                        return params
        except Exception as e:
            print(f"Warning: Failed to load from SCE-UA results: {e}")

    # 4. Try loading from calibrate_params.txt
    params_file = os.path.join(param_dir, f"{basin_id}_calibrate_params.txt")
    if os.path.exists(params_file):
        try:
            params_array = np.loadtxt(params_file)
            if len(params_array.shape) > 1:
                params_array = params_array[1:].flatten()

            if parameter_names is not None:
                params = OrderedDict(zip(parameter_names, params_array))
            else:
                # Use generic parameter names if no parameter_names provided
                params = OrderedDict(
                    (f"param_{i}", val) for i, val in enumerate(params_array)
                )
            print(
                f"Loaded parameters for basin {basin_id} from calibrate_params.txt"
            )
            return params
        except Exception as e:
            print(f"Warning: Failed to load from calibrate_params.txt: {e}")

    # If we reach here, no valid parameter file was found
    raise FileNotFoundError(
        f"Could not find calibrated parameters for basin {basin_id} in {param_dir}. "
        f"Tried: calibration_results.json, {basin_id}_ga.csv, {basin_id}_scipy.csv, "
        f"{basin_id}_sceua.csv, {basin_id}_calibrate_params.txt"
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
    data_path: Optional[str] = None,
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
        # Use provided data_path (already resolved by UnifiedDataLoader) or fall back to config
        data_dir = (
            data_path
            if data_path is not None
            else data_config.get("data_source_path", "")
        )
        basin_area = get_basin_area(basins, data_type, data_dir)

        # Convert to m³/s - process each basin separately to avoid broadcasting issues
        target_unit = "m^3/s"

        # Initialize arrays to store converted results
        qsim_converted = np.zeros_like(qsim)
        qobs_converted = np.zeros_like(qobs)

        # Process each basin separately
        for i, basin_id in enumerate(basins):
            # Extract single basin data
            ds_single_basin = xr.Dataset(
                {
                    "qsim": (["time"], qsim[:, i]),
                    "qobs": (["time"], qobs[:, i]),
                },
                coords={"time": times, "basin": [basin_id]},
            )
            ds_single_basin["qsim"].attrs["units"] = "mm/day"
            ds_single_basin["qobs"].attrs["units"] = "mm/day"

            # Get single basin area
            single_basin_area = basin_area.isel(basin=i)

            # Convert units for this basin
            ds_qsim_single = streamflow_unit_conv(
                ds_single_basin[["qsim"]],
                single_basin_area,
                target_unit=target_unit,
                inverse=True,
            )
            ds_qobs_single = streamflow_unit_conv(
                ds_single_basin[["qobs"]],
                single_basin_area,
                target_unit=target_unit,
                inverse=True,
            )

            # Store converted values
            qsim_converted[:, i] = ds_qsim_single["qsim"].values
            qobs_converted[:, i] = ds_qobs_single["qobs"].values

        # Update dataset with converted values
        ds["qsim"].values = qsim_converted
        ds["qsim"].attrs["units"] = target_unit
        ds["qobs"].values = qobs_converted
        ds["qobs"].attrs["units"] = target_unit

    # Save to NetCDF
    output_file = os.path.join(
        output_dir, f"{model_name}_evaluation_results.nc"
    )
    ds.to_netcdf(output_file)
    print(f"   💾 Evaluation results (NetCDF): {output_file}")


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
                    processed_metrics[key] = (
                        value.flatten()[0] if value.size > 0 else value
                    )
                else:
                    processed_metrics[key] = value
            metrics_list.append(processed_metrics)

    if not metrics_list:
        print("Warning: No metrics to save!")
        return

    metrics_df = pd.DataFrame(metrics_list, index=basin_ids)
    metrics_file = os.path.join(output_dir, "basins_metrics.csv")
    metrics_df.to_csv(metrics_file, sep=",", index=True, header=True)

    print(f"   💾 Metrics summary (CSV): {metrics_file}")

    print("\n📊 " + "=" * 77)
    print("📊 EVALUATION METRICS SUMMARY")
    print("📊 " + "=" * 77)
    print(f"\n📈 Metrics for each basin:")
    print("-" * 80)

    # Print with better formatting
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(metrics_df)

    # Also print summary statistics
    if len(basin_ids) > 1:
        print("\n" + "-" * 80)
        print("📊 Summary Statistics Across All Basins:")
        print("-" * 80)
        print(metrics_df.describe().loc[["mean", "std", "min", "max"]])

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
        print(
            "Warning: No parameter names available. Skipping parameter summary."
        )
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
                        (param_ranges[name][1] - param_ranges[name][0])
                        * params[name]
                        + param_ranges[name][0]
                        for name in parameter_names
                    ]
                    denorm_params_list.append(denorm_params)
                except (KeyError, TypeError):
                    has_denorm = False
                    print(
                        f"Warning: Could not denormalize parameters for basin {basin_id}"
                    )

    # Save normalized parameters
    norm_params_df = pd.DataFrame(
        norm_params_list, columns=parameter_names, index=basin_ids
    )
    norm_file = os.path.join(output_dir, "basins_norm_params.csv")
    norm_params_df.to_csv(norm_file, sep=",", index=True, header=True)
    print(f"   💾 Parameters summary (normalized): {norm_file}")

    # Save denormalized parameters if available
    if has_denorm and denorm_params_list:
        denorm_params_df = pd.DataFrame(
            denorm_params_list, columns=parameter_names, index=basin_ids
        )
        denorm_file = os.path.join(output_dir, "basins_denorm_params.csv")
        denorm_params_df.to_csv(denorm_file, sep=",", index=True, header=True)
        print(f"   💾 Parameters summary (denormalized): {denorm_file}")

        print("-" * 50)
        print("Normalized Parameters:")
        print(norm_params_df)
        print("-" * 50)
        print("Denormalized Parameters:")
        print(denorm_params_df)
    else:
        print(
            "Note: Parameter denormalization skipped (param_range.yaml not available)"
        )
        print("-" * 50)
        print("Parameters (normalized [0,1] or as-is):")
        print(norm_params_df)
