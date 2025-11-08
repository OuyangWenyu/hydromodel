r"""
Author: Wenyu Ouyang
Date: 2025-08-07
LastEditTime: 2025-08-11 17:40:30
LastEditors: Wenyu Ouyang
Description: Simplified unified calibration interface for all hydrological models
FilePath: /hydromodel/hydromodel/trainers/unified_calibrate.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import sys
import io
import json
import numpy as np
import pandas as pd
import spotpy
import pickle
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
from contextlib import redirect_stdout
from scipy.optimize import minimize
from spotpy.parameter import Uniform, ParameterSet
from tqdm import tqdm

# Import DEAP for genetic algorithm
try:
    from deap import base, creator, tools

    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    print("Warning: DEAP not available. Genetic Algorithm will not work.")

from hydromodel.models.model_config import read_model_param_dict
from hydromodel.models.model_dict import LOSS_DICT, MODEL_DICT
from hydromodel.datasets.unified_data_loader import UnifiedDataLoader
from hydromodel.trainers.unified_simulate import UnifiedSimulator
from hydromodel.trainers.calibrate_sceua import SpotSetup


class ModelSetupBase(ABC):
    """Base class for model setup adapters."""

    def __init__(
        self, data, model_config, loss_config, warmup_length=0, **kwargs
    ):
        self.data = data
        self.model_config = model_config
        self.loss_config = loss_config
        self.warmup_length = warmup_length
        self.kwargs = kwargs

    @abstractmethod
    def get_parameter_names(self) -> List[str]:
        """Return list of parameter names."""
        pass

    @abstractmethod
    def get_parameter_bounds(self) -> List[tuple]:
        """Return list of parameter bounds as (min, max) tuples."""
        pass

    @abstractmethod
    def simulate(self, params: np.ndarray) -> np.ndarray:
        """Run model simulation with given parameters."""
        pass

    @abstractmethod
    def calculate_objective(
        self, simulation: np.ndarray, observation: np.ndarray
    ) -> float:
        """Calculate objective function value."""
        pass


class UnifiedModelSetup(ModelSetupBase):
    """
    Completely unified setup for all hydrological models using the MODEL_DICT interface.

    This class provides a truly unified interface where all models (XAJ, GR series, etc.) are treated identically.
    The only difference is in parameter setup and normalization - the simulation interface is now completely unified with no conditional logic.

    Key unified features:
    - All models use the same MODEL_DICT calling convention
    - All models receive param_range
    - Return value normalization handles tuple vs single array returns
    - No if/else model type distinctions in simulation method
    """

    def __init__(
        self,
        data_config,
        model_config,
        loss_config,
        training_config=None,
        **kwargs,
    ):
        # Get warmup length from data config
        warmup_length = data_config.get("warmup_length", 365)

        super().__init__(
            None, model_config, loss_config, warmup_length, **kwargs
        )

        self.model_name = model_config["name"]
        self.data_config = data_config
        self.training_config = training_config or {}
        self.warmup_length = warmup_length

        # Load data using unified data loader
        self.data_loader = UnifiedDataLoader(data_config)
        self.p_and_e_full, qobs_full = self.data_loader.load_data()

        # Check if using separate data for each basin (for multi-basin flood events)
        self.use_separate_data = (
            hasattr(self.data_loader, "basin_data_separate")
            and self.data_loader.basin_data_separate is not None
        )

        if self.use_separate_data:
            # For multi-basin flood events, we'll extract data per basin from basin_data_separate
            # during set_basin_index() call
            self.p_and_e = None  # Will be set per basin
            self.qobs_full = None  # Will be set per basin
            self.true_obs = None  # Will be set per basin
        else:
            # Store full data for multi-basin support
            self.p_and_e = self.p_and_e_full
            self.qobs_full = qobs_full  # Keep full qobs for basin extraction

        # Basin index for single-basin calibration (None means use all basins)
        self.basin_index = None

        # Store whether this is event data for special handling
        self.is_event_data = data_config.get("is_event_data", False)

        # Store observation data with different handling for continuous vs event data
        if not self.use_separate_data:
            if self.is_event_data:
                # For event data: keep full observation time series
                # Simulation results only have values during events (warmup periods are zeros)
                # Loss calculation will naturally focus on event periods where both have values
                self.true_obs = qobs_full  # Keep complete time series
            else:
                # For continuous data: remove warmup period from both simulation and observation
                self.true_obs = qobs_full[
                    warmup_length:, :, :
                ]  # Remove warmup period

        # Traditional models (XAJ, GR series, etc.)
        param_file = self.training_config.get("param_range_file")
        self._setup_model_params(param_file)

        # Create base model config for UnifiedSimulator (without specific parameter values)
        self.base_model_config = {
            "type": "lumped",
            "model_name": self.model_name,
            "model_params": model_config.copy(),
            "parameters": {},  # Will be filled in during simulation
        }

        # Get basin configurations from data loader
        self.basin_configs = self.data_loader.get_basin_configs()

        # Initialize UnifiedSimulator once during setup
        # Use dummy parameters initially - will be updated during simulation
        dummy_params = {name: 0.5 for name in self.parameter_names}
        init_config = self.base_model_config.copy()
        init_config["parameters"] = dummy_params

        # For now, use the first basin's config for initialization
        # Multi-basin support can be added later
        first_basin_id = (
            str(self.data_loader.basin_ids[0])
            if self.data_loader.basin_ids
            else "default"
        )
        basin_config = self.basin_configs.get(
            first_basin_id, {"basin_area": 1000.0}
        )

        self.simulator = UnifiedSimulator(init_config, basin_config)

        # Create spotpy parameters
        self.params = []
        self.params.extend(
            Uniform(par_name, low=bound[0], high=bound[1])
            for par_name, bound in zip(
                self.parameter_names, self.parameter_bounds
            )
        )

    def _setup_model_params(self, param_file):
        """Setup parameters for models."""
        # Load parameter configuration
        self.param_range = read_model_param_dict(param_file)
        self.parameter_names = self.param_range[self.model_name]["param_name"]
        self.parameter_bounds = [(0.0, 1.0) for _ in self.parameter_names]

    def get_parameter_names(self) -> List[str]:
        return self.parameter_names

    def get_parameter_bounds(self) -> List[tuple]:
        return self.parameter_bounds

    def set_basin_index(self, basin_index: int):
        """Set the basin index for single-basin calibration.

        This method extracts data for a single basin from the full multi-basin dataset.
        It updates self.p_and_e and self.true_obs to contain only the target basin's data.

        Args:
            basin_index: Index of the basin to calibrate (0-based)
        """
        self.basin_index = basin_index

        # Check if using separate data
        if self.use_separate_data:
            # Extract data from basin_data_separate
            basin_id = self.data_loader.basin_ids[basin_index]
            basin_data = self.data_loader.basin_data_separate[basin_id]

            # Get basin-specific data (no padding, correct time alignment)
            # p_and_e shape: [time, features] -> [time, 1, features]
            # qobs shape: [time, features] -> [time, 1, features]
            self.p_and_e = basin_data["p_and_e"][:, np.newaxis, :]
            qobs_basin = basin_data["qobs"][:, np.newaxis, :]

            # Update true_obs based on whether it's event data
            if self.is_event_data:
                # For event data: keep full observation time series
                self.true_obs = qobs_basin
            else:
                # For continuous data: remove warmup period
                self.true_obs = qobs_basin[self.warmup_length :, :, :]
        else:
            # Extract single basin data from full dataset
            # p_and_e shape: [time, basin, features]
            # qobs shape: [time, basin, 1]
            self.p_and_e = self.p_and_e_full[
                :, basin_index : basin_index + 1, :
            ]

            # Update true_obs based on whether it's event data
            if self.is_event_data:
                # For event data: keep full observation time series
                self.true_obs = self.qobs_full[
                    :, basin_index : basin_index + 1, :
                ]
            else:
                # For continuous data: extract single basin and remove warmup period
                self.true_obs = self.qobs_full[
                    self.warmup_length :, basin_index : basin_index + 1, :
                ]

    def simulate(self, params: np.ndarray) -> np.ndarray:
        """Run model simulation using pre-initialized UnifiedSimulator."""
        # Traditional models: preserve parameter order and keep normalized values in [0,1]
        from collections import OrderedDict

        ordered_params = OrderedDict(
            (name, float(params[i]))
            for i, name in enumerate(self.parameter_names)
        )

        # Update parameters in existing simulator (much more efficient than recreating)
        self.simulator.update_parameters(ordered_params)

        # Run simulation with flexible interface
        results = self.simulator.simulate(
            inputs=self.p_and_e,
            warmup_length=self.warmup_length,
            is_event_data=self.is_event_data,
        )

        # Extract simulation output (qsim) - most calibration only needs this
        # This avoids returning the full results dict every time
        if isinstance(results, dict) and "qsim" in results:
            return results["qsim"]
        elif isinstance(results, np.ndarray):
            return results
        else:
            # Fallback: return first available result
            return (
                list(results.values())[0]
                if isinstance(results, dict)
                else results
            )

    # _params_array_to_dict removed: models now receive normalized params and param ranges, and handle scaling internally

    # Event data simulation is now handled by UnifiedSimulator

    # _find_event_segments method removed - now handled by UnifiedSimulator

    def calculate_objective(
        self, simulation: np.ndarray, observation: np.ndarray
    ) -> float:
        """Calculate objective function for all models."""
        if self.loss_config["type"] == "time_series":
            return LOSS_DICT[self.loss_config["obj_func"]](
                observation, simulation
            )
        else:
            # TODO: Implement event-based objective function
            raise NotImplementedError(
                "Event-based objective function not implemented yet"
            )


def calibrate(config, **kwargs) -> Dict[str, Any]:
    """
    Unified calibration interface for all hydrological models.

    Parameters
    ----------
    config : Dict
        Configuration dictionary containing all settings.
        Must contain 'data_cfgs', 'model_cfgs', 'training_cfgs' keys
        Optional in training_cfgs: 'save_config' (bool, default: True)
    **kwargs
        Additional arguments

    Returns
    -------
    Dict[str, Any]
        Dictionary containing calibration results
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
    # Extract model config
    model_cfgs = config["model_cfgs"]
    model_config = {
        "name": model_cfgs.get("model_name"),
        **model_cfgs.get("model_params", {}),
    }
    training_config = config["training_cfgs"]

    # Extract components from training_config
    algorithm_config = {
        "name": training_config.get("algorithm_name", "SCE_UA"),
        **training_config.get("algorithm_params", {}),
    }
    loss_config = training_config.get(
        "loss_config", {"type": "time_series", "obj_func": "RMSE"}
    )

    # Create output directory
    output_dir = os.path.join(
        training_config.get("output_dir", "results"),
        training_config.get("experiment_name", "experiment"),
    )
    os.makedirs(output_dir, exist_ok=True)

    # Create unified model setup
    model_setup = UnifiedModelSetup(
        data_config=data_config,
        model_config=model_config,
        loss_config=loss_config,
        training_config=training_config,
        **kwargs,
    )

    results = {}
    # Get basin IDs from data loader or config
    if (
        hasattr(model_setup.data_loader, "basin_ids")
        and model_setup.data_loader.basin_ids
    ):
        basin_ids = model_setup.data_loader.basin_ids
    elif "basin_ids" in data_config:
        basin_ids = data_config["basin_ids"]
    elif model_setup.p_and_e is not None:
        # Fallback to p_and_e shape
        basin_ids = [f"basin_{i}" for i in range(model_setup.p_and_e.shape[1])]
    else:
        raise ValueError(
            "Cannot determine basin_ids: not found in config, data_loader, or p_and_e"
        )

    # For multi-basin calibration, we can either:
    # 1. Calibrate all basins together (TODO - future implementation)
    # 2. Calibrate each basin separately (current implementation)

    # Currently using approach 2 - calibrate each basin separately
    # This is the default approach that hydromodel supports

    total_basins = len(basin_ids)
    print(f"\n🚀 {'='*60}")
    print(f"🚀 Starting calibration for {total_basins} basin(s)")
    print(f"🚀 {'='*60}\n")

    for i, basin_id in enumerate(basin_ids):
        print(f"▶️  {'='*60}")
        print(f"▶️  Basin {i+1}/{total_basins}: {basin_id}")
        print(f"▶️  {'='*60}")

        basin_result = _calibrate_model(
            model_setup,
            algorithm_config,
            output_dir,
            basin_id,
            basin_index=i,
            **kwargs,
        )
        results[basin_id] = basin_result

        print(f"✅ Basin {i+1}/{total_basins} completed: {basin_id}")

    # Save calibration results to JSON file for evaluation
    print(f"\n Saving calibration results...")
    results_file = os.path.join(output_dir, "calibration_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f" Results saved to: {results_file}")

    # Save configuration files if requested (default: True)
    save_config = training_config.get("save_config", True)
    if save_config:
        _save_calibration_config(
            config, output_dir, model_setup.model_name, model_setup.param_range
        )

    print(f"\n {'='*60}")
    print(f" All calibrations completed successfully!")
    print(f" Total basins calibrated: {total_basins}")
    print(f" {'='*60}")

    return results


def _save_calibration_config(
    config: Dict[str, Any],
    output_dir: str,
    model_name: str,
    param_range: Dict[str, Any],
) -> None:
    """
    Save calibration configuration and parameter ranges to output directory.

    Parameters
    ----------
    config : Dict
        Configuration dictionary
    output_dir : str
        Output directory path
    model_name : str
        Name of the model being calibrated
    param_range : Dict
        Parameter ranges dictionary (full MODEL_PARAM_DICT)
    """
    import yaml
    import copy

    # Save parameter range file for the current model only
    param_range_saved_path = None
    if param_range and model_name in param_range:
        param_range_file = os.path.join(output_dir, "param_range.yaml")
        # Only save the current model's parameter range
        model_param_range = {model_name: param_range[model_name]}
        with open(param_range_file, "w", encoding="utf-8") as f:
            yaml.dump(
                model_param_range,
                f,
                default_flow_style=False,
                allow_unicode=True,
            )
        param_range_saved_path = param_range_file
        print(
            f"💾 Saved parameter range for {model_name} to: {param_range_file}"
        )
    else:
        print(
            f"⚠️  Warning: Parameter range for model '{model_name}' not found. "
            "Skipping param_range.yaml generation."
        )

    # Save calibration config
    config_output_path = os.path.join(output_dir, "calibration_config.yaml")
    # Make a copy to avoid modifying the original config
    config_copy = copy.deepcopy(config)
    # Update param_range_file to the actual saved path
    if "training_cfgs" in config_copy and param_range_saved_path:
        config_copy["training_cfgs"][
            "param_range_file"
        ] = param_range_saved_path

    with open(config_output_path, "w", encoding="utf-8") as f:
        yaml.dump(config_copy, f, default_flow_style=False, allow_unicode=True)
    print(f"💾 Saved calibration config to: {config_output_path}")


def _calibrate_model(
    model_setup: UnifiedModelSetup,
    algorithm_config: Dict,
    output_dir: str,
    basin_id: str,
    basin_index: int = 0,
    **kwargs,
) -> Dict[str, Any]:
    """Calibrate any model using the unified setup."""

    algorithm_name = algorithm_config["name"]

    if algorithm_name in ["scipy_minimize", "scipy", "Scipy"]:
        return _calibrate_with_scipy(
            model_setup, algorithm_config, output_dir, basin_id
        )
    elif algorithm_name in ["SCE_UA", "sceua"]:
        return _calibrate_with_sceua(
            model_setup, algorithm_config, output_dir, basin_id
        )
    elif algorithm_name in ["genetic_algorithm", "GA"]:
        if not DEAP_AVAILABLE:
            raise ImportError(
                "DEAP is required for genetic algorithm. Install with: pip install deap"
            )
        return _calibrate_with_ga(
            model_setup, algorithm_config, output_dir, basin_id
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")


def _calibrate_with_scipy(model_setup, algorithm_config, output_dir, basin_id):
    """Calibrate using scipy.optimize.minimize."""

    # Determine basin index from config
    basin_ids = model_setup.data_config.get(
        "basin_ids",
        [f"basin_{i}" for i in range(model_setup.p_and_e_full.shape[1])],
    )
    try:
        basin_index = basin_ids.index(str(basin_id))
    except Exception:
        basin_index = 0

    # Set basin index to extract single-basin data
    model_setup.set_basin_index(basin_index)

    print(f"\n🔧 {'='*60}")
    print(f"🔧 Starting Scipy optimization for basin: {basin_id}")
    print(f"🔧 {'='*60}\n")

    # Get algorithm parameters
    method = algorithm_config.get("method", "SLSQP")
    max_iterations = algorithm_config.get("max_iterations", 500)
    ftol = algorithm_config.get("ftol", 1e-6)  # Function tolerance
    gtol = algorithm_config.get("gtol", 1e-5)  # Gradient tolerance

    # Get parameter bounds and names
    bounds = model_setup.get_parameter_bounds()
    param_names = model_setup.parameter_names
    n_params = len(param_names)

    # Initialize from middle of bounds
    initial_params = np.array([np.mean(bound) for bound in bounds])

    print(f"📋 Scipy parameters:")
    print(f"   • Method: {method}")
    print(f"   • Maximum iterations: {max_iterations}")
    print(f"   • Function tolerance (ftol): {ftol}")
    print(f"   • Gradient tolerance (gtol): {gtol}")
    print(f"   • Number of parameters: {n_params}")
    print(f"   • Initial parameters: {initial_params}\n")

    # Storage for iteration history and progress tracking
    iteration_history = []
    iteration_count = [0]  # Use list to allow modification in nested function
    best_objective = [float("inf")]  # Track best objective value

    # Create progress bar
    pbar = tqdm(total=max_iterations, desc="Scipy Progress", unit="iter")

    # Objective function for scipy with single basin evaluation
    def objective_func(params):
        try:
            params_array = np.array(params)
            # Simulate with single-basin data (already extracted by set_basin_index)
            sim_basin = model_setup.simulate(params_array)  # [time, 1, 1]
            # Get observation for target basin (already extracted by set_basin_index)
            obs_basin = model_setup.true_obs
            # Calculate objective
            obj_value = model_setup.calculate_objective(sim_basin, obs_basin)

            # Ensure scalar return
            if isinstance(obj_value, (list, np.ndarray)):
                obj_value = (
                    float(obj_value[0])
                    if len(obj_value) > 0
                    else float(obj_value)
                )
            else:
                obj_value = float(obj_value)

            # Record iteration
            iteration_count[0] += 1
            iteration_history.append(
                {
                    "iteration": iteration_count[0],
                    "objective_value": obj_value,
                    **{
                        f"param_{name}": float(params[i])
                        for i, name in enumerate(param_names)
                    },
                }
            )

            # Update progress bar
            if obj_value < best_objective[0]:
                best_objective[0] = obj_value
            pbar.set_postfix({"best": f"{best_objective[0]:.6f}"})
            pbar.update(1)

            return obj_value

        except Exception as e:
            print(
                f"Warning: Evaluation failed at iteration {iteration_count[0]} with error: {e}"
            )
            return 1e10  # Return large penalty for failed simulations

    # Prepare results storage
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"{basin_id}_scipy.csv")

    # Run optimization
    print("Starting optimization...\n")
    constraints = None  # Can be extended in future

    # Build options dict based on method
    options = {
        "disp": False,
        "maxiter": max_iterations,
    }

    # Add method-specific tolerance parameters
    if method in ["SLSQP", "L-BFGS-B", "TNC"]:
        options["ftol"] = ftol
    if method in ["L-BFGS-B", "TNC", "CG", "BFGS"]:
        options["gtol"] = gtol

    try:
        result = minimize(
            objective_func,
            initial_params,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options=options,
        )
    finally:
        # Always close progress bar
        pbar.close()

    print(f"\n✅ Scipy optimization completed!\n")

    # Save iteration history to CSV
    if iteration_history:
        df_history = pd.DataFrame(iteration_history)
        df_history.to_csv(results_file, index=False)
        print(f"Results saved to: {results_file}")

    # Process results
    print(f"\n✅ {'='*60}")
    if result.success:
        best_params = dict(zip(param_names, result.x))

        print(
            f"✅ Scipy optimization completed successfully for basin: {basin_id}"
        )
        print(f"   📊 Convergence: {result.message}")
        print(f"   📊 Total iterations: {result.nit}")
        print(f"   📊 Function evaluations: {result.nfev}")
        print(f"   📊 Best objective value: {result.fun:.6f}")
        print(f"   📊 Best parameters (actual ranges):")
        # Denormalize using the same formula as process_parameters in param_utils.py
        param_ranges = model_setup.param_range[model_setup.model_name][
            "param_range"
        ]
        for name, norm_value in best_params.items():
            min_val, max_val = param_ranges[name]
            denorm_value = min_val + norm_value * (max_val - min_val)
            print(f"      • {name}: {denorm_value:.6f}")
        print(f"✅ {'='*60}\n")

        return {
            "convergence": "success",
            "objective_value": result.fun,
            "best_params": {model_setup.model_name: best_params},
            "algorithm_info": {
                "method": method,
                "iterations": result.nit,
                "function_evaluations": result.nfev,
                "message": result.message,
                "max_iterations": max_iterations,
            },
        }
    else:
        print(f"Scipy optimization failed for basin: {basin_id}")
        print(f"Status: {result.message}")
        print(f"Total iterations: {result.nit}")
        print(f"Function evaluations: {result.nfev}")

        # Even if failed, return best found parameters if available
        best_params = (
            dict(zip(param_names, result.x)) if hasattr(result, "x") else None
        )
        best_obj = result.fun if hasattr(result, "fun") else float("inf")

        if best_params:
            print(f"Best objective value found: {best_obj:.6f}")
            print(f"Best parameters found (actual ranges):")
            # Denormalize using the same formula as process_parameters in param_utils.py
            param_ranges = model_setup.param_range[model_setup.model_name][
                "param_range"
            ]
            for name, norm_value in best_params.items():
                min_val, max_val = param_ranges[name]
                denorm_value = min_val + norm_value * (max_val - min_val)
                print(f"  {name}: {denorm_value:.6f}")

        print(f"{'='*60}\n")

        return {
            "convergence": "failed",
            "objective_value": best_obj,
            "best_params": (
                {model_setup.model_name: best_params} if best_params else None
            ),
            "algorithm_info": {
                "method": method,
                "iterations": result.nit if hasattr(result, "nit") else 0,
                "function_evaluations": (
                    result.nfev if hasattr(result, "nfev") else 0
                ),
                "message": (
                    result.message
                    if hasattr(result, "message")
                    else "Unknown error"
                ),
                "max_iterations": max_iterations,
            },
        }


def _calibrate_with_sceua(model_setup, algorithm_config, output_dir, basin_id):
    """Calibrate using SCE-UA via SpotPy, subclassing SpotSetup to use unified simulate."""

    # Determine basin index from config (fallback to 0)
    basin_ids = model_setup.data_config.get(
        "basin_ids",
        [f"basin_{i}" for i in range(model_setup.p_and_e_full.shape[1])],
    )
    try:
        basin_index = basin_ids.index(str(basin_id))
    except Exception:
        basin_index = 0

    # Set basin index to extract single-basin data
    model_setup.set_basin_index(basin_index)

    print(f"\n🔧 {'='*60}")
    print(f"🔧 Starting SCE-UA optimization for basin: {basin_id}")
    print(f"🔧 {'='*60}\n")

    # Sampler configuration
    rep = algorithm_config.get("rep", 1000)
    ngs = algorithm_config.get("ngs", 1000)
    kstop = algorithm_config.get("kstop", 500)
    peps = algorithm_config.get("peps", 0.1)
    pcento = algorithm_config.get("pcento", 0.1)
    random_seed = algorithm_config.get("random_seed", 1234)

    print(f"📋 SCE-UA parameters:")
    print(f"   • rep (max repetitions): {rep}")
    print(f"   • ngs (# of complexes): {ngs}")
    print(f"   • kstop (convergence criterion): {kstop}")
    print(f"   • peps (convergence threshold): {peps}")
    print(f"   • pcento (percentage change): {pcento}")
    print(f"   • random_seed: {random_seed}\n")

    class UnifiedSpotSetup(SpotSetup):
        def __init__(self, model_setup, total_iterations):
            # Do not call super().__init__ to avoid reloading or storing redundant data
            self.model_setup = model_setup
            # Reuse parameter definitions created by UnifiedModelSetup
            self.params = model_setup.params
            self.parameter_names = model_setup.parameter_names
            # Progress tracking
            self.iteration_count = 0
            self.total_iterations = total_iterations
            # Configure tqdm to output to stderr (default) to avoid stdout conflicts
            # This allows us to suppress SpotPy's stdout without affecting tqdm
            self.pbar = tqdm(
                total=total_iterations,
                desc="SCE-UA Progress",
                unit="iter",
                position=0,
                leave=True,
                dynamic_ncols=True,
                # file parameter omitted - defaults to sys.stderr
            )
            self.best_objective = float("inf")

        def simulation(self, x):
            # x comes as ParameterSet; convert to numpy array of parameter values
            try:
                params_array = np.array([v for v in x])
            except Exception:
                params_array = np.array(x)
            # Run simulation with single-basin data (already extracted by set_basin_index)
            return self.model_setup.simulate(params_array)  # [time, 1, 1]

        def evaluation(self):
            # Return true_obs (already extracted for single basin by set_basin_index)
            return self.model_setup.true_obs

        def objectivefunction(self, simulation, evaluation, params=None):
            obj_value = self.model_setup.calculate_objective(
                simulation, evaluation
            )

            # Ensure scalar return
            if isinstance(obj_value, (list, np.ndarray)):
                obj_value = (
                    float(obj_value[0])
                    if len(obj_value) > 0
                    else float(obj_value)
                )
            else:
                obj_value = float(obj_value)

            # Update progress bar
            self.iteration_count += 1
            if self.pbar is not None:
                # Update best objective if improved
                if obj_value < self.best_objective:
                    self.best_objective = obj_value
                self.pbar.set_postfix({"best": f"{self.best_objective:.6f}"})
                self.pbar.update(1)

            return obj_value

        def __del__(self):
            # Close progress bar when done
            if self.pbar is not None:
                self.pbar.close()

    # Initialize setup with estimated total iterations
    # SCE-UA iterations are complex to estimate, use rep as approximation
    spot_setup = UnifiedSpotSetup(model_setup, rep)

    os.makedirs(output_dir, exist_ok=True)
    dbname = os.path.join(output_dir, f"{basin_id}_sceua")

    sampler = spotpy.algorithms.sceua(
        spot_setup, dbname=dbname, dbformat="csv", random_state=random_seed
    )

    # Run optimization with suppressed SpotPy output
    # SpotPy's internal print statements interfere with tqdm's single-line update
    # Suppress SpotPy's verbose output to prevent interference with tqdm
    # The progress bar will show all necessary information
    with redirect_stdout(io.StringIO()):
        sampler.sample(rep, ngs=ngs, kstop=kstop, peps=peps, pcento=pcento)

    # Close progress bar explicitly
    if hasattr(spot_setup, "pbar") and spot_setup.pbar is not None:
        spot_setup.pbar.close()

    print("\n✅ SCE-UA optimization completed!\n")

    # Extract results
    results = sampler.getdata()
    df_results = pd.DataFrame(results)
    if "like1" in df_results.columns and len(df_results) > 0:
        best_run = df_results.loc[df_results["like1"].idxmin()]
        best_like = float(best_run["like1"])
    else:
        best_like = float(getattr(sampler.status, "objectivefunction", np.nan))

    # Determine best parameters
    param_names = model_setup.get_parameter_names()
    best_params = {}
    if "like1" in df_results.columns and len(df_results) > 0:
        # Try parx1.. order
        cols = []
        for j in range(len(param_names)):
            col = f"parx{j+1}"
            if col in df_results.columns:
                cols.append(col)
        if len(cols) == len(param_names):
            for j, name in enumerate(param_names):
                best_params[name] = float(best_run[cols[j]])
        else:
            # Try par{name}
            for name in param_names:
                col = f"par{name}"
                if col in df_results.columns:
                    best_params[name] = float(best_run[col])

    if not best_params:
        try:
            best_sim = np.array(getattr(sampler.status, "params"))
            best_params = dict(zip(param_names, best_sim))
        except Exception:
            best_params = {name: np.nan for name in param_names}

    # Print results
    print(f"\n✅ {'='*60}")
    print(
        f"✅ SCE-UA optimization completed successfully for basin: {basin_id}"
    )
    print(f"   📊 Best objective value: {best_like:.6f}")
    print(f"   📊 Best parameters (actual ranges):")
    # Denormalize using the same formula as process_parameters in param_utils.py
    param_ranges = model_setup.param_range[model_setup.model_name][
        "param_range"
    ]
    for name, norm_value in best_params.items():
        min_val, max_val = param_ranges[name]
        denorm_value = min_val + norm_value * (max_val - min_val)
        print(f"      • {name}: {denorm_value:.6f}")
    print(f"✅ {'='*60}\n")

    return {
        "convergence": "success",
        "objective_value": best_like,
        "best_params": {model_setup.model_name: best_params},
        "algorithm_info": {
            "rep": rep,
            "ngs": ngs,
            "kstop": kstop,
            "peps": peps,
            "pcento": pcento,
            "random_seed": random_seed,
        },
    }


def _calibrate_with_ga(model_setup, algorithm_config, output_dir, basin_id):
    """Calibrate using genetic algorithm via DEAP."""

    # Determine basin index from config
    basin_ids = model_setup.data_config.get(
        "basin_ids",
        [f"basin_{i}" for i in range(model_setup.p_and_e_full.shape[1])],
    )
    try:
        basin_index = basin_ids.index(str(basin_id))
    except Exception:
        basin_index = 0

    # Set basin index to extract single-basin data
    model_setup.set_basin_index(basin_index)

    print(f"\n🧬 {'='*60}")
    print(f"🧬 Starting GA calibration for basin: {basin_id}")
    print(f"🧬 {'='*60}\n")

    # Set up random seeds
    random_seed = algorithm_config.get("random_seed", 1234)
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Problem configuration
    n_params = len(model_setup.parameter_names)
    bounds = model_setup.parameter_bounds
    param_names = model_setup.parameter_names

    # Algorithm parameters
    pop_size = algorithm_config.get("pop_size", 80)
    n_generations = algorithm_config.get("n_generations", 50)
    cx_prob = algorithm_config.get("cx_prob", 0.7)
    mut_prob = algorithm_config.get("mut_prob", 0.2)

    print(f"📋 GA parameters:")
    print(f"   • Population size: {pop_size}")
    print(f"   • Generations: {n_generations}")
    print(f"   • Crossover probability: {cx_prob}")
    print(f"   • Mutation probability: {mut_prob}")
    print(f"   • Number of parameters: {n_params}\n")

    # Create DEAP types (avoid re-creating if already exists)
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Individual and population initialization
    for i, (low, high) in enumerate(bounds):
        toolbox.register(f"attr_{i}", random.uniform, low, high)

    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        [getattr(toolbox, f"attr_{i}") for i in range(n_params)],
        n=1,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation function for single basin
    def evaluate_individual(ind):
        """Evaluate fitness of an individual for the target basin."""
        try:
            params_array = np.array(ind)
            # Simulate with single-basin data (already extracted by set_basin_index)
            sim_basin = model_setup.simulate(params_array)  # [time, 1, 1]
            # Get observation for target basin (already extracted by set_basin_index)
            obs_basin = model_setup.true_obs
            # Calculate objective
            obj_value = model_setup.calculate_objective(sim_basin, obs_basin)
            # Ensure scalar return
            if isinstance(obj_value, (list, np.ndarray)):
                obj_value = (
                    float(obj_value[0])
                    if len(obj_value) > 0
                    else float(obj_value)
                )
            return (float(obj_value),)
        except Exception as e:
            print(f"Warning: Evaluation failed with error: {e}")
            return (1e10,)  # Return large penalty for failed evaluations

    # Custom mutation with boundary constraints
    def mutate_bounded(individual, mu, sigma, indpb):
        """Gaussian mutation with boundary constraints."""
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] += random.gauss(mu, sigma)
                # Enforce bounds
                low, high = bounds[i]
                individual[i] = max(low, min(high, individual[i]))
        return (individual,)

    # Register genetic operators
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate_bounded, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize population
    print("Initializing population...")
    population = toolbox.population(n=pop_size)

    # Prepare results storage
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"{basin_id}_ga.csv")

    # Storage for generation statistics
    gen_stats = []

    # Evaluate initial population
    print("Evaluating initial population...")
    fitnesses = []
    for ind in tqdm(population, desc="Initial evaluation"):
        fit = toolbox.evaluate(ind)
        fitnesses.append(fit)

    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Record initial generation
    fits = [ind.fitness.values[0] for ind in population]
    best_ind = tools.selBest(population, 1)[0]
    gen_stat = {
        "generation": 0,
        "objective_value": best_ind.fitness.values[
            0
        ],  # Use same column name as scipy
        "min_fitness": min(fits),
        "mean_fitness": np.mean(fits),
        "max_fitness": max(fits),
    }
    # Add best individual's parameters
    for i, name in enumerate(param_names):
        gen_stat[f"param_{name}"] = float(best_ind[i])
    gen_stats.append(gen_stat)
    print(
        f"Generation 0: Best = {best_ind.fitness.values[0]:.6f}, Mean = {np.mean(fits):.6f}"
    )

    # Evolution process
    print(f"\nStarting evolution for {n_generations} generations...")
    for generation in tqdm(range(1, n_generations + 1), desc="GA Evolution"):
        # Selection
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate offspring with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = [toolbox.evaluate(ind) for ind in invalid_ind]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population
        population[:] = offspring

        # Record statistics
        fits = [ind.fitness.values[0] for ind in population]
        best_ind = tools.selBest(population, 1)[0]
        gen_stat = {
            "generation": generation,
            "objective_value": best_ind.fitness.values[
                0
            ],  # Use same column name as scipy
            "min_fitness": min(fits),
            "mean_fitness": np.mean(fits),
            "max_fitness": max(fits),
        }
        # Add best individual's parameters
        for i, name in enumerate(param_names):
            gen_stat[f"param_{name}"] = float(best_ind[i])
        gen_stats.append(gen_stat)

        # Print progress every 10 generations
        if generation % 10 == 0 or generation == n_generations:
            print(
                f"Generation {generation}: Best = {best_ind.fitness.values[0]:.6f}, Mean = {np.mean(fits):.6f}"
            )

    # Save generation statistics to CSV
    df_stats = pd.DataFrame(gen_stats)
    df_stats.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")

    # Get best individual from final population
    best_ind = tools.selBest(population, 1)[0]
    best_params = dict(zip(param_names, best_ind))

    print(f"\n✅ {'='*60}")
    print(f"✅ GA Calibration completed for basin: {basin_id}")
    print(f"   📊 Best objective value: {best_ind.fitness.values[0]:.6f}")
    print(f"   📊 Best parameters (actual ranges):")
    # Denormalize using the same formula as process_parameters in param_utils.py
    param_ranges = model_setup.param_range[model_setup.model_name][
        "param_range"
    ]
    for name, norm_value in best_params.items():
        min_val, max_val = param_ranges[name]
        denorm_value = min_val + norm_value * (max_val - min_val)
        print(f"      • {name}: {denorm_value:.6f}")
    print(f"✅ {'='*60}\n")

    return {
        "convergence": "success",
        "objective_value": best_ind.fitness.values[0],
        "best_params": {model_setup.model_name: best_params},
        "algorithm_info": {
            "generations": n_generations,
            "population_size": pop_size,
            "crossover_prob": cx_prob,
            "mutation_prob": mut_prob,
            "random_seed": random_seed,
        },
    }
