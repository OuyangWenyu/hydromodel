"""
Author: Wenyu Ouyang
Date: 2025-08-06
LastEditTime: 2025-08-07 14:51:43
LastEditors: Wenyu Ouyang
Description: Unified calibration interface for all hydrological models
FilePath: \hydromodel\hydromodel\trainers\unified_calibrate.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import json
import numpy as np
import pandas as pd
import spotpy
import pickle
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
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
from hydromodel.configs.unified_config import UnifiedConfig
from hydromodel.models.unit_hydrograph import (
    unit_hydrograph,
    categorized_unit_hydrograph,
    init_unit_hydrograph,
)
from hydromodel.trainers.unit_hydrograph_trainer import (
    objective_function_multi_event,
    categorize_floods_by_peak,
    optimize_uh_for_group,
    evaluate_single_event_from_uh,
)


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


class TraditionalModelSetup(ModelSetupBase):
    """Setup for traditional models (XAJ, GR series, etc.) using SpotSetup pattern."""

    def __init__(
        self,
        p_and_e,
        qobs,
        model_config,
        loss_config,
        warmup_length=365,
        param_file=None,
        **kwargs,
    ):
        super().__init__(
            None, model_config, loss_config, warmup_length, **kwargs
        )

        # Load parameter configuration
        self.param_range = read_model_param_dict(param_file)
        self.parameter_names = self.param_range[model_config["name"]][
            "param_name"
        ]

        # Store data
        self.p_and_e = p_and_e
        self.true_obs = qobs[
            warmup_length:, :, :
        ]  # Remove warmup period from observation

        # Create spotpy parameters
        self.params = []
        self.params.extend(
            Uniform(par_name, low=0.0, high=1.0)
            for par_name in self.parameter_names
        )

    def get_parameter_names(self) -> List[str]:
        return self.parameter_names

    def get_parameter_bounds(self) -> List[tuple]:
        return [(0.0, 1.0) for _ in self.parameter_names]

    def simulate(self, params: np.ndarray) -> np.ndarray:
        """Run traditional model simulation."""
        # Reshape parameters for model input
        params_reshaped = params.reshape(1, -1)

        # Run model simulation
        sim, _ = MODEL_DICT[self.model_config["name"]](
            self.p_and_e,
            params_reshaped,
            warmup_length=self.warmup_length,
            **self.model_config,
            **self.param_range,
        )
        return sim

    def calculate_objective(
        self, simulation: np.ndarray, observation: np.ndarray
    ) -> float:
        """Calculate objective function for traditional models."""
        if self.loss_config["type"] == "time_series":
            return LOSS_DICT[self.loss_config["obj_func"]](
                observation, simulation
            )
        else:
            # TODO: Implement event-based objective function
            raise NotImplementedError(
                "Event-based objective function not implemented yet"
            )


class UnitHydrographSetup(ModelSetupBase):
    """Setup for Unit Hydrograph models with warmup period handling."""

    def __init__(
        self,
        all_event_data,
        model_config,
        loss_config,
        warmup_length=0,
        **kwargs,
    ):
        super().__init__(
            all_event_data, model_config, loss_config, warmup_length, **kwargs
        )

        # Unit hydrograph specific parameters
        self.common_n_uh = model_config.get("n_uh", 24)
        self.smoothing_factor = model_config.get("smoothing_factor", 0.1)
        self.peak_violation_weight = model_config.get(
            "peak_violation_weight", 10000.0
        )
        self.apply_peak_penalty = model_config.get("apply_peak_penalty", True)

        # Process event data - remove warmup period for unit hydrograph
        self.processed_event_data = self._process_event_data_for_uh()

    def _process_event_data_for_uh(self):
        """
        Process event data for unit hydrograph model.
        Remove warmup period from each event as UH doesn't need warmup.
        """
        processed_events = []

        for event_data in self.data:
            if self.warmup_length > 0:
                # Create a copy of event data without warmup period
                event_copy = {}
                for key, value in event_data.items():
                    if key in [
                        "P_eff",
                        "net_rain",
                        "Q_obs_eff",
                        "obs_discharge",
                    ] and isinstance(value, np.ndarray):
                        # Remove warmup period from time series data
                        event_copy[key] = value[self.warmup_length :]
                    else:
                        # Keep metadata as is
                        event_copy[key] = value
                processed_events.append(event_copy)
            else:
                processed_events.append(event_data)

        return processed_events

    def get_parameter_names(self) -> List[str]:
        return [f"uh_{i+1}" for i in range(self.common_n_uh)]

    def get_parameter_bounds(self) -> List[tuple]:
        return [(0.0, 1.0) for _ in range(self.common_n_uh)]

    def simulate(self, params: np.ndarray) -> np.ndarray:
        """
        Simulate using unified unit hydrograph model interface.
        """
        # For unit hydrograph, we need to convert event data to proper format
        # This is a simplified version - full implementation would handle all events
        if len(self.processed_event_data) == 0:
            return np.array([])

        # Use first event as example (full implementation would handle all events)
        event_data = self.processed_event_data[0]
        net_rain_key = self.model_config.get("net_rain_name", "P_eff")

        if net_rain_key not in event_data:
            return np.array([])

        net_rain = event_data[net_rain_key]

        # Convert to expected format: [time, basin, features]
        inputs = np.array(net_rain).reshape(-1, 1, 1)

        # Call unified model interface
        simulated_flow = unit_hydrograph(
            inputs=inputs,
            params=params.reshape(1, -1),  # [1 basin, n_uh]
            warmup_length=0,  # Already removed warmup in preprocessing
            return_state=False,
        )

        return (
            simulated_flow.flatten()
            if simulated_flow.ndim > 1
            else simulated_flow
        )

    def calculate_objective(
        self, params: np.ndarray, observation=None
    ) -> float:
        """Calculate objective function for unit hydrograph using processed data."""
        return objective_function_multi_event(
            params,
            self.processed_event_data,  # Use processed data without warmup
            self.smoothing_factor,
            self.peak_violation_weight,
            self.apply_peak_penalty,
            self.common_n_uh,
            net_rain_name=self.model_config.get("net_rain_name", "P_eff"),
            obs_flow_name=self.model_config.get("obs_flow_name", "Q_obs_eff"),
        )


class CategorizedUnitHydrographSetup(ModelSetupBase):
    """Setup for Categorized Unit Hydrograph models (multi-class UH)."""

    def __init__(
        self,
        all_event_data,
        model_config,
        loss_config,
        warmup_length=0,
        **kwargs,
    ):
        super().__init__(
            all_event_data, model_config, loss_config, warmup_length, **kwargs
        )

        # Categorization parameters
        self.category_weights = model_config.get(
            "category_weights",
            {
                "small": {
                    "smoothing_factor": 0.1,
                    "peak_violation_weight": 100.0,
                },
                "medium": {
                    "smoothing_factor": 0.5,
                    "peak_violation_weight": 500.0,
                },
                "large": {
                    "smoothing_factor": 1.0,
                    "peak_violation_weight": 1000.0,
                },
            },
        )

        self.uh_lengths = model_config.get(
            "uh_lengths", {"small": 8, "medium": 16, "large": 24}
        )

        # Process event data - remove warmup period for categorized UH
        if warmup_length > 0:
            self.processed_event_data = self._process_event_data_for_uh()
        else:
            self.processed_event_data = all_event_data

        # Categorize events by peak
        self.categorized_events, self.thresholds = categorize_floods_by_peak(
            self.processed_event_data
        )

        if self.categorized_events is None:
            raise ValueError("Failed to categorize flood events")

        # Store category information for parameter management
        self.categories = list(self.categorized_events.keys())
        self.total_params = sum(
            self.uh_lengths[cat]
            for cat in self.categories
            if cat in self.uh_lengths
        )

    def _process_event_data_for_uh(self):
        """Process event data by removing warmup period."""
        processed_events = []

        for event_data in self.data:
            if self.warmup_length > 0:
                event_copy = {}
                for key, value in event_data.items():
                    if key in [
                        "P_eff",
                        "net_rain",
                        "Q_obs_eff",
                        "obs_discharge",
                    ] and isinstance(value, np.ndarray):
                        event_copy[key] = value[self.warmup_length :]
                    else:
                        event_copy[key] = value
                processed_events.append(event_copy)
            else:
                processed_events.append(event_data)

        return processed_events

    def get_parameter_names(self) -> List[str]:
        """Return list of parameter names for all categories."""
        param_names = []
        for category in self.categories:
            n_uh = self.uh_lengths.get(category, 24)
            for i in range(n_uh):
                param_names.append(f"uh_{category}_{i+1}")
        return param_names

    def get_parameter_bounds(self) -> List[tuple]:
        """Return list of parameter bounds for all categories."""
        bounds = []
        for category in self.categories:
            n_uh = self.uh_lengths.get(category, 24)
            bounds.extend([(0.0, 1.0) for _ in range(n_uh)])
        return bounds

    def simulate(self, params: np.ndarray) -> np.ndarray:
        """For categorized UH, simulation is handled within objective function."""
        return params

    def calculate_objective(
        self, params: np.ndarray, observation=None
    ) -> float:
        """Calculate objective function for categorized unit hydrograph."""
        # Split parameters by category
        param_dict = self._split_params_by_category(params)

        total_objective = 0.0
        total_events = 0

        # Calculate objective for each category
        for category, events in self.categorized_events.items():
            if category not in param_dict:
                continue

            category_params = param_dict[category]
            weights = self.category_weights.get(category, {})

            # Normalize category parameters
            if category_params.sum() > 0:
                category_params = category_params / category_params.sum()

            # Calculate objective for this category
            category_objective = objective_function_multi_event(
                category_params,
                events,
                weights.get("smoothing_factor", 0.1),
                weights.get("peak_violation_weight", 1000.0),
                len(category_params) > 2,  # apply_peak_penalty
                len(category_params),
                net_rain_name=self.model_config.get("net_rain_name", "P_eff"),
                obs_flow_name=self.model_config.get(
                    "obs_flow_name", "Q_obs_eff"
                ),
            )

            # Weight by number of events in category
            total_objective += category_objective * len(events)
            total_events += len(events)

        # Return weighted average objective
        return (
            total_objective / total_events
            if total_events > 0
            else float("inf")
        )

    def _split_params_by_category(
        self, params: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Split flattened parameters by category."""
        param_dict = {}
        start_idx = 0

        for category in self.categories:
            n_uh = self.uh_lengths.get(category, 24)
            end_idx = start_idx + n_uh
            param_dict[category] = params[start_idx:end_idx]
            start_idx = end_idx

        return param_dict

    def get_category_results(
        self, params: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Get optimized UH for each category."""
        param_dict = self._split_params_by_category(params)

        # Normalize each category separately
        for category, category_params in param_dict.items():
            if category_params.sum() > 0:
                param_dict[category] = category_params / category_params.sum()

        return param_dict


class SpotpyAdapter:
    """Adapter to make ModelSetup work with spotpy."""

    def __init__(self, model_setup: ModelSetupBase):
        self.model_setup = model_setup
        self.parameter_names = model_setup.get_parameter_names()

        # Create spotpy parameters
        bounds = model_setup.get_parameter_bounds()
        self.params = []
        for i, (name, (low, high)) in enumerate(
            zip(self.parameter_names, bounds)
        ):
            self.params.append(Uniform(name, low=low, high=high))

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, x: ParameterSet) -> Union[list, np.array]:
        params = np.array(x)
        return self.model_setup.simulate(params)

    def evaluation(self) -> Union[list, np.array]:
        if hasattr(self.model_setup, "true_obs"):
            return self.model_setup.true_obs
        else:
            # For unit hydrograph, evaluation is handled differently
            return np.array([0])  # Dummy value

    def objectivefunction(
        self,
        simulation: Union[list, np.array],
        evaluation: Union[list, np.array],
        params=None,
    ) -> float:
        if isinstance(
            self.model_setup,
            (UnitHydrographSetup, CategorizedUnitHydrographSetup),
        ):
            # For unit hydrograph models, simulation contains the UH parameters
            return self.model_setup.calculate_objective(simulation)
        else:
            return self.model_setup.calculate_objective(simulation, evaluation)


def calibrate_with_config(
    config: UnifiedConfig,
    data: Union[np.ndarray, List[Dict]],
    **kwargs,
) -> Dict[str, Any]:
    """
    Unified calibration interface using UnifiedConfig.

    Parameters
    ----------
    config : UnifiedConfig
        Unified configuration object containing all settings
    data : Union[np.ndarray, List[Dict]]
        Input data. For traditional models: (p_and_e, qobs) tuple.
        For unit hydrograph: List of event data dictionaries.
    **kwargs
        Additional arguments

    Returns
    -------
    Dict[str, Any]
        Calibration results dictionary
    """
    # Extract configurations
    model_config = config.get_model_config()
    algorithm_config = config.get_algorithm_config()
    loss_config = config.get_loss_config()

    # Extract other parameters from config
    data_cfgs = config.data_cfgs
    training_cfgs = config.training_cfgs

    output_dir = os.path.join(
        training_cfgs.get("output_dir", "results"),
        training_cfgs.get("experiment_name", "experiment"),
    )

    return calibrate(
        data=data,
        model_config=model_config,
        algorithm_config=algorithm_config,
        loss_config=loss_config,
        output_dir=output_dir,
        warmup_length=data_cfgs.get("warmup_length", 0),
        param_file=data_cfgs.get("param_range_file"),
        basin_ids=data_cfgs.get("basin_ids", []),
        **kwargs,
    )


def calibrate(
    data: Union[np.ndarray, List[Dict]],
    model_config: Dict,
    algorithm_config: Dict,
    loss_config: Dict,
    output_dir: str,
    warmup_length: int = 0,
    param_file: Optional[str] = None,
    basin_ids: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Unified calibration interface for all hydrological models.

    Parameters
    ----------
    data : Union[np.ndarray, List[Dict]]
        Input data. For traditional models: (p_and_e, qobs) tuple.
        For unit hydrograph: List of event data dictionaries.
    model_config : Dict
        Model configuration including name and parameters.
        For unit hydrograph: {"name": "unit_hydrograph", "n_uh": 24, "smoothing_factor": 0.1, ...}
        For traditional: {"name": "xaj_mz", "source_type": "sources", ...}
    algorithm_config : Dict
        Algorithm configuration: {"name": "SCE_UA", "rep": 1000, ...}
        Supported algorithms: "SCE_UA", "scipy_minimize", "genetic_algorithm"
    loss_config : Dict
        Loss function configuration: {"type": "time_series", "obj_func": "RMSE", ...}
    output_dir : str
        Directory to save calibration results
    warmup_length : int, default=0
        Warmup period length. Note: Unit hydrograph will handle warmup internally.
    param_file : str, optional
        Parameter range file for traditional models
    basin_ids : List[str], optional
        Basin identifiers for multi-basin calibration
    **kwargs
        Additional arguments

    Returns
    -------
    Dict[str, Any]
        Dictionary containing calibration results, best parameters, and metadata
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Determine model type and create appropriate setup
    model_name = model_config["name"]

    # Check if this is a unit hydrograph model (with legacy support)
    if model_name == "unit_hydrograph" and isinstance(data, list):
        # Legacy unit hydrograph model setup using event data
        model_setup = UnitHydrographSetup(
            all_event_data=data,
            model_config=model_config,
            loss_config=loss_config,
            warmup_length=warmup_length,
            **kwargs,
        )

        return _calibrate_unit_hydrograph(
            model_setup, algorithm_config, output_dir, **kwargs
        )

    elif model_name == "categorized_unit_hydrograph" and isinstance(
        data, list
    ):
        # Legacy categorized unit hydrograph model setup using event data
        model_setup = CategorizedUnitHydrographSetup(
            all_event_data=data,
            model_config=model_config,
            loss_config=loss_config,
            warmup_length=warmup_length,
            **kwargs,
        )

        return _calibrate_categorized_unit_hydrograph(
            model_setup, algorithm_config, output_dir, **kwargs
        )

    else:
        # Unified model setup (XAJ, GR series, unit hydrograph with unified interface)
        # All models now use the same interface through MODEL_DICT
        if isinstance(data, list):
            # Convert event data to standard format for unified interface
            # This is needed for unit hydrograph models using new unified interface
            p_and_e, qobs = _convert_event_data_to_standard_format(data)
        else:
            p_and_e, qobs = data

        results = {}
        if basin_ids is None:
            basin_ids = [f"basin_{i}" for i in range(p_and_e.shape[1])]

        for i, basin_id in enumerate(basin_ids):
            model_setup = TraditionalModelSetup(
                p_and_e=p_and_e[:, i : i + 1, :],
                qobs=qobs[:, i : i + 1, :],
                model_config=model_config,
                loss_config=loss_config,
                warmup_length=warmup_length,
                param_file=param_file,
                **kwargs,
            )

            basin_result = _calibrate_traditional_model(
                model_setup, algorithm_config, output_dir, basin_id, **kwargs
            )
            results[basin_id] = basin_result

        return results


def _convert_event_data_to_standard_format(event_data: List[Dict]) -> tuple:
    """
    Convert event data list to standard (p_and_e, qobs) format for unified interface.

    This helper function bridges the gap between event-based unit hydrograph data
    and the standard model interface format used by traditional models.
    """
    if not event_data:
        raise ValueError("Event data list is empty")

    # For now, use the first event as a representative
    # TODO: Implement proper multi-event handling
    event = event_data[0]

    net_rain = np.array(event.get("P_eff", event.get("net_rain", [])))
    obs_flow = np.array(event.get("Q_obs_eff", event.get("obs_discharge", [])))

    if len(net_rain) == 0 or len(obs_flow) == 0:
        raise ValueError("Event data does not contain required time series")

    # Convert to standard format: [time, basin=1, features]
    p_and_e = net_rain.reshape(-1, 1, 1)  # Only net rain for unit hydrograph
    qobs = obs_flow.reshape(-1, 1, 1)

    return p_and_e, qobs


def _calibrate_unit_hydrograph(
    model_setup: UnitHydrographSetup,
    algorithm_config: Dict,
    output_dir: str,
    **kwargs,
) -> Dict[str, Any]:
    """Calibrate unit hydrograph model."""

    algorithm_name = algorithm_config["name"]

    if algorithm_name == "scipy_minimize":
        # Use scipy.optimize.minimize (current default for UH)
        U_initial_guess = init_unit_hydrograph(model_setup.common_n_uh)
        bounds = model_setup.get_parameter_bounds()
        constraints = {"type": "eq", "fun": lambda U: np.sum(U) - 1}

        max_iterations = algorithm_config.get("max_iterations", 500)
        method = algorithm_config.get("method", "SLSQP")

        result = minimize(
            model_setup.calculate_objective,
            U_initial_guess,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options={"disp": True, "maxiter": max_iterations},
        )

        if result.success or result.status in [0, 2]:
            best_params = {
                "unit_hydrograph": dict(
                    zip(model_setup.get_parameter_names(), result.x)
                )
            }
            objective_value = result.fun
        else:
            best_params = None
            objective_value = float("inf")

    elif algorithm_name == "SCE_UA":
        # Use SCE-UA via spotpy
        spotpy_adapter = SpotpyAdapter(model_setup)

        sampler = spotpy.algorithms.sceua(
            spotpy_adapter,
            dbname=os.path.join(output_dir, "unit_hydrograph"),
            dbformat="csv",
            random_state=algorithm_config.get("random_seed", 1234),
        )

        rep = algorithm_config.get("rep", 1000)
        sampler.sample(rep)

        # Extract best parameters
        results_data = sampler.getdata()
        df_results = pd.DataFrame(results_data)
        best_run = df_results.loc[df_results["like1"].idxmin()]

        param_names = model_setup.get_parameter_names()
        best_params = {"unit_hydrograph": {}}

        for j, param_name in enumerate(param_names):
            param_col = f"parx{j+1}"
            if param_col in df_results.columns:
                best_params["unit_hydrograph"][param_name] = float(
                    best_run[param_col]
                )

        objective_value = float(best_run["like1"])

    elif algorithm_name == "genetic_algorithm":
        # Use Genetic Algorithm
        if not DEAP_AVAILABLE:
            raise ImportError(
                "DEAP package is required for genetic algorithm. Please install with: pip install deap"
            )

        best_params, objective_value, ga_results = _calibrate_with_ga(
            model_setup, algorithm_config, output_dir
        )

    else:
        raise ValueError(
            f"Unsupported algorithm for unit hydrograph: {algorithm_name}"
        )

    # Save results
    results = {
        "model_config": model_setup.model_config,
        "algorithm_config": algorithm_config,
        "best_params": best_params,
        "objective_value": objective_value,
        "convergence": "success" if best_params is not None else "failed",
    }

    # Save to JSON
    results_file = os.path.join(
        output_dir, "unit_hydrograph_calibration_results.json"
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    return results


def _calibrate_categorized_unit_hydrograph(
    model_setup: CategorizedUnitHydrographSetup,
    algorithm_config: Dict,
    output_dir: str,
    **kwargs,
) -> Dict[str, Any]:
    """Calibrate categorized unit hydrograph model."""

    algorithm_name = algorithm_config["name"]

    if algorithm_name == "scipy_minimize":
        # Use scipy.optimize.minimize
        # Initialize parameters for all categories
        U_initial_guess = []
        bounds = []
        constraints_list = []

        start_idx = 0
        for category in model_setup.categories:
            n_uh = model_setup.uh_lengths.get(category, 24)

            # Initialize this category's UH
            category_init = init_unit_hydrograph(n_uh)
            U_initial_guess.extend(category_init)

            # Add bounds for this category
            bounds.extend([(0.0, 1.0) for _ in range(n_uh)])

            # Add normalization constraint for this category
            end_idx = start_idx + n_uh
            constraints_list.append(
                {
                    "type": "eq",
                    "fun": lambda U, start=start_idx, end=end_idx: np.sum(
                        U[start:end]
                    )
                    - 1,
                }
            )
            start_idx = end_idx

        U_initial_guess = np.array(U_initial_guess)
        max_iterations = algorithm_config.get("max_iterations", 500)
        method = algorithm_config.get("method", "SLSQP")

        result = minimize(
            model_setup.calculate_objective,
            U_initial_guess,
            method=method,
            bounds=bounds,
            constraints=constraints_list,
            options={"disp": True, "maxiter": max_iterations},
        )

        if result.success or result.status in [0, 2]:
            # Split parameters by category and create result structure
            param_dict = model_setup.get_category_results(result.x)

            best_params = {"categorized_unit_hydrograph": {}}
            for category, category_params in param_dict.items():
                category_dict = {}
                for i, param_value in enumerate(category_params):
                    category_dict[f"uh_{category}_{i+1}"] = float(param_value)
                best_params["categorized_unit_hydrograph"][
                    category
                ] = category_dict

            objective_value = result.fun
        else:
            best_params = None
            objective_value = float("inf")

    elif algorithm_name == "SCE_UA":
        # Use SCE-UA via spotpy
        spotpy_adapter = SpotpyAdapter(model_setup)

        sampler = spotpy.algorithms.sceua(
            spotpy_adapter,
            dbname=os.path.join(output_dir, "categorized_unit_hydrograph"),
            dbformat="csv",
            random_state=algorithm_config.get("random_seed", 1234),
        )

        rep = algorithm_config.get("rep", 1000)
        sampler.sample(rep)

        # Extract best parameters
        results_data = sampler.getdata()
        df_results = pd.DataFrame(results_data)
        best_run = df_results.loc[df_results["like1"].idxmin()]

        param_names = model_setup.get_parameter_names()

        # Reconstruct parameters array
        params_array = np.zeros(len(param_names))
        for j, param_name in enumerate(param_names):
            param_col = f"parx{j+1}"
            if param_col in df_results.columns:
                params_array[j] = float(best_run[param_col])

        # Split parameters by category
        param_dict = model_setup.get_category_results(params_array)

        best_params = {"categorized_unit_hydrograph": {}}
        for category, category_params in param_dict.items():
            category_dict = {}
            for i, param_value in enumerate(category_params):
                category_dict[f"uh_{category}_{i+1}"] = float(param_value)
            best_params["categorized_unit_hydrograph"][
                category
            ] = category_dict

        objective_value = float(best_run["like1"])

    elif algorithm_name == "genetic_algorithm":
        # Use Genetic Algorithm
        if not DEAP_AVAILABLE:
            raise ImportError(
                "DEAP package is required for genetic algorithm. Please install with: pip install deap"
            )

        best_params, objective_value, ga_results = (
            _calibrate_categorized_with_ga(
                model_setup, algorithm_config, output_dir
            )
        )

    else:
        raise ValueError(
            f"Unsupported algorithm for categorized unit hydrograph: {algorithm_name}"
        )

    # Save results with categorization information
    results = {
        "model_config": model_setup.model_config,
        "algorithm_config": algorithm_config,
        "best_params": best_params,
        "objective_value": objective_value,
        "convergence": "success" if best_params is not None else "failed",
        "categorization_info": {
            "categories": model_setup.categories,
            "thresholds": model_setup.thresholds,
            "uh_lengths": model_setup.uh_lengths,
            "category_weights": model_setup.category_weights,
            "events_per_category": {
                cat: len(events)
                for cat, events in model_setup.categorized_events.items()
            },
        },
    }

    # Save to JSON
    results_file = os.path.join(
        output_dir, "categorized_unit_hydrograph_calibration_results.json"
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    return results


def _calibrate_traditional_model(
    model_setup: TraditionalModelSetup,
    algorithm_config: Dict,
    output_dir: str,
    basin_id: str,
    **kwargs,
) -> Dict[str, Any]:
    """Calibrate traditional hydrological model."""

    algorithm_name = algorithm_config["name"]

    if algorithm_name == "SCE_UA":
        # Use existing SCE-UA implementation
        spotpy_adapter = SpotpyAdapter(model_setup)

        random_seed = algorithm_config.get("random_seed", 1234)
        np.random.seed(random_seed)

        db_path = os.path.join(output_dir, basin_id)
        sampler = spotpy.algorithms.sceua(
            spotpy_adapter,
            dbname=db_path,
            dbformat="csv",
            random_state=random_seed,
        )

        # Sample parameters
        rep = algorithm_config.get("rep", 1000)
        ngs = algorithm_config.get("ngs", 1000)
        kstop = algorithm_config.get("kstop", 500)
        peps = algorithm_config.get("peps", 0.1)
        pcento = algorithm_config.get("pcento", 0.1)

        sampler.sample(rep, ngs=ngs, kstop=kstop, peps=peps, pcento=pcento)

        # Extract best parameters
        results_data = sampler.getdata()
        df_results = pd.DataFrame(results_data)
        best_run = df_results.loc[df_results["like1"].idxmin()]

        param_names = model_setup.get_parameter_names()
        best_params = {basin_id: {}}

        for j, param_name in enumerate(param_names):
            param_col = f"parx{j+1}"
            if param_col in df_results.columns:
                best_params[basin_id][param_name] = float(best_run[param_col])

        objective_value = float(best_run["like1"])

    elif algorithm_name == "genetic_algorithm":
        # Use Genetic Algorithm for traditional models
        if not DEAP_AVAILABLE:
            raise ImportError(
                "DEAP package is required for genetic algorithm. Please install with: pip install deap"
            )

        best_params, objective_value, ga_results = (
            _calibrate_traditional_with_ga(
                model_setup, algorithm_config, output_dir, basin_id
            )
        )

    else:
        raise ValueError(
            f"Unsupported algorithm for traditional model: {algorithm_name}"
        )

    # Save results
    results = {
        "basin_id": basin_id,
        "model_config": model_setup.model_config,
        "algorithm_config": algorithm_config,
        "best_params": best_params,
        "objective_value": objective_value,
        "convergence": "success",
    }

    return results


# Backward compatibility function
def calibrate_by_sceua(
    basins,
    p_and_e,
    qobs,
    dbname,
    warmup_length=365,
    model=None,
    algorithm=None,
    loss=None,
    param_file=None,
):
    """
    Backward compatibility wrapper for existing calibrate_by_sceua function.

    This function maintains the original interface while using the new unified system.
    """

    # Set defaults to match original function
    if model is None:
        model = {
            "name": "xaj_mz",
            "source_type": "sources5mm",
            "source_book": "HF",
            "kernel_size": 15,
            "time_interval_hours": 24,
        }
    if algorithm is None:
        algorithm = {
            "name": "SCE_UA",
            "random_seed": 1234,
            "rep": 1000,
            "ngs": 1000,
            "kstop": 500,
            "peps": 0.1,
            "pcento": 0.1,
        }
    if loss is None:
        loss = {
            "type": "time_series",
            "obj_func": "RMSE",
            "events": None,
        }

    # Use new unified calibrate function
    results = calibrate(
        data=(p_and_e, qobs),
        model_config=model,
        algorithm_config=algorithm,
        loss_config=loss,
        output_dir=dbname,
        warmup_length=warmup_length,
        param_file=param_file,
        basin_ids=basins,
    )

    # Convert results to match original return format (list of samplers)
    # Note: This is a simplified conversion for compatibility
    return [{"results": results, "basin_id": basin} for basin in basins]


def calibrate_unit_hydrograph(
    all_event_data: List[Dict],
    model_config: Dict,
    algorithm_config: Dict,
    output_dir: str,
    warmup_length: int = 0,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenient function specifically for unit hydrograph calibration.

    Parameters
    ----------
    all_event_data : List[Dict]
        List of flood event data dictionaries
    model_config : Dict
        Unit hydrograph model configuration
        Example: {
            "name": "unit_hydrograph",
            "n_uh": 24,
            "smoothing_factor": 0.1,
            "peak_violation_weight": 10000.0,
            "apply_peak_penalty": True,
            "net_rain_name": "P_eff",
            "obs_flow_name": "Q_obs_eff"
        }
    algorithm_config : Dict
        Algorithm configuration
        Example: {"name": "scipy_minimize", "method": "SLSQP", "max_iterations": 500}
        Or: {"name": "SCE_UA", "rep": 1000, "random_seed": 1234}
    output_dir : str
        Directory to save results
    warmup_length : int, default=0
        Warmup period length (will be removed from event data for UH model)

    Returns
    -------
    Dict[str, Any]
        Calibration results including best parameters and objective value
    """

    loss_config = {"type": "time_series", "obj_func": "RMSE"}

    return calibrate(
        data=all_event_data,
        model_config=model_config,
        algorithm_config=algorithm_config,
        loss_config=loss_config,
        output_dir=output_dir,
        warmup_length=warmup_length,
        **kwargs,
    )


# ====================================================================================
# Genetic Algorithm Implementation
# ====================================================================================


def _check_bounds_decorator(min_val, max_val):
    """
    A decorator to set bounds for individuals in a population.

    Parameters
    ----------
    min_val : float
        The lower bound of individuals
    max_val : float
        The upper bound of individuals

    Returns
    -------
    function
        A wrapper for clipping data into a given bound
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > max_val:
                        child[i] = max_val
                    elif child[i] < min_val:
                        child[i] = min_val
            return offspring

        return wrapper

    return decorator


def _evaluate_unit_hydrograph_individual(
    individual, model_setup: UnitHydrographSetup
):
    """
    Evaluate individual (parameters) for unit hydrograph model.

    Parameters
    ----------
    individual : list
        Individual parameters (unit hydrograph values)
    model_setup : UnitHydrographSetup
        Model setup containing processed event data and configuration

    Returns
    -------
    tuple
        Fitness value (as tuple for DEAP)
    """
    # Normalize individual to ensure sum equals 1
    individual_array = np.array(individual)
    if individual_array.sum() > 0:
        individual_array = individual_array / individual_array.sum()
    else:
        individual_array = np.ones_like(individual_array) / len(
            individual_array
        )

    # Calculate objective function
    fitness = model_setup.calculate_objective(individual_array)

    return (fitness,)  # DEAP expects tuple


def _evaluate_traditional_individual(
    individual, model_setup: TraditionalModelSetup
):
    """
    Evaluate individual for traditional hydrological model.

    Parameters
    ----------
    individual : list
        Individual parameters
    model_setup : TraditionalModelSetup
        Model setup for traditional model

    Returns
    -------
    tuple
        Fitness value (as tuple for DEAP)
    """
    params = np.array(individual).reshape(1, -1)

    # Run model simulation
    sim = model_setup.simulate(params)

    # Calculate objective function
    fitness = model_setup.calculate_objective(sim, model_setup.true_obs)

    return (fitness,)


def _setup_ga_toolbox(param_count: int, evaluation_func, ga_config: dict):
    """
    Set up DEAP toolbox for genetic algorithm.

    Parameters
    ----------
    param_count : int
        Number of parameters
    evaluation_func : function
        Function to evaluate individuals
    ga_config : dict
        GA configuration parameters

    Returns
    -------
    toolbox, stats, halloffame
        DEAP components
    """
    # Clear any existing creators
    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin
    if hasattr(creator, "Individual"):
        del creator.Individual

    # Create fitness and individual classes
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attribute,
        n=param_count,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluation_func)

    # Apply bounds
    toolbox.decorate("mate", _check_bounds_decorator(0.0, 1.0))
    toolbox.decorate("mutate", _check_bounds_decorator(0.0, 1.0))

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("std", np.std)

    # Hall of fame
    halloffame = tools.HallOfFame(maxsize=1)

    return toolbox, stats, halloffame


def _calibrate_with_ga(
    model_setup: UnitHydrographSetup, algorithm_config: Dict, output_dir: str
) -> tuple[Dict[str, Any], float, Dict]:
    """
    Calibrate unit hydrograph model using genetic algorithm.

    Parameters
    ----------
    model_setup : UnitHydrographSetup
        Unit hydrograph model setup
    algorithm_config : Dict
        GA configuration
    output_dir : str
        Output directory for results

    Returns
    -------
    tuple
        (best_params, objective_value, ga_results)
    """

    # GA parameters
    random_seed = algorithm_config.get("random_seed", 1234)
    pop_size = algorithm_config.get("pop_size", 50)
    n_generations = algorithm_config.get("n_generations", 40)
    cx_prob = algorithm_config.get("cx_prob", 0.5)
    mut_prob = algorithm_config.get("mut_prob", 0.2)
    save_freq = algorithm_config.get("save_freq", 5)

    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    param_count = model_setup.common_n_uh

    # Create evaluation function
    def evaluate_individual(individual):
        return _evaluate_unit_hydrograph_individual(individual, model_setup)

    # Setup GA
    toolbox, stats, halloffame = _setup_ga_toolbox(
        param_count, evaluate_individual, algorithm_config
    )

    # Initialize population
    pop = toolbox.population(n=pop_size)

    # Logbook for statistics
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    print(f"Initializing GA population with {pop_size} individuals...")

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    halloffame.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(pop), **record)

    print(f"Generation 0: Min={record['min']:.6f}, Avg={record['avg']:.6f}")

    # Save initial state
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        "population": pop,
        "generation": 0,
        "halloffame": halloffame,
        "logbook": logbook,
        "rndstate": random.getstate(),
    }

    with open(os.path.join(output_dir, "ga_checkpoint_gen0.pkl"), "wb") as f:
        pickle.dump(checkpoint, f)

    # Evolution
    for gen in tqdm(range(1, n_generations + 1), desc="GA Evolution"):
        # Select next generation
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update population
        pop[:] = offspring
        halloffame.update(pop)

        # Record statistics
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        print(
            f"Generation {gen}: Min={record['min']:.6f}, Avg={record['avg']:.6f}, "
            f"Best Individual: {halloffame[0].fitness.values[0]:.6f}"
        )

        # Save checkpoint
        if gen % save_freq == 0 or gen == n_generations:
            checkpoint = {
                "population": pop,
                "generation": gen,
                "halloffame": halloffame,
                "logbook": logbook,
                "rndstate": random.getstate(),
            }

            with open(
                os.path.join(output_dir, f"ga_checkpoint_gen{gen}.pkl"), "wb"
            ) as f:
                pickle.dump(checkpoint, f)

    # Extract best parameters
    best_individual = halloffame[0]
    best_individual_array = np.array(best_individual)

    # Normalize to ensure sum equals 1
    if best_individual_array.sum() > 0:
        best_individual_array = (
            best_individual_array / best_individual_array.sum()
        )

    param_names = model_setup.get_parameter_names()
    best_params = {
        "unit_hydrograph": dict(zip(param_names, best_individual_array))
    }

    objective_value = halloffame[0].fitness.values[0]

    # GA results summary
    ga_results = {
        "final_population_size": len(pop),
        "generations": n_generations,
        "best_fitness_history": logbook.select("min"),
        "avg_fitness_history": logbook.select("avg"),
    }

    print(f"GA completed! Best objective value: {objective_value:.6f}")

    return best_params, objective_value, ga_results


def _calibrate_traditional_with_ga(
    model_setup: TraditionalModelSetup,
    algorithm_config: Dict,
    output_dir: str,
    basin_id: str,
) -> tuple[Dict[str, Any], float, Dict]:
    """
    Calibrate traditional model using genetic algorithm.

    Parameters
    ----------
    model_setup : TraditionalModelSetup
        Traditional model setup
    algorithm_config : Dict
        GA configuration
    output_dir : str
        Output directory for results
    basin_id : str
        Basin identifier

    Returns
    -------
    tuple
        (best_params, objective_value, ga_results)
    """

    # GA parameters
    random_seed = algorithm_config.get("random_seed", 1234)
    pop_size = algorithm_config.get("pop_size", 50)
    n_generations = algorithm_config.get("n_generations", 40)
    cx_prob = algorithm_config.get("cx_prob", 0.5)
    mut_prob = algorithm_config.get("mut_prob", 0.2)
    save_freq = algorithm_config.get("save_freq", 5)

    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    param_count = len(model_setup.get_parameter_names())

    # Create evaluation function
    def evaluate_individual(individual):
        return _evaluate_traditional_individual(individual, model_setup)

    # Setup GA
    toolbox, stats, halloffame = _setup_ga_toolbox(
        param_count, evaluate_individual, algorithm_config
    )

    # Initialize population
    pop = toolbox.population(n=pop_size)

    # Logbook for statistics
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    print(
        f"Initializing GA population for {basin_id} with {pop_size} individuals..."
    )

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    halloffame.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(pop), **record)

    print(f"Generation 0: Min={record['min']:.6f}, Avg={record['avg']:.6f}")

    # Save initial state
    basin_output_dir = os.path.join(output_dir, basin_id)
    os.makedirs(basin_output_dir, exist_ok=True)

    checkpoint = {
        "population": pop,
        "generation": 0,
        "halloffame": halloffame,
        "logbook": logbook,
        "rndstate": random.getstate(),
    }

    with open(
        os.path.join(basin_output_dir, "ga_checkpoint_gen0.pkl"), "wb"
    ) as f:
        pickle.dump(checkpoint, f)

    # Evolution
    for gen in tqdm(
        range(1, n_generations + 1), desc=f"GA Evolution {basin_id}"
    ):
        # Select next generation
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update population
        pop[:] = offspring
        halloffame.update(pop)

        # Record statistics
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        print(
            f"{basin_id} - Generation {gen}: Min={record['min']:.6f}, Avg={record['avg']:.6f}"
        )

        # Save checkpoint
        if gen % save_freq == 0 or gen == n_generations:
            checkpoint = {
                "population": pop,
                "generation": gen,
                "halloffame": halloffame,
                "logbook": logbook,
                "rndstate": random.getstate(),
            }

            with open(
                os.path.join(basin_output_dir, f"ga_checkpoint_gen{gen}.pkl"),
                "wb",
            ) as f:
                pickle.dump(checkpoint, f)

    # Extract best parameters
    best_individual = list(halloffame[0])
    param_names = model_setup.get_parameter_names()
    best_params = {basin_id: dict(zip(param_names, best_individual))}

    objective_value = halloffame[0].fitness.values[0]

    # GA results summary
    ga_results = {
        "final_population_size": len(pop),
        "generations": n_generations,
        "best_fitness_history": logbook.select("min"),
        "avg_fitness_history": logbook.select("avg"),
    }

    print(
        f"GA completed for {basin_id}! Best objective value: {objective_value:.6f}"
    )

    return best_params, objective_value, ga_results


def _calibrate_categorized_with_ga(
    model_setup: CategorizedUnitHydrographSetup,
    algorithm_config: Dict,
    output_dir: str,
) -> tuple[Dict[str, Any], float, Dict]:
    """
    Calibrate categorized unit hydrograph model using genetic algorithm.

    Parameters
    ----------
    model_setup : CategorizedUnitHydrographSetup
        Categorized unit hydrograph model setup
    algorithm_config : Dict
        GA configuration
    output_dir : str
        Output directory for results

    Returns
    -------
    tuple
        (best_params, objective_value, ga_results)
    """

    # GA parameters
    random_seed = algorithm_config.get("random_seed", 1234)
    pop_size = algorithm_config.get("pop_size", 50)
    n_generations = algorithm_config.get("n_generations", 40)
    cx_prob = algorithm_config.get("cx_prob", 0.5)
    mut_prob = algorithm_config.get("mut_prob", 0.2)
    save_freq = algorithm_config.get("save_freq", 5)

    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    param_count = model_setup.total_params

    # Create evaluation function
    def evaluate_individual(individual):
        return _evaluate_categorized_individual(individual, model_setup)

    # Setup GA
    toolbox, stats, halloffame = _setup_ga_toolbox(
        param_count, evaluate_individual, algorithm_config
    )

    # Initialize population
    pop = toolbox.population(n=pop_size)

    # Logbook for statistics
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    print(
        f"Initializing Categorized GA population with {pop_size} individuals..."
    )
    print(
        f"Total parameters: {param_count} (across {len(model_setup.categories)} categories)"
    )

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    halloffame.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(pop), **record)

    print(f"Generation 0: Min={record['min']:.6f}, Avg={record['avg']:.6f}")

    # Save initial state
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        "population": pop,
        "generation": 0,
        "halloffame": halloffame,
        "logbook": logbook,
        "rndstate": random.getstate(),
    }

    with open(
        os.path.join(output_dir, "categorized_ga_checkpoint_gen0.pkl"), "wb"
    ) as f:
        pickle.dump(checkpoint, f)

    # Evolution
    for gen in tqdm(
        range(1, n_generations + 1), desc="Categorized GA Evolution"
    ):
        # Select next generation
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update population
        pop[:] = offspring
        halloffame.update(pop)

        # Record statistics
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        print(
            f"Generation {gen}: Min={record['min']:.6f}, Avg={record['avg']:.6f}, "
            f"Best Individual: {halloffame[0].fitness.values[0]:.6f}"
        )

        # Save checkpoint
        if gen % save_freq == 0 or gen == n_generations:
            checkpoint = {
                "population": pop,
                "generation": gen,
                "halloffame": halloffame,
                "logbook": logbook,
                "rndstate": random.getstate(),
            }

            with open(
                os.path.join(
                    output_dir, f"categorized_ga_checkpoint_gen{gen}.pkl"
                ),
                "wb",
            ) as f:
                pickle.dump(checkpoint, f)

    # Extract best parameters
    best_individual = halloffame[0]
    best_individual_array = np.array(best_individual)

    # Split parameters by category and normalize each category
    param_dict = model_setup.get_category_results(best_individual_array)

    # Create result structure
    best_params = {"categorized_unit_hydrograph": {}}
    for category, category_params in param_dict.items():
        category_dict = {}
        for i, param_value in enumerate(category_params):
            category_dict[f"uh_{category}_{i+1}"] = float(param_value)
        best_params["categorized_unit_hydrograph"][category] = category_dict

    objective_value = halloffame[0].fitness.values[0]

    # GA results summary
    ga_results = {
        "final_population_size": len(pop),
        "generations": n_generations,
        "best_fitness_history": logbook.select("min"),
        "avg_fitness_history": logbook.select("avg"),
        "category_info": {
            "categories": model_setup.categories,
            "uh_lengths": model_setup.uh_lengths,
            "total_params": param_count,
        },
    }

    print(
        f"Categorized GA completed! Best objective value: {objective_value:.6f}"
    )

    return best_params, objective_value, ga_results


def _evaluate_categorized_individual(
    individual, model_setup: CategorizedUnitHydrographSetup
):
    """
    Evaluate individual for categorized unit hydrograph model.

    Parameters
    ----------
    individual : list
        Individual parameters for all categories
    model_setup : CategorizedUnitHydrographSetup
        Categorized model setup

    Returns
    -------
    tuple
        Fitness value (as tuple for DEAP)
    """
    # Convert individual to numpy array
    individual_array = np.array(individual)

    # Normalize each category separately
    param_dict = model_setup._split_params_by_category(individual_array)

    # Normalize each category to sum to 1
    normalized_params = []
    for category in model_setup.categories:
        category_params = param_dict[category]
        if category_params.sum() > 0:
            category_params = category_params / category_params.sum()
        else:
            category_params = np.ones_like(category_params) / len(
                category_params
            )
        normalized_params.extend(category_params)

    normalized_array = np.array(normalized_params)

    # Calculate objective function
    fitness = model_setup.calculate_objective(normalized_array)

    return (fitness,)  # DEAP expects tuple


def calibrate_categorized_unit_hydrograph(
    all_event_data: List[Dict],
    model_config: Dict,
    algorithm_config: Dict,
    output_dir: str,
    warmup_length: int = 0,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenient function specifically for categorized unit hydrograph calibration.

    Parameters
    ----------
    all_event_data : List[Dict]
        List of flood event data dictionaries
    model_config : Dict
        Categorized unit hydrograph model configuration
        Example: {
            "name": "categorized_unit_hydrograph",
            "category_weights": {
                "small": {"smoothing_factor": 0.1, "peak_violation_weight": 100.0},
                "medium": {"smoothing_factor": 0.5, "peak_violation_weight": 500.0},
                "large": {"smoothing_factor": 1.0, "peak_violation_weight": 1000.0}
            },
            "uh_lengths": {"small": 8, "medium": 16, "large": 24},
            "net_rain_name": "P_eff",
            "obs_flow_name": "Q_obs_eff"
        }
    algorithm_config : Dict
        Algorithm configuration
        Example: {"name": "scipy_minimize", "method": "SLSQP", "max_iterations": 500}
        Or: {"name": "genetic_algorithm", "pop_size": 100, "n_generations": 50}
    output_dir : str
        Directory to save results
    warmup_length : int, default=0
        Warmup period length (will be removed from event data for categorized UH model)

    Returns
    -------
    Dict[str, Any]
        Calibration results including best parameters for each category
    """

    loss_config = {"type": "time_series", "obj_func": "RMSE"}

    return calibrate(
        data=all_event_data,
        model_config=model_config,
        algorithm_config=algorithm_config,
        loss_config=loss_config,
        output_dir=output_dir,
        warmup_length=warmup_length,
        **kwargs,
    )
