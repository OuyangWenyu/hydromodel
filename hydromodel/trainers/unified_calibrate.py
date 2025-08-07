"""
Author: Wenyu Ouyang
Date: 2025-08-07
LastEditTime: 2025-08-07 16:49:35
LastEditors: Wenyu Ouyang
Description: Simplified unified calibration interface for all hydrological models
FilePath: /hydromodel/hydromodel/trainers/unified_calibrate.py
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
from typing import Dict, List, Optional, Union, Any, Tuple
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
from hydromodel.datasets.unified_data_loader import UnifiedDataLoader


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
        self, params: np.ndarray, observation=None
    ) -> float:
        """Calculate objective function value."""
        pass


class UnifiedModelSetup(ModelSetupBase):
    """
    Completely unified setup for all hydrological models using the MODEL_DICT interface.

    This class provides a truly unified interface where all models (unit hydrograph,
    categorized unit hydrograph, XAJ, GR series, etc.) are treated identically.
    The only difference is in parameter setup and normalization - the simulation
    interface is now completely unified with no conditional logic.

    Key unified features:
    - All models use the same MODEL_DICT calling convention
    - All models receive param_range (empty dict for unit hydrograph models)
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

        # Load data using unified data loader
        self.data_loader = UnifiedDataLoader(data_config)
        self.p_and_e, qobs = self.data_loader.load_data()

        # Store observation data (remove warmup period)
        self.true_obs = qobs[
            warmup_length:, :, :
        ]  # Remove warmup period from observation

        # Store whether this is event data for special handling
        self.is_event_data = self.data_loader.is_event_data()

        # Handle unit hydrograph models with legacy parameter configuration
        if self.model_name in [
            "unit_hydrograph",
            "categorized_unit_hydrograph",
        ]:
            self._setup_unit_hydrograph_params()
            # Unit hydrograph models don't need param_range
            self.param_range = {}
        else:
            # Traditional models (XAJ, GR series, etc.)
            param_file = self.training_config.get("param_range_file")
            self._setup_traditional_model_params(param_file)

        # Create spotpy parameters
        self.params = []
        self.params.extend(
            Uniform(par_name, low=bound[0], high=bound[1])
            for par_name, bound in zip(
                self.parameter_names, self.parameter_bounds
            )
        )

    def _setup_unit_hydrograph_params(self):
        """Setup parameters for unit hydrograph models."""
        if self.model_name == "unit_hydrograph":
            n_uh = self.model_config.get("n_uh", 24)
            self.parameter_names = [f"uh_{i+1}" for i in range(n_uh)]
            self.parameter_bounds = [(0.001, 1.0) for _ in range(n_uh)]

        elif self.model_name == "categorized_unit_hydrograph":
            # Handle categorized unit hydrograph parameters
            uh_lengths = self.model_config.get(
                "uh_lengths", {"small": 8, "medium": 16, "large": 24}
            )
            self.parameter_names = []
            self.parameter_bounds = []

            for category, length in uh_lengths.items():
                for i in range(length):
                    self.parameter_names.append(f"uh_{category}_{i+1}")
                    self.parameter_bounds.append((0.001, 1.0))

    def _setup_traditional_model_params(self, param_file):
        """Setup parameters for traditional models."""
        # Load parameter configuration
        self.param_range = read_model_param_dict(param_file)
        self.parameter_names = self.param_range[self.model_name]["param_name"]
        self.parameter_bounds = [(0.0, 1.0) for _ in self.parameter_names]

    def get_parameter_names(self) -> List[str]:
        return self.parameter_names

    def get_parameter_bounds(self) -> List[tuple]:
        return self.parameter_bounds

    def simulate(self, params: np.ndarray) -> np.ndarray:
        """Run model simulation using unified MODEL_DICT interface."""
        # Handle event data vs continuous data
        if self.is_event_data and self.model_name not in [
            "unit_hydrograph",
            "categorized_unit_hydrograph",
        ]:
            # For traditional models (like XAJ) with event data, we need special handling
            return self._simulate_event_data(params)
        else:
            # Standard simulation for continuous data or unit hydrograph models
            return self._simulate_continuous_data(params)

    def _simulate_continuous_data(self, params: np.ndarray) -> np.ndarray:
        """Standard simulation for continuous data."""
        # Reshape parameters for model input
        params_reshaped = params.reshape(1, -1)

        # Handle special parameter processing for unit hydrograph models
        if self.model_name == "unit_hydrograph":
            # For unit hydrograph, normalize parameters to sum to 1.0
            params_normalized = params / np.sum(params)
            params_reshaped = params_normalized.reshape(1, -1)

        elif self.model_name == "categorized_unit_hydrograph":
            # For categorized unit hydrograph, convert to dictionary format
            uh_lengths = self.model_config.get(
                "uh_lengths", {"small": 8, "medium": 16, "large": 24}
            )

            param_dict = {}
            param_idx = 0
            for category, length in uh_lengths.items():
                category_params = params[param_idx : param_idx + length]
                # Normalize category parameters to sum to 1.0
                param_dict[category] = (
                    category_params / np.sum(category_params)
                ).reshape(1, -1)
                param_idx += length

            # Add thresholds if specified
            thresholds = self.model_config.get(
                "thresholds", {"small_medium": 10.0, "medium_large": 25.0}
            )
            param_dict["thresholds"] = thresholds

            # Pass dictionary as parameters (MODEL_DICT will handle this)
            params_reshaped = param_dict

        # Run model simulation through MODEL_DICT using unified interface
        model_function = MODEL_DICT[self.model_name]

        # Unified interface: always pass param_range and handle return values consistently
        model_result = model_function(
            self.p_and_e,
            params_reshaped,
            warmup_length=self.warmup_length,
            **self.model_config,
            **self.param_range,  # Empty dict for unit hydrograph models
        )

        # Normalize return format: extract simulation output regardless of return type
        if isinstance(model_result, tuple):
            # Traditional models return tuple (simulation, states, ...)
            sim_output = model_result[0]
        else:
            # Unit hydrograph models return single array
            sim_output = model_result

        return sim_output

    def _simulate_event_data(self, params: np.ndarray) -> np.ndarray:
        """
        Special simulation for event data with traditional models like XAJ.

        For event data, we need to:
        1. Identify event segments (non-zero periods)
        2. Run model on each event separately
        3. Combine results while maintaining timeline

        This handles the fact that traditional models expect continuous data
        but event data has gaps between events.
        """
        # Reshape parameters for model input
        params_reshaped = params.reshape(1, -1)

        # Get model function
        model_function = MODEL_DICT[self.model_name]

        # Extract precipitation data to identify events
        # For event data, precipitation is in first channel (net rain)
        net_rain = self.p_and_e[:, :, 0]  # [time, basin]

        # Initialize output array
        output_shape = (self.p_and_e.shape[0], self.p_and_e.shape[1], 1)
        sim_output = np.zeros(output_shape)

        # Process each basin separately
        for basin_idx in range(self.p_and_e.shape[1]):
            basin_rain = net_rain[:, basin_idx]

            # Find event segments (continuous non-zero periods)
            event_segments = self._find_event_segments(basin_rain)

            # Process each event segment
            for start_idx, end_idx in event_segments:
                # Extract event data
                event_p_and_e = self.p_and_e[
                    start_idx : end_idx + 1, basin_idx : basin_idx + 1, :
                ]

                # Run model on this event segment
                try:
                    event_result = model_function(
                        event_p_and_e,
                        params_reshaped,
                        warmup_length=0,  # No warmup for event segments
                        **self.model_config,
                        **self.param_range,
                    )

                    # Extract simulation output
                    if isinstance(event_result, tuple):
                        event_sim = event_result[0]
                    else:
                        event_sim = event_result

                    # Store in output array
                    sim_output[
                        start_idx : end_idx + 1, basin_idx : basin_idx + 1, :
                    ] = event_sim

                except Exception as e:
                    print(
                        f"Warning: Event simulation failed for basin {basin_idx}, "
                        f"segment {start_idx}-{end_idx}: {e}"
                    )
                    # Fill with zeros on failure
                    sim_output[
                        start_idx : end_idx + 1, basin_idx : basin_idx + 1, :
                    ] = 0.0

        return sim_output

    def _find_event_segments(
        self, rain_series: np.ndarray, min_gap_length: int = 1
    ) -> List[Tuple[int, int]]:
        """
        Find continuous event segments in rain time series.

        Parameters
        ----------
        rain_series : np.ndarray
            1D array of precipitation values
        min_gap_length : int
            Minimum gap length to consider as event separation

        Returns
        -------
        List[Tuple[int, int]]
            List of (start_idx, end_idx) tuples for each event segment
        """
        # Find non-zero indices
        non_zero_indices = np.where(rain_series > 0)[0]

        if len(non_zero_indices) == 0:
            return []

        # Find gaps in the indices
        gaps = np.diff(non_zero_indices) > min_gap_length

        # Split indices by gaps
        split_points = np.where(gaps)[0] + 1
        split_indices = np.split(non_zero_indices, split_points)

        # Convert to start-end pairs
        segments = []
        for indices in split_indices:
            if len(indices) > 0:
                # Expand segment to include some context if possible
                start_idx = max(0, indices[0])
                end_idx = min(len(rain_series) - 1, indices[-1])
                segments.append((start_idx, end_idx))

        return segments

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
    config : UnifiedConfig or Dict
        Configuration object or dictionary containing all settings.
        If UnifiedConfig object: Uses the structured configuration
        If Dict: Should contain 'data_cfgs', 'model_cfgs', 'training_cfgs' keys
    **kwargs
        Additional arguments

    Returns
    -------
    Dict[str, Any]
        Dictionary containing calibration results
    """

    # Handle different config types
    if hasattr(config, "data_cfgs"):
        # UnifiedConfig object
        data_config = config.data_cfgs
        model_config = config.get_model_config()
        training_config = config.training_cfgs
    elif isinstance(config, dict):
        # Dictionary with expected structure
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
    else:
        raise ValueError(
            "Config must be either a UnifiedConfig object or a dictionary with "
            "'data_cfgs', 'model_cfgs', 'training_cfgs' keys"
        )

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
    basin_ids = data_config.get(
        "basin_ids",
        [f"basin_{i}" for i in range(model_setup.p_and_e.shape[1])],
    )

    # For multi-basin calibration, we can either:
    # 1. Calibrate all basins together (current implementation)
    # 2. Calibrate each basin separately (loop approach)

    # Currently using approach 1 - calibrate all basins together
    # This is more efficient and allows for shared parameters

    basin_result = _calibrate_model(
        model_setup, algorithm_config, output_dir, "multi_basin", **kwargs
    )

    # Store results for all basins
    for basin_id in basin_ids:
        results[basin_id] = basin_result

    return results


def _calibrate_model(
    model_setup: UnifiedModelSetup,
    algorithm_config: Dict,
    output_dir: str,
    basin_id: str,
    **kwargs,
) -> Dict[str, Any]:
    """Calibrate any model using the unified setup."""

    algorithm_name = algorithm_config["name"]

    if algorithm_name == "scipy_minimize":
        return _calibrate_with_scipy(
            model_setup, algorithm_config, output_dir, basin_id
        )
    elif algorithm_name == "SCE_UA":
        return _calibrate_with_sceua(
            model_setup, algorithm_config, output_dir, basin_id
        )
    elif algorithm_name == "genetic_algorithm":
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

    # Get initial parameters
    bounds = model_setup.get_parameter_bounds()
    initial_params = np.array([np.mean(bound) for bound in bounds])

    # Special handling for unit hydrograph models
    if model_setup.model_name == "unit_hydrograph":
        # Initialize with gamma distribution
        from hydromodel.models.unit_hydrograph import init_unit_hydrograph

        n_uh = len(model_setup.parameter_names)
        initial_params = init_unit_hydrograph(n_uh)

        # Add constraint for sum to 1.0
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}
    else:
        constraints = None

    # Objective function for scipy
    def objective_func(params):
        try:
            simulation = model_setup.simulate(params)
            return model_setup.calculate_objective(
                simulation, model_setup.true_obs
            )
        except Exception as e:
            return 1e6  # Return large penalty for failed simulations

    # Run optimization
    method = algorithm_config.get("method", "SLSQP")
    max_iterations = algorithm_config.get("max_iterations", 500)

    result = minimize(
        objective_func,
        initial_params,
        method=method,
        bounds=bounds,
        constraints=constraints,
        options={"disp": True, "maxiter": max_iterations},
    )

    if result.success:
        best_params = dict(zip(model_setup.parameter_names, result.x))
        return {
            "convergence": "success",
            "objective_value": result.fun,
            "best_params": {model_setup.model_name: best_params},
            "algorithm_info": {
                "iterations": result.nit,
                "message": result.message,
            },
        }
    else:
        return {
            "convergence": "failed",
            "objective_value": float("inf"),
            "best_params": None,
            "algorithm_info": {
                "iterations": result.nit,
                "message": result.message,
            },
        }


def _calibrate_with_sceua(model_setup, algorithm_config, output_dir, basin_id):
    """Calibrate using SCE-UA algorithm via spotpy."""

    class SpotPySetup:
        def __init__(self, model_setup):
            self.model_setup = model_setup
            self.params = model_setup.params

        def simulation(self, vector):
            params_array = np.array([param for param in vector])
            simulation = self.model_setup.simulate(params_array)
            return simulation.flatten()

        def evaluation(self):
            return self.model_setup.true_obs.flatten()

        def objectivefunction(self, simulation, evaluation):
            return self.model_setup.calculate_objective(simulation, evaluation)

    # Create setup
    spot_setup = SpotPySetup(model_setup)

    # Configure SCE-UA
    rep = algorithm_config.get("rep", 1000)
    sampler = spotpy.algorithms.sceua(
        spot_setup, dbname=f"{output_dir}/{basin_id}_sceua", dbformat="csv"
    )

    # Run optimization
    sampler.sample(rep)

    # Get results
    results = sampler.getdata()
    best_sim = sampler.status.params
    best_like = sampler.status.objectivefunction

    best_params = dict(zip(model_setup.parameter_names, best_sim))

    return {
        "convergence": "success",
        "objective_value": best_like,
        "best_params": {model_setup.model_name: best_params},
        "algorithm_info": {"rep": rep},
    }


def _calibrate_with_ga(model_setup, algorithm_config, output_dir, basin_id):
    """Calibrate using genetic algorithm via DEAP."""

    # Set up DEAP genetic algorithm
    random.seed(algorithm_config.get("random_seed", 1234))
    np.random.seed(algorithm_config.get("random_seed", 1234))

    # Problem configuration
    n_params = len(model_setup.parameter_names)
    bounds = model_setup.parameter_bounds

    # Create DEAP types
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Individual and population
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

    # Genetic operators
    toolbox.register(
        "evaluate",
        lambda ind: (
            model_setup.calculate_objective(
                model_setup.simulate(np.array(ind)), model_setup.true_obs
            ),
        ),
    )
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Parameters
    pop_size = algorithm_config.get("pop_size", 80)
    n_generations = algorithm_config.get("n_generations", 50)
    cx_prob = algorithm_config.get("cx_prob", 0.7)
    mut_prob = algorithm_config.get("mut_prob", 0.2)

    # Run GA
    population = toolbox.population(n=pop_size)

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Evolution
    for generation in range(n_generations):
        # Selection and reproduction
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population
        population[:] = offspring

    # Get best individual
    best_ind = tools.selBest(population, 1)[0]
    best_params = dict(zip(model_setup.parameter_names, best_ind))

    return {
        "convergence": "success",
        "objective_value": best_ind.fitness.values[0],
        "best_params": {model_setup.model_name: best_params},
        "algorithm_info": {
            "generations": n_generations,
            "population_size": pop_size,
        },
    }
