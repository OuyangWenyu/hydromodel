"""
Author: Wenyu Ouyang
Date: 2025-08-07
LastEditTime: 2025-08-07 15:53:18
LastEditors: Wenyu Ouyang
Description: Simplified unified calibration interface for all hydrological models
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

        self.model_name = model_config["name"]

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
            self._setup_traditional_model_params(param_file)

        # Store data
        self.p_and_e = p_and_e
        self.true_obs = qobs[
            warmup_length:, :, :
        ]  # Remove warmup period from observation

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
    Simplified unified calibration interface for all hydrological models.

    Parameters
    ----------
    data : Union[np.ndarray, List[Dict]]
        Input data. For traditional models: (p_and_e, qobs) tuple.
        For unit hydrograph: List of event data dictionaries.
    model_config : Dict
        Model configuration including name and parameters.
    algorithm_config : Dict
        Algorithm configuration: {"name": "SCE_UA", "rep": 1000, ...}
    loss_config : Dict
        Loss function configuration: {"type": "time_series", "obj_func": "RMSE", ...}
    output_dir : str
        Directory to save calibration results
    warmup_length : int, default=0
        Warmup period length
    param_file : str, optional
        Parameter range file for traditional models
    basin_ids : List[str], optional
        Basin identifiers for multi-basin calibration
    **kwargs
        Additional arguments

    Returns
    -------
    Dict[str, Any]
        Dictionary containing calibration results
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert event data to standard format if needed
    if isinstance(data, list):
        # Convert event data to standard (p_and_e, qobs) format
        p_and_e, qobs = _convert_event_data_to_standard_format(data)
    else:
        p_and_e, qobs = data

    results = {}
    if basin_ids is None:
        basin_ids = [f"basin_{i}" for i in range(p_and_e.shape[1])]

    for i, basin_id in enumerate(basin_ids):
        model_setup = UnifiedModelSetup(
            p_and_e=p_and_e[:, i : i + 1, :],
            qobs=qobs[:, i : i + 1, :],
            model_config=model_config,
            loss_config=loss_config,
            warmup_length=warmup_length,
            param_file=param_file,
            **kwargs,
        )

        basin_result = _calibrate_model(
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
