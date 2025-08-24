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
from hydromodel.core.unified_simulate import UnifiedSimulator
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
        self.warmup_length = warmup_length

        # Load data using unified data loader
        self.data_loader = UnifiedDataLoader(data_config)
        self.p_and_e, qobs = self.data_loader.load_data()

        # Store observation data (remove warmup period); seq-first data
        self.true_obs = qobs[
            warmup_length:, :, :
        ]  # Remove warmup period from observation

        # Store whether this is event data for special handling
        self.is_event_data = self.data_loader.is_event_data()

        # Handle unit hydrograph models vs traditional models
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

        # Create base model config for UnifiedSimulator (without specific parameter values)
        self.base_model_config = {
            "model_name": self.model_name,
            "model_params": model_config.copy(),
            "parameters": {},  # Will be filled in during simulation
        }

        # For traditional models, embed the per-model param dict into model_params so simulator forwards it
        if self.model_name not in [
            "unit_hydrograph",
            "categorized_unit_hydrograph",
        ]:
            if (
                isinstance(self.param_range, dict)
                and self.model_name in self.param_range
            ):
                self.base_model_config["model_params"][self.model_name] = (
                    self.param_range[self.model_name]
                )

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
        """Run model simulation using unified UnifiedSimulator interface."""
        # Build parameters without denormalization; models will handle scaling using provided param ranges
        if self.model_name == "unit_hydrograph":
            params_normalized = params / np.sum(params)
            parameter_value = {"uh_values": params_normalized.tolist()}
        elif self.model_name == "categorized_unit_hydrograph":
            uh_lengths = self.model_config.get(
                "uh_lengths", {"small": 8, "medium": 16, "large": 24}
            )
            uh_categories = {}
            param_idx = 0
            for category, length in uh_lengths.items():
                category_params = params[param_idx : param_idx + length]
                uh_categories[category] = (
                    category_params / np.sum(category_params)
                ).tolist()
                param_idx += length
            thresholds = self.model_config.get(
                "thresholds", {"small_medium": 10.0, "medium_large": 25.0}
            )
            parameter_value = {
                "uh_categories": uh_categories,
                "thresholds": thresholds,
            }
        else:
            # Traditional models: preserve parameter order and keep normalized values in [0,1]
            from collections import OrderedDict

            ordered_params = OrderedDict(
                (name, float(params[i]))
                for i, name in enumerate(self.parameter_names)
            )
            parameter_value = ordered_params

        # Create model config with specific parameters for this simulation
        model_config = self.base_model_config.copy()
        model_config["parameters"] = parameter_value

        # Create simulator instance
        simulator = UnifiedSimulator(model_config)

        # Run simulation with flexible interface
        results = simulator.simulate(
            inputs=self.p_and_e,
            warmup_length=self.warmup_length,
            is_event_data=self.is_event_data,
        )

        return results

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
    # 1. Calibrate all basins together (TODO - future implementation)
    # 2. Calibrate each basin separately (current implementation)

    # Currently using approach 2 - calibrate each basin separately
    # This is the default approach that hydromodel supports

    for i, basin_id in enumerate(basin_ids):
        basin_result = _calibrate_model(
            model_setup,
            algorithm_config,
            output_dir,
            basin_id,
            basin_index=i,
            **kwargs,
        )
        results[basin_id] = basin_result

    return results


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
    """Calibrate using SCE-UA via SpotPy, subclassing SpotSetup to use unified simulate."""

    # Determine basin index from config (fallback to 0)
    basin_ids = model_setup.data_config.get(
        "basin_ids",
        [f"basin_{i}" for i in range(model_setup.p_and_e.shape[0])],
    )
    try:
        basin_index = basin_ids.index(str(basin_id))
    except Exception:
        basin_index = 0

    class UnifiedSpotSetup(SpotSetup):
        def __init__(self, model_setup, basin_index: int):
            # Do not call super().__init__ to avoid reloading or storing redundant data
            self.model_setup = model_setup
            self.basin_index = int(basin_index)
            # Reuse parameter definitions created by UnifiedModelSetup
            self.params = model_setup.params
            self.parameter_names = model_setup.parameter_names

        def simulation(self, x):
            # x comes as ParameterSet; convert to numpy array of parameter values
            try:
                params_array = np.array([v for v in x])
            except Exception:
                params_array = np.array(x)
            # Run unified simulation over all basins, then slice this basin
            sim_all = self.model_setup.simulate(
                params_array
            )  # [time, basin, 1]
            return sim_all[:, self.basin_index : self.basin_index + 1, :]

        def evaluation(self):
            # true_obs stored as seq-first data
            return self.model_setup.true_obs[
                :, self.basin_index : self.basin_index + 1, :
            ]

        def objectivefunction(self, simulation, evaluation, params=None):
            return self.model_setup.calculate_objective(simulation, evaluation)

    # Initialize setup
    spot_setup = UnifiedSpotSetup(model_setup, basin_index)

    # Sampler configuration
    rep = algorithm_config.get("rep", 1000)
    ngs = algorithm_config.get("ngs", 1000)
    kstop = algorithm_config.get("kstop", 500)
    peps = algorithm_config.get("peps", 0.1)
    pcento = algorithm_config.get("pcento", 0.1)
    random_seed = algorithm_config.get("random_seed", 1234)

    os.makedirs(output_dir, exist_ok=True)
    dbname = os.path.join(output_dir, f"{basin_id}_sceua")

    sampler = spotpy.algorithms.sceua(
        spot_setup, dbname=dbname, dbformat="csv", random_state=random_seed
    )

    # Run optimization
    sampler.sample(rep, ngs=ngs, kstop=kstop, peps=peps, pcento=pcento)

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
