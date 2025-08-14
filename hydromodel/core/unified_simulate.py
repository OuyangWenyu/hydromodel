r"""
Author: Wenyu Ouyang
Date: 2025-08-07
LastEditTime: 2025-08-12 11:00:08
LastEditors: Wenyu Ouyang
Description: Unified simulation interface for all hydrological models
FilePath: \hydromodel\hydromodel\core\unified_simulate.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from collections import OrderedDict
from abc import ABC, abstractmethod

from hydromodel.models.model_dict import MODEL_DICT
from hydromodel.configs.unified_config import UnifiedConfig
from hydromodel.datasets.unified_data_loader import UnifiedDataLoader


class UnifiedSimulator:
    """
    Unified simulator for all hydrological models.

    This class provides a single interface for running simulations with any model type.
    The key design principle is separation of concerns:
    - Model configuration is handled during initialization
    - Input data is provided as arguments to simulate() method
    - This allows one simulator instance to run multiple different datasets

    Key features:
    - Single interface for all models (XAJ, GR series, unit hydrograph, etc.)
    - Flexible data input: simulate(inputs, **kwargs)
    - One-time model setup, multiple simulations
    - Direct parameter specification
    - Consistent output format across all models
    - Support for both single and multi-basin simulations
    """

    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the unified simulator with model configuration only.

        Parameters
        ----------
        model_config : Dict[str, Any]
            Model configuration containing:
            - model_name: Name of the model to use
            - model_params: Model-specific parameters (structure, etc.)
            - parameters: Specific parameter values for simulation
        """
        self.model_config = model_config

        # Extract model information
        self.model_name = self.model_config["model_name"]
        self.model_params = self.model_config.get("model_params", {})
        self.parameters = OrderedDict(self.model_config.get("parameters", {}))

        # Validate model exists
        if self.model_name not in MODEL_DICT:
            raise ValueError(
                f"Model '{self.model_name}' not found in MODEL_DICT"
            )

        # Get model function
        self.model_function = MODEL_DICT[self.model_name]

        # Setup parameter handling (convert dict to array format)
        self._setup_parameters()

    def _setup_parameters(self):
        """
        Setup model parameters for simulation.
        Convert parameter dictionary to array format expected by models.
        """
        if not self.parameters:
            raise ValueError(
                f"Model {self.model_name} requires parameters to be specified"
            )

        if self.model_name in [
            "unit_hydrograph",
            "categorized_unit_hydrograph",
        ]:
            # Unit hydrograph models: parameters are the UH values
            self._setup_unit_hydrograph_params()
        else:
            # Traditional models: parameters from parameter dictionary
            self._setup_traditional_model_params()

    def _setup_unit_hydrograph_params(self):
        """Setup parameters for unit hydrograph models."""
        if self.model_name == "unit_hydrograph":
            # Single unit hydrograph: expect array or list of values
            uh_values = self.parameters.get("uh_values")
            if uh_values is None:
                raise ValueError(
                    "unit_hydrograph model requires 'uh_values' in parameters"
                )

            # Convert to numpy array and normalize to sum to 1
            uh_array = np.array(uh_values)
            uh_array = uh_array / np.sum(uh_array)

            # Store as base parameters that will be replicated per basin as needed
            self.base_uh_params = uh_array

        elif self.model_name == "categorized_unit_hydrograph":
            # Categorized unit hydrograph: expect dictionary with category values
            uh_categories = self.parameters.get("uh_categories")
            thresholds = self.parameters.get(
                "thresholds", {"small_medium": 10.0, "medium_large": 25.0}
            )

            if uh_categories is None:
                raise ValueError(
                    "categorized_unit_hydrograph model requires 'uh_categories' in parameters"
                )

            # Convert to expected format
            param_dict = {}
            for category, values in uh_categories.items():
                normalized_values = np.array(values) / np.sum(values)
                param_dict[category] = normalized_values.reshape(1, -1)

            param_dict["thresholds"] = thresholds
            self.param_array = param_dict

    def _setup_traditional_model_params(self):
        """Setup parameters for traditional models (XAJ, GR series, etc.)."""
        # Convert parameter dictionary to list format
        param_names = list(self.parameters.keys())
        param_values = list(self.parameters.values())

        # Store parameter info for later use when we know number of basins
        self.param_names = param_names
        self.param_values = param_values

        # Check if basin-specific parameters are provided
        self.has_basin_specific_params = self.parameters.get(
            "basin_specific", False
        ) and isinstance(param_values[0], (list, np.ndarray))

    def simulate(
        self,
        inputs: np.ndarray,
        qobs: Optional[np.ndarray] = None,
        warmup_length: int = 365,
        is_event_data: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run model simulation with provided input data.

        Parameters
        ----------
        inputs : np.ndarray
            Input data array with shape [time, basin, features]
            Features typically include [precipitation, potential_evapotranspiration]
        qobs : np.ndarray, optional
            Observed streamflow data with shape [time, basin, 1]
        warmup_length : int, default 365
            Length of warmup period (time steps)
        is_event_data : bool, default False
            Whether input data represents event-based data
        **kwargs
            Additional arguments passed to the model function

        Returns
        -------
        Dict[str, Any]
            Dictionary containing simulation results:
            - simulation: Model simulation output [time, basin, 1]
            - observation: Observed data (if provided) [time, basin, 1]
            - input_data: Input data used for simulation [time, basin, n_features]
            - metadata: Simulation metadata and configuration
        """
        # Validate inputs
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)

        if inputs.ndim != 3:
            raise ValueError(
                f"Input data must be 3D array [time, basin, features], got shape {inputs.shape}"
            )

        # Get number of basins and setup parameters accordingly
        n_basins = inputs.shape[1]
        self.param_array = self._prepare_param_array(n_basins)

        # Handle different simulation scenarios
        if is_event_data and self.model_name not in [
            "unit_hydrograph",
            "categorized_unit_hydrograph",
        ]:
            # Event data with traditional models
            simulation_output = self._simulate_event_data(
                inputs, warmup_length, **kwargs
            )
        else:
            # Standard simulation
            simulation_output = self._simulate_continuous_data(
                inputs, warmup_length, **kwargs
            )

        # Prepare results
        results = {
            "simulation": simulation_output,
            "observation": qobs,
            "input_data": inputs,
            "metadata": {
                "model_name": self.model_name,
                "model_params": self.model_params,
                "parameters": self.parameters,
                "warmup_length": warmup_length,
                "simulation_shape": simulation_output.shape,
                "time_steps": simulation_output.shape[0],
                "n_basins": simulation_output.shape[1],
                "is_event_data": is_event_data,
            },
        }

        return results

    def _prepare_param_array(self, n_basins: int) -> np.ndarray:
        """
        Prepare parameter array for the given number of basins.

        Parameters
        ----------
        n_basins : int
            Number of basins in the input data

        Returns
        -------
        np.ndarray or dict
            Parameter array with shape [n_basins, n_params] for traditional models
            or parameter dictionary for unit hydrograph models
        """
        if self.model_name == "unit_hydrograph":
            # Unit hydrograph: replicate base parameters for each basin
            if hasattr(self, "base_uh_params"):
                return np.tile(self.base_uh_params, (n_basins, 1))
            else:
                raise ValueError(
                    "Unit hydrograph parameters not properly initialized"
                )

        elif self.model_name == "categorized_unit_hydrograph":
            # Categorized unit hydrograph already has param_array as dict
            if hasattr(self, "param_array"):
                return self.param_array
            else:
                raise ValueError(
                    "Categorized unit hydrograph parameters not properly initialized"
                )

        else:
            # Traditional models: create param array from parameter values
            if n_basins == 1:
                param_array = np.array(self.param_values).reshape(1, -1)
            else:
                if self.has_basin_specific_params:
                    # Basin-specific parameters provided
                    param_array = np.array(self.param_values)
                    if param_array.shape[0] != n_basins:
                        raise ValueError(
                            f"Basin-specific parameters shape {param_array.shape[0]} "
                            f"does not match number of basins {n_basins}"
                        )
                else:
                    # Replicate same parameters for all basins
                    param_array = np.tile(self.param_values, (n_basins, 1))

            return param_array

    def _simulate_continuous_data(
        self, inputs: np.ndarray, warmup_length: int, **kwargs
    ) -> np.ndarray:
        """Standard simulation for continuous data."""
        # Prepare model configuration
        model_config = dict(self.model_params)
        model_config.update(kwargs)

        # Run model simulation
        model_result = self.model_function(
            inputs,
            self.param_array,
            warmup_length=warmup_length,
            **model_config,
        )

        # Handle different return formats
        if isinstance(model_result, tuple):
            # Traditional models return (simulation, states, ...)
            simulation_output = model_result[0]
        else:
            # Unit hydrograph models return single array
            simulation_output = model_result

        return simulation_output

    def _simulate_event_data(
        self, inputs: np.ndarray, warmup_length: int, **kwargs
    ) -> np.ndarray:
        """Special simulation for event data with traditional models."""
        # Validate that flood_event markers are present
        if inputs.shape[2] < 3:
            raise ValueError(
                "Event data simulation requires flood_event markers. "
                f"Expected input shape [time, basin, 3] with features [rain, pet, flood_event], "
                f"but got shape {inputs.shape}."
            )

        # Initialize output array
        output_shape = (inputs.shape[0], inputs.shape[1], 1)
        simulation_output = np.zeros(output_shape)

        # Process each basin separately
        for basin_idx in range(inputs.shape[1]):
            # Find event segments using flood_event markers (including warmup period)
            event_segments = self._find_event_segments(
                inputs, basin_idx, warmup_length
            )

            # Get basin-specific parameters
            if self.param_array.shape[0] > 1:
                basin_params = self.param_array[basin_idx : basin_idx + 1, :]
            else:
                basin_params = self.param_array

            # Process each event segment
            for (
                extended_start,
                extended_end,
                original_start,
                original_end,
            ) in event_segments:
                # Extract event data (including warmup period)
                event_inputs = inputs[
                    extended_start : extended_end + 1,
                    basin_idx : basin_idx + 1,
                    :,
                ]

                try:
                    # Run model on this event segment
                    model_config = dict(self.model_params)
                    model_config.update(kwargs)

                    event_result = self.model_function(
                        event_inputs,
                        basin_params,
                        warmup_length=warmup_length,
                        **model_config,
                    )

                    # Extract simulation output
                    if isinstance(event_result, tuple):
                        event_sim = event_result[0]
                    else:
                        event_sim = event_result

                    # Ensure event_sim has the correct shape for storage
                    if event_sim.ndim == 1:
                        # Convert 1D output to (time, 1, 1) for compatibility
                        event_sim = event_sim.reshape(-1, 1, 1)
                    elif event_sim.ndim == 2:
                        if event_sim.shape[1] == 1:
                            # Convert (time, 1) to (time, 1, 1)
                            event_sim = event_sim.reshape(-1, 1, 1)
                        else:
                            # If it's (time, basin) but basin != 1, take the first basin and add feature dim
                            event_sim = event_sim[:, 0:1].reshape(-1, 1, 1)

                    # Store only the event period output (excluding warmup period)
                    # The model output should already exclude warmup period
                    event_output_length = original_end - original_start + 1
                    if event_sim.shape[0] == event_output_length:
                        # Model correctly handled warmup and returned only event period output
                        simulation_output[
                            original_start : original_end + 1,
                            basin_idx : basin_idx + 1,
                            :,
                        ] = event_sim
                    else:
                        # Fallback: store whatever the model returned, starting from original event start
                        actual_length = min(
                            event_sim.shape[0], event_output_length
                        )
                        simulation_output[
                            original_start : original_start + actual_length,
                            basin_idx : basin_idx + 1,
                            :,
                        ] = event_sim[:actual_length]

                except Exception as e:
                    print(
                        f"Warning: Event simulation failed for basin {basin_idx}, "
                        f"segment {original_start}-{original_end}: {e}"
                    )
                    # Fill with zeros on failure (only for the original event period)
                    simulation_output[
                        original_start : original_end + 1,
                        basin_idx : basin_idx + 1,
                        :,
                    ] = 0.0

        return simulation_output

    def _find_event_segments(
        self,
        inputs: np.ndarray,
        basin_idx: int,
        warmup_length: int = 0,
    ) -> List[Tuple[int, int, int, int]]:
        """Find continuous event segments using flood_event markers, including warmup period.

        Returns
        -------
        List[Tuple[int, int, int, int]]
            List of (extended_start, extended_end, original_start, original_end) tuples.
            extended_start includes warmup period, original_start is the actual event start.
        """
        # For event data, flood_event markers are mandatory
        # Use flood_event markers (feature index 2)
        flood_event_series = inputs[:, basin_idx, 2]
        # Find non-zero indices (both warmup and actual event periods)
        event_indices = np.where(flood_event_series > 0)[0]

        if len(event_indices) == 0:
            return []

        # Find continuous segments
        segments = []
        if len(event_indices) > 0:
            # Find gaps in the indices
            gaps = np.diff(event_indices) > 1

            # Split indices by gaps
            split_points = np.where(gaps)[0] + 1
            split_indices = np.split(event_indices, split_points)

            # Convert to start-end pairs and extend with warmup period
            for indices in split_indices:
                if len(indices) > 0:
                    original_start_idx = indices[0]
                    original_end_idx = indices[-1]

                    # Extend start index to include warmup period
                    extended_start_idx = max(
                        0, original_start_idx - warmup_length
                    )

                    # Return (extended_start, extended_end, original_start, original_end)
                    segments.append(
                        (
                            extended_start_idx,
                            original_end_idx,
                            original_start_idx,
                            original_end_idx,
                        )
                    )

        return segments


def simulate(
    config: Optional[Dict[str, Any]] = None,
    inputs: Optional[np.ndarray] = None,
    qobs: Optional[np.ndarray] = None,
    model_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Unified simulation interface for all hydrological models.

    This is the main entry point for model simulation, providing flexible usage:
    1. Traditional config-based approach (for backward compatibility)
    2. New flexible approach with separate model config and input data

    Parameters
    ----------
    config : Dict[str, Any], optional
        Traditional simulation configuration containing data_cfgs and model_cfgs
        (used for backward compatibility)
    inputs : np.ndarray, optional
        Input data array with shape [time, basin, features]
        Features typically include [precipitation, potential_evapotranspiration]
    qobs : np.ndarray, optional
        Observed streamflow data with shape [time, basin, 1]
    model_config : Dict[str, Any], optional
        Model configuration containing model_name, model_params, and parameters
    **kwargs
        Additional arguments (warmup_length, is_event_data, etc.)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing simulation results and metadata

    Examples
    --------
    # New flexible approach (recommended)
    >>> model_config = {
    ...     "model_name": "xaj_mz",
    ...     "model_params": {"source_type": "sources", "source_book": "HF"},
    ...     "parameters": {
    ...         "K": 0.5, "B": 0.3, "IM": 0.01, "UM": 20, "LM": 80,
    ...         "DM": 120, "C": 0.15, "SM": 50, "EX": 1.0, "KI": 0.3,
    ...         "KG": 0.2, "A": 0.8, "THETA": 0.2, "CI": 0.8, "CG": 0.15
    ...     }
    ... }
    >>> # Create simulator once
    >>> simulator = UnifiedSimulator(model_config)
    >>> # Run with different inputs
    >>> results1 = simulator.simulate(inputs1, qobs=qobs1, warmup_length=365)
    >>> results2 = simulator.simulate(inputs2, qobs=qobs2, warmup_length=365)

    # Traditional approach (backward compatibility)
    >>> config = {
    ...     "data_cfgs": {
    ...         "data_source_type": "selfmadehydrodataset",
    ...         "data_source_path": "/path/to/data",
    ...         "basin_ids": ["basin_001"],
    ...         "warmup_length": 365
    ...     },
    ...     "model_cfgs": {
    ...         "model_name": "xaj_mz",
    ...         "model_params": {"source_type": "sources", "source_book": "HF"},
    ...         "parameters": {...}
    ...     }
    ... }
    >>> results = simulate(config)
    """

    # Handle different usage patterns
    if config is not None:
        # Traditional config-based approach (backward compatibility)
        return _simulate_with_config(config, **kwargs)
    elif model_config is not None and inputs is not None:
        # New flexible approach
        return _simulate_with_inputs(model_config, inputs, qobs, **kwargs)
    else:
        raise ValueError(
            "Must provide either 'config' (traditional approach) or "
            "'model_config' and 'inputs' (flexible approach)"
        )


def _simulate_with_config(config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Traditional config-based simulation for backward compatibility."""
    # Validate configuration
    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary")

    if "data_cfgs" not in config:
        raise ValueError("Config must contain 'data_cfgs' section")

    if "model_cfgs" not in config:
        raise ValueError("Config must contain 'model_cfgs' section")

    model_cfgs = config["model_cfgs"]
    if "model_name" not in model_cfgs:
        raise ValueError("model_cfgs must contain 'model_name'")

    if "parameters" not in model_cfgs:
        raise ValueError(
            "model_cfgs must contain 'parameters' with specific parameter values"
        )

    # Load data using the traditional approach
    data_config = config["data_cfgs"]
    data_loader = UnifiedDataLoader(data_config)
    inputs, qobs = data_loader.load_data()

    # Extract simulation parameters
    warmup_length = data_config.get("warmup_length", 365)
    is_event_data = data_loader.is_event_data()

    # Create and run simulator with the loaded data
    simulator = UnifiedSimulator(model_cfgs)
    results = simulator.simulate(
        inputs=inputs,
        qobs=qobs,
        warmup_length=warmup_length,
        is_event_data=is_event_data,
        **kwargs,
    )

    # Add traditional metadata
    results["metadata"]["basin_ids"] = data_config.get("basin_ids", [])
    results["metadata"]["data_source_type"] = data_config.get(
        "data_source_type"
    )

    return results


def _simulate_with_inputs(
    model_config: Dict[str, Any],
    inputs: np.ndarray,
    qobs: Optional[np.ndarray] = None,
    **kwargs,
) -> Dict[str, Any]:
    """New flexible simulation with separate model config and input data."""
    # Create and run simulator
    simulator = UnifiedSimulator(model_config)
    results = simulator.simulate(inputs=inputs, qobs=qobs, **kwargs)

    return results
