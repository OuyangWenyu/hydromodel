"""
Author: Wenyu Ouyang
Date: 2025-08-07
LastEditTime: 2025-08-07 21:00:00
LastEditors: Wenyu Ouyang
Description: Unified simulation interface for all hydrological models
FilePath: /hydromodel/hydromodel/core/unified_simulate.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from abc import ABC, abstractmethod

from hydromodel.models.model_dict import MODEL_DICT
from hydromodel.configs.unified_config import UnifiedConfig
from hydromodel.datasets.unified_data_loader import UnifiedDataLoader


class UnifiedSimulator:
    """
    Unified simulator for all hydrological models.
    
    This class provides a single interface for running simulations with any model type
    using a simplified configuration format. Unlike calibration, simulation requires
    specific parameter values rather than parameter ranges.
    
    Key features:
    - Single interface for all models (XAJ, GR series, unit hydrograph, etc.)
    - Simplified configuration (no training/evaluation configs needed)
    - Direct parameter specification
    - Consistent output format across all models
    - Support for both single and multi-basin simulations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the unified simulator.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Simulation configuration containing:
            - data_cfgs: Data configuration (same as calibration)
            - model_cfgs: Model configuration with specific parameter values
            - simulation_cfgs: Optional simulation-specific settings
        """
        self.config = config
        
        # Extract configuration sections
        self.data_config = config["data_cfgs"]
        self.model_config = config["model_cfgs"]
        self.simulation_config = config.get("simulation_cfgs", {})
        
        # Extract model information
        self.model_name = self.model_config["model_name"]
        self.model_params = self.model_config.get("model_params", {})
        self.parameters = self.model_config.get("parameters", {})
        
        # Get warmup length
        self.warmup_length = self.data_config.get("warmup_length", 365)
        
        # Load data using unified data loader
        self.data_loader = UnifiedDataLoader(self.data_config)
        self.p_and_e, self.qobs = self.data_loader.load_data()
        
        # Store whether this is event data
        self.is_event_data = self.data_loader.is_event_data()
        
        # Setup parameter handling
        self._setup_parameters()
        
    def _setup_parameters(self):
        """Setup model parameters for simulation."""
        if self.model_name in ["unit_hydrograph", "categorized_unit_hydrograph"]:
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
                raise ValueError("unit_hydrograph model requires 'uh_values' in parameters")
            
            # Convert to numpy array and normalize to sum to 1
            self.param_array = np.array(uh_values)
            self.param_array = self.param_array / np.sum(self.param_array)
            self.param_array = self.param_array.reshape(1, -1)
            
        elif self.model_name == "categorized_unit_hydrograph":
            # Categorized unit hydrograph: expect dictionary with category values
            uh_categories = self.parameters.get("uh_categories")
            thresholds = self.parameters.get("thresholds", {
                "small_medium": 10.0, "medium_large": 25.0
            })
            
            if uh_categories is None:
                raise ValueError("categorized_unit_hydrograph model requires 'uh_categories' in parameters")
            
            # Convert to expected format
            param_dict = {}
            for category, values in uh_categories.items():
                normalized_values = np.array(values) / np.sum(values)
                param_dict[category] = normalized_values.reshape(1, -1)
            
            param_dict["thresholds"] = thresholds
            self.param_array = param_dict
    
    def _setup_traditional_model_params(self):
        """Setup parameters for traditional models (XAJ, GR series, etc.)."""
        if not self.parameters:
            raise ValueError(f"Model {self.model_name} requires parameters to be specified")
        
        # Convert parameter dictionary to array format expected by models
        param_names = list(self.parameters.keys())
        param_values = list(self.parameters.values())
        
        # Create parameter array [1, n_params] for single basin
        # For multi-basin, this would be [n_basins, n_params]
        n_basins = self.p_and_e.shape[1]
        if n_basins == 1:
            self.param_array = np.array(param_values).reshape(1, -1)
        else:
            # For multi-basin, replicate parameters for each basin
            # Or user should provide basin-specific parameters
            basin_params = self.parameters.get("basin_specific", False)
            if basin_params and isinstance(param_values[0], (list, np.ndarray)):
                # Basin-specific parameters provided
                self.param_array = np.array(param_values)
                if self.param_array.shape[0] != n_basins:
                    raise ValueError(f"Basin-specific parameters shape {self.param_array.shape[0]} "
                                   f"does not match number of basins {n_basins}")
            else:
                # Replicate same parameters for all basins
                self.param_array = np.tile(param_values, (n_basins, 1))
    
    def simulate(self) -> Dict[str, Any]:
        """
        Run model simulation using the configured parameters.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing simulation results:
            - simulation: Model simulation output [time, basin, 1]
            - observation: Original observations (if available) [time, basin, 1]
            - input_data: Input data used for simulation [time, basin, n_features]
            - metadata: Simulation metadata and configuration
        """
        # Get model function
        model_function = MODEL_DICT[self.model_name]
        
        # Handle different simulation scenarios
        if self.is_event_data and self.model_name not in [
            "unit_hydrograph", "categorized_unit_hydrograph"
        ]:
            # Event data with traditional models
            simulation_output = self._simulate_event_data(model_function)
        else:
            # Standard simulation
            simulation_output = self._simulate_continuous_data(model_function)
        
        # Prepare results
        results = {
            "simulation": simulation_output,
            "observation": self.qobs,
            "input_data": self.p_and_e,
            "metadata": {
                "model_name": self.model_name,
                "model_params": self.model_params,
                "parameters": self.parameters,
                "warmup_length": self.warmup_length,
                "basin_ids": self.data_config.get("basin_ids", []),
                "data_source_type": self.data_config.get("data_source_type"),
                "simulation_shape": simulation_output.shape,
                "time_steps": simulation_output.shape[0],
                "n_basins": simulation_output.shape[1],
            }
        }
        
        return results
    
    def _simulate_continuous_data(self, model_function) -> np.ndarray:
        """Standard simulation for continuous data."""
        # Prepare model configuration
        model_config = dict(self.model_params)
        
        # Add parameter range for traditional models (empty for unit hydrograph)
        if self.model_name not in ["unit_hydrograph", "categorized_unit_hydrograph"]:
            # For traditional models, we don't need param_range during simulation
            # The parameters are already provided as specific values
            param_range = {}
        else:
            param_range = {}
        
        # Run model simulation
        model_result = model_function(
            self.p_and_e,
            self.param_array,
            warmup_length=self.warmup_length,
            **model_config,
            **param_range,
        )
        
        # Handle different return formats
        if isinstance(model_result, tuple):
            # Traditional models return (simulation, states, ...)
            simulation_output = model_result[0]
        else:
            # Unit hydrograph models return single array
            simulation_output = model_result
        
        return simulation_output
    
    def _simulate_event_data(self, model_function) -> np.ndarray:
        """Special simulation for event data with traditional models."""
        # Extract precipitation data to identify events
        net_rain = self.p_and_e[:, :, 0]  # [time, basin]
        
        # Initialize output array
        output_shape = (self.p_and_e.shape[0], self.p_and_e.shape[1], 1)
        simulation_output = np.zeros(output_shape)
        
        # Process each basin separately
        for basin_idx in range(self.p_and_e.shape[1]):
            basin_rain = net_rain[:, basin_idx]
            
            # Find event segments
            event_segments = self._find_event_segments(basin_rain)
            
            # Get basin-specific parameters
            if self.param_array.shape[0] > 1:
                basin_params = self.param_array[basin_idx:basin_idx+1, :]
            else:
                basin_params = self.param_array
            
            # Process each event segment
            for start_idx, end_idx in event_segments:
                # Extract event data
                event_p_and_e = self.p_and_e[
                    start_idx:end_idx+1, basin_idx:basin_idx+1, :
                ]
                
                try:
                    # Run model on this event segment
                    event_result = model_function(
                        event_p_and_e,
                        basin_params,
                        warmup_length=0,  # No warmup for event segments
                        **self.model_params,
                    )
                    
                    # Extract simulation output
                    if isinstance(event_result, tuple):
                        event_sim = event_result[0]
                    else:
                        event_sim = event_result
                    
                    # Store in output array
                    simulation_output[
                        start_idx:end_idx+1, basin_idx:basin_idx+1, :
                    ] = event_sim
                    
                except Exception as e:
                    print(f"Warning: Event simulation failed for basin {basin_idx}, "
                          f"segment {start_idx}-{end_idx}: {e}")
                    # Fill with zeros on failure
                    simulation_output[
                        start_idx:end_idx+1, basin_idx:basin_idx+1, :
                    ] = 0.0
        
        return simulation_output
    
    def _find_event_segments(self, rain_series: np.ndarray, 
                            min_gap_length: int = 1) -> List[Tuple[int, int]]:
        """Find continuous event segments in rain time series."""
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
                start_idx = max(0, indices[0])
                end_idx = min(len(rain_series) - 1, indices[-1])
                segments.append((start_idx, end_idx))
        
        return segments


def simulate(config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Unified simulation interface for all hydrological models.
    
    This is the main entry point for model simulation, providing a single
    function interface similar to the calibrate() function.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Simulation configuration dictionary containing:
        - data_cfgs: Data configuration (path, basins, time periods, etc.)
        - model_cfgs: Model configuration with specific parameter values
        - simulation_cfgs: Optional simulation-specific settings
    **kwargs
        Additional arguments passed to the simulator
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing simulation results and metadata
    
    Examples
    --------
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
    ...         "parameters": {
    ...             "K": 0.5, "B": 0.3, "IM": 0.01, "UM": 20, "LM": 80, 
    ...             "DM": 120, "C": 0.15, "SM": 50, "EX": 1.0, "KI": 0.3, 
    ...             "KG": 0.2, "A": 0.8, "THETA": 0.2, "CI": 0.8, "CG": 0.15
    ...         }
    ...     }
    ... }
    >>> results = simulate(config)
    >>> print(f"Simulation shape: {results['simulation'].shape}")
    """
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
        raise ValueError("model_cfgs must contain 'parameters' with specific parameter values")
    
    # Create and run simulator
    simulator = UnifiedSimulator(config, **kwargs)
    results = simulator.simulate()
    
    return results