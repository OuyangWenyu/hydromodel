r"""
Author: Wenyu Ouyang
Date: 2025-08-07
LastEditTime: 2025-08-12 11:00:08
LastEditors: Wenyu Ouyang
Description: Unified simulation interface for all hydrological models
FilePath: \hydromodel\hydromodel\core\unified_simulate.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import numpy as np
from typing import Dict, Any, Optional, Union
from collections import OrderedDict

from hydroutils.hydro_event import find_flood_event_segments_as_tuples
from hydromodel.models.model_dict import MODEL_DICT
from .basin import Basin


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

    def __init__(
        self,
        model_config: Dict[str, Any],
        basin_config: Optional[Union[Basin, Dict[str, Any]]] = None,
    ):
        """
        Initialize the unified simulator with model configuration and basin information.

        Parameters
        ----------
        model_config : Dict[str, Any]
            Model configuration containing:
            - model_name: Name of the model to use
            - model_params: Model-specific parameters (structure, etc.)
            - parameters: Specific parameter values for simulation
        basin_config : Basin or Dict[str, Any], optional
            Basin configuration for unit conversion and modeling approach.
            If dict provided, it will be converted to Basin instance.
            If None, unit conversion will be disabled.
        """
        self.model_config = model_config

        # Extract model information
        self.model_name = self.model_config["model_name"]
        self.model_params = self.model_config.get("model_params", {})
        self.parameters = OrderedDict(self.model_config.get("parameters", {}))

        # Store basin configuration
        if basin_config is not None:
            if isinstance(basin_config, dict):
                self.basin = Basin.from_config(basin_config)
            else:
                self.basin = basin_config
        else:
            self.basin = None

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
        return_intermediate: bool = True,
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
        return_intermediate : bool, default True
            Whether to return intermediate results from model computation
        **kwargs
            Additional arguments passed to the model function

        Returns
        -------
        Dict[str, Any]
            Dictionary containing simulation results:
            - simulation: Model simulation output [time, basin, 1]
            - observation: Observed data (if provided) [time, basin, 1]
            - input_data: Input data used for simulation [time, basin, n_features]
            - intermediate: Intermediate results (if return_intermediate=True)
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
            simulation_result = self._simulate_event_data(
                inputs, warmup_length, return_intermediate, **kwargs
            )
        else:
            # Standard simulation
            simulation_result = self._simulate_continuous_data(
                inputs, warmup_length, return_intermediate, **kwargs
            )

        # Extract simulation output and intermediate results
        if isinstance(simulation_result, dict):
            simulation_output = simulation_result["simulation"]
            intermediate_results = simulation_result.get("intermediate", None)
        else:
            # Backward compatibility: if only simulation array is returned
            simulation_output = simulation_result
            intermediate_results = None

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
                "return_intermediate": return_intermediate,
            },
        }

        # Add intermediate results if available
        if intermediate_results is not None:
            results["intermediate"] = intermediate_results

        # Apply unit conversion if basin configuration and output unit are available
        results = self._trans_sim_results_unit(
            results,
            output_unit=kwargs.get("output_unit", "m^3/s"),
            time_step_hours=kwargs.get("time_step_hours", 3.0),
        )

        return results

    def _trans_sim_results_unit(
        self,
        results,
        output_unit="m^3/s",
        time_step_hours=3.0,
    ):
        """
        Apply unit conversion to simulation results using basin configuration.

        Parameters
        ----------
        results : Dict[str, Any]
            Simulation results dictionary
        output_unit : str, default "m^3/s"
            Target output unit for conversion
        time_step_hours : float, default 3.0
            Time step in hours for the data
        time_series : Optional
            Time series data for detecting time interval (optional)

        Returns
        -------
        Dict[str, Any]
            Updated results with unit conversion applied
        """
        if self.basin is not None and output_unit == "m^3/s":
            try:
                from hydroutils.hydro_units import streamflow_unit_conv

                # Get simulation results
                simulation = results.get("simulation")
                if simulation is not None:
                    # Detect time interval from time series or use provided time step
                    # Convert time_step_hours to integer format for time_interval
                    if time_step_hours.is_integer():
                        time_interval = f"{int(time_step_hours)}h"
                    else:
                        # Handle fractional hours by converting to minutes if < 1 hour
                        if time_step_hours < 1:
                            time_interval_minutes = int(time_step_hours * 60)
                            time_interval = f"{time_interval_minutes}m"
                        else:
                            # Round to nearest hour for other cases
                            time_interval = f"{round(time_step_hours)}h"

                    # Convert simulation results from mm/time to m³/s
                    # simulation shape is [time, basin, 1]
                    converted_simulation = np.zeros_like(simulation)

                    for basin_idx in range(simulation.shape[1]):
                        basin_simulation = simulation[
                            :, basin_idx, 0
                        ]  # Extract time series for this basin

                        # Get basin area for this unit (supports semi-distributed)
                        basin_area_km2 = self.basin.unit_areas

                        converted_discharge = streamflow_unit_conv(
                            data=basin_simulation,
                            area=basin_area_km2,
                            target_unit=output_unit,
                            source_unit=f"mm/{time_interval}",
                            area_unit="km^2",
                        )

                        converted_simulation[:, basin_idx, 0] = (
                            converted_discharge
                        )

                    # Update results with converted simulation
                    results["simulation"] = converted_simulation

                    # Add conversion metadata
                    results["metadata"]["unit_conversion"] = {
                        "applied": True,
                        "source_unit": f"mm/{time_interval}",
                        "target_unit": output_unit,
                        "basin_info": {
                            "basin_id": self.basin.basin_id,
                            "basin_name": self.basin.basin_name,
                            "total_area_km2": self.basin.total_area_km2,
                            "modeling_approach": self.basin.modeling_approach,
                        },
                        "time_interval": time_interval,
                    }

            except ImportError:
                # If hydroutils is not available, just add a note
                results["metadata"]["unit_conversion"] = {
                    "applied": False,
                    "error": "hydroutils not available for unit conversion",
                }
            except Exception as e:
                # If conversion fails, add error info but don't fail the simulation
                results["metadata"]["unit_conversion"] = {
                    "applied": False,
                    "error": f"Unit conversion failed: {str(e)}",
                }
        else:
            # No unit conversion applied
            reason = "No basin configuration provided"
            if self.basin is not None:
                reason = (
                    f"Output unit '{output_unit}' does not require conversion"
                )

            results["metadata"]["unit_conversion"] = {
                "applied": False,
                "reason": reason,
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
        self,
        inputs: np.ndarray,
        warmup_length: int,
        return_intermediate: bool,
        **kwargs,
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """Standard simulation for continuous data."""
        # Prepare model configuration
        model_config = dict(self.model_params)
        model_config.update(kwargs)

        # Run model simulation
        model_result = self.model_function(
            inputs,
            self.param_array,
            warmup_length=warmup_length,
            return_state=return_intermediate,
            **model_config,
        )
        if return_intermediate:
            return model_result
        # Handle different return formats
        if isinstance(model_result, tuple):
            # Traditional models return (simulation, states, ...)
            simulation_output = model_result[0]
        else:
            # Unit hydrograph models return single array
            simulation_output = model_result

        return simulation_output

    def _simulate_event_data(
        self,
        inputs: np.ndarray,
        warmup_length: int,
        return_intermediate: bool,
        **kwargs,
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
        outputs = []

        # Process each basin separately
        for basin_idx in range(inputs.shape[1]):
            # Find event segments using flood_event markers (including warmup period)
            flood_event_array = inputs[
                :, basin_idx, 2
            ]  # feature index 2 is flood_event
            event_segments = find_flood_event_segments_as_tuples(
                flood_event_array, warmup_length
            )

            # Get basin-specific parameters
            if self.param_array.shape[0] > 1:
                basin_params = self.param_array[basin_idx : basin_idx + 1, :]
            else:
                basin_params = self.param_array

            # Process each event segment
            for j, (
                extended_start,
                extended_end,
                original_start,
                original_end,
            ) in enumerate(event_segments):
                # Extract event data (including warmup period)
                event_inputs = inputs[
                    extended_start : extended_end + 1,
                    basin_idx : basin_idx + 1,
                    :,
                ]
                # Run model on this event segment
                model_config = dict(self.model_params)
                model_config.update(kwargs)

                event_result = self.model_function(
                    event_inputs,
                    basin_params,
                    warmup_length=warmup_length,
                    return_state=return_intermediate,
                    **model_config,
                )
                event_sim = np.concatenate(event_result, axis=-1)
                if j == 0:
                    simulation_output = np.zeros(
                        (inputs.shape[0], inputs.shape[1], event_sim.shape[-1])
                    )
                # save the event result to its location in long time series data
                simulation_output[
                    original_start : original_end + 1,
                    basin_idx : basin_idx + 1,
                    :,
                ] = event_sim
            outputs.append(simulation_output)
        output_arr = np.concatenate(outputs, axis=1)
        return output_arr
