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


def get_model_output_names(model_name, return_state=False):
    """
    Get the names of output variables for different models.

    Parameters
    ----------
    model_name : str
        Name of the model
    return_state : bool
        Whether state variables are returned

    Returns
    -------
    list
        List of output variable names
    """
    # Base output for most models
    base_outputs = {
        "xaj": (
            ["qsim", "es"]
            if not return_state
            else ["qsim", "es", "w", "s", "fr", "qi", "qg"]
        ),
        "xaj_mz": (
            ["qsim", "es"]
            if not return_state
            else ["qsim", "es", "w", "s", "fr", "qi", "qg"]
        ),
        "dhf": (
            ["qsim"]
            if not return_state
            else [
                "qsim",
                "runoff",
                "y0",
                "yu",
                "yl",
                "y",
                "sa",
                "ua",
                "Pa",
            ]
        ),
        "hymod": (
            ["qsim", "et"]
            if not return_state
            else ["qsim", "et", "x_slow", "x_quick", "x_loss"]
        ),
        "gr1a": (
            ["qsim", "ets"] if not return_state else ["qsim", "ets", "s"]
        ),
        "gr2m": (
            ["qsim", "ets"] if not return_state else ["qsim", "ets", "s"]
        ),
        "gr3j": (
            ["qsim", "ets"] if not return_state else ["qsim", "ets", "s", "r"]
        ),
        "gr4j": (
            ["qsim", "ets"] if not return_state else ["qsim", "ets", "s", "r"]
        ),
        "gr5j": (
            ["qsim", "ets"] if not return_state else ["qsim", "ets", "s", "r"]
        ),
        "gr6j": (
            ["qsim", "ets"] if not return_state else ["qsim", "ets", "s", "r"]
        ),
        "semi_xaj": ["qsim"],  # Semi-XAJ typically returns only streamflow
        "unit_hydrograph": ["qsim"],
        "categorized_unit_hydrograph": ["qsim"],
    }

    return base_outputs.get(
        model_name, ["output_0", "output_1", "output_2"]
    )  # fallback names


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
        NOTE: Now we only support single basin simulation.

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
        # Convert parameter dictionary to list format
        param_names = list(self.parameters.keys())
        param_values = list(self.parameters.values())

        # Store parameter info for later use when we know number of basins
        self.param_names = param_names
        self.param_values = np.expand_dims(param_values, axis=0)

    def simulate(
        self,
        inputs: np.ndarray,
        qobs: Optional[np.ndarray] = None,
        warmup_length: int = 365,
        is_event_data: bool = False,
        return_intermediate: bool = True,
        return_warmup_states: bool = False,
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
        return_warmup_states : bool, default False
            Whether to return initial states after warmup period.
            Returns warmup states in simulation result dict
        **kwargs
            Additional arguments passed to the model function.
            Can include 'initial_states': Dict[str, Any] - Dictionary of initial
            state values to override after warmup. For DHF model, keys can
            include: "sa0", "ua0", "ya0"

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

        # Handle different simulation scenarios
        if is_event_data:
            # Event data with traditional models
            simulation_result = self._simulate_event_data(
                inputs,
                warmup_length,
                return_intermediate,
                return_warmup_states,
                **kwargs,
            )
        else:
            # Standard simulation
            simulation_result = self._simulate_continuous_data(
                inputs,
                warmup_length,
                return_intermediate,
                return_warmup_states,
                **kwargs,
            )

        return simulation_result

    def _process_model_result(
        self,
        model_result,
        output_names,
        return_warmup_states,
    ):
        """
        Process model result into dictionary format.

        Parameters
        ----------
        model_result : tuple or np.ndarray
            Raw model output
        output_names : list
            List of output variable names
        return_warmup_states : bool
            Whether warmup states should be included

        Returns
        -------
        dict
            Dictionary containing processed model results
        """
        if isinstance(model_result, tuple):
            # Check if last element is warmup_states dict
            if (
                return_warmup_states
                and len(model_result) > len(output_names)
                and isinstance(model_result[-1], dict)
            ):
                # Extract warmup states and process remaining results
                warmup_states = model_result[-1]
                model_arrays = model_result[:-1]
                result_dict = {
                    name: arr for name, arr in zip(output_names, model_arrays)
                }
                result_dict["warmup_states"] = warmup_states
            else:
                # Traditional case without warmup states
                result_dict = {
                    name: arr for name, arr in zip(output_names, model_result)
                }
        else:
            # Handle single array or tuple with warmup_states
            if return_warmup_states and isinstance(model_result, tuple):
                # Single array + warmup_states case
                result_dict = {output_names[0]: model_result[0]}
                result_dict["warmup_states"] = model_result[1]
            else:
                # Unit hydrograph models return single array
                result_dict = {output_names[0]: model_result}

        return result_dict

    def trans_sim_results_unit(
        self,
        simulation,
        output_unit="m^3/s",
        time_interval_hours=3.0,
    ):
        """
        Apply unit conversion to simulation results using basin configuration.

        Parameters
        ----------
        simulation : np.ndarray
            Simulation results
        output_unit : str, default "m^3/s"
            Target output unit for conversion
        time_interval_hours : float, default 3.0
            Time step in hours for the data

        Returns
        -------
        tuple
            tuple of (converted_simulation, unitconv_metadata)
        """
        if self.basin is not None and output_unit == "m^3/s":
            from hydroutils.hydro_units import streamflow_unit_conv

            # Get simulation results
            if simulation is not None:
                # Detect time interval from time series or use provided time step
                # Convert time_step_hours to integer format for time_interval
                if time_interval_hours.is_integer():
                    time_interval = f"{int(time_interval_hours)}h"
                else:
                    # Handle fractional hours by converting to minutes if < 1 hour
                    if time_interval_hours < 1:
                        time_interval_minutes = int(time_interval_hours * 60)
                        time_interval = f"{time_interval_minutes}m"
                    else:
                        # Round to nearest hour for other cases
                        time_interval = f"{round(time_interval_hours)}h"

                # Convert simulation results from mm/time to m³/s
                # simulation shape is [time, basin, 1]
                converted_simulation = np.zeros_like(simulation)

                for basin_idx in range(simulation.shape[1]):
                    # TODO: Only support 1 basin for now
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

                    converted_simulation[:, basin_idx, 0] = converted_discharge

                    # Add conversion metadata
                    unitconv_metadata = {
                        "applied": True,
                        "source_unit": f"mm/{time_interval}",
                        "target_unit": output_unit,
                    }

        else:
            # No unit conversion applied
            reason = "No basin configuration provided"
            if self.basin is not None:
                reason = (
                    f"Output unit '{output_unit}' does not require conversion"
                )
            unitconv_metadata = {
                "applied": False,
                "reason": reason,
                "source_unit": None,
                "target_unit": output_unit,
            }
        return converted_simulation, unitconv_metadata

    def _simulate_continuous_data(
        self,
        inputs: np.ndarray,
        warmup_length: int,
        return_intermediate: bool,
        return_warmup_states: bool,
        **kwargs,
    ) -> Dict[str, Any]:
        """Standard simulation for continuous data."""
        # Prepare model configuration
        model_config = dict(self.model_params)
        model_config.update(kwargs)

        # Run model simulation
        model_result = self.model_function(
            inputs,
            self.param_values,
            warmup_length=warmup_length,
            return_state=return_intermediate,
            return_warmup_states=return_warmup_states,
            **model_config,
        )

        # Convert model_result to dictionary based on model output names
        output_names = get_model_output_names(
            self.model_name, return_intermediate
        )

        result_dict = self._process_model_result(
            model_result, output_names, return_warmup_states
        )

        return result_dict

    def _simulate_event_data(
        self,
        inputs: np.ndarray,
        warmup_length: int,
        return_intermediate: bool,
        return_warmup_states: bool,
        **kwargs,
    ) -> Dict[str, Any]:
        """Special simulation for event data with traditional models."""
        # Validate that flood_event markers are present
        if inputs.shape[2] < 3:
            raise ValueError(
                "Event data simulation requires flood_event markers. "
                f"Expected input shape [time, basin, 3] with features [rain, pet, flood_event], "
                f"but got shape {inputs.shape}."
            )

        # Initialize output array and warmup states storage
        outputs = []
        event_warmup_states = None

        # Process each basin separately
        for basin_idx in range(inputs.shape[1]):
            # Find event segments using flood_event markers (including warmup period)
            flood_event_array = inputs[
                :, basin_idx, 2
            ]  # feature index 2 is flood_event
            event_segments = find_flood_event_segments_as_tuples(
                flood_event_array, warmup_length
            )

            basin_params = self.param_values

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
                    return_warmup_states=return_warmup_states,
                    **model_config,
                )

                # Convert event_result tuple to dictionary based on model output names
                output_names = get_model_output_names(
                    self.model_name, return_intermediate
                )

                event_dict = self._process_model_result(
                    event_result, output_names, return_warmup_states
                )

                # Store warmup states for later use (only from first event)
                if j == 0 and "warmup_states" in event_dict:
                    event_warmup_states = event_dict["warmup_states"]

                if j == 0:
                    # Initialize output dictionaries for each variable (excluding warmup_states)
                    simulation_output = {}
                    for name, arr in event_dict.items():
                        if (
                            name != "warmup_states"
                        ):  # Skip warmup_states as it's not a time series
                            simulation_output[name] = np.zeros(
                                (inputs.shape[0], inputs.shape[1], 1)
                            )

                # Save the event result to its location in long time series data (excluding warmup_states)
                for name, arr in event_dict.items():
                    if (
                        name != "warmup_states"
                    ):  # Skip warmup_states as it's not a time series
                        simulation_output[name][
                            original_start : original_end + 1,
                            basin_idx : basin_idx + 1,
                            :,
                        ] = arr

            outputs.append(simulation_output)

        # Combine outputs from all basins into final output dictionary
        final_output = {}
        for var_name in outputs[0].keys():
            basin_arrays = [output[var_name] for output in outputs]
            final_output[var_name] = np.concatenate(basin_arrays, axis=1)

        # Add warmup states if requested and available
        if return_warmup_states and event_warmup_states is not None:
            final_output["warmup_states"] = event_warmup_states

        return final_output
