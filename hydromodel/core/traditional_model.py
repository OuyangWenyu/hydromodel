import numpy as np
from typing import Dict, Any, Optional, Union
from collections import OrderedDict

from joblib.testing import param

from hydromodel.models.model_dict import MODEL_DICT
from hydromodel.core.unified_simulate import get_model_output_names, UnifiedSimulator
from .basin import Basin


class TraditionalModel:
    """
    传统水文模型接口
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
        self.model_type = self.model_config["type"]
        self.model_name = self.model_config["model_name"]
        self.model_params = self.model_config.get("model_params", {})
        self.parameters = OrderedDict(self.model_config.get("parameters", {}))

        #Store basin configuration
        if basin_config is not None:
            if isinstance(basin_config, dict):
                self.basin = Basin.from_config(basin_config)
            else:
                self.basin = basin_config
        else:
            self.basin = None

        # Validata model exists
        if self.model_name not in MODEL_DICT:
            raise ValueError(
                f"Model '{self.model_name}' not found in MODEL_DICT"
            )

        # Get model function
        self.model_function = MODEL_DICT[self.model_name]

    def _setup_parameters(self):
        """
        Setup model parameters for simulation.
        Convert parameter dictionary to array format expected by models
        """
        if not self.parameters:
            raise ValueError(
                f"Model '{self.model_name}' requires parameters to be specified"
            )
        #Convert parameter dictionary to lisr format
        param_names = list(self.parameters.keys())
        param_values = self.parameters.values()

        #Store parameter info for later use when we know number of basins
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
                f"Input data must be 3D array [time, basin, features], got shape {inputs.ndim}"
            )

        model_result = self.model_function(
            inputs,
            self.param_values,
            warmup_length=warmup_length,
            return_state=return_intermediate,
            return_warmup_states=return_warmup_states,
            **self.model_params
        )

        output_names = get_model_output_names(self.model_name, return_intermediate)

        # This processing logic is simplified from UnifiedSimulator._process_model_result
        if isinstance(model_result, tuple):
            result_dict = {name: arr for name, arr in zip(output_names, model_result)}
        else:
            result_dict = {output_names[0]: model_result}

        return result_dict


