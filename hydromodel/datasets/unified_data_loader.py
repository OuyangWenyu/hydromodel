"""
Author: Wenyu Ouyang
Date: 2025-01-22
LastEditTime: 2025-08-07 16:48:21
LastEditors: Wenyu Ouyang
Description: Unified data loading interface for all hydrological models
FilePath: /hydromodel/hydromodel/datasets/unified_data_loader.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path

# Import different datasource types
try:
    from hydrodatasource.reader.floodevent import FloodEventDatasource

    FLOODEVENT_AVAILABLE = True
except ImportError:
    FLOODEVENT_AVAILABLE = False
    FloodEventDatasource = None

try:
    from hydrodataset import Camels
    CAMELS_AVAILABLE = True
except ImportError:
    CAMELS_AVAILABLE = False
    Camels = None

try:
    from hydrodatasource.reader.data_source import SelfMadeHydroDataset

    SELFMADE_AVAILABLE = True
except ImportError:
    SELFMADE_AVAILABLE = False
    SelfMadeHydroDataset = None

HYDRODATASOURCE_AVAILABLE = (
    FLOODEVENT_AVAILABLE or CAMELS_AVAILABLE or SELFMADE_AVAILABLE
)


class UnifiedDataLoader:
    """
    Unified data loader that handles both continuous time series and flood events data.

    This class provides a consistent interface for loading different types of hydrological data:
    - Continuous time series (traditional models like XAJ, GR series)
    - Flood event series (unit hydrograph models)

    Key features:
    - Uses hydrodatasource.read_ts_xrdataset() as the unified interface
    - Handles data type detection and appropriate loader selection
    - Converts all data to the standard (p_and_e, qobs) format
    - Supports both event-based and continuous data seamlessly
    """

    def __init__(self, data_config: Dict[str, Any]):
        """
        Initialize the unified data loader.

        Parameters
        ----------
        data_config : Dict[str, Any]
            Data configuration dictionary containing:
            - data_type: Type of data source ('floodevent', 'camels', 'selfmade', etc.)
            - data_path: Path to the data
            - basin_ids: List of basin identifiers
            - time_periods: Time period configuration
            - variables: List of variables to load
            - warmup_length: Warmup period length
            - **kwargs: Additional datasource-specific parameters
        """
        self.config = data_config
        self.data_type = data_config.get("data_type", "selfmade")
        self.data_path = data_config.get("data_path")
        self.basin_ids = data_config.get("basin_ids", [])
        self.warmup_length = data_config.get("warmup_length", 365)

        # Get time periods
        time_periods = data_config.get("time_periods", {})
        self.time_range = time_periods.get(
            "calibration", ["2014-10-01", "2019-09-30"]
        )

        # Get variable names
        self.variables = data_config.get(
            "variables", ["prcp", "PET", "streamflow"]
        )

        # Initialize the appropriate datasource
        self.datasource = self._create_datasource()

    def _create_datasource(self) -> Any:
        """Create the appropriate datasource based on data_type."""
        if not HYDRODATASOURCE_AVAILABLE:
            raise ImportError(
                "hydrodatasource package is required for unified data loading"
            )

        if self.data_type == "floodevent":
            # Flood event data source
            if not FLOODEVENT_AVAILABLE:
                raise ImportError(
                    "FloodEventDatasource not available. Please install hydrodatasource."
                )
            return FloodEventDatasource(
                data_path=self.data_path,
                dataset_name=self.config.get("dataset_name", "events"),
                time_unit=self.config.get("time_unit", ["3h"]),
                rain_key=self.config.get("rain_key", "rain"),
                net_rain_key=self.config.get("net_rain_key", "P_eff"),
                obs_flow_key=self.config.get("obs_flow_key", "Q_obs_eff"),
                warmup_length=self.warmup_length,
                **self.config.get("datasource_kwargs", {}),
            )
        elif self.data_type == "camels":
            # CAMELS data source
            if not CAMELS_AVAILABLE:
                raise ImportError(
                    "Camels not available. Please install hydrodatasource."
                )
            return Camels(self.data_path)
        elif self.data_type == "selfmade":
            # Self-made hydro dataset
            if not SELFMADE_AVAILABLE:
                raise ImportError(
                    "SelfMadeHydroDataset not available. Please install hydrodatasource."
                )
            return SelfMadeHydroDataset(
                data_path=self.data_path,
                download=False,
                time_unit=self.config.get("time_unit", ["1D"]),
                dataset_name=self.config.get("dataset_name", "selfmade"),
                **self.config.get("datasource_kwargs", {}),
            )
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data using the unified interface and return in standard format.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (p_and_e, qobs) tuple where:
            - p_and_e: Precipitation and PET data [time, basin, features=2]
            - qobs: Observed streamflow [time, basin, features=1]
        """
        # Use read_ts_xrdataset as the unified interface
        ts_data = self.datasource.read_ts_xrdataset(
            gage_id_lst=self.basin_ids,
            t_range=self.time_range,
            var_lst=self.variables,
            **self.config.get("read_kwargs", {}),
        )

        # Handle different return types from read_ts_xrdataset
        if isinstance(ts_data, dict):
            # Multiple time units - select the first one
            time_unit = list(ts_data.keys())[0]
            xr_dataset = ts_data[time_unit]
        else:
            # Single xarray dataset
            xr_dataset = ts_data

        # Convert to standard (p_and_e, qobs) format
        return self._convert_xr_to_standard_format(xr_dataset)

    def _convert_xr_to_standard_format(
        self, xr_dataset: xr.Dataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert xarray dataset to standard (p_and_e, qobs) format.

        Parameters
        ----------
        xr_dataset : xr.Dataset
            Input xarray dataset from read_ts_xrdataset

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (p_and_e, qobs) in standard format
        """
        if self.data_type == "floodevent":
            return self._convert_event_data(xr_dataset)
        else:
            return self._convert_continuous_data(xr_dataset)

    def _convert_event_data(
        self, xr_dataset: xr.Dataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert flood event data to standard format.

        For flood events, the data comes with event markers and may have gaps.
        We need to handle this appropriately for different model types.
        """
        # Extract variables based on data_type configuration
        net_rain_key = self.config.get("net_rain_key", "P_eff")
        obs_flow_key = self.config.get("obs_flow_key", "Q_obs_eff")

        # Check if we have the expected variables
        if net_rain_key not in xr_dataset.data_vars:
            raise ValueError(
                f"Expected variable '{net_rain_key}' not found in dataset"
            )
        if obs_flow_key not in xr_dataset.data_vars:
            raise ValueError(
                f"Expected variable '{obs_flow_key}' not found in dataset"
            )

        # For event data, we typically only have net rain, not separate P and PET
        # So we create a dummy PET array of zeros
        net_rain = xr_dataset[net_rain_key].values  # [time, basin]
        obs_flow = xr_dataset[obs_flow_key].values  # [time, basin]

        # Create dummy PET (zeros) to maintain the standard format
        dummy_pet = np.zeros_like(net_rain)

        # Stack P and E: [time, basin, features=2]
        p_and_e = np.stack([net_rain, dummy_pet], axis=2)

        # Expand qobs: [time, basin, features=1]
        qobs = np.expand_dims(obs_flow, axis=2)

        return p_and_e, qobs

    def _convert_continuous_data(
        self, xr_dataset: xr.Dataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert continuous time series data to standard format.

        This handles traditional continuous data sources like CAMELS.
        """
        # Standard variable names for continuous data
        var_mapping = {
            "prcp": ["prcp", "precipitation", "P"],
            "pet": ["PET", "pet", "potential_evapotranspiration", "E"],
            "flow": ["streamflow", "flow", "Q", "discharge"],
        }

        # Find the actual variable names in the dataset
        prcp_var = self._find_variable_name(xr_dataset, var_mapping["prcp"])
        pet_var = self._find_variable_name(xr_dataset, var_mapping["pet"])
        flow_var = self._find_variable_name(xr_dataset, var_mapping["flow"])

        if not all([prcp_var, pet_var, flow_var]):
            raise ValueError(
                f"Could not find required variables in dataset. Available: {list(xr_dataset.data_vars)}"
            )

        # Extract data
        prcp = xr_dataset[prcp_var].values
        pet = xr_dataset[pet_var].values
        flow = xr_dataset[flow_var].values

        # Stack P and E: [time, basin, features=2]
        p_and_e = np.stack([prcp, pet], axis=2)

        # Expand qobs: [time, basin, features=1]
        qobs = np.expand_dims(flow, axis=2)

        return p_and_e, qobs

    def _find_variable_name(
        self, xr_dataset: xr.Dataset, possible_names: List[str]
    ) -> Optional[str]:
        """Find the actual variable name from a list of possible names."""
        for name in possible_names:
            if name in xr_dataset.data_vars:
                return name
        return None

    def get_event_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get event metadata if available (for flood event data).

        Returns
        -------
        Optional[Dict[str, Any]]
            Event metadata if available, None otherwise
        """
        if self.data_type == "floodevent" and hasattr(
            self.datasource, "get_event_metadata"
        ):
            return self.datasource.get_event_metadata()
        return None

    def is_event_data(self) -> bool:
        """Check if this is event-based data."""
        return self.data_type == "floodevent"

    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded data."""
        return {
            "data_type": self.data_type,
            "data_path": self.data_path,
            "basin_ids": self.basin_ids,
            "time_range": self.time_range,
            "variables": self.variables,
            "warmup_length": self.warmup_length,
            "is_event_data": self.is_event_data(),
        }


def create_data_loader(data_config: Dict[str, Any]) -> UnifiedDataLoader:
    """
    Factory function to create a unified data loader.

    Parameters
    ----------
    data_config : Dict[str, Any]
        Data configuration dictionary

    Returns
    -------
    UnifiedDataLoader
        Configured data loader instance
    """
    return UnifiedDataLoader(data_config)


def load_data_from_config(
    data_config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenient function to load data directly from config.

    Parameters
    ----------
    data_config : Dict[str, Any]
        Data configuration dictionary

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (p_and_e, qobs) in standard format
    """
    loader = create_data_loader(data_config)
    return loader.load_data()
