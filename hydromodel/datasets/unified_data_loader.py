r"""
Author: Wenyu Ouyang
Date: 2025-01-22
LastEditTime: 2025-08-08 20:41:16
LastEditors: Wenyu Ouyang
Description: Unified data loading interface for all hydrological models
FilePath: /hydromodel/hydromodel/datasets/unified_data_loader.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import re
import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Tuple, Any
import os
import yaml
from pathlib import Path
from hydroutils.hydro_units import streamflow_unit_conv

# Import different datasource types
try:
    from hydrodataset import Camels
    from hydrodataset import CamelsUs

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

HYDRODATASOURCE_AVAILABLE = CAMELS_AVAILABLE or SELFMADE_AVAILABLE


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

    # Unified variable mapping for all data sources
    VAR_MAPPING = {
        "prcp": ["prcp", "precipitation", "P", "rain", "rainfall"],
        "pet": ["PET", "pet", "potential_evapotranspiration", "ES"],
        "flow": [
            "streamflow",
            "flow",
            "Q",
            "discharge",
            "inflow",
            "Q_obs",
        ],
        "area": [
            "area_gages2",
            "area",
            "basin_area",
            "drainage_area",
        ],
        "flood_event": ["flood_event", "event", "flood_indicator"],
    }

    def __init__(
        self, data_config: Dict[str, Any], is_train_val_test: str = "train"
    ):
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
        is_train_val_test : str
            "train", "valid", "test"
        """
        self.config = data_config
        # Support both naming conventions for backward compatibility
        self.data_type = data_config.get("data_source_type")
        self.data_path = data_config.get("data_source_path")

        # Handle None data_path: try hydro_setting.yml first, then use default
        if self.data_path is None:
            self.data_path = self._get_default_data_path()

        self.basin_ids = data_config.get("basin_ids", [])
        self.warmup_length = data_config.get("warmup_length", 365)

        # Get time periods
        if is_train_val_test == "train":
            self.time_range = data_config.get("train_period", None)
        elif is_train_val_test == "valid":
            self.time_range = data_config.get("valid_period", None)
        elif is_train_val_test == "test":
            self.time_range = data_config.get("test_period", None)
        else:
            raise ValueError(f"Invalid is_train_val_test: {is_train_val_test}")
        self.is_train_val_test = is_train_val_test
        self.variables = data_config.get(
            "variables", ["prcp", "PET", "streamflow"]
        )

        # Initialize the appropriate datasource
        self.datasource = self._create_datasource()

    def _get_default_data_path(self) -> str:
        """
        Get default data path based on data_source_type.
        Tries to load from hydro_setting.yml first, then uses default ~/hydromodel_data/.

        Returns
        -------
        str
            Data path
        """
        data_path = None

        # Try to load from hydro_setting.yml
        try:
            setting_file = os.path.join(Path.home(), "hydro_setting.yml")
            if os.path.exists(setting_file):
                with open(setting_file, "r", encoding="utf-8") as f:
                    settings = yaml.safe_load(f)

                if settings and "local_data_path" in settings:
                    datasets_origin = settings["local_data_path"].get("datasets-origin")
                    basins_origin = settings["local_data_path"].get("basins-origin")

                    # Determine path based on data_source_type
                    if self.data_type in ["selfmadehydrodataset", "floodevent"]:
                        # For custom data, use basins-origin directly
                        if basins_origin:
                            data_path = basins_origin
                    else:
                        # For standard datasets (camels_us, etc.), use datasets-origin directly
                        # aqua_fetch will automatically append the dataset directory name (e.g., CAMELS_US)
                        if datasets_origin:
                            data_path = datasets_origin

                    if data_path:
                        print(f"使用 hydro_setting.yml 中的路径: {data_path}")
        except Exception as e:
            print(f"Warning: 无法从 hydro_setting.yml 加载路径: {e}")

        # If still None, use default path (consistent with hydromodel.__init__.py)
        if data_path is None:
            default_root = os.path.join(Path.home(), "hydromodel_data")

            if self.data_type in ["selfmadehydrodataset", "floodevent"]:
                # For custom data
                data_path = os.path.join(default_root, "basins-origin")
            else:
                # For standard datasets: use datasets-origin directly
                # aqua_fetch will automatically append the dataset directory name (e.g., CAMELS_US)
                data_path = os.path.join(default_root, "datasets-origin")

            print(f"使用默认路径: {data_path}")

        return data_path

    def _create_datasource(self) -> Any:
        """Create the appropriate datasource based on data_type."""
        if not HYDRODATASOURCE_AVAILABLE:
            raise ImportError(
                "hydrodatasource package is required for unified data loading"
            )

        if self.data_type == "camels_us":
            # CAMELS data source
            if not CAMELS_AVAILABLE:
                raise ImportError(
                    "Camels not available. Please install hydrodatasource."
                )
            return CamelsUs(self.data_path)
        elif self.data_type in ["floodevent", "selfmadehydrodataset"]:
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
            - p_and_e: Precipitation and PET data [basin, time, features=2]
            - qobs: Observed streamflow [basin, time, features=1]
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

        # Check and convert units before converting to standard format
        xr_dataset = self._check_and_convert_units(xr_dataset)

        # Store the xarray dataset for later use (e.g., in evaluation)
        self.ds = xr_dataset

        # Convert to standard (p_and_e, qobs) format
        return self._xrdataset_to_ndarray(xr_dataset)

    def _xrdataset_to_ndarray(
        self, xr_dataset: xr.Dataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert continuous time series data to standard format.

        This handles traditional continuous data sources and event data.
        """
        # Find the actual variable names in the dataset using unified mapping
        prcp_var = self._find_variable_name(
            xr_dataset, self.VAR_MAPPING["prcp"]
        )
        pet_var = self._find_variable_name(xr_dataset, self.VAR_MAPPING["pet"])
        flow_var = self._find_variable_name(
            xr_dataset, self.VAR_MAPPING["flow"]
        )

        if not all([prcp_var, pet_var, flow_var]):
            raise ValueError(
                f"Could not find required variables in dataset. Available: {list(xr_dataset.data_vars)}"
            )

        # Extract basic data with explicit dimension order [time, basin]
        prcp = xr_dataset[prcp_var].transpose("time", "basin").values
        pet = xr_dataset[pet_var].transpose("time", "basin").values
        flow = xr_dataset[flow_var].transpose("time", "basin").values

        # Check if flood_event variable exists
        flood_event_var = self._find_variable_name(
            xr_dataset, self.VAR_MAPPING["flood_event"]
        )

        if flood_event_var:
            # Event data with flood_event: [time, basin, features=3]
            flood_event = (
                xr_dataset[flood_event_var].transpose("time", "basin").values
            )
            p_and_e = np.stack([prcp, pet, flood_event], axis=2)
        else:
            # Traditional continuous data: [time, basin, features=2]
            p_and_e = np.stack([prcp, pet], axis=2)

        # Expand qobs: [time, basin, features=1]
        qobs = np.expand_dims(flow, axis=2)

        return p_and_e, qobs

    def get_basin_configs(self) -> Dict[str, Any]:
        """
        Get basin configuration information for all basins.

        Parameters
        ----------
        attr_list : List[str], optional
            List of specific attributes to read. If None, defaults to ["area"]

        Returns
        -------
        Dict[str, Any]
            Dictionary containing basin configurations for each basin
        """
        basin_configs = {}

        # Get all basin attributes at once
        areas = self.datasource.read_area(
            gage_id_lst=self.basin_ids,
        )

        # Process each basin
        for basin_id in self.basin_ids:
            # Get attributes for this basin from xarray.Dataset
            basin_attrs = {}
            if isinstance(areas, xr.Dataset):
                basin_idx = list(areas.basin.values).index(basin_id)
                # Extract all data variables for this basin
                for var_name in areas.data_vars:
                    value = float(areas[var_name].isel(basin=basin_idx).values)
                    # Map area variables to basin_area
                    if var_name in self.VAR_MAPPING["area"]:
                        basin_attrs["basin_area"] = value
                    else:
                        basin_attrs[var_name] = value
            else:
                raise ValueError(f"Unsupported areas type: {type(areas)}")

            # Create basin config from attributes
            basin_config = {
                "basin_id": basin_id,
                "basin_name": f"Basin_{basin_id}",
                **basin_attrs,  # Add all attributes directly
            }

            basin_configs[basin_id] = basin_config

        return basin_configs

    def _find_variable_name(
        self, xr_dataset: xr.Dataset, possible_names: List[str]
    ) -> Optional[str]:
        """Find the actual variable name from a list of possible names."""
        for name in possible_names:
            if name in xr_dataset.data_vars:
                return name
        return None

    def _check_and_convert_units(self, xr_dataset: xr.Dataset) -> xr.Dataset:
        """
        Check units between precipitation and streamflow data and convert if necessary.

        This function ensures that precipitation and streamflow data have consistent units
        before performing hydrological modeling calculations.

        Parameters
        ----------
        xr_dataset : xr.Dataset
            Input xarray dataset containing hydrological variables

        Returns
        -------
        xr.Dataset
            Dataset with consistent units between precipitation and streamflow
        """

        def standardize_unit(unit):
            """Standardize unit strings for comparison."""
            unit = unit.lower()  # convert to lower case
            unit = re.sub(r"day", "d", unit)
            unit = re.sub(r"hour", "h", unit)
            return unit

        # For continuous data, check precipitation and streamflow units
        prcp_var = self._find_variable_name(
            xr_dataset, self.VAR_MAPPING["prcp"]
        )
        flow_var = self._find_variable_name(
            xr_dataset, self.VAR_MAPPING["flow"]
        )

        if prcp_var and flow_var:
            # Get units from attributes
            prcp_unit = xr_dataset[prcp_var].attrs.get("units", "mm/d")
            flow_unit = xr_dataset[flow_var].attrs.get("units", "m3/s")

            # Standardize units for comparison
            standardized_prcp_unit = standardize_unit(prcp_unit)
            standardized_flow_unit = standardize_unit(flow_unit)

            if standardized_prcp_unit != standardized_flow_unit:
                # Convert streamflow to match precipitation unit
                if hasattr(self.datasource, "read_area"):
                    basin_areas = self.datasource.read_area(self.basin_ids)
                    flow_dataset = xr_dataset[[flow_var]]
                    converted_flow_dataset = streamflow_unit_conv(
                        flow_dataset, basin_areas, target_unit=prcp_unit
                    )
                    xr_dataset[flow_var] = converted_flow_dataset[flow_var]
                else:
                    print(
                        f"Warning: Cannot convert units - datasource doesn't support read_area()"
                    )
                    print(
                        f"Precipitation unit: {prcp_unit}, Flow unit: {flow_unit}"
                    )

        return xr_dataset
