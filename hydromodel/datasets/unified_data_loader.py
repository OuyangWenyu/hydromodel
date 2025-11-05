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
import importlib
from pathlib import Path
from hydroutils.hydro_units import streamflow_unit_conv

# Import dataset mapping from dataset_dict
from .dataset_dict import DATASET_MAPPING, get_dataset_category

# Check availability
try:
    import hydrodataset
    HYDRODATASET_AVAILABLE = True
except ImportError:
    HYDRODATASET_AVAILABLE = False

try:
    import hydrodatasource
    HYDRODATASOURCE_AVAILABLE = True
except ImportError:
    HYDRODATASOURCE_AVAILABLE = False


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
            "Area",
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
            "variables", ["precipitation", "potential_evapotranspiration", "streamflow"]
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

        # Get dataset category from DATASET_MAPPING
        dataset_category = self._get_dataset_category()

        # Try to load from hydro_setting.yml
        try:
            setting_file = os.path.join(Path.home(), "hydro_setting.yml")
            if os.path.exists(setting_file):
                with open(setting_file, "r", encoding="utf-8") as f:
                    settings = yaml.safe_load(f)

                if settings and "local_data_path" in settings:
                    datasets_origin = settings["local_data_path"].get(
                        "datasets-origin"
                    )
                    basins_origin = settings["local_data_path"].get(
                        "basins-origin"
                    )

                    # Determine path based on dataset category
                    if dataset_category == "hydrodatasource":
                        # For custom data from hydrodatasource
                        if basins_origin:
                            data_path = basins_origin
                    elif dataset_category == "hydrodataset":
                        # For public datasets from hydrodataset
                        if datasets_origin:
                            data_path = datasets_origin

                    if data_path:
                        print(
                            f"Using data paths in hydro_setting.yml : {data_path}"
                        )
        except Exception as e:
            print(f"Warning: unable to load path from hydro_setting.yml: {e}")

        # If still None, use default path
        if data_path is None:
            default_root = os.path.join(Path.home(), "hydromodel_data")

            if dataset_category == "hydrodatasource":
                # For custom data
                data_path = os.path.join(default_root, "basins-interim")
            else:
                # For public datasets: use datasets-origin directly
                # aqua_fetch will automatically append the dataset directory name (e.g., CAMELS_US)
                data_path = os.path.join(default_root, "datasets-origin")

            print(f"Using default data paths: {data_path}")

        return data_path

    def _get_dataset_category(self) -> str:
        """
        Get dataset category from DATASET_MAPPING.

        Returns
        -------
        str
            Dataset category: "hydrodataset" or "hydrodatasource"
        """
        category = get_dataset_category(self.data_type)
        # Default to hydrodataset for backward compatibility if not found
        return category if category is not None else "hydrodataset"

    def _create_datasource(self) -> Any:
        """
        Create the appropriate datasource based on data_type using dynamic imports.

        This method uses DATASET_MAPPING to dynamically import and instantiate the
        correct dataset class, supporting all datasets from hydrodataset and hydrodatasource.
        """
        # Check if data_type is in DATASET_MAPPING
        if self.data_type not in DATASET_MAPPING:
            raise ValueError(
                f"Unsupported data_type: {self.data_type}\n"
                f"Supported datasets: {list(DATASET_MAPPING.keys())}"
            )

        module_name, class_name, category = DATASET_MAPPING[self.data_type]

        # Check package availability
        if category == "hydrodataset" and not HYDRODATASET_AVAILABLE:
            raise ImportError(
                f"hydrodataset package is required for '{self.data_type}' dataset. "
                "Install with: pip install hydrodataset"
            )
        elif category == "hydrodatasource" and not HYDRODATASOURCE_AVAILABLE:
            raise ImportError(
                f"hydrodatasource package is required for '{self.data_type}' dataset. "
                "Install with: pip install hydrodatasource"
            )

        # Dynamic import
        try:
            module = importlib.import_module(module_name)
            dataset_class = getattr(module, class_name)
        except ImportError as e:
            raise ImportError(
                f"Failed to import {class_name} from {module_name}: {e}\n"
                f"Make sure the required package is installed."
            )
        except AttributeError as e:
            raise AttributeError(
                f"Class {class_name} not found in module {module_name}: {e}"
            )

        # Instantiate dataset based on category
        if category == "hydrodataset":
            # Public datasets from hydrodataset - simple initialization
            return dataset_class(self.data_path)
        elif category == "hydrodatasource":
            # Custom datasets from hydrodatasource - requires additional config
            return dataset_class(
                data_path=self.data_path,
                download=False,
                time_unit=self.config.get("time_unit", ["1D"]),
                dataset_name=self.config.get("dataset_name", "selfmadehydrodataset"),
                **self.config.get("datasource_kwargs", {}),
            )
        else:
            raise ValueError(f"Unknown dataset category: {category}")

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

                    # Store original dimension order
                    original_dims = xr_dataset[flow_var].dims

                    # Ensure flow data has correct dimension order [time, basin] for processing
                    if (
                        "time" in xr_dataset[flow_var].dims
                        and "basin" in xr_dataset[flow_var].dims
                    ):
                        flow_data_transposed = xr_dataset[flow_var].transpose(
                            "time", "basin"
                        )
                    else:
                        flow_data_transposed = xr_dataset[flow_var]

                    # Process each basin separately to avoid broadcasting issues with hydroutils
                    basin_list = xr_dataset.basin.values
                    num_basins = len(basin_list)
                    time_values = xr_dataset.time.values
                    num_times = len(time_values)

                    # Initialize array with shape [time, basin]
                    converted_flow_values = np.zeros((num_times, num_basins))

                    for i, basin_id in enumerate(basin_list):
                        # Extract single basin data
                        flow_single = flow_data_transposed.sel(basin=basin_id)

                        # Create single-basin dataset
                        flow_dataset_single = xr.Dataset(
                            {
                                flow_var: (
                                    ["time", "basin"],
                                    flow_single.values.reshape(-1, 1),
                                )
                            },
                            coords={"time": time_values, "basin": [basin_id]},
                        )
                        flow_dataset_single[flow_var].attrs = xr_dataset[
                            flow_var
                        ].attrs

                        # Get single basin area
                        single_basin_area = basin_areas.isel(basin=i)

                        # Convert units for this basin
                        converted_single = streamflow_unit_conv(
                            flow_dataset_single,
                            single_basin_area,
                            target_unit=prcp_unit,
                        )

                        # Store converted values
                        converted_flow_values[:, i] = converted_single[
                            flow_var
                        ].values.flatten()

                    # Create new DataArray with converted values in [time, basin] order
                    converted_da = xr.DataArray(
                        converted_flow_values,
                        dims=["time", "basin"],
                        coords={"time": time_values, "basin": basin_list},
                    )

                    # Transpose back to original dimension order if needed
                    if original_dims != ("time", "basin"):
                        converted_da = converted_da.transpose(*original_dims)

                    # Update dataset
                    xr_dataset[flow_var] = converted_da
                    xr_dataset[flow_var].attrs["units"] = prcp_unit
                else:
                    print(
                        f"Warning: Cannot convert units - datasource doesn't support read_area()"
                    )
                    print(
                        f"Precipitation unit: {prcp_unit}, Flow unit: {flow_unit}"
                    )

        return xr_dataset
