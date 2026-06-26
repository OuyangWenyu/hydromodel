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
import importlib
import importlib.util
import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List, Optional, Tuple, Any
from hydroutils.hydro_units import streamflow_unit_conv

HYDRODATASET_AVAILABLE = importlib.util.find_spec("hydrodataset") is not None
HYDRODATASOURCE_AVAILABLE = (
    importlib.util.find_spec("hydrodatasource") is not None
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
            - dataset: Dataset identifier (e.g. 'camels_us')
            - source: 'local' or 'cloud' (default 'local')
            - basin_ids: List of basin identifiers
            - train_period / valid_period / test_period: Time period config
            - variables: List of variables to load
            - warmup_length: Warmup period length
            - **kwargs: Additional datasource-specific parameters
        is_train_val_test : str
            "train", "valid", "test"
        """
        from hydrodatasource.configs.data_resolver import open_dataset

        self.config = data_config
        self.dataset = data_config.get("dataset")
        if not self.dataset:
            raise ValueError("data_cfgs.dataset is required")
        source = data_config.get("source", "local")

        # Build reader_kwargs from config
        reader_kwargs = {}
        if "time_unit" in data_config:
            reader_kwargs["time_unit"] = data_config["time_unit"]
        if "dataset_name" in data_config:
            reader_kwargs["dataset_name"] = data_config["dataset_name"]
        if "datasource_kwargs" in data_config:
            reader_kwargs.update(data_config["datasource_kwargs"])

        # When uri is explicitly provided (self-made / test datasets),
        # bypass registry lookup and delegate to open_dataset with a
        # one-off ResolverContext that points to the explicit path.
        explicit_uri = data_config.get("uri")
        if explicit_uri:
            from pathlib import Path as _Path
            from hydrodataset.configs.data_resolver import ResolverContext

            uri_path = _Path(explicit_uri)
            reader_name = data_config.get("reader", self.dataset)
            ctx = ResolverContext(
                storage={
                    "local": {"root": str(uri_path.parent)},
                },
                registry={
                    self.dataset: {
                        "reader": reader_name,
                        "path": uri_path.name,
                    }
                },
            )
            self.datasource = open_dataset(
                self.dataset, source=source, ctx=ctx, **reader_kwargs
            )
        else:
            # Default path: open_dataset resolves via built-in registries
            self.datasource = open_dataset(
                self.dataset, source=source, **reader_kwargs
            )

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
            "variables",
            ["precipitation", "potential_evapotranspiration", "streamflow"],
        )

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data using the unified interface and return in standard format.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (p_and_e, qobs) tuple where:
            - p_and_e: Precipitation and PET data [time, basin, features=2or3]
            - qobs: Observed streamflow [time, basin, features=1]

        Notes
        -----
        - For continuous data: features=2 (precipitation, PET)
        - For flood event data: features=3 (precipitation, PET, flood_event_marker)
        """
        # Detect flood event datasource by capability, not by name
        if hasattr(self.datasource, "load_1basin_flood_events"):
            # Use specialized flood event loading
            return self._load_flood_event_data()
        else:
            # Use standard continuous data loading
            return self._load_continuous_data()

    def _load_continuous_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load continuous time series data (traditional approach).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (p_and_e, qobs) tuple in standard format
        """
        # Use read_ts_xrdataset as the unified interface
        # Support both read_kwargs (legacy) and datasource_kwargs (preferred)
        read_kwargs = self.config.get("read_kwargs", {})
        datasource_kwargs = self.config.get("datasource_kwargs", {})
        # Merge both, with datasource_kwargs taking precedence
        merged_kwargs = {**read_kwargs, **datasource_kwargs}

        ts_data = self.datasource.read_ts_xrdataset(
            gage_id_lst=self.basin_ids,
            t_range=self.time_range,
            var_lst=self.variables,
            **merged_kwargs,
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

    def _load_flood_event_data(
        self, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load flood event data using FloodEventDatasource.

        This method loads all flood events for the specified basin(s) and concatenates
        them into a single continuous time series format, compatible with the model's
        expected input shape.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (p_and_e, qobs) tuple where:
            - p_and_e: [time, basin, features=3] - (precipitation, PET, flood_event_marker)
            - qobs: [time, basin, features=1] - observed streamflow

        Notes
        -----
        - Flood event marker: 1 during flood events, 0 during warmup/non-event periods
        - All events are concatenated into a continuous time series
        - NaN values in warmup periods are expected and will be handled by the model
        """
        if not hasattr(self.datasource, "load_1basin_flood_events"):
            raise ValueError(
                f"Datasource {type(self.datasource)} does not support flood event loading. "
                "Please use FloodEventDatasource."
            )

        # Get time unit and calculate flow unit
        time_unit = self.config.get("time_unit", ["1D"])[0]
        time_unit_hours = self._parse_time_unit_to_hours(time_unit)
        flow_unit = f"mm/{time_unit_hours}h"

        # Load events for each basin
        all_basin_data = []

        for basin_id in self.basin_ids:
            # Load flood events for this basin
            # Pass datasource_kwargs to load_1basin_flood_events
            load_kwargs = self.config.get("datasource_kwargs", {}).copy()
            load_kwargs.update(kwargs)  # Merge with any additional kwargs

            events = self.datasource.load_1basin_flood_events(
                station_id=basin_id,
                flow_unit=flow_unit,
                include_peak_obs=True,
                verbose=True,
                **load_kwargs,
            )

            if events is None or len(events) == 0:
                raise ValueError(f"No flood events found for basin {basin_id}")

            # Filter events by time_range (test_period/train_period)
            if self.time_range is not None and len(self.time_range) == 2:
                time_start = pd.Timestamp(self.time_range[0])
                time_end = pd.Timestamp(self.time_range[1])

                filtered_events = []
                for event in events:
                    event_time = event.get("time", None)
                    if event_time is not None and len(event_time) > 0:
                        # Check if event overlaps with time_range
                        event_time_pd = pd.to_datetime(event_time)
                        event_start = event_time_pd[0]
                        event_end = event_time_pd[-1]

                        # Keep event if it overlaps with time_range
                        if event_end >= time_start and event_start <= time_end:
                            filtered_events.append(event)

                if len(filtered_events) == 0:
                    raise ValueError(
                        f"No flood events found for basin {basin_id} in time range "
                        f"{self.time_range[0]} to {self.time_range[1]}"
                    )

                events = filtered_events

            # Convert events to continuous time series format
            basin_p_and_e, basin_qobs, basin_time = (
                self._convert_events_to_arrays(events, time_unit=time_unit)
            )

            all_basin_data.append((basin_p_and_e, basin_qobs, basin_time))

        # Stack data from all basins: [time, basin, features]
        # For multi-basin flood events, DO NOT pad - store each basin's data separately
        # Evaluation will process them one by one and merge results into single NetCDF

        if len(all_basin_data) > 1:
            # Store data separately for each basin (no padding)
            # This will be handled by the evaluator
            self.basin_data_separate = {}
            for idx, (basin_p_and_e, basin_qobs, basin_time) in enumerate(
                all_basin_data
            ):
                basin_id = self.basin_ids[idx]
                self.basin_data_separate[basin_id] = {
                    "p_and_e": basin_p_and_e,
                    "qobs": basin_qobs,
                    "time": basin_time,
                }

            # Return dummy data - evaluator will use basin_data_separate instead
            # Just return first basin's data for compatibility
            p_and_e = all_basin_data[0][0][
                :, np.newaxis, :
            ]  # [time, 1, features]
            qobs = all_basin_data[0][1][
                :, np.newaxis, :
            ]  # [time, 1, features]
            self.time_array = all_basin_data[0][2]
            self.ds = None
            return p_and_e, qobs

        # Single basin case - proceed as normal
        # Find the maximum length across all basins
        max_length = max(data[0].shape[0] for data in all_basin_data)

        # Pad all basins to the same length (only needed for single basin, no actual padding)
        padded_p_and_e = []
        padded_qobs = []
        padded_times = []

        # Get time_unit for padding
        time_unit = self.config.get("time_unit", ["3h"])[0]
        pd_freq = time_unit.replace("H", "h").replace(
            "D", "d"
        )  # pandas uses lowercase

        for idx, (basin_p_and_e, basin_qobs, basin_time) in enumerate(
            all_basin_data
        ):
            current_length = basin_p_and_e.shape[0]

            if current_length < max_length:
                # Pad with zeros (non-event periods)
                pad_length = max_length - current_length

                if len(all_basin_data) > 1:
                    basin_id = self.basin_ids[idx]
                    print(
                        f"  Basin {basin_id}: {current_length} -> {max_length} timesteps (padded {pad_length})"
                    )

                # Pad p_and_e: [time, features]
                p_and_e_padded = np.pad(
                    basin_p_and_e,
                    ((0, pad_length), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

                # Pad qobs: [time, features]
                qobs_padded = np.pad(
                    basin_qobs,
                    ((0, pad_length), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

                # Pad time array: extend with regular intervals
                last_time = pd.Timestamp(basin_time[-1])
                pad_time = pd.date_range(
                    start=last_time + pd.Timedelta(pd_freq),
                    periods=pad_length,
                    freq=pd_freq,
                ).values
                time_padded = np.concatenate([basin_time, pad_time])

                padded_p_and_e.append(p_and_e_padded)
                padded_qobs.append(qobs_padded)
                padded_times.append(time_padded)
            else:
                if len(all_basin_data) > 1:
                    basin_id = self.basin_ids[idx]
                    print(
                        f"  Basin {basin_id}: {current_length} timesteps (no padding needed)"
                    )
                padded_p_and_e.append(basin_p_and_e)
                padded_qobs.append(basin_qobs)
                padded_times.append(basin_time)

        # Store all basin time arrays (each basin may have different valid time ranges)
        # All arrays have same length after padding, but padding timestamps may differ
        self.time_arrays = padded_times  # List of time arrays, one per basin

        # For backward compatibility, also store first basin's time array
        time_array = padded_times[0]

        # Now stack with consistent shapes
        p_and_e = np.stack(padded_p_and_e, axis=1)
        qobs = np.stack(padded_qobs, axis=1)

        # Store time array for flood event data (first basin's time for backward compat)
        self.time_array = time_array

        # Set ds attribute to None for flood event data (not xarray format)
        # This is needed for compatibility with evaluation code
        self.ds = None

        return p_and_e, qobs

    def _convert_events_to_arrays(
        self, events: List[Dict], time_unit: str = "3h"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert list of flood events to concatenated numpy arrays.

        Following data_augment.py approach:
        - Keep COMPLETE events (including NaN warmup periods)
        - Do NOT filter out warmup periods
        - Concatenate all events with gaps
        - Let the model handle warmup internally

        Parameters
        ----------
        events : List[Dict]
            List of flood event dictionaries, each containing:
            - rain: rainfall data (with NaN in warmup period)
            - ES: evapotranspiration data
            - inflow: streamflow data (with NaN in warmup period)
            - flood_event_markers: event indicator (NaN=warmup, 1=event)
            - time: time array (optional)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (p_and_e, qobs, time_array) where:
            - p_and_e: [time, features=4] - complete events with gaps
              features: [prcp, pet, marker, event_id]
            - qobs: [time, features=1] - complete streamflow with gaps
            - time_array: [time] - time coordinates (datetime64 or None)
        """
        all_prcp = []
        all_pet = []
        all_flow = []
        all_markers = []
        all_event_ids = []
        all_times = []

        GAP_LENGTH = 10  # Add gap between events

        for idx, event in enumerate(events):
            # Extract COMPLETE time series data from event (including warmup NaN)
            prcp = np.array(event.get("rain", []), dtype=float)
            pet = np.array(event.get("ES", []), dtype=float)
            flow = np.array(event.get("inflow", []), dtype=float)

            # Get flood event markers (NaN for warmup, 1 for flood period)
            markers = event.get("flood_event_markers", np.ones(len(prcp)))
            markers = np.array(markers, dtype=float)

            # Get event ID (to group peaks from same flood event)
            event_id = event.get("event_id", idx + 1)
            event_ids = np.full(len(prcp), event_id, dtype=float)

            # Extract time array if available
            event_time = event.get("time", None)
            if event_time is not None:
                event_time = pd.to_datetime(event_time).values
            else:
                # Generate dummy time if not available
                # Convert time_unit to pandas frequency (use lowercase h/d)
                pd_freq = time_unit.replace("H", "h").replace("D", "d")
                event_time = pd.date_range(
                    "2000-01-01", periods=len(prcp), freq=pd_freq
                ).values

            # Keep COMPLETE event (do NOT filter)
            all_prcp.append(prcp)
            all_pet.append(pet)
            all_flow.append(flow)
            all_markers.append(markers)
            all_event_ids.append(event_ids)
            all_times.append(event_time)

            # Add gap between events
            gap_prcp = np.zeros(GAP_LENGTH)
            gap_pet = np.ones(GAP_LENGTH) * 0.27
            gap_flow = np.zeros(GAP_LENGTH)
            gap_markers = np.zeros(GAP_LENGTH)
            gap_event_ids = np.zeros(GAP_LENGTH)  # gap has event_id = 0

            # Generate gap time (continuing from last event time)
            # Convert time_unit to pandas frequency (use lowercase h/d)
            pd_freq = time_unit.replace("H", "h").replace("D", "d")
            if event_time is not None and len(event_time) > 0:
                last_time = pd.Timestamp(event_time[-1])
                gap_time = pd.date_range(
                    start=last_time + pd.Timedelta(pd_freq),
                    periods=GAP_LENGTH,
                    freq=pd_freq,
                ).values
            else:
                gap_time = pd.date_range(
                    "2000-01-01", periods=GAP_LENGTH, freq=pd_freq
                ).values

            all_prcp.append(gap_prcp)
            all_pet.append(gap_pet)
            all_flow.append(gap_flow)
            all_markers.append(gap_markers)
            all_event_ids.append(gap_event_ids)
            all_times.append(gap_time)

        # Remove last gap
        all_prcp = all_prcp[:-1]
        all_pet = all_pet[:-1]
        all_flow = all_flow[:-1]
        all_markers = all_markers[:-1]
        all_event_ids = all_event_ids[:-1]
        all_times = all_times[:-1]

        # Concatenate
        prcp_concat = np.concatenate(all_prcp)
        pet_concat = np.concatenate(all_pet)
        flow_concat = np.concatenate(all_flow)
        markers_concat = np.concatenate(all_markers)
        event_ids_concat = np.concatenate(all_event_ids)
        time_concat = np.concatenate(all_times)

        # Stack into [time, features=4]: [prcp, pet, marker, event_id]
        p_and_e = np.stack(
            [prcp_concat, pet_concat, markers_concat, event_ids_concat], axis=1
        )
        qobs = np.expand_dims(flow_concat, axis=1)

        return p_and_e, qobs, time_concat

    @staticmethod
    def _parse_time_unit_to_hours(time_unit: str) -> int:
        """
        Parse time unit string to hours.

        Parameters
        ----------
        time_unit : str
            Time unit string (e.g., "1h", "3h", "1D")

        Returns
        -------
        int
            Number of hours
        """
        import re

        match = re.match(r"(\d+)([hHdD])", time_unit)
        if not match:
            raise ValueError(f"Invalid time unit format: {time_unit}")

        value, unit = match.groups()
        value = int(value)

        if unit.lower() == "h":
            return value
        elif unit.lower() == "d":
            return value * 24
        else:
            raise ValueError(f"Unsupported time unit: {unit}")

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
