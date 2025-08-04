"""
Author: Wenyu Ouyang
Date: 2025-01-20 10:00:00
LastEditTime: 2025-08-04 08:46:59
LastEditors: Wenyu Ouyang
Description: Hydrological Data Augmentation Module - Generate synthetic flood events based on unit hydrograph and net rainfall
FilePath: \hydromodel\hydromodel\models\data_augment.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import numpy as np
import pandas as pd
import pint
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Tuple
import copy
from abc import ABC, abstractmethod

from hydrodatasource.configs.config import SETTING
from hydrodatasource.reader.floodevent import FloodEventDatasource
from hydrodatasource.utils.utils import streamflow_unit_conv
from hydromodel.models.unit_hydrograph import (
    save_dataframe_to_csv,
    uh_conv,
    evaluate_single_event_from_uh,
    optimize_shared_unit_hydrograph,
    optimize_uh_for_group,
    categorize_floods_by_peak,
)


class BaseDataAugmenter(ABC):
    """
    Abstract base class for hydrological data augmentation.
    """

    @abstractmethod
    def validate_data(self, data):
        """Validate input data format and content."""
        pass

    @abstractmethod
    def augment_data(self, data, n_samples_per_event: int = None):
        """Generate augmented flood events from input data."""
        pass


class HydrologicalDataAugmenter(BaseDataAugmenter):
    """
    Hydrological Data Augmentation Class

    Generates synthetic flood events based on unit hydrographs and net rainfall patterns
    from optimal flood events. Supports multiple watersheds and maintains temporal structure
    while generating new years starting from next year.

    Features:
    - Progressive rainfall addition (1st period, 1st+2nd, 1st+2nd+3rd, etc.)
    - Multiple scaling factors for rainfall intensity variation
    - Automatic time assignment with preserved month/day/hour but incremented years
    - Multi-watershed support
    - Unit hydrograph convolution for flow generation
    """

    # Constants for data column names and time step
    NET_RAIN = "P_eff"
    OBS_FLOW = "Q_obs_eff"
    DELTA_T_HOURS = 3.0

    def __init__(
        self,
        scaling_factors: List[float] = None,
        start_year_offset: int = 1,
        data_path: str = None,
        station_id: str = "songliao_21401550",
        optimization_mode: str = "shared",
        top_n_events: int = 10,
        min_nse_threshold: float = 0.7,
        uh_length: int = 24,
        verbose: bool = True,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the HydrologicalDataAugmenter

        Parameters:
        ----------
            scaling_factors: List of scaling factors for rainfall intensity (default: [0.5, 0.8, 1.2, 1.5, 2.0])
            start_year_offset: Years to add to current year for starting augmented data (default: 1)

            data_path: Path to data directory (defaults to SETTING path)
            station_id: Station ID for data loading (default: "songliao_21401550")
            optimization_mode: "shared" or "categorized" unit hydrograph optimization (default: "shared")
            top_n_events: Number of top events to extract for augmentation (default: 10)
            min_nse_threshold: Minimum NSE threshold for selecting events (default: 0.7)
            uh_length: Unit hydrograph length for shared mode (default: 24)
            verbose: Whether to print detailed information during real data loading (default: True)
            random_state: Random seed for reproducibility (default: None)
        """
        self.scaling_factors = scaling_factors or [
            0.5,
            0.8,
            1.0,
            1.2,
            1.5,
            2.0,
        ]
        self.start_year_offset = start_year_offset

        # Basin area will be set during data initialization
        self.basin_area_km2 = None

        # Internal state
        self.optimal_events_ = None
        self.unit_hydrographs_ = None
        self.watershed_info_ = None
        self.is_fitted_ = False

        if random_state is not None:
            np.random.seed(random_state)

        # Auto-load real data
        if verbose:
            print("🚀 Auto-loading real hydrological data...")

        try:
            # Load and optimize from scratch (results_file option removed)
            data = load_real_hydrological_data(
                data_path=data_path,
                station_id=station_id,
                optimization_mode=optimization_mode,
                top_n_events=top_n_events,
                min_nse_threshold=min_nse_threshold,
                uh_length=uh_length,
                verbose=verbose,
            )

            # Auto-initialize with loaded data
            self._initialize_with_data(data)

            if verbose:
                print("✅ Auto-loading and initialization completed!")

        except Exception as e:
            if verbose:
                print(f"❌ Failed to auto-load real data: {e}")
                print(
                    "💡 You can still manually initialize data using augment_data() method"
                )
            # Don't raise the exception, allow manual initialization later

    def validate_data(self, data: Dict):
        """
        Validate input data format and content.

        Parameters
        ----------
        data : dict
            Dictionary containing:
                - 'optimal_events': List of optimal flood events (each with P_eff, Q_obs_eff, etc.)
                - 'unit_hydrographs': Dict mapping event names to unit hydrograph arrays
                - 'watershed_info': Optional dict with watershed metadata.
                - 'basin_area_km2': Basin area in km² (for unit conversion, optional)

        Raises
        ------
        ValueError
            If required keys are missing or data format is invalid.
        """
        required_keys = ["optimal_events", "unit_hydrographs"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key in data: {key}")

        # Validate data consistency
        self._validate_input_data_format(data)

    def _initialize_with_data(self, data: Dict):
        """
        Initialize augmenter with validated data.

        Parameters
        ----------
        data : dict
            Validated data dictionary.
        """
        # Validate first
        self.validate_data(data)

        self.optimal_events_ = data["optimal_events"]
        self.unit_hydrographs_ = data["unit_hydrographs"]
        self.watershed_info_ = data.get("watershed_info", {})

        # Get basin area from data if available (from datasource)
        self.basin_area_km2 = data.get("basin_area_km2")
        if self.basin_area_km2 is None:
            # Try to get it from watershed_info as fallback
            self.basin_area_km2 = self.watershed_info_.get("basin_area_km2")

        if self.basin_area_km2 is None:
            print("⚠️ Warning: basin_area_km2 is not available from datasource")

        self.is_fitted_ = True

        print(
            f"✅ Augmenter initialized successfully with {len(self.optimal_events_)} events"
        )
        if self.basin_area_km2 is not None:
            print(f"📊 Basin area available: {self.basin_area_km2} km²")

    def _validate_input_data_format(self, data: Dict):
        """Validate input data consistency"""
        optimal_events = data["optimal_events"]
        unit_hydrographs = data["unit_hydrographs"]

        if not optimal_events:
            raise ValueError("No optimal events provided")

        for i, event in enumerate(optimal_events):
            required_fields = [self.NET_RAIN, self.OBS_FLOW, "filepath"]
            for field in required_fields:
                if field not in event:
                    raise ValueError(
                        f"Event {i} missing required field: {field}"
                    )

            event_name = event["filepath"]
            if event_name not in unit_hydrographs:
                raise ValueError(
                    f"No unit hydrograph found for event: {event_name}"
                )

    def augment_data(
        self, data: Dict, n_samples_per_event: int = None
    ) -> List[Dict]:
        """
        Generate augmented flood events from input data.

        Parameters
        ----------
        data : dict
            Dictionary containing optimal events and unit hydrographs.
        n_samples_per_event : int, optional
            Number of samples per event. If None, auto-calculated based on scaling factors.

        Returns
        -------
        list of dict
            List of augmented event dictionaries with generated synthetic data.
        """
        # Initialize with data if not already done
        if not self.is_fitted_:
            self._initialize_with_data(data)

        augmented_events = []
        current_year = datetime.now().year + self.start_year_offset
        sample_counter = 0

        for event_idx, original_event in enumerate(self.optimal_events_):
            print(
                f"🔄 Processing event {event_idx + 1}/{len(self.optimal_events_)}: {original_event['filepath']}"
            )

            # Generate progressive rainfall patterns and scaling variations
            event_augmentations = self._generate_event_variations(
                original_event
            )

            for aug_data in event_augmentations:
                # Create new event with updated timing
                new_event = self._create_augmented_event(
                    original_event,
                    aug_data,
                    current_year + sample_counter,
                    sample_counter,
                )

                augmented_events.append(new_event)
                sample_counter += 1

        print(f"✅ Generated {len(augmented_events)} augmented flood events")

        # Convert flow units from mm/3h to m³/s (always required)
        if self.basin_area_km2 is None:
            raise ValueError(
                "Basin area (basin_area_km2) could not be read from datasource, required for unit conversion to m³/s"
            )

        print("🔄 Converting flow units from mm/3h to m³/s...")
        augmented_events = self._convert_flow_units_to_cms(augmented_events)
        print("✅ Unit conversion completed")

        return augmented_events

    def _generate_event_variations(self, original_event: Dict) -> List[Dict]:
        """
        Generate all variations for a single event:
        1. Progressive rainfall addition (1st period, 1st+2nd, etc.)
        2. Multiple scaling factors for each pattern

        Args:
            original_event: Original event dictionary

        Returns:
            List of variation dictionaries with P_eff and generated Q_sim
        """
        variations = []
        original_rain = original_event[self.NET_RAIN]
        unit_hydrograph = self.unit_hydrographs_[original_event["filepath"]]

        # Find effective rainfall periods
        effective_periods = self._find_effective_periods(original_rain)

        if not effective_periods:
            print(
                f"⚠️ Warning: No effective rainfall periods found for {original_event['filepath']}"
            )
            return variations

        # Progressive rainfall addition
        for period_end in range(1, len(effective_periods) + 1):
            # Create progressive rainfall pattern (1st, 1st+2nd, 1st+2nd+3rd, etc.)
            progressive_rain = np.zeros_like(original_rain)
            for i in range(period_end):
                period_idx = effective_periods[i]
                progressive_rain[period_idx] = original_rain[period_idx]

            # Apply scaling factors
            for scale_factor in self.scaling_factors:
                scaled_rain = progressive_rain * scale_factor

                # Generate flow using unit hydrograph convolution
                generated_flow = uh_conv(
                    scaled_rain, unit_hydrograph, truncate=False
                )

                variation = {
                    self.NET_RAIN: scaled_rain,
                    "Q_sim": generated_flow,
                    "scale_factor": scale_factor,
                    "periods_used": period_end,
                    "generation_method": "progressive_scaling",
                }
                variations.append(variation)

        return variations

    def _find_effective_periods(
        self, rainfall: np.ndarray, threshold: float = 1e-6
    ) -> List[int]:
        """Find indices of effective rainfall periods"""
        return [i for i, rain in enumerate(rainfall) if rain > threshold]

    def _create_augmented_event(
        self,
        original_event: Dict,
        aug_data: Dict,
        new_year: int,
        sample_id: int,
    ) -> Dict:
        """
        Create a new augmented event with proper time assignment

        Args:
            original_event: Original event dictionary
            aug_data: Augmentation data with P_eff and Q_sim
            new_year: Year for the new event
            sample_id: Unique sample identifier

        Returns:
            New event dictionary
        """
        # Create new event as a copy of original
        new_event = copy.deepcopy(original_event)

        # Update with augmented data
        new_event[self.NET_RAIN] = aug_data[self.NET_RAIN]
        # Keep original observed flow separate from generated flow
        new_event["Q_obs_original"] = original_event[
            self.OBS_FLOW
        ]  # Store original observed flow
        new_event[self.OBS_FLOW] = aug_data[
            "Q_sim"
        ]  # Generated flow becomes "observed"
        new_event["Q_sim"] = aug_data["Q_sim"]  # Also keep as simulated

        # Update metadata
        new_event["augmentation_metadata"] = {
            "source_event": original_event["filepath"],
            "scale_factor": aug_data["scale_factor"],
            "periods_used": aug_data["periods_used"],
            "generation_method": aug_data["generation_method"],
            "sample_id": sample_id,
            "is_augmented": True,
        }

        # Generate new event name and filepath
        original_name = original_event["filepath"].replace(".csv", "")
        new_name = f"{original_name}_aug_{sample_id:04d}"
        new_event["filepath"] = f"{new_name}.csv"

        # Update time information (always preserve temporal structure)
        new_event = self._update_temporal_info(new_event, new_year)

        # Update peak observation
        if len(aug_data["Q_sim"]) > 0:
            new_event["peak_obs"] = np.max(aug_data["Q_sim"])

        return new_event

    def _update_temporal_info(self, event: Dict, new_year: int) -> Dict:
        """
        Update temporal information preserving month/day/hour but changing year
        for new filename format: event_YYYYMMDDHH_YYYYMMDDHH_aug_XXXX.csv

        Parameters
        ----------
        event: Dict
            Event dictionary
        new_year: int
            New year to assign

        Returns
        -------
        Dict: Updated event dictionary
        """
        # Try to extract start and end times from original filepath
        try:
            filepath = event["filepath"]
            if "event_" in filepath and "_aug_" in filepath:
                # Extract the time part: event_1994081520_1994081805_aug_0000.csv
                parts = filepath.split("_aug_")
                time_part = parts[0].replace(
                    "event_", ""
                )  # 1994081520_1994081805

                if "_" in time_part:
                    start_time_str, end_time_str = time_part.split("_")

                    # Parse start time: YYYYMMDDHH
                    if len(start_time_str) == 10:
                        start_month = int(start_time_str[4:6])
                        start_day = int(start_time_str[6:8])
                        start_hour = int(start_time_str[8:10])
                    else:
                        # Default values if parsing fails
                        start_month, start_day, start_hour = 7, 15, 0

                    # Parse end time: YYYYMMDDHH
                    if len(end_time_str) == 10:
                        end_month = int(end_time_str[4:6])
                        end_day = int(end_time_str[6:8])
                        end_hour = int(end_time_str[8:10])
                    else:
                        # Default values if parsing fails
                        end_month, end_day, end_hour = 7, 16, 0
                else:
                    # Fallback to default values
                    start_month, start_day, start_hour = 7, 15, 0
                    end_month, end_day, end_hour = 7, 16, 0
            else:
                # Default to summer flood season
                start_month, start_day, start_hour = 7, 15, 0
                end_month, end_day, end_hour = 7, 16, 0

        except (ValueError, IndexError):
            # Default to summer flood season
            start_month, start_day, start_hour = 7, 15, 0
            end_month, end_day, end_hour = 7, 16, 0

        # Create new time strings with updated year
        new_start_time_str = (
            f"{new_year:04d}{start_month:02d}{start_day:02d}{start_hour:02d}"
        )
        new_end_time_str = (
            f"{new_year:04d}{end_month:02d}{end_day:02d}{end_hour:02d}"
        )

        # Update filepath with new times
        if "_aug_" in event["filepath"]:
            # Extract the augmentation part
            parts = event["filepath"].split("_aug_")
            aug_suffix = parts[1]  # Keep augmentation ID
            event["filepath"] = (
                f"event_{new_start_time_str}_{new_end_time_str}_aug_{aug_suffix}"
            )

        # Add temporal metadata with start and end time strings
        event["temporal_info"] = {
            "start_time_string": new_start_time_str,
            "end_time_string": new_end_time_str,
        }

        return event

    def _convert_flow_units_to_cms(
        self, augmented_events: List[Dict]
    ) -> List[Dict]:
        """
        Convert flow units from mm/time to m³/s using streamflow_unit_conv

        Args:
            augmented_events: List of augmented event dictionaries

        Returns:
            List of events with flow units converted to m³/s
        """
        converted_events = []

        for event in augmented_events:
            # Create a copy to avoid modifying original
            converted_event = copy.deepcopy(event)

            # Convert OBS_FLOW (observed flow)
            if self.OBS_FLOW in converted_event:
                converted_event[self.OBS_FLOW] = (
                    self._convert_single_flow_array(
                        converted_event[self.OBS_FLOW]
                    )
                )

            # Convert Q_sim (simulated flow) if present
            if "Q_sim" in converted_event:
                converted_event["Q_sim"] = self._convert_single_flow_array(
                    converted_event["Q_sim"]
                )

            # Convert Q_obs_original (original observed flow) if present
            if "Q_obs_original" in converted_event:
                converted_event["Q_obs_original"] = (
                    self._convert_single_flow_array(
                        converted_event["Q_obs_original"]
                    )
                )

            # Update peak_obs if present
            if (
                "peak_obs" in converted_event
                and self.OBS_FLOW in converted_event
            ):
                converted_event["peak_obs"] = np.max(
                    converted_event[self.OBS_FLOW]
                )

            converted_events.append(converted_event)

        return converted_events

    def _convert_single_flow_array(self, flow_array: np.ndarray) -> np.ndarray:
        """
        Convert a single flow array from mm/time to m³/s

        Args:
            flow_array: Flow data in mm/time format

        Returns:
            Flow data converted to m³/s
        """
        area_quantity = self.basin_area_km2["area"].values

        # Convert using streamflow_unit_conv with inverse=True (from mm/3h to m³/s)
        converted_quantity = streamflow_unit_conv(
            streamflow=flow_array,
            area=area_quantity,
            target_unit="m^3/s",
            inverse=True,
            source_unit="mm/3h",
        )

        return converted_quantity

    def save_augmented_events(
        self,
        augmented_events: List[Dict],
        output_dir: str = "results/augmented_events",
        format: str = "csv",
    ) -> None:
        """
        Save augmented events to files

        Args:
            augmented_events: List of augmented event dictionaries
            output_dir: Output directory path
            format: Output format ('csv' or 'pkl')
        """
        os.makedirs(output_dir, exist_ok=True)

        if format == "csv":
            self._save_as_csv(augmented_events, output_dir)
        elif format == "pkl":
            self._save_as_pickle(augmented_events, output_dir)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _save_as_csv(self, augmented_events: List[Dict], output_dir: str):
        """Save events as individual CSV files with timestamps and metadata comments"""
        for event in augmented_events:
            # Get temporal information
            temporal_info = event.get("temporal_info", {})

            if (
                "start_time_string" in temporal_info
                and "end_time_string" in temporal_info
            ):
                # Parse start and end times
                start_time_str = temporal_info[
                    "start_time_string"
                ]  # YYYYMMDDHH
                end_time_str = temporal_info["end_time_string"]  # YYYYMMDDHH

                # Parse datetime components
                start_year = int(start_time_str[0:4])
                start_month = int(start_time_str[4:6])
                start_day = int(start_time_str[6:8])
                start_hour = int(start_time_str[8:10])

                end_year = int(end_time_str[0:4])
                end_month = int(end_time_str[4:6])
                end_day = int(end_time_str[6:8])
                end_hour = int(end_time_str[8:10])

                # Create datetime objects
                start_datetime = datetime(
                    start_year, start_month, start_day, start_hour, 0, 0
                )
                end_datetime = datetime(
                    end_year, end_month, end_day, end_hour, 0, 0
                )

                # Calculate time steps (assuming 3-hour intervals: 2, 5, 8, 11, 14, 17, 20, 23)
                # Start time must be on the 2-5-8-... timeline
                if start_hour not in [2, 5, 8, 11, 14, 17, 20, 23]:
                    # Adjust to nearest valid hour
                    valid_hours = [2, 5, 8, 11, 14, 17, 20, 23]
                    start_hour = min(
                        valid_hours, key=lambda x: abs(x - start_hour)
                    )
                    start_datetime = datetime(
                        start_year, start_month, start_day, start_hour, 0, 0
                    )

                # Generate timestamps for the time range
                timestamps = []
                current_datetime = start_datetime

                while current_datetime <= end_datetime:
                    timestamps.append(
                        current_datetime.strftime("%Y-%m-%d %H:%M:%S")
                    )
                    current_datetime += timedelta(hours=3)  # 3-hour intervals

                # Get the original data
                original_rain = event[self.NET_RAIN]
                original_flow = event[self.OBS_FLOW]

                # Truncate data to match the time range
                data_length = len(timestamps)
                truncated_rain = original_rain[:data_length]
                truncated_flow = original_flow[:data_length]

                # Pad if necessary
                if len(truncated_rain) < data_length:
                    truncated_rain = np.pad(
                        truncated_rain,
                        (0, data_length - len(truncated_rain)),
                        "constant",
                    )
                if len(truncated_flow) < data_length:
                    truncated_flow = np.pad(
                        truncated_flow,
                        (0, data_length - len(truncated_flow)),
                        "constant",
                    )

            else:
                # Fallback: use original data without temporal truncation
                max_length = max(
                    len(event[self.NET_RAIN]), len(event[self.OBS_FLOW])
                )
                timestamps = [f"T{i * 3:06.1f}h" for i in range(max_length)]
                truncated_rain = np.pad(
                    event[self.NET_RAIN],
                    (0, max_length - len(event[self.NET_RAIN])),
                    "constant",
                )
                truncated_flow = np.pad(
                    event[self.OBS_FLOW],
                    (0, max_length - len(event[self.OBS_FLOW])),
                    "constant",
                )

            # Create DataFrame
            # Check if we have original observed data
            if "Q_obs_original" in event:
                # Ensure original obs data has the same length as other data
                original_obs = event["Q_obs_original"]
                if len(original_obs) < data_length:
                    original_obs = np.pad(
                        original_obs,
                        (0, data_length - len(original_obs)),
                        "constant",
                    )
                elif len(original_obs) > data_length:
                    original_obs = original_obs[:data_length]

                df = pd.DataFrame(
                    {
                        "time": timestamps,
                        "net_rain": truncated_rain,
                        "gen_discharge": truncated_flow,
                        "obs_discharge": original_obs,
                    }
                )
            else:
                # Fallback for cases where original obs data is not available
                df = pd.DataFrame(
                    {
                        "time": timestamps,
                        "net_rain": truncated_rain,
                        "gen_discharge": truncated_flow,
                    }
                )

            # Create metadata comments
            metadata_lines = [
                f"# Augmented Event: {event['filepath']}",
                f"# Source: {event['augmentation_metadata']['source_event']}",
                f"# Scale Factor: {event['augmentation_metadata']['scale_factor']}",
                f"# Periods Used: {event['augmentation_metadata']['periods_used']}",
                f"# Sample ID: {event['augmentation_metadata']['sample_id']}",
            ]

            if "temporal_info" in event:
                metadata_lines.append(
                    f"# Start Time: {event['temporal_info']['start_time_string']}"
                )
                metadata_lines.append(
                    f"# End Time: {event['temporal_info']['end_time_string']}"
                )

            filepath = os.path.join(output_dir, event["filepath"])

            save_dataframe_to_csv(df, filepath, metadata_lines=metadata_lines)

        print(
            f"✅ Saved {len(augmented_events)} augmented events to {output_dir}"
        )

    def _save_as_pickle(self, augmented_events: List[Dict], output_dir: str):
        """Save events as pickle file"""
        import pickle

        filepath = os.path.join(output_dir, "augmented_events.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(augmented_events, f)
        print(
            f"✅ Saved {len(augmented_events)} augmented events to {filepath}"
        )

    def get_augmentation_summary(
        self, augmented_events: List[Dict]
    ) -> pd.DataFrame:
        """
        Generate summary statistics of augmented events

        Args:
            augmented_events: List of augmented events

        Returns:
            DataFrame with summary statistics
        """
        summary_data = []

        for event in augmented_events:
            metadata = event["augmentation_metadata"]
            temporal_info = event.get("temporal_info", {})

            summary_data.append(
                {
                    "event_name": event["filepath"],
                    "source_event": metadata["source_event"],
                    "scale_factor": metadata["scale_factor"],
                    "periods_used": metadata["periods_used"],
                    "sample_id": metadata["sample_id"],
                    "peak_flow": event.get("peak_obs", 0),
                    "total_rainfall": np.sum(event[self.NET_RAIN]),
                    "total_flow": np.sum(event[self.OBS_FLOW]),
                    "start_time": temporal_info.get(
                        "start_time_string", "Unknown"
                    ),
                    "end_time": temporal_info.get(
                        "end_time_string", "Unknown"
                    ),
                }
            )

        return pd.DataFrame(summary_data)


def load_real_hydrological_data(
    data_path: str = None,
    station_id: str = "songliao_21401550",
    optimization_mode: str = "shared",
    top_n_events: int = 10,
    min_nse_threshold: float = 0.7,
    uh_length: int = 24,
    smoothing_factor: float = 0.1,
    peak_violation_weight: float = 10000.0,
    verbose: bool = True,
) -> Dict:
    """
    Load real hydrological data and extract optimal events with unit hydrographs

    Args:
        data_path: Path to data directory (defaults to SETTING path)
        station_id: Station ID for data loading
        optimization_mode: "shared" for shared UH, "categorized" for class-based UH
        top_n_events: Number of top events to extract for augmentation
        min_nse_threshold: Minimum NSE threshold for selecting events
        uh_length: Unit hydrograph length for shared mode
        smoothing_factor: Smoothing factor for optimization
        peak_violation_weight: Peak violation penalty weight
        verbose: Whether to print detailed information

    Returns:
        Dict containing optimal events and unit hydrographs ready for augmentation
    """
    if verbose:
        print("🔄 Loading real hydrological data for data augmentation...")

    # Use default data path if not provided
    if data_path is None:
        data_path = os.path.join(
            SETTING["local_data_path"]["datasets-interim"], "songliaorrevent"
        )

    # 1. Load event data and get basin area
    if verbose:
        print(f"📂 Loading data from: {data_path}")
        print(f"🏭 Station ID: {station_id}")

    # Create datasource instance to read basin area
    dataset = FloodEventDatasource(
        data_path,
        trange4cache=["1960-01-01 02", "2024-12-31 23"],
    )
    # Get basin area from datasource
    basin_area_km2 = None
    if station_id:
        basin_area_km2 = dataset.read_area([station_id])

    all_event_data = dataset.load_1basin_flood_events(
        station_id=station_id,
        flow_unit="mm/3h",
        include_peak_obs=True,
        verbose=verbose,
    )

    if all_event_data is None or len(all_event_data) == 0:
        raise ValueError("No event data loaded")

    if verbose:
        print(f"✅ Loaded {len(all_event_data)} flood events")

    # 2. Optimize unit hydrographs based on mode
    optimal_events = []
    unit_hydrographs = {}

    if optimization_mode == "shared":
        # Shared unit hydrograph approach
        if verbose:
            print(
                f"⚙️ Optimizing shared unit hydrograph (length: {uh_length})..."
            )

        U_optimized = optimize_shared_unit_hydrograph(
            all_event_data,
            uh_length,
            smoothing_factor,
            peak_violation_weight,
            apply_peak_penalty=(uh_length > 2),
            max_iterations=500,
            verbose=verbose,
        )

        if U_optimized is None:
            raise ValueError("Unit hydrograph optimization failed")

        # Evaluate all events with the shared UH
        event_evaluations = []
        for event in all_event_data:
            result = evaluate_single_event_from_uh(event, U_optimized)
            if result["NSE"] >= min_nse_threshold:
                event_evaluations.append((event, result["NSE"]))

        # Sort by NSE and take top events
        event_evaluations.sort(key=lambda x: x[1], reverse=True)
        top_events = event_evaluations[:top_n_events]

        # Prepare data for augmentation
        for i, (event, nse) in enumerate(top_events):
            optimal_events.append(event)
            event_name = event.get("filepath", f"event_{i:04d}.csv")
            unit_hydrographs[event_name] = U_optimized.copy()

        if verbose:
            print(
                f"📊 Selected {len(optimal_events)} optimal events (NSE ≥ {min_nse_threshold})"
            )

    elif optimization_mode == "categorized":
        # Categorized unit hydrograph approach
        if verbose:
            print("🔄 Categorizing events by flood peak...")

        categorized_events, (threshold_low, threshold_high) = (
            categorize_floods_by_peak(all_event_data)
        )

        if categorized_events is None:
            raise ValueError("Event categorization failed")

        if verbose:
            print(
                f"📊 Categorization thresholds: small ≤ {threshold_low:.2f} < medium ≤ {threshold_high:.2f} < large"
            )

        # Define category weights
        category_weights = {
            "small": {"smoothing_factor": 0.1, "peak_violation_weight": 100.0},
            "medium": {
                "smoothing_factor": 0.5,
                "peak_violation_weight": 500.0,
            },
            "large": {
                "smoothing_factor": 1.0,
                "peak_violation_weight": 1000.0,
            },
        }

        uh_length_by_category = {
            "small": max(8, uh_length // 3),
            "medium": max(16, uh_length // 2),
            "large": uh_length,
        }

        # Optimize UH for each category
        for category_name, events in categorized_events.items():
            if len(events) < 3:
                if verbose:
                    print(
                        f"⚠️ Skipping category '{category_name}' (insufficient events: {len(events)})"
                    )
                continue

            if verbose:
                print(
                    f"⚙️ Optimizing {category_name} category UH ({len(events)} events)..."
                )

            weights = category_weights[category_name]
            n_uh = uh_length_by_category[category_name]

            U_optimized_cat = optimize_uh_for_group(
                events, category_name, weights, n_uh
            )

            if U_optimized_cat is None:
                if verbose:
                    print(
                        f"❌ Failed to optimize UH for category: {category_name}"
                    )
                continue

            # Evaluate events in this category
            category_evaluations = []
            for event in events:
                result = evaluate_single_event_from_uh(event, U_optimized_cat)
                if result["NSE"] >= min_nse_threshold:
                    category_evaluations.append((event, result["NSE"]))

            # Take top events from this category
            category_evaluations.sort(key=lambda x: x[1], reverse=True)
            top_category_events = category_evaluations[
                : max(1, top_n_events // 3)
            ]

            # Add to optimal events
            for event, nse in top_category_events:
                optimal_events.append(event)
                event_name = event.get(
                    "filepath",
                    f"event_{category_name}_{len(optimal_events):04d}.csv",
                )
                unit_hydrographs[event_name] = U_optimized_cat.copy()

        if verbose:
            print(
                f"📊 Selected {len(optimal_events)} optimal events across all categories"
            )

    else:
        raise ValueError(f"Unknown optimization mode: {optimization_mode}")

    if len(optimal_events) == 0:
        raise ValueError(
            f"No events meet the NSE threshold of {min_nse_threshold}"
        )

    # 3. Prepare watershed info
    watershed_info = {
        "name": f"Station_{station_id}",
        "station_id": station_id,
        "data_path": data_path,
        "optimization_mode": optimization_mode,
        "total_events_loaded": len(all_event_data),
        "optimal_events_selected": len(optimal_events),
        "min_nse_threshold": min_nse_threshold,
        "basin_area_km2": basin_area_km2,  # Add basin area to watershed info
    }

    if verbose:
        print("✅ Real data loading completed successfully!")
        print(f"   📈 Events selected: {len(optimal_events)}")
        print(f"   🔧 Unit hydrographs generated: {len(unit_hydrographs)}")

        # Show some statistics
        if optimal_events:
            nse_values = []
            for event in optimal_events:
                event_name = event.get("filepath", "unknown")
                if event_name in unit_hydrographs:
                    result = evaluate_single_event_from_uh(
                        event, unit_hydrographs[event_name]
                    )
                    nse_values.append(result["NSE"])

            if nse_values:
                print(
                    f"   📊 NSE range: {min(nse_values):.3f} - {max(nse_values):.3f}"
                )
                print(f"   📊 Average NSE: {np.mean(nse_values):.3f}")

    return {
        "optimal_events": optimal_events,
        "unit_hydrographs": unit_hydrographs,
        "watershed_info": watershed_info,
        "basin_area_km2": basin_area_km2,  # Return basin area directly
    }


# Example usage and convenience functions
def augment_multiple_watersheds(
    watershed_data: Dict[str, Dict], config: Dict = None
) -> Dict[str, List[Dict]]:
    """
    Augment data for multiple watersheds

    Args:
        watershed_data: Dict mapping watershed names to their data
        config: Augmentation configuration

    Returns:
        Dict mapping watershed names to their augmented events
    """
    results = {}

    for watershed_name, data in watershed_data.items():
        print(f"\n🏞️ Processing watershed: {watershed_name}")

        # Create augmenter with config parameters
        config = config or {}
        augmenter = HydrologicalDataAugmenter(
            scaling_factors=config.get(
                "scaling_factors", [0.5, 0.8, 1.2, 1.5, 2.0]
            ),
            start_year_offset=config.get("start_year_offset", 1),
            random_state=config.get("random_state", None),
        )
        augmented_events = augmenter.augment_data(data)

        results[watershed_name] = augmented_events

    return results


if __name__ == "__main__":
    # Example usage
    print("🚀 Hydrological Data Augmentation Module")
    print("This module provides tools for generating synthetic flood events")
    print("based on unit hydrographs and net rainfall patterns.")
    print("\nKey features:")
    print("- Progressive rainfall addition")
    print("- Multiple scaling factors")
    print("- Temporal structure preservation")
    print("- Multi-watershed support")
