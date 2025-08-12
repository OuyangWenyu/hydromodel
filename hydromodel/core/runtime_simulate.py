"""
Runtime simulation utilities using RuntimeDataLoader.

This module provides utilities for real-time hydrological model simulation
using the new RuntimeDataLoader interface, designed to complement the
traditional UnifiedDataLoader for training scenarios.

Key Features:
- Atomic data loading for specific variables and time ranges
- Multi-source support (CSV, SQL, streams, memory)
- Optimized for real-time and operational scenarios
- Model-agnostic utilities for all hydrological models

Author: Wenyu Ouyang
Date: 2025-08-12
"""

import os
import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

from hydromodel.core.unified_simulate import UnifiedSimulator

# Import RuntimeDataLoader
try:
    from hydrodatasource.runtime import RuntimeDataLoader, load_runtime_data

    RUNTIME_DATA_AVAILABLE = True
except ImportError:
    RUNTIME_DATA_AVAILABLE = False
    RuntimeDataLoader = None


class RuntimeSimulationError(Exception):
    """Custom exception for runtime simulation errors."""

    pass


def create_runtime_simulation_config(
    model_name: str,
    parameters: Dict[str, Union[float, int, List]],
    model_params: Optional[Dict[str, Any]] = None,
    simulation_settings: Optional[Dict[str, Any]] = None,
    data_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create simulation configuration for RuntimeDataLoader approach.

    This creates a standardized configuration structure that works with
    UnifiedSimulator and can be easily extended for different models.

    Parameters
    ----------
    model_name : str
        Name of the model to simulate
    parameters : Dict[str, Union[float, int, List]]
        Model parameters with specific values for simulation
    model_params : Optional[Dict[str, Any]], default None
        Model-specific configuration parameters
    simulation_settings : Optional[Dict[str, Any]], default None
        Simulation settings (warmup_length, output options, etc.)
    data_info : Optional[Dict[str, Any]], default None
        Information about the data source for reference

    Returns
    -------
    Dict[str, Any]
        Standardized configuration dictionary

    Examples
    --------
    >>> config = create_runtime_simulation_config(
    ...     model_name="xaj",
    ...     parameters={"K": 0.5, "B": 0.3, "IM": 0.01},
    ...     simulation_settings={"warmup_length": 365}
    ... )
    """
    config = {
        "model_cfgs": {
            "model_name": model_name,
            "model_params": model_params or {},
            "parameters": parameters,
        },
        "simulation_cfgs": simulation_settings or {},
        "data_info": data_info or {},
    }
    return config


def save_simulation_results(
    results: Dict[str, Any],
    basin_ids: List[str],
    output_dir: Union[str, Path],
    experiment_name: str = "simulation",
    time_range: Optional[Tuple[str, str]] = None,
    freq: str = "D",
) -> List[Path]:
    """
    Save simulation results to files.

    Parameters
    ----------
    results : Dict[str, Any]
        Simulation results from run_runtime_simulation()
    basin_ids : List[str]
        Basin identifiers for output files
    output_dir : Union[str, Path]
        Output directory for results
    experiment_name : str, default "simulation"
        Experiment name for output files
    time_range : Optional[Tuple[str, str]], default None
        Time range for creating time index (start_date, end_date)
    freq : str, default "D"
        Frequency for time index generation

    Returns
    -------
    List[Path]
        List of saved file paths

    Examples
    --------
    >>> saved_files = save_simulation_results(
    ...     results=simulation_results,
    ...     basin_ids=["basin_001"],
    ...     output_dir="results",
    ...     experiment_name="xaj_simulation",
    ...     time_range=("2024-01-01", "2024-12-31")
    ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    simulation = results["simulation"]

    # Save simulation output for each basin
    for basin_idx, basin_id in enumerate(basin_ids):
        basin_sim = simulation[:, basin_idx, 0]

        # Create time index
        if time_range:
            start_date = pd.to_datetime(time_range[0])
            time_index = pd.date_range(
                start_date, periods=len(basin_sim), freq=freq
            )
        else:
            time_index = pd.RangeIndex(len(basin_sim), name="time_step")

        # Create DataFrame
        df = pd.DataFrame(
            {
                "simulation": basin_sim,
            },
            index=time_index,
        )

        # Add observations if available
        if "observation" in results and results["observation"] is not None:
            basin_obs = results["observation"][:, basin_idx, 0]
            df["observation"] = basin_obs

        # Save to CSV
        csv_path = output_dir / f"{basin_id}_{experiment_name}.csv"
        df.to_csv(csv_path, index=True)
        saved_files.append(csv_path)
        print(f"💾 Saved simulation results: {csv_path}")

    # Save metadata
    metadata_path = output_dir / f"{experiment_name}_metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(results["metadata"], f, default_flow_style=False)
    saved_files.append(metadata_path)
    print(f"💾 Saved simulation metadata: {metadata_path}")

    return saved_files


def print_simulation_summary(
    results: Dict[str, Any],
    basin_ids: List[str],
    model_name: str = "Model",
    verbose: bool = True,
) -> None:
    """
    Print comprehensive simulation results summary.

    Parameters
    ----------
    results : Dict[str, Any]
        Simulation results dictionary
    basin_ids : List[str]
        Basin identifiers for detailed output
    model_name : str, default "Model"
        Model name for display
    verbose : bool, default True
        Whether to print detailed statistics

    Examples
    --------
    >>> print_simulation_summary(
    ...     results=simulation_results,
    ...     basin_ids=["basin_001"],
    ...     model_name="XAJ"
    ... )
    """
    if not verbose:
        return

    metadata = results["metadata"]
    simulation = results["simulation"]

    print("\n" + "=" * 60)
    print(f"{model_name.upper()} SIMULATION RESULTS SUMMARY")
    print("=" * 60)

    print(f"Model: {metadata.get('model_name', model_name)}")
    print(
        f"Simulation shape: {metadata.get('simulation_shape', simulation.shape)}"
    )
    print(f"Time steps: {metadata.get('time_steps', simulation.shape[0])}")
    print(f"Number of basins: {metadata.get('n_basins', simulation.shape[1])}")
    print(f"Warmup length: {metadata.get('warmup_length', 'Unknown')}")

    # Overall statistics
    sim_stats = {
        "Mean": np.nanmean(simulation),
        "Std": np.nanstd(simulation),
        "Min": np.nanmin(simulation),
        "Max": np.nanmax(simulation),
    }

    print("\nOverall Simulation Statistics:")
    for stat, value in sim_stats.items():
        print(f"  {stat}: {value:.4f}")

    # Basin-specific summary
    if basin_ids and len(basin_ids) > 0:
        print("\nBasin-specific Statistics:")
        for basin_idx, basin_id in enumerate(basin_ids):
            if basin_idx < simulation.shape[1]:
                basin_sim = simulation[:, basin_idx, 0]
                basin_mean = np.nanmean(basin_sim)
                basin_std = np.nanstd(basin_sim)
                basin_max = np.nanmax(basin_sim)
                basin_min = np.nanmin(basin_sim)
                print(
                    f"  {basin_id}: Mean={basin_mean:.3f}, Std={basin_std:.3f}, "
                    f"Min={basin_min:.3f}, Max={basin_max:.3f}"
                )


def validate_runtime_simulation_inputs(
    model_name: str,
    parameters: Dict[str, Union[float, int]],
    inputs: np.ndarray,
    basin_ids: List[str],
) -> Dict[str, Any]:
    """
    Validate inputs for runtime simulation.

    Parameters
    ----------
    model_name : str
        Model name to validate
    parameters : Dict[str, Union[float, int]]
        Model parameters
    inputs : np.ndarray
        Input data array
    basin_ids : List[str]
        Basin identifiers

    Returns
    -------
    Dict[str, Any]
        Validation results with 'valid' flag and any issues found
    """
    validation = {"valid": True, "issues": [], "warnings": []}

    # Check model name
    from hydromodel.models.model_dict import MODEL_DICT

    if model_name not in MODEL_DICT:
        validation["issues"].append(
            f"Model '{model_name}' not found in MODEL_DICT"
        )
        validation["valid"] = False

    # Check parameters
    if not parameters:
        validation["issues"].append("No parameters provided")
        validation["valid"] = False

    # Check input data dimensions
    if not isinstance(inputs, np.ndarray):
        validation["issues"].append("Inputs must be numpy array")
        validation["valid"] = False
    elif inputs.ndim != 3:
        validation["issues"].append(
            f"Inputs must be 3D [time, basin, features], got shape {inputs.shape}"
        )
        validation["valid"] = False
    elif inputs.shape[1] != len(basin_ids):
        validation["warnings"].append(
            f"Input basins ({inputs.shape[1]}) != basin_ids length ({len(basin_ids)})"
        )

    # Check for NaN values
    if np.isnan(inputs).any():
        nan_ratio = np.isnan(inputs).sum() / inputs.size
        validation["warnings"].append(
            f"Input data contains {nan_ratio:.2%} NaN values"
        )

    return validation
