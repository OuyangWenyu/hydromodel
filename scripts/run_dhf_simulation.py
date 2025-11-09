#!/usr/bin/env python3
r"""
Author: Wenyu Ouyang
Date: 2025-08-12
LastEditTime: 2025-08-12 14:00:00
LastEditors: Wenyu Ouyang
Description: DHF (大伙房) model simulation script using refactored runtime simulation utilities
FilePath: \hydromodel\scripts\run_dhf_simulation.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import argparse
import sys
import os
import json
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple

# Add hydromodel to path
repo_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_path)

# Add local dependency paths
workspace_root = os.path.dirname(repo_path)
for local_pkg in ["hydroutils", "hydrodatasource", "hydrodataset"]:
    local_path = os.path.join(workspace_root, local_pkg)
    if os.path.exists(local_path):
        sys.path.insert(0, local_path)

# Import the new runtime simulation utilities
from hydromodel.trainers.unified_simulate import UnifiedSimulator

# Import RuntimeDataLoader directly
try:
    from hydrodatasource.runtime import load_runtime_data

    RUNTIME_DATA_AVAILABLE = True
except ImportError:
    RUNTIME_DATA_AVAILABLE = False
    load_runtime_data = None
from hydromodel.configs.config_manager import *


def parse_arguments():
    """Parse command line arguments for DHF simulation"""
    parser = argparse.ArgumentParser(
        description="DHF (大伙房) Model Simulation using Runtime Data Loading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DHF Model - 19-parameter Chinese watershed model with dual-layer runoff generation

Data Source Types (RuntimeDataLoader):
  - csv: CSV files with time series data (recommended for DHF)
  - parquet: Parquet files for large datasets
  - memory: In-memory data (DataFrame, Dict, Arrays)
  - sql: SQL databases (PostgreSQL, MySQL, SQLite)
  - stream: Real-time data streams (for operational scenarios)

Usage Examples:
  # Basic DHF simulation with CSV data
  python run_dhf_simulation.py --data-path /path/to/data.csv --basin-ids basin_001

  # Database-driven simulation
  python run_dhf_simulation.py --data-source sql --sql-connection "postgresql://user:pass@host/db" --sql-table hydro_data
        """,
    )

    # Model configuration
    parser.add_argument(
        "--model-type",
        type=str,
        default="dhf",
        choices=["dhf"],
        help="DHF model type (default: dhf)",
    )

    # Data source configuration
    parser.add_argument(
        "--data-source",
        type=str,
        default="csv",
        choices=["csv", "parquet", "json", "memory", "sql", "stream"],
        help="Data source type for RuntimeDataLoader (default: csv)",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        help="Data file path (for csv, parquet, json sources)",
        default=r"E:\data\ClassC\songliaorrevent\timeseries\3h\songliao_21100150.csv",
    )

    parser.add_argument(
        "--sql-connection",
        type=str,
        help="SQL connection string (for sql source)",
    )

    parser.add_argument(
        "--sql-table",
        type=str,
        default="hydro_data",
        help="SQL table name (default: hydro_data)",
    )

    parser.add_argument(
        "--basin-ids",
        nargs="+",
        default=["songliao_21100150"],
        help="Basin IDs to simulate (default: songliao_21100150)",
    )

    parser.add_argument(
        "--variables",
        nargs="+",
        default=[
            "rain",
            "ES",
            "inflow",
            "flood_event",
        ],
        help="Variables for simulation (default: rain, ES, inflow, flood_event)",
    )

    parser.add_argument(
        "--time-range",
        nargs=2,
        default=["2020-01-01", "2020-12-31"],
        help="Time range for simulation: start_date end_date (default: 2020-01-01 2020-12-31)",
    )

    parser.add_argument(
        "--time-column",
        type=str,
        default="time",
        help="Name of time column in data (default: time)",
    )

    parser.add_argument(
        "--basin-column",
        type=str,
        default="basin",
        help="Name of basin column in data (default: basin)",
    )

    # Model parameters
    parser.add_argument(
        "--params-file",
        type=str,
        default=r"D:\Code\songliaodb_analysis\results\json_parameters\songliao_21100150_params.json",
        help="JSON/YAML file containing DHF model parameters",
    )

    # Simulation settings
    parser.add_argument(
        "--warmup-length",
        type=int,
        default=480,
        help="Warmup period length (default: 365 days)",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/dhf_runtime_simulation",
        help="Output directory for results",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="dhf_runtime",
        help="Experiment name for output files",
    )

    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save simulation results to files",
    )

    # Control options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running simulation",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet mode - minimal output",
    )

    parser.add_argument(
        "--validate-inputs",
        action="store_true",
        help="Validate simulation inputs before running",
    )

    return parser.parse_args()


def load_dhf_parameters_from_file(params_file: str) -> dict:
    """
    Load DHF-specific parameters from JSON file.

    DHF model has 18 parameters in specific order:
    [S0, U0, D0, K, KW, K2, KA, G, A, B, B0, K0, N, DD, CC, COE, DDL, CCL]
    """
    try:
        if params_file.endswith(".json"):
            with open(params_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract DHF parameters in order
            dhf_params = OrderedDict(
                {
                    "S0": float(data["S0"]),
                    "U0": float(data["U0"]),
                    "D0": float(data["D0"]),
                    "K": float(data["K"]),
                    "KW": float(data["KW"]),
                    "K2": float(data["K2"]),
                    "KA": float(data["KA"]),
                    "G": float(data["G"]),
                    "A": float(data["A"]),
                    "B": float(data["B"]),
                    "B0": float(data["B0"]),
                    "K0": float(data["K0"]),
                    "N": float(data["N"]),
                    "DD": float(data["DD"]),
                    "CC": float(data["CC"]),
                    "COE": float(data["COE"]),
                    "DDL": float(data["DDL"]),
                    "CCL": float(data["CCL"]),
                }
            )
            return dhf_params
        else:
            return load_config_from_file(params_file)
    except Exception as e:
        raise ValueError(
            f"Failed to load DHF parameters from {params_file}: {e}"
        )


def get_dhf_default_parameters() -> dict:
    """Get default parameters for DHF model testing"""
    return {
        "S0": 50.0,
        "U0": 100.0,
        "D0": 80.0,
        "K": 0.8,
        "KW": 0.5,
        "K2": 0.3,
        "KA": 0.7,
        "G": 0.2,
        "A": 0.4,
        "B": 0.6,
        "B0": 1.5,
        "K0": 0.5,
        "N": 2.0,
        "DD": 1.2,
        "CC": 0.8,
        "COE": 0.6,
        "DDL": 1.0,
        "CCL": 0.7,
    }


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


def main():
    """Main execution function using refactored runtime simulation utilities"""
    args = parse_arguments()
    verbose = not args.quiet

    try:
        # Load DHF model parameters
        if args.params_file and os.path.exists(args.params_file):
            parameters = load_dhf_parameters_from_file(args.params_file)
            if verbose:
                print(f"📋 Loaded DHF parameters from: {args.params_file}")
        else:
            parameters = get_dhf_default_parameters()
            if verbose:
                print("📋 Using default DHF parameters")
                if args.params_file:
                    print(f"   (Parameter file not found: {args.params_file})")

        if verbose:
            print(f"   DHF model: {len(parameters)} parameters loaded")

        if args.dry_run:
            print("\n🔍 Dry run completed - configuration is valid")
            print(f"   Model: {args.model_type}")
            print(f"   Data source: {args.data_source}")
            print(f"   Parameters: {len(parameters)}")
            return 0

        # Prepare data source configuration
        source_config = {}
        if args.data_source in ["csv", "parquet", "json"]:
            if not args.data_path:
                print(
                    f"❌ ERROR: --data-path required for {args.data_source} source"
                )
                return 1
            source_config["file_path"] = args.data_path
            source_config["time_column"] = args.time_column
            source_config["basin_column"] = args.basin_column
        elif args.data_source == "sql":
            if not args.sql_connection:
                print("❌ ERROR: --sql-connection required for sql source")
                return 1
            source_config.update(
                {
                    "connection_string": args.sql_connection,
                    "table_name": args.sql_table,
                    "time_column": args.time_column,
                    "basin_column": args.basin_column,
                }
            )

        # Check if RuntimeDataLoader is available
        if not RUNTIME_DATA_AVAILABLE:
            print(
                "❌ ERROR: RuntimeDataLoader not available. Please ensure hydrodatasource with runtime module is installed."
            )
            return 1

        # DHF simulation using direct approach (recommended)
        if verbose:
            print(f"\n🚀 Starting DHF simulation...")
            print(f"   Source: {args.data_source}")
            print(f"   Basins: {args.basin_ids}")
            print(f"   Variables: {args.variables}")
            print(
                f"   Time range: {args.time_range[0]} to {args.time_range[1]}"
            )

        # Load data directly using RuntimeDataLoader
        try:
            data = load_runtime_data(
                variables=args.variables,
                basin_ids=args.basin_ids,
                time_range=tuple(args.time_range),
                source_type=args.data_source,
                source_config=source_config,
                return_format="arrays",  # Return in (p_and_e, qobs) format
            )

            if isinstance(data, tuple):
                inputs, qobs = data
                # For pure simulation, set qobs to None
                qobs = None
                if verbose:
                    print(f"✅ Data loaded successfully:")
                    print(
                        f"   Input shape: {inputs.shape} [time, basin, features]"
                    )
                    print("   Observations: None (simulation mode)")
            else:
                print("❌ ERROR: Expected array format from RuntimeDataLoader")
                return 1

        except Exception as e:
            print(f"❌ ERROR: Failed to load data: {e}")
            return 1

        # Validate inputs if requested
        if args.validate_inputs:
            validation = validate_runtime_simulation_inputs(
                model_name=args.model_type,
                parameters=parameters,
                inputs=inputs,
                basin_ids=args.basin_ids,
            )

            if not validation["valid"]:
                print("❌ Validation failed:")
                for issue in validation["issues"]:
                    print(f"   - {issue}")
                return 1

            if validation["warnings"] and verbose:
                print("⚠️  Validation warnings:")
                for warning in validation["warnings"]:
                    print(f"   - {warning}")

        # Create model configuration and run simulation directly
        model_config = {
            "model_name": args.model_type,
            "model_params": {
                "main_river_length": 155.763,
                "basin_area": 5482.0,
            },
            "parameters": parameters,
        }

        if verbose:
            print(f"⚙️  Initializing {args.model_type} simulator...")
            print(f"   Parameters: {len(parameters)} model parameters")
            print(f"   Input shape: {inputs.shape}")
            print(f"   Warmup length: {args.warmup_length}")

        # Create and run simulator directly
        simulator = UnifiedSimulator(model_config)
        results = simulator.simulate(
            inputs=inputs,
            qobs=qobs,
            warmup_length=args.warmup_length,
            is_event_data=True,
        )

        if verbose:
            sim_shape = results["qsim"].shape
            print(f"✅ Simulation completed: {sim_shape}")

        if verbose:
            print(f"\n✅ DHF simulation completed successfully!")
            print(f"   Final simulation shape: {results['qsim'].shape}")

        return 0

    except KeyboardInterrupt:
        print("\n👋 DHF simulation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: DHF simulation failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
