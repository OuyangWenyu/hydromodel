#!/usr/bin/env python3
"""
Author: Wenyu Ouyang
Date: 2025-08-08
LastEditTime: 2025-08-08 18:30:22
LastEditors: Wenyu Ouyang
Description: DHF (大伙房) model simulation script using the new flexible interface
FilePath: \hydromodel\scripts\run_unified_simulation.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import numpy as np
from datetime import datetime

# Add hydromodel to path
repo_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_path)

# Add local dependency paths
workspace_root = os.path.dirname(repo_path)
for local_pkg in ["hydroutils", "hydrodatasource", "hydrodataset"]:
    local_path = os.path.join(workspace_root, local_pkg)
    if os.path.exists(local_path):
        sys.path.insert(0, local_path)

from hydromodel.core.unified_simulate import (
    UnifiedSimulator,
    _simulate_with_config,
)
from hydromodel.configs.config_manager import ConfigManager
from hydromodel.configs.script_utils import ScriptUtils


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="DHF (大伙房) Model Simulation using Latest Unified Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DHF Model Types Supported:
  - dhf: 22-parameter Chinese watershed model with dual-layer runoff generation

Data Source Types:
  - csv_json: CSV time series data + JSON parameter files (recommended for DHF)
  - selfmadehydrodataset: Custom hydrological datasets
  - floodevent: Event-based data

Usage Examples:
  # Basic DHF simulation with default settings
  python run_dhf_simulation.py --model-type dhf --data-path /path/to/data

  # Configuration file approach (recommended)
  python run_dhf_simulation.py --config config.yaml
        """,
    )

    # Add common arguments
    ScriptUtils.add_common_arguments(parser)

    # DHF-specific arguments
    parser.add_argument(
        "--data-source-type",
        type=str,
        default="csv_json",
        choices=["csv_json", "selfmadehydrodataset", "floodevent"],
        help="Dataset type (default: csv_json for DHF model)",
    )

    parser.add_argument(
        "--data-source-path",
        type=str,
        default=None,
        help="Data directory path (uses default if not specified)",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="dhf_simulation_data",
        help="Dataset name for DHF simulation (default: dhf_simulation_data)",
    )

    parser.add_argument(
        "--basin-ids",
        nargs="+",
        default=["basin_001"],
        help="Basin IDs to simulate (default: basin_001)",
    )

    parser.add_argument(
        "--variables",
        nargs="+",
        default=["prcp", "PET", "streamflow"],
        help="Variables for simulation (default: prcp, PET, streamflow)",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="dhf",
        choices=["dhf"],
        help="DHF model type (default: dhf)",
    )

    parser.add_argument(
        "--warmup-length",
        type=int,
        default=30,
        help="Warmup period length in time steps (default: 30 for DHF)",
    )

    parser.add_argument(
        "--time-unit",
        type=str,
        default="1d",
        help="Time unit for data (default: 1d for daily DHF model)",
    )

    # Parameter input methods
    parser.add_argument(
        "--params-file",
        type=str,
        default=None,
        help="JSON/YAML file containing model parameters",
    )

    parser.add_argument(
        "--calibration-results",
        type=str,
        default=None,
        help="Use parameters from calibration results file",
    )

    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save simulation results to files",
    )

    parser.add_argument(
        "--plot-results",
        action="store_true",
        help="Generate simulation plots",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility (default: 1234)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet mode - minimal output",
    )

    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save configuration to file after run",
    )

    return parser.parse_args()


def load_parameters_from_file(params_file: str) -> dict:
    """Load DHF parameters from JSON file"""
    try:
        if params_file.endswith(".json"):
            import json

            with open(params_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract DHF parameters in order
            # [S0, U0, D0, K, KW, K2, KA, G, A, B, B0, K0, N, L, DD, CC, COE, DDL, CCL, SA0, UA0, YA0]
            dhf_params = {
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
                "L": float(data["L"]),
                "DD": float(data["DD"]),
                "CC": float(data["CC"]),
                "COE": float(data["COE"]),
                "DDL": float(data["DDL"]),
                "CCL": float(data["CCL"]),
                "SA0": float(data["SA0"]),
                "UA0": float(data["UA0"]),
                "YA0": float(data["YA0"]),
            }
            return dhf_params
        else:
            return ConfigManager.load_config_from_file(params_file)
    except Exception as e:
        raise ValueError(
            f"Failed to load DHF parameters from {params_file}: {e}"
        )


def load_parameters_from_calibration(
    results_file: str, model_name: str, basin_id: str = None
) -> dict:
    """Load parameters from calibration results"""
    try:
        with open(results_file, "r") as f:
            if results_file.endswith(".json"):
                import json

                results = json.load(f)
            else:
                results = yaml.safe_load(f)

        # Extract parameters from calibration results
        if basin_id:
            basin_results = results.get(basin_id, {})
        else:
            # Use first basin if no specific basin requested
            basin_results = next(iter(results.values()), {})

        best_params = basin_results.get("best_params", {})
        model_params = best_params.get(model_name, {})

        if not model_params:
            raise ValueError(
                f"No parameters found for model {model_name} in results file"
            )

        return model_params

    except Exception as e:
        raise ValueError(
            f"Failed to load parameters from calibration results: {e}"
        )


def get_default_parameters(model_name: str) -> dict:
    """Get default parameters for DHF model testing"""
    if model_name == "dhf":
        # DHF model default parameters - 22 parameters
        # [S0, U0, D0, K, KW, K2, KA, G, A, B, B0, K0, N, L, DD, CC, COE, DDL, CCL, SA0, UA0, YA0]
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
            "L": 100.0,
            "DD": 1.2,
            "CC": 0.8,
            "COE": 0.6,
            "DDL": 1.0,
            "CCL": 0.7,
            "SA0": 20.0,
            "UA0": 40.0,
            "YA0": 10.0,
        }
    else:
        return ConfigManager.get_model_default_parameters(model_name)


def save_simulation_results(results: dict, config: dict, output_dir: str):
    """Save simulation results to files"""
    os.makedirs(output_dir, exist_ok=True)

    # Save simulation output as CSV
    simulation = results["simulation"]
    basin_ids = config["data_cfgs"]["basin_ids"]

    for basin_idx, basin_id in enumerate(basin_ids):
        basin_sim = simulation[:, basin_idx, 0]

        # Create DataFrame with time index
        import pandas as pd

        df = pd.DataFrame(
            {
                "simulation": basin_sim,
            }
        )

        if "observation" in results and results["observation"] is not None:
            basin_obs = results["observation"][:, basin_idx, 0]
            df["observation"] = basin_obs

        # Save to CSV
        csv_path = os.path.join(output_dir, f"{basin_id}_simulation.csv")
        df.to_csv(csv_path, index=True)
        print(f"Saved simulation results: {csv_path}")

    # Save metadata
    metadata_path = os.path.join(output_dir, "simulation_metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.dump(results["metadata"], f, default_flow_style=False)
    print(f"Saved simulation metadata: {metadata_path}")


def print_results_summary(results: dict, config: dict, verbose: bool = True):
    """Print simulation results summary"""
    if not verbose:
        return

    metadata = results["metadata"]
    simulation = results["simulation"]

    print("\n" + "=" * 60)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 60)

    print(f"Model: {metadata['model_name']}")
    print(f"Simulation shape: {metadata['simulation_shape']}")
    print(f"Time steps: {metadata['time_steps']}")
    print(f"Number of basins: {metadata['n_basins']}")
    print(f"Warmup length: {metadata['warmup_length']}")

    # Basic statistics
    sim_stats = {
        "Mean": np.nanmean(simulation),
        "Std": np.nanstd(simulation),
        "Min": np.nanmin(simulation),
        "Max": np.nanmax(simulation),
    }

    print("\nSimulation Statistics:")
    for stat, value in sim_stats.items():
        print(f"  {stat}: {value:.4f}")

    # Basin-specific summary
    basin_ids = config["data_cfgs"]["basin_ids"]
    for basin_idx, basin_id in enumerate(basin_ids):
        basin_sim = simulation[:, basin_idx, 0]
        basin_mean = np.nanmean(basin_sim)
        basin_max = np.nanmax(basin_sim)
        print(
            f"  Basin {basin_id}: Mean={basin_mean:.4f}, Max={basin_max:.4f}"
        )


def main():
    """Main execution function"""
    args = parse_arguments()
    verbose = not args.quiet

    try:
        # Setup configuration using unified workflow
        # Ensure correct model defaults for quick setup
        if not getattr(args, "model_type", None) and not getattr(
            args, "model", None
        ):
            args.model = "dhf"

        # Add parameter handling to args namespace before config setup
        if args.params_file:
            parameters = load_parameters_from_file(args.params_file)
            args.model_parameters = parameters
        elif args.calibration_results:
            basin_id = args.basin_ids[0] if args.basin_ids else None
            parameters = load_parameters_from_calibration(
                args.calibration_results, getattr(args, "model_type", "dhf"), basin_id
            )
            args.model_parameters = parameters
        else:
            # Use default parameters
            parameters = get_default_parameters(getattr(args, "model_type", "dhf"))
            if not parameters:
                raise ValueError(
                    f"No default parameters available for DHF model. "
                    "Please provide --params-file or --calibration-results"
                )
            args.model_parameters = parameters

        config = ScriptUtils.setup_configuration(args)
        if config is None:
            return 1

        # Apply overrides
        ScriptUtils.apply_overrides(config, args.override)

        # Apply command line overrides for output settings
        if args.output_dir:
            config["training_cfgs"]["output_dir"] = args.output_dir
        if args.experiment_name:
            config["training_cfgs"]["experiment_name"] = args.experiment_name

        # Validate configuration
        if not ScriptUtils.validate_and_show_config(
            config, verbose, "DHF Model"
        ):
            return 1

        if args.dry_run:
            print("\n🔍 Dry run completed - configuration is valid")
            return 0

        # Run DHF simulation using unified interface
        if verbose:
            print("\n🚀 Starting DHF model simulation with unified architecture...")
            print("📦 Using unified simulate interface")
            print("DHF Model: 22-parameter Chinese watershed model with dual-layer runoff")

        # Use the unified simulation interface
        results = _simulate_with_config(config)

        # Process and display results
        print_results_summary(results, config, verbose)

        # Save results if requested
        if args.save_results or config.get("simulation_cfgs", {}).get("save_results", False):
            output_dir = os.path.join(
                config.get("training_cfgs", {}).get("output_dir", "results/simulations"),
                config.get("training_cfgs", {}).get("experiment_name", "dhf_simulation"),
            )
            save_simulation_results(results, config, output_dir)

        # Save configuration file if requested
        if args.save_config:
            training_cfgs = config.get("training_cfgs", {})
            output_dir = os.path.join(
                training_cfgs.get("output_dir", "results"),
                training_cfgs.get("experiment_name", "experiment"),
            )
            config_output_path = os.path.join(
                output_dir, "dhf_simulation_config.yaml"
            )
            ScriptUtils.save_config_file(config, config_output_path)

        ScriptUtils.print_completion_message(config, "DHF simulation")
        return 0

    except KeyboardInterrupt:
        print("\n👋 Simulation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: Simulation failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
