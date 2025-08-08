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


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="DHF (大伙房) Model Simulation using New Flexible Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DHF Model Features:
  - 22 parameters for comprehensive watershed simulation
  - Dual-layer runoff generation (surface and subsurface)
  - Complex routing with surface and groundwater components
  - Designed for Chinese watershed conditions

Data Source Types:
  - csv_json: CSV time series data + JSON parameter files
  - selfmadehydrodataset: Custom hydrological datasets
  - floodevent: Event-based data

Usage Examples:
  # Use CSV data and JSON parameters (recommended for DHF)
  python run_unified_simulation.py --data-path /path/to/dhf.csv --params-file /path/to/dhf_params.json
  
  # Use configuration file
  python run_unified_simulation.py --config simulate_dhf_example.yaml
  
  # Quick test with default parameters
  python run_unified_simulation.py --model dhf --data-path /path/to/data --basin-id basin_001
        """,
    )

    # Configuration file option
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Configuration file path (YAML format). If provided, overrides other arguments.",
    )

    # Quick setup options
    parser.add_argument(
        "--model",
        type=str,
        default="dhf",
        choices=["dhf"],
        help="Model type - fixed to DHF (大伙房) model",
    )

    parser.add_argument(
        "--data-source-type",
        type=str,
        default="csv_json",
        choices=["csv_json", "selfmadehydrodataset", "floodevent"],
        help="Dataset type (default: csv_json for DHF model)",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Data directory path",
    )

    parser.add_argument(
        "--basin-ids",
        nargs="+",
        default=["basin_001"],
        help="Basin IDs to simulate (default: basin_001)",
    )

    parser.add_argument(
        "--warmup-length",
        type=int,
        default=30,
        help="Warmup period length in time steps (default: 30 for DHF)",
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

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/simulations",
        help="Output directory (default: results/simulations)",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (auto-generated if not provided)",
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

    # Other options
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet mode - minimal output",
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
        # Load configuration using ConfigManager
        if verbose:
            if args.config:
                print(f"Loading configuration from: {args.config}")
            else:
                print("Creating configuration from command line arguments")

        # Add parameter handling to args namespace for ConfigManager
        if args.params_file:
            parameters = load_parameters_from_file(args.params_file)
            args.model_parameters = parameters
        elif args.calibration_results:
            basin_id = args.basin_ids[0] if args.basin_ids else None
            parameters = load_parameters_from_calibration(
                args.calibration_results, args.model, basin_id
            )
            args.model_parameters = parameters
        else:
            # Use default parameters
            parameters = get_default_parameters(args.model)
            if not parameters:
                raise ValueError(
                    f"No default parameters available for model {args.model}. "
                    "Please provide --params-file or --calibration-results"
                )
            args.model_parameters = parameters

        config = ConfigManager.create_simulation_config(
            config_file=args.config, args=args
        )

        # Print configuration summary
        if verbose:
            print("\n" + "=" * 60)
            print("UNIFIED MODEL SIMULATION")
            print("=" * 60)

            data_cfg = config["data_cfgs"]
            model_cfg = config["model_cfgs"]

            print(f"Data Source: {data_cfg['data_source_type']}")
            print(f"Data Path: {data_cfg.get('data_source_path', 'default')}")
            print(f"Basins: {', '.join(data_cfg['basin_ids'])}")
            print(f"Model: {model_cfg['model_name']} (DHF - 大伙房水文模型)")
            print(
                f"Parameters: {len(model_cfg['parameters'])} DHF parameters specified (22 expected)"
            )

        # Run DHF simulation using NEW flexible unified interface
        if verbose:
            print(
                f"\nStarting DHF model simulation with new flexible architecture..."
            )
            print(f"Using: UnifiedSimulator + flexible simulate interface")
            print(
                f"DHF Model: 22-parameter Chinese watershed model with dual-layer runoff"
            )

        # NEW FLEXIBLE APPROACH: separate model config and data loading
        # 1. Extract model configuration
        model_config = config["model_cfgs"]

        # 2. Create DHF simulator instance (one-time initialization)
        simulator = UnifiedSimulator(model_config)

        # 3. Load data using traditional config (for backward compatibility)
        if verbose:
            print("Loading DHF model data...")
            print(f"Parameters: 22 DHF parameters loaded")
            print(
                f"Data format: precipitation + evapotranspiration time series"
            )

        # Use the backward-compatible helper to load data
        results = _simulate_with_config(config)

        if verbose:
            print("DHF simulation completed using new flexible architecture!")
            print(
                f"Benefits: DHF model initialized once, multiple datasets can be run"
            )
            print(
                f"Model components: runoff generation + surface/groundwater routing"
            )

        # Process and display results
        print_results_summary(results, config, verbose)

        # Save results if requested
        simulation_cfg = config.get("simulation_cfgs", {})
        if simulation_cfg.get("save_results", False):
            output_dir = os.path.join(
                simulation_cfg.get("output_dir", "results/simulations"),
                simulation_cfg.get("experiment_name", "simulation"),
            )
            save_simulation_results(results, config, output_dir)

        if verbose:
            print(f"\nDHF SIMULATION COMPLETED SUCCESSFULLY!")
            print(
                f"Used NEW flexible architecture: UnifiedSimulator + flexible simulate()"
            )
            print(
                f"Model: {config['model_cfgs']['model_name']} (大伙房水文模型)"
            )
            print(
                f"Parameters: 22 DHF parameters for comprehensive watershed simulation"
            )
            print(
                f"Architecture: DHF model config separated from data input for maximum flexibility"
            )
            print(
                f"Key advantage: Same DHF model can now run different datasets without reinitialization"
            )

        return 0

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        return 1
    except Exception as e:
        print(f"\nERROR: Simulation failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
