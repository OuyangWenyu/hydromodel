"""
Author: Wenyu Ouyang
Date: 2025-08-07
LastEditTime: 2025-08-07 20:35:52
LastEditors: Wenyu Ouyang
Description: XAJ model calibration script using the latest unified architecture
FilePath: \hydromodel\scripts\run_xaj_calibration_unified.py
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

from hydromodel.trainers.unified_calibrate import calibrate, DEAP_AVAILABLE
from hydromodel.configs.config_manager import ConfigManager


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="XAJ Model Calibration using Latest Unified Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
XAJ Model Types Supported:
  - xaj: XinAnJiang model (original version with routing)
  - xaj_mz: XinAnJiang model (with mizuRoute routing)

Algorithm Types Supported:
  - SCE_UA: Shuffled Complex Evolution via spotpy (default)
  - genetic_algorithm: Genetic algorithm via DEAP (if installed)
  - scipy_minimize: SciPy optimization methods

Data Source Types:
  - camels: CAMELS dataset
  - selfmadehydrodataset: Custom hydrological datasets
  - owndata: User-defined data format

Usage Examples:
  # Basic XAJ calibration with default settings
  python run_xaj_calibration_unified.py --model-type xaj_mz --algorithm SCE_UA

  # XAJ with genetic algorithm (requires DEAP)
  python run_xaj_calibration_unified.py --model-type xaj_mz --algorithm genetic_algorithm

  # Custom data directory and basin
  python run_xaj_calibration_unified.py --data-dir /path/to/data --basin-id basin_001

  # Configuration file approach (recommended)
  python run_xaj_calibration_unified.py --config config.yaml
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

    # Data configuration
    parser.add_argument(
        "--data-source-type",
        type=str,
        default="camels",
        choices=["camels", "selfmadehydrodataset", "owndata"],
        help="Dataset type (default: selfmadehydrodataset)",
    )

    parser.add_argument(
        "--data-source-path",
        type=str,
        default=None,
        help="Data directory path (uses default if not specified)",
    )

    parser.add_argument(
        "--basin-ids",
        nargs="+",
        default=["01013500"],
        help="Basin IDs to calibrate (default: basin_001)",
    )

    parser.add_argument(
        "--warmup-length",
        type=int,
        default=365,
        help="Warmup period length in days (default: 365)",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["prcp", "pet", "streamflow"],
        help="Variables to calibrate (default: prcp, pet, streamflow)",
    )
    # Model configuration
    parser.add_argument(
        "--model-type",
        type=str,
        default="xaj_mz",
        choices=["xaj", "xaj_mz"],
        help="XAJ model variant (default: xaj_mz)",
    )

    parser.add_argument(
        "--source-type",
        type=str,
        default="sources",
        choices=["sources", "sources5mm"],
        help="XAJ source data type (default: sources)",
    )

    parser.add_argument(
        "--source-book",
        type=str,
        default="HF",
        choices=["HF", "EH"],
        help="XAJ computation method: HF=Hydrological Forecast, EH=Engineering Hydrology (default: HF)",
    )

    parser.add_argument(
        "--kernel-size",
        type=int,
        default=15,
        help="XAJ convolutional kernel size (default: 15)",
    )

    # Algorithm configuration
    parser.add_argument(
        "--algorithm",
        type=str,
        default="SCE_UA",
        choices=["SCE_UA", "genetic_algorithm", "scipy_minimize"],
        help="Optimization algorithm (default: SCE_UA)",
    )

    # SCE-UA specific parameters
    parser.add_argument(
        "--rep",
        type=int,
        default=5000,
        help="SCE-UA repetitions (default: 5000)",
    )

    parser.add_argument(
        "--ngs",
        type=int,
        default=1000,
        help="SCE-UA number of complexes (default: 1000)",
    )

    # Genetic Algorithm parameters
    parser.add_argument(
        "--pop-size",
        type=int,
        default=80,
        help="GA population size (default: 80)",
    )

    parser.add_argument(
        "--n-generations",
        type=int,
        default=50,
        help="GA number of generations (default: 50)",
    )

    # SciPy parameters
    parser.add_argument(
        "--scipy-method",
        type=str,
        default="L-BFGS-B",
        help="SciPy optimization method (default: L-BFGS-B)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Maximum iterations for scipy (default: 1000)",
    )

    # Loss function configuration
    parser.add_argument(
        "--obj-func",
        type=str,
        default="RMSE",
        choices=["RMSE", "NSE", "KGE"],
        help="Objective function (default: RMSE)",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory (default: results)",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (auto-generated if not provided)",
    )

    # Parameter range file
    parser.add_argument(
        "--param-range-file",
        type=str,
        default=None,
        help="Parameter range file path (uses default if not specified)",
    )

    # Other options
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


# This function is now replaced by ConfigManager.create_calibration_config


# Configuration validation is now handled by ConfigManager


def print_config_summary(config: dict, verbose: bool = True):
    """Print configuration summary"""
    if not verbose:
        return

    print("=" * 80)
    print("🚀 XAJ Model Calibration - Latest Unified Architecture")
    print("=" * 80)

    # Data configuration
    data_cfg = config["data_cfgs"]
    print(f"📁 Data Source: {data_cfg['data_source_type']}")
    print(f"📂 Data Path: {data_cfg.get('data_source_path', 'default')}")
    print(f"🏭 Basins: {', '.join(data_cfg['basin_ids'])}")
    print(f"⏱️ Warmup: {data_cfg.get('warmup_length', 365)} days")

    # Model configuration
    model_cfg = config["model_cfgs"]
    print(f"🤖 Model: {model_cfg['model_name']}")
    model_params = model_cfg.get("model_params", {})
    if model_params:
        print(f"   📋 Parameters: {model_params}")

    # Training configuration
    training_cfg = config["training_cfgs"]
    print(f"🔧 Algorithm: {training_cfg['algorithm_name']}")
    print(f"📊 Objective: {training_cfg['loss_config']['obj_func']}")
    print(f"🎯 Experiment: {training_cfg.get('experiment_name', 'default')}")

    # Algorithm parameters
    algo_params = training_cfg.get("algorithm_params", {})
    if algo_params:
        print(f"⚙️ Algorithm Parameters:")
        for key, value in algo_params.items():
            print(f"   {key}: {value}")

    # Special warnings/info
    if training_cfg["algorithm_name"] == "genetic_algorithm":
        print(f"🧬 DEAP Available: {DEAP_AVAILABLE}")
        if not DEAP_AVAILABLE:
            print("⚠️ Genetic algorithm requires DEAP: pip install deap")

    print("-" * 80)


def save_config_file(config: dict, output_path: str):
    """Save configuration to YAML file"""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"💾 Configuration saved: {output_path}")
    except Exception as e:
        print(f"⚠️ Warning: Could not save config file: {e}")


def process_results(results: dict, config: dict, verbose: bool = True):
    """Process and display calibration results"""
    if not verbose:
        return

    print("\n" + "=" * 80)
    print("📈 CALIBRATION RESULTS SUMMARY")
    print("=" * 80)

    training_cfg = config["training_cfgs"]
    algorithm_name = training_cfg["algorithm_name"]
    obj_func = training_cfg["loss_config"]["obj_func"]

    successful_basins = 0
    total_basins = len(config["data_cfgs"]["basin_ids"])
    objective_values = []

    for basin_id, result in results.items():
        print(f"\n🏭 Basin: {basin_id}")
        print("-" * 40)

        convergence = result.get("convergence", "unknown")
        objective_value = result.get("objective_value", float("inf"))
        best_params = result.get("best_params", {})
        algorithm_info = result.get("algorithm_info", {})

        # Convergence status
        if convergence == "success":
            print(f"✅ Convergence: SUCCESS")
            successful_basins += 1
            objective_values.append(objective_value)
        else:
            print(f"❌ Convergence: FAILED")

        # Objective value
        print(f"🎯 Best {obj_func}: {objective_value:.6f}")

        # Algorithm info
        if algorithm_info:
            if algorithm_name == "SCE_UA":
                iterations = algorithm_info.get("rep", "N/A")
                print(f"🔄 Iterations: {iterations}")
            elif algorithm_name == "genetic_algorithm":
                generations = algorithm_info.get("generations", "N/A")
                pop_size = algorithm_info.get("population_size", "N/A")
                print(f"🧬 Generations: {generations}, Population: {pop_size}")
            elif algorithm_name == "scipy_minimize":
                iterations = algorithm_info.get("iterations", "N/A")
                message = algorithm_info.get("message", "")
                print(f"🔄 Iterations: {iterations}")
                if message:
                    print(f"💬 Message: {message}")

        # Parameter summary
        if best_params and basin_id in best_params:
            basin_params = best_params[basin_id]
            print(f"📋 Parameters: {len(basin_params)} optimized")

            # Show first few parameters as example
            param_items = list(basin_params.items())[:5]
            for param_name, param_value in param_items:
                print(f"   {param_name}: {param_value:.6f}")
            if len(basin_params) > 5:
                print(f"   ... and {len(basin_params) - 5} more parameters")

    # Overall summary
    print(f"\n" + "=" * 80)
    print("📊 OVERALL SUMMARY")
    print("=" * 80)
    print(f"✅ Successful basins: {successful_basins}/{total_basins}")
    print(f"🔧 Algorithm used: {algorithm_name}")
    print(f"📊 Objective function: {obj_func}")

    if objective_values:
        print(f"🎯 Best {obj_func}: {min(objective_values):.6f}")
        print(f"📈 Average {obj_func}: {np.mean(objective_values):.6f}")
        print(f"📊 Std {obj_func}: {np.std(objective_values):.6f}")

    # Save location
    output_dir = training_cfg.get("output_dir", "results")
    experiment_name = training_cfg.get("experiment_name", "experiment")
    result_path = os.path.join(output_dir, experiment_name)
    print(f"💾 Results saved to: {result_path}")


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
        
        config = ConfigManager.create_calibration_config(
            config_file=args.config,
            args=args
        )

        # Print configuration summary
        print_config_summary(config, verbose)

        # Check algorithm availability
        training_cfg = config["training_cfgs"]
        if (
            training_cfg["algorithm_name"] == "genetic_algorithm"
            and not DEAP_AVAILABLE
        ):
            print(
                "❌ ERROR: Genetic algorithm requested but DEAP is not available"
            )
            print("💡 Install DEAP with: pip install deap")
            return 1

        # Run calibration using unified interface
        if verbose:
            print(f"\n🚀 Starting calibration with unified architecture...")
            print(
                f"📦 Using: hydromodel.trainers.unified_calibrate.calibrate()"
            )

        # The new unified calibration call - single function, single parameter!
        results = calibrate(config)

        # Process and display results
        process_results(results, config, verbose)

        # Save configuration file if requested
        if args.save_config or args.config is None:
            output_dir = training_cfg.get("output_dir", "results")
            experiment_name = training_cfg.get("experiment_name", "experiment")
            config_output_path = os.path.join(
                output_dir, experiment_name, "calibration_config.yaml"
            )
            save_config_file(config, config_output_path)

        print(f"\n🎉 XAJ calibration completed successfully!")
        print(f"✨ Used latest unified architecture: calibrate(config)")
        print(
            f"🔧 Model: {config['model_cfgs']['model_name']} | Algorithm: {training_cfg['algorithm_name']}"
        )

        return 0

    except KeyboardInterrupt:
        print("\nCalibration interrupted by user")
        return 1
    except Exception as e:
        print(f"\nERROR: Calibration failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
