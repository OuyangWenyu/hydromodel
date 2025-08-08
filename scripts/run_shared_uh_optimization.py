"""
Author: Wenyu Ouyang
Date: 2025-08-07
LastEditTime: 2025-08-08 09:21:26
LastEditors: Wenyu Ouyang
Description: Shared Unit Hydrograph calibration script using unified architecture
FilePath: \hydromodel\scripts\run_shared_uh_optimization.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

# Add hydromodel to path
repo_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_path)

from hydromodel.configs.config_manager import ConfigManager
from hydromodel.trainers.unified_calibrate import (
    calibrate,
    DEAP_AVAILABLE,
)
from hydromodel.core.results_manager import results_manager

# Optional imports - handle missing dependencies gracefully
try:
    from hydrodatasource.reader.floodevent import FloodEventDatasource

    FLOODEVENT_AVAILABLE = True
except ImportError:
    print(
        "Warning: hydrodatasource not available - flood event loading disabled"
    )
    FLOODEVENT_AVAILABLE = False

try:
    from hydromodel.trainers.unit_hydrograph_trainer import (
        evaluate_single_event_from_uh,
        print_report_preview,
        save_results_to_csv,
    )

    UH_TRAINER_AVAILABLE = True
except ImportError:
    print("Warning: unit hydrograph trainer functions not available")
    UH_TRAINER_AVAILABLE = False

try:
    from hydroutils.hydro_plot import (
        plot_unit_hydrograph,
        setup_matplotlib_chinese,
    )

    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: hydroutils plotting not available - plotting disabled")
    PLOTTING_AVAILABLE = False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Shared Unit Hydrograph Calibration with Unified Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Unified Architecture Unit Hydrograph Calibration

Uses the latest unified calibrate(config) interface with ConfigManager.

Configuration File Mode:
  python run_shared_uh_optimization.py --config configs/unit_hydrograph_config.yaml

Quick Setup Mode:
  python run_shared_uh_optimization.py --quick-setup --station-id songliao_21401550 --algorithm scipy_minimize

Create Configuration Template:
  python run_shared_uh_optimization.py --create-template my_uh_config.yaml

Advanced Usage:
  # Override specific parameters
  python run_shared_uh_optimization.py --config my_config.yaml --override model_cfgs.model_params.n_uh=32
  
  # Use different algorithms
  python run_shared_uh_optimization.py --config my_config.yaml --override training_cfgs.algorithm_name=genetic_algorithm
        """,
    )

    # Configuration file mode
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to configuration file (YAML or JSON)",
    )

    parser.add_argument(
        "--create-template",
        type=str,
        help="Create unit hydrograph configuration template and exit",
    )

    # Quick setup mode
    parser.add_argument(
        "--quick-setup",
        action="store_true",
        help="Quick setup mode using command line arguments",
    )

    # Quick setup parameters
    parser.add_argument(
        "--data-path",
        type=str,
        help="Data directory path (quick setup mode)",
    )

    parser.add_argument(
        "--station-id",
        type=str,
        default="songliao_21401550",
        help="Station ID for calibration (quick setup mode)",
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="scipy_minimize",
        choices=["scipy_minimize", "SCE_UA", "genetic_algorithm"],
        help="Optimization algorithm (quick setup mode)",
    )

    parser.add_argument(
        "--n-uh",
        type=int,
        default=24,
        help="Unit hydrograph length (quick setup mode)",
    )

    parser.add_argument(
        "--warmup-length",
        type=int,
        default=480,
        help="Warmup length in time steps (quick setup mode)",
    )

    # Common options
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        help="Override config values (e.g., -o model_cfgs.model_params.n_uh=32)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Override experiment name",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run - validate config and show what would be done",
    )

    parser.add_argument(
        "--plot-results",
        action="store_true",
        help="Generate plots of results",
    )

    parser.add_argument(
        "--save-evaluation",
        action="store_true",
        help="Save detailed evaluation results to CSV",
    )

    return parser.parse_args()


def create_unit_hydrograph_template(template_file: str):
    """Create a unit hydrograph configuration template"""
    config = {
        "data_cfgs": {
            "data_source_type": "floodevent",
            "data_source_path": "D:\\data\\waterism\\datasets-interim\\songliaorrevent",
            "basin_ids": ["songliao_21401550"],
            "warmup_length": 480,  # 8 hours * 60 minutes / 3 hours for 3h data
            "variables": ["P_eff", "Q_obs_eff"],
            "time_range": ["1960-01-01", "2024-12-31"],
        },
        "model_cfgs": {
            "model_name": "unit_hydrograph",
            "model_params": {
                "n_uh": 24,
                "smoothing_factor": 0.1,
                "peak_violation_weight": 10000.0,
                "apply_peak_penalty": True,
                "net_rain_name": "P_eff",
                "obs_flow_name": "Q_obs_eff",
            },
        },
        "training_cfgs": {
            "algorithm_name": "scipy_minimize",
            "algorithm_params": {"method": "SLSQP", "max_iterations": 500},
            "loss_config": {
                "type": "event_based",
                "obj_func": "RMSE",
            },
            "output_dir": "results",
            "experiment_name": "unit_hydrograph_experiment",
            "random_seed": 1234,
        },
        "evaluation_cfgs": {
            "metrics": [
                "RMSE",
                "NSE",
                "flood_peak_error",
                "flood_volume_error",
            ],
            "save_results": True,
            "plot_results": True,
        },
    }

    ConfigManager.save_config_to_file(config, template_file)
    return config


def create_quick_setup_config(args):
    """Create configuration from quick setup arguments"""

    # Create a minimal args namespace for ConfigManager
    class QuickArgs:
        def __init__(self):
            self.data_source_type = "floodevent"
            self.data_source_path = args.data_path
            self.basin_ids = [args.station_id]
            self.warmup_length = args.warmup_length
            self.variables = ["P_eff", "Q_obs_eff"]
            self.model = (
                "unit_hydrograph"  # Use 'model' instead of 'model_type'
            )
            self.algorithm = args.algorithm
            self.output_dir = args.output_dir or "results"
            self.experiment_name = (
                args.experiment_name
                or f"uh_{args.station_id}_{args.algorithm}"
            )
            self.random_seed = 1234

            # Algorithm-specific parameters
            if args.algorithm == "scipy_minimize":
                self.scipy_method = "SLSQP"
                self.max_iterations = 500
            elif args.algorithm == "SCE_UA":
                self.rep = 1000
                self.ngs = 1000
            elif args.algorithm == "genetic_algorithm":
                self.pop_size = 80
                self.n_generations = 50

    quick_args = QuickArgs()

    # Use ConfigManager to create the configuration
    config = ConfigManager.create_calibration_config(args=quick_args)

    # Add unit hydrograph specific parameters
    config["model_cfgs"]["model_params"].update(
        {
            "n_uh": args.n_uh,
            "smoothing_factor": 0.1,
            "peak_violation_weight": 10000.0,
            "apply_peak_penalty": True,
            "net_rain_name": "P_eff",
            "obs_flow_name": "Q_obs_eff",
        }
    )

    return config


def apply_overrides(config: dict, overrides: list):
    """Apply command line overrides to configuration"""
    if not overrides:
        return

    print("🔧 Applying configuration overrides:")

    for override in overrides:
        if "=" not in override:
            print(f"❌ Invalid override format: {override}")
            continue

        key_path, value = override.split("=", 1)
        keys = key_path.split(".")

        # Navigate to the nested key and set the value
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Convert value to appropriate type
        final_key = keys[-1]
        try:
            import ast

            current[final_key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            current[final_key] = value

        print(f"   ✅ {key_path} = {value}")


def validate_and_show_config(config: dict, verbose: bool = True) -> bool:
    """Validate configuration and show summary"""
    if verbose:
        print("🔍 Unit Hydrograph Configuration Summary:")
        print("=" * 60)

        data_cfgs = config.get("data_cfgs", {})
        model_cfgs = config.get("model_cfgs", {})
        training_cfgs = config.get("training_cfgs", {})
        eval_cfgs = config.get("evaluation_cfgs", {})

        print("📊 Data Configuration:")
        print(f"   📂 Data directory: {data_cfgs.get('data_source_path')}")
        print(f"   🏭 Station ID: {', '.join(data_cfgs.get('basin_ids', []))}")
        print(f"   ⏱️ Warmup length: {data_cfgs.get('warmup_length')} steps")
        print(f"   📋 Variables: {data_cfgs.get('variables', [])}")

        print("\n🔧 Model Configuration:")
        model_params = model_cfgs.get("model_params", {})
        print(f"   🏷️ Model name: {model_cfgs.get('model_name')}")
        print(f"   📏 Unit hydrograph length: {model_params.get('n_uh')}")
        print(
            f"   🔀 Smoothing factor: {model_params.get('smoothing_factor')}"
        )

        print("\n🎯 Training Configuration:")
        print(f"   🔬 Algorithm: {training_cfgs.get('algorithm_name')}")
        print(
            f"   📊 Objective: {training_cfgs.get('loss_config', {}).get('obj_func')}"
        )
        print(
            f"   📁 Output: {training_cfgs.get('output_dir')}/{training_cfgs.get('experiment_name')}"
        )

        # Check algorithm availability
        algorithm_name = training_cfgs.get("algorithm_name")
        if algorithm_name == "genetic_algorithm" and not DEAP_AVAILABLE:
            print(
                f"\n❌ ERROR: Algorithm '{algorithm_name}' requires DEAP package"
            )
            print("💡 Install with: pip install deap")
            return False

        print("\n✅ Configuration validation passed")

    return True


def load_flood_events_data(config: dict, verbose: bool = True):
    """Load flood events data based on configuration"""
    if not FLOODEVENT_AVAILABLE:
        raise ImportError(
            "FloodEventDatasource not available - install hydrodatasource package"
        )

    data_cfgs = config.get("data_cfgs", {})

    if verbose:
        print(f"\n🔄 Loading flood events data...")

    # Load flood events
    dataset = FloodEventDatasource(
        data_cfgs.get("data_source_path"),
        time_unit=["3h"],
        trange4cache=["1960-01-01 02", "2024-12-31 23"],
        warmup_length=data_cfgs.get("warmup_length", 480),
    )

    basin_ids = data_cfgs.get("basin_ids", [])
    if not basin_ids:
        raise ValueError("Basin IDs must be specified")

    all_event_data = dataset.load_1basin_flood_events(
        station_id=basin_ids[0],
        flow_unit="mm/3h",
        include_peak_obs=True,
        verbose=verbose,
    )

    if all_event_data is None:
        raise ValueError(f"No flood events found for basin {basin_ids[0]}")

    # Check for NaN values (excluding warmup period)
    dataset.check_event_data_nan(all_event_data, exclude_warmup=True)

    if verbose:
        print(f"   ✅ Loaded {len(all_event_data)} flood events")

    return all_event_data


def process_results(results, config: dict, args):
    """Process and display calibration results using unified ResultsManager"""
    # Use the unified results manager
    processed_results = results_manager.process_results(results, config, args)
    
    # Return processed results for potential further use
    return processed_results


def main():
    """Main function"""
    args = parse_arguments()

    # Handle template creation
    if args.create_template:
        print(
            f"Creating unit hydrograph configuration template: {args.create_template}"
        )
        config = create_unit_hydrograph_template(args.create_template)
        print(f"✅ Template saved to: {args.create_template}")
        print("\n📋 Configuration template preview:")
        print(config)
        return

    # Load or create configuration
    if args.config:
        # Configuration file mode
        try:
            config = ConfigManager.load_config_from_file(args.config)
            print(f"✅ Loaded configuration: {args.config}")
        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
            return
    elif args.quick_setup:
        # Quick setup mode
        print(
            "🚀 Quick setup mode - creating configuration from command line arguments"
        )
        config = create_quick_setup_config(args)
    else:
        # No config provided - show helpful information
        print("🔍 Unit Hydrograph Calibration with Unified Config")
        print("=" * 60)
        print()
        print("📋 Available options:")
        print()
        print("1️⃣ Use existing configuration file:")
        print(
            "   python run_unit_hydrograph_with_config.py --config my_config.yaml"
        )
        print()
        print("2️⃣ Quick setup mode (no config file needed):")
        print("   python run_unit_hydrograph_with_config.py --quick-setup")
        print(
            "   python run_unit_hydrograph_with_config.py --quick-setup --station-id songliao_21401550 --algorithm scipy_minimize"
        )
        print()
        print("3️⃣ Create a configuration template:")
        print(
            "   python run_unit_hydrograph_with_config.py --create-template my_uh_config.yaml"
        )
        print()
        print("4️⃣ Use example configurations:")

        # Check for example configs
        examples_dir = Path(repo_path) / "configs" / "examples"
        if examples_dir.exists():
            example_files = list(examples_dir.glob("*unit_hydrograph*.yaml"))
            if example_files:
                print("   Available examples:")
                for example_file in example_files:
                    print(
                        f"   - python run_unit_hydrograph_with_config.py --config {example_file}"
                    )
            else:
                print(
                    "   - python run_unit_hydrograph_with_config.py --config configs/examples/unit_hydrograph_example.yaml"
                )
        else:
            print(
                "   - python run_unit_hydrograph_with_config.py --config configs/examples/unit_hydrograph_example.yaml"
            )

        print()
        print(
            "💡 For more options, run: python run_unit_hydrograph_with_config.py --help"
        )
        print()

        # Offer to run quick setup with defaults
        try:
            response = (
                input(
                    "Would you like to run with default quick setup? (y/n): "
                )
                .lower()
                .strip()
            )
            if response in ["y", "yes", ""]:
                print("🚀 Using default quick setup configuration...")
                args.quick_setup = True
                config = create_quick_setup_config(args)
            else:
                return
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return

    # Apply overrides
    apply_overrides(config, args.override or [])

    # Apply command line overrides for output settings
    if args.output_dir:
        config["training_cfgs"]["output_dir"] = args.output_dir
    if args.experiment_name:
        config["training_cfgs"]["experiment_name"] = args.experiment_name

    # Validate configuration
    if not validate_and_show_config(config, args.verbose):
        return

    if args.dry_run:
        print("\n🔍 Dry run completed - configuration is valid")
        return

    try:
        # Note: For unit hydrograph models with flood events, we don't need to
        # load data separately as the unified calibrate() function handles it
        print(f"\n🚀 Starting unit hydrograph calibration...")
        print(f"📦 Using unified calibrate(config) interface")

        # Run calibration using unified interface
        results = calibrate(config)

        # Process results using unified ResultsManager
        processed_results = process_results(results, config, args)

        training_cfgs = config.get("training_cfgs", {})
        output_path = os.path.join(
            training_cfgs.get("output_dir", "results"),
            training_cfgs.get("experiment_name", "experiment"),
        )

        print(f"\n🎉 Unit hydrograph calibration completed!")
        print(f"✨ Used latest unified architecture: calibrate(config)")
        print(f"💾 Results saved to: {output_path}")

    except Exception as e:
        print(f"❌ Calibration failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
