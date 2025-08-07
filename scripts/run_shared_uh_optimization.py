"""
Author: Wenyu Ouyang
Date: 2025-08-07
LastEditTime: 2025-08-07 14:19:27
LastEditors: Wenyu Ouyang
Description: Unit Hydrograph calibration script using unified config system
FilePath: \hydromodel\scripts\run_unit_hydrograph_with_config.py
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

from hydromodel.configs.unified_config import (
    UnifiedConfig,
    load_config,
    create_default_config,
)
from hydromodel.trainers.unified_calibrate import (
    calibrate_with_config,
    DEAP_AVAILABLE,
)
from hydrodatasource.reader.floodevent import FloodEventDatasource
from hydromodel.trainers.unit_hydrograph_trainer import (
    evaluate_single_event_from_uh,
    print_report_preview,
    save_results_to_csv,
)
from hydroutils.hydro_plot import (
    plot_unit_hydrograph,
    setup_matplotlib_chinese,
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Unit Hydrograph Calibration with Unified Config System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration-based Unit Hydrograph Calibration

This script supports both configuration file mode and quick setup mode:

Configuration File Mode:
  python run_unit_hydrograph_with_config.py --config configs/examples/unit_hydrograph_example.yaml

Quick Setup Mode:
  python run_unit_hydrograph_with_config.py --quick-setup --station-id songliao_21401550 --algorithm scipy_minimize

Create Configuration Template:
  python run_unit_hydrograph_with_config.py --create-template my_uh_config.yaml

Advanced Usage:
  # Override specific parameters
  python run_unit_hydrograph_with_config.py --config my_config.yaml --override model_cfgs.model_params.n_uh=32
  
  # Use different algorithms
  python run_unit_hydrograph_with_config.py --config my_config.yaml --override training_cfgs.algorithm_name=genetic_algorithm
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
            "data_type": "flood_events",
            "data_dir": "D:\\data\\waterism\\datasets-interim\\songliaorrevent",
            "basin_ids": ["songliao_21401550"],
            "warmup_length": 480,  # 8 hours * 60 minutes / 3 hours for 3h data
            "time_periods": {
                "overall": ["1960-01-01", "2024-12-31"],
                "calibration": ["1960-01-01", "2020-12-31"],
                "testing": ["2021-01-01", "2024-12-31"],
            },
            "cross_validation": {"enabled": False, "folds": 1},
            "param_range_file": None,
            "random_seed": 1234,
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
            "output_dir": "results",
            "experiment_name": "unit_hydrograph_experiment",
        },
        "evaluation_cfgs": {
            "loss_type": "event_based",
            "objective_function": "RMSE",
            "metrics": [
                "RMSE",
                "NSE",
                "flood_peak_error",
                "flood_volume_error",
            ],
            "events_config": {"include_peak_obs": True},
            "evaluation_period": "testing",
        },
    }

    unified_config = UnifiedConfig(config_dict=config)
    unified_config.save_config(template_file)

    return unified_config


def create_quick_setup_config(args):
    """Create configuration from quick setup arguments"""
    from hydrodatasource.configs.config import SETTING

    # Default data path if not provided
    data_path = args.data_path or os.path.join(
        SETTING["local_data_path"]["datasets-interim"], "songliaorrevent"
    )

    config_dict = {
        "data_cfgs": {
            "data_type": "flood_events",
            "data_dir": data_path,
            "basin_ids": [args.station_id],
            "warmup_length": args.warmup_length,
            "time_periods": {
                "overall": ["1960-01-01", "2024-12-31"],
                "calibration": ["1960-01-01", "2020-12-31"],
                "testing": ["2021-01-01", "2024-12-31"],
            },
            "cross_validation": {"enabled": False, "folds": 1},
            "param_range_file": None,
            "random_seed": 1234,
        },
        "model_cfgs": {
            "model_name": "unit_hydrograph",
            "model_params": {
                "n_uh": args.n_uh,
                "smoothing_factor": 0.1,
                "peak_violation_weight": 10000.0,
                "apply_peak_penalty": True,
                "net_rain_name": "P_eff",
                "obs_flow_name": "Q_obs_eff",
            },
        },
        "training_cfgs": {
            "algorithm_name": args.algorithm,
            "algorithm_params": {
                # Will be filled based on algorithm type
            },
            "output_dir": args.output_dir or "results",
            "experiment_name": args.experiment_name
            or f"uh_{args.station_id}_{args.algorithm}",
        },
        "evaluation_cfgs": {
            "loss_type": "event_based",
            "objective_function": "RMSE",
            "metrics": [
                "RMSE",
                "NSE",
                "flood_peak_error",
                "flood_volume_error",
            ],
            "events_config": {"include_peak_obs": True},
            "evaluation_period": "testing",
        },
    }

    # Set algorithm-specific parameters
    if args.algorithm == "scipy_minimize":
        config_dict["training_cfgs"]["algorithm_params"].update(
            {"method": "SLSQP", "max_iterations": 500}
        )
    elif args.algorithm == "SCE_UA":
        config_dict["training_cfgs"]["algorithm_params"].update(
            {
                "random_seed": 1234,
                "rep": 1000,
                "ngs": 1000,
                "kstop": 50,
                "peps": 0.1,
                "pcento": 0.1,
            }
        )
    elif args.algorithm == "genetic_algorithm":
        config_dict["training_cfgs"]["algorithm_params"].update(
            {
                "random_seed": 1234,
                "pop_size": 80,
                "n_generations": 50,
                "cx_prob": 0.7,
                "mut_prob": 0.2,
                "save_freq": 5,
            }
        )

    return UnifiedConfig(config_dict=config_dict)


def apply_overrides(config: UnifiedConfig, overrides: list):
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
        current = config.config
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


def validate_and_show_config(
    config: UnifiedConfig, verbose: bool = True
) -> bool:
    """Validate configuration and show summary"""
    if verbose:
        print("🔍 Unit Hydrograph Configuration Summary:")
        print("=" * 60)

        data_cfgs = config.data_cfgs
        model_cfgs = config.model_cfgs
        training_cfgs = config.training_cfgs
        eval_cfgs = config.evaluation_cfgs

        print("📊 Data Configuration:")
        print(f"   📂 Data directory: {data_cfgs.get('data_dir')}")
        print(f"   🏭 Station ID: {', '.join(data_cfgs.get('basin_ids', []))}")
        print(f"   ⏱️ Warmup length: {data_cfgs.get('warmup_length')} steps")

        print("\n🔧 Model Configuration:")
        print(
            f"   📏 Unit hydrograph length: {model_cfgs.get('model_params', {}).get('n_uh')}"
        )
        print(
            f"   🔀 Smoothing factor: {model_cfgs.get('model_params', {}).get('smoothing_factor')}"
        )

        print("\n🎯 Training Configuration:")
        print(f"   🔬 Algorithm: {training_cfgs.get('algorithm_name')}")
        print(
            f"   📁 Output: {training_cfgs.get('output_dir')}/{training_cfgs.get('experiment_name')}"
        )

        print("\n📈 Evaluation Configuration:")
        print(f"   📉 Objective: {eval_cfgs.get('objective_function')}")

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


def load_flood_events_data(config: UnifiedConfig, verbose: bool = True):
    """Load flood events data based on configuration"""
    data_cfgs = config.data_cfgs

    if verbose:
        print(f"\n🔄 Loading flood events data...")

    # Load flood events
    dataset = FloodEventDatasource(
        data_cfgs.get("data_dir"),
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


def process_results(results, config: UnifiedConfig, args):
    """Process and display calibration results"""
    print(f"\n📈 Unit Hydrograph Calibration Results:")
    print("=" * 60)

    convergence = results.get("convergence", "unknown")
    objective_value = results.get("objective_value", float("inf"))
    best_params = results.get("best_params", {})

    print(f"✅ Convergence: {convergence}")
    print(f"🎯 Best RMSE: {objective_value:.6f}")

    if convergence == "success" and "unit_hydrograph" in best_params:
        uh_params_dict = best_params["unit_hydrograph"]
        model_cfgs = config.model_cfgs
        n_uh = model_cfgs.get("model_params", {}).get("n_uh", 24)

        # Extract unit hydrograph parameters
        uh_params = [uh_params_dict[f"uh_{i+1}"] for i in range(n_uh)]

        print(f"📊 Unit Hydrograph Parameters ({len(uh_params)} values):")
        for i, param in enumerate(uh_params[:5]):  # Show first 5 values
            print(f"   uh_{i+1}: {param:.6f}")
        if len(uh_params) > 5:
            print(f"   ... ({len(uh_params)-5} more parameters)")

        # Plot results if requested
        if args.plot_results:
            setup_matplotlib_chinese()
            plot_unit_hydrograph(uh_params, "Calibrated Unit Hydrograph")
            print("📈 Unit hydrograph plot generated")

        # Save evaluation results if requested
        if args.save_evaluation:
            try:
                # Load data again for evaluation
                all_event_data = load_flood_events_data(config, verbose=False)

                # Evaluate each event
                evaluation_results = []
                for event in all_event_data:
                    result = evaluate_single_event_from_uh(
                        event,
                        uh_params,
                        net_rain_key="P_eff",
                        obs_flow_key="Q_obs_eff",
                    )
                    if result:
                        evaluation_results.append(result)

                if evaluation_results:
                    # Create DataFrame and save
                    df = pd.DataFrame(evaluation_results)
                    df_sorted = df.sort_values("NSE", ascending=False)

                    training_cfgs = config.training_cfgs
                    output_dir = os.path.join(
                        training_cfgs.get("output_dir", "results"),
                        training_cfgs.get("experiment_name", "experiment"),
                    )
                    os.makedirs(output_dir, exist_ok=True)

                    csv_file = os.path.join(
                        output_dir, "unit_hydrograph_evaluation.csv"
                    )
                    save_results_to_csv(
                        df_sorted, csv_file, "Unit Hydrograph Evaluation"
                    )

                    # Show preview
                    print_report_preview(
                        df_sorted, "Unit Hydrograph Evaluation", top_n=5
                    )

                    print(f"💾 Detailed evaluation saved to: {csv_file}")
                else:
                    print("⚠️ No valid evaluation results found")

            except Exception as e:
                print(f"❌ Failed to save evaluation: {e}")
    else:
        print("❌ Calibration failed - no valid parameters found")


def main():
    """Main function"""
    args = parse_arguments()

    # Handle template creation
    if args.create_template:
        print(
            f"🔧 Creating unit hydrograph configuration template: {args.create_template}"
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
            config = load_config(args.config)
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
        config.config["training_cfgs"]["output_dir"] = args.output_dir
    if args.experiment_name:
        config.config["training_cfgs"][
            "experiment_name"
        ] = args.experiment_name

    # Validate configuration
    if not validate_and_show_config(config, args.verbose):
        return

    if args.dry_run:
        print("\n🔍 Dry run completed - configuration is valid")
        return

    try:
        # Load data
        data = load_flood_events_data(config, args.verbose)

        # Run calibration
        print(f"\n🚀 Starting unit hydrograph calibration...")
        results = calibrate_with_config(config, data)

        # Process results
        process_results(results, config, args)

        training_cfgs = config.training_cfgs
        output_path = os.path.join(
            training_cfgs.get("output_dir", "results"),
            training_cfgs.get("experiment_name", "experiment"),
        )

        print(f"\n🎉 Unit hydrograph calibration completed!")
        print(f"✨ Used unified config-driven interface")
        print(f"💾 Results saved to: {output_path}")

    except Exception as e:
        print(f"❌ Calibration failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
