"""
Author: Wenyu Ouyang
Date: 2025-08-07
LastEditTime: 2025-08-07 14:25:05
LastEditors: Wenyu Ouyang
Description: Categorized Unit Hydrograph calibration script using unified config system
FilePath: \hydromodel\scripts\run_categorized_uh_with_config.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import sys
import argparse
import json
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
from hydromodel.models.unit_hydrograph import (
    evaluate_single_event_from_uh,
    print_report_preview,
    save_results_to_csv,
    print_category_statistics,
    categorize_floods_by_peak,
)
from hydroutils.hydro_plot import (
    plot_unit_hydrograph,
    setup_matplotlib_chinese,
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Categorized Unit Hydrograph Calibration with Unified Config System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration-based Categorized Unit Hydrograph Calibration

This script supports both configuration file mode and quick setup mode:

Configuration File Mode:
  python run_categorized_uh_with_config.py --config configs/examples/categorized_uh_example.yaml

Quick Setup Mode:
  python run_categorized_uh_with_config.py --quick-setup --station-id songliao_21401550 --algorithm genetic_algorithm

Create Configuration Template:
  python run_categorized_uh_with_config.py --create-template my_categorized_uh_config.yaml

Advanced Usage:
  # Override specific parameters
  python run_categorized_uh_with_config.py --config my_config.yaml --override model_cfgs.model_params.uh_lengths='{"small":8,"medium":16,"large":24}'
  
  # Use different category weights scheme
  python run_categorized_uh_with_config.py --config my_config.yaml --category-weights balanced
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
        help="Create categorized unit hydrograph configuration template and exit",
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
        default="genetic_algorithm",
        choices=["scipy_minimize", "SCE_UA", "genetic_algorithm"],
        help="Optimization algorithm (quick setup mode)",
    )

    parser.add_argument(
        "--category-weights",
        type=str,
        default="default",
        choices=["default", "balanced", "aggressive"],
        help="Category weight scheme (quick setup mode)",
    )

    parser.add_argument(
        "--uh-lengths",
        type=str,
        default='{"small":8,"medium":16,"large":24}',
        help="UH lengths for categories as JSON (quick setup mode)",
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
        help="Override config values (e.g., -o model_cfgs.model_params.uh_lengths='{\"small\":12}')",
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


def get_category_weights_scheme(scheme_name: str):
    """Get predefined category weight schemes"""
    schemes = {
        "default": {
            "small": {"smoothing_factor": 0.1, "peak_violation_weight": 100.0},
            "medium": {
                "smoothing_factor": 0.5,
                "peak_violation_weight": 500.0,
            },
            "large": {
                "smoothing_factor": 1.0,
                "peak_violation_weight": 1000.0,
            },
        },
        "balanced": {
            "small": {"smoothing_factor": 0.2, "peak_violation_weight": 200.0},
            "medium": {
                "smoothing_factor": 0.2,
                "peak_violation_weight": 200.0,
            },
            "large": {"smoothing_factor": 0.2, "peak_violation_weight": 200.0},
        },
        "aggressive": {
            "small": {"smoothing_factor": 0.05, "peak_violation_weight": 50.0},
            "medium": {
                "smoothing_factor": 0.1,
                "peak_violation_weight": 100.0,
            },
            "large": {
                "smoothing_factor": 0.5,
                "peak_violation_weight": 2000.0,
            },
        },
    }
    return schemes.get(scheme_name, schemes["default"])


def create_categorized_uh_template(template_file: str):
    """Create a categorized unit hydrograph configuration template"""
    config = {
        "data_cfgs": {
            "data_type": "flood_events",
            "data_dir": "D:\\data\\waterism\\datasets-interim\\songliaorrevent",
            "basin_ids": ["songliao_21401550"],
            "warmup_length": 480,
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
            "model_name": "categorized_unit_hydrograph",
            "model_params": {
                "net_rain_name": "P_eff",
                "obs_flow_name": "Q_obs_eff",
                "category_weights": {
                    "small": {
                        "smoothing_factor": 0.1,
                        "peak_violation_weight": 100.0,
                    },
                    "medium": {
                        "smoothing_factor": 0.5,
                        "peak_violation_weight": 500.0,
                    },
                    "large": {
                        "smoothing_factor": 1.0,
                        "peak_violation_weight": 1000.0,
                    },
                },
                "uh_lengths": {"small": 8, "medium": 16, "large": 24},
            },
        },
        "training_cfgs": {
            "algorithm_name": "genetic_algorithm",
            "algorithm_params": {
                "random_seed": 1234,
                "pop_size": 80,
                "n_generations": 50,
                "cx_prob": 0.7,
                "mut_prob": 0.2,
                "save_freq": 5,
            },
            "output_dir": "results",
            "experiment_name": "categorized_uh_experiment",
        },
        "evaluation_cfgs": {
            "loss_type": "event_based",
            "objective_function": "multi_category_loss",
            "metrics": [
                "RMSE",
                "NSE",
                "flood_peak_error",
                "flood_volume_error",
                "category_performance",
            ],
            "events_config": {
                "include_peak_obs": True,
                "categorization_method": "peak_magnitude",
            },
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

    # Parse UH lengths JSON
    try:
        uh_lengths = json.loads(args.uh_lengths)
    except json.JSONDecodeError:
        print(f"❌ Invalid UH lengths JSON: {args.uh_lengths}")
        uh_lengths = {"small": 8, "medium": 16, "large": 24}

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
            "model_name": "categorized_unit_hydrograph",
            "model_params": {
                "net_rain_name": "P_eff",
                "obs_flow_name": "Q_obs_eff",
                "category_weights": get_category_weights_scheme(
                    args.category_weights
                ),
                "uh_lengths": uh_lengths,
            },
        },
        "training_cfgs": {
            "algorithm_name": args.algorithm,
            "algorithm_params": {},
            "output_dir": args.output_dir or "results",
            "experiment_name": args.experiment_name
            or f"categorized_uh_{args.station_id}_{args.algorithm}",
        },
        "evaluation_cfgs": {
            "loss_type": "event_based",
            "objective_function": "multi_category_loss",
            "metrics": [
                "RMSE",
                "NSE",
                "flood_peak_error",
                "flood_volume_error",
                "category_performance",
            ],
            "events_config": {
                "include_peak_obs": True,
                "categorization_method": "peak_magnitude",
            },
            "evaluation_period": "testing",
        },
    }

    # Set algorithm-specific parameters
    if args.algorithm == "scipy_minimize":
        config_dict["training_cfgs"]["algorithm_params"].update(
            {"method": "SLSQP", "max_iterations": 1000}
        )
    elif args.algorithm == "SCE_UA":
        config_dict["training_cfgs"]["algorithm_params"].update(
            {
                "random_seed": 1234,
                "rep": 2000,
                "ngs": 1000,
                "kstop": 100,
                "peps": 0.01,
                "pcento": 0.01,
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
        print("🔍 Categorized Unit Hydrograph Configuration Summary:")
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
        model_params = model_cfgs.get("model_params", {})
        uh_lengths = model_params.get("uh_lengths", {})
        print(f"   📏 UH lengths: {uh_lengths}")
        print(
            f"   🏷️ Category weights available: {list(model_params.get('category_weights', {}).keys())}"
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
        include_peak_obs=True,  # Required for categorization
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
    """Process and display categorized unit hydrograph calibration results"""
    print(f"\n📈 Categorized Unit Hydrograph Calibration Results:")
    print("=" * 60)

    convergence = results.get("convergence", "unknown")
    objective_value = results.get("objective_value", float("inf"))
    best_params = results.get("best_params", {})
    categorization_info = results.get("categorization_info", {})

    print(f"✅ Convergence: {convergence}")
    print(f"🎯 Best objective value: {objective_value:.6f}")

    # Display categorization information
    if categorization_info:
        print(f"\n📊 Flood Categorization Results:")
        categories = categorization_info.get("categories", [])
        thresholds = categorization_info.get("thresholds", {})
        events_per_category = categorization_info.get(
            "events_per_category", {}
        )

        print(f"   🏷️ Categories: {categories}")
        print(f"   📐 Thresholds: {thresholds}")
        print(f"   📊 Events per category:")
        for category, count in events_per_category.items():
            print(f"      {category.capitalize()}: {count} events")

    if (
        convergence == "success"
        and "categorized_unit_hydrograph" in best_params
    ):
        cat_uh_params = best_params["categorized_unit_hydrograph"]

        print(f"\n📏 Unit Hydrograph Parameters by Category:")
        for category, params_dict in cat_uh_params.items():
            if isinstance(params_dict, dict):
                uh_params = list(params_dict.values())
                print(
                    f"   📈 {category.capitalize()}: {len(uh_params)} parameters"
                )
                print(f"      First 3 values: {uh_params[:3]}")

        # Plot results if requested
        if args.plot_results:
            setup_matplotlib_chinese()

            for category, params_dict in cat_uh_params.items():
                if isinstance(params_dict, dict):
                    uh_params = list(params_dict.values())
                    plot_unit_hydrograph(
                        uh_params,
                        f"Categorized Unit Hydrograph - {category.capitalize()}",
                    )

            print("📈 Categorized unit hydrograph plots generated")

        # Save evaluation results if requested
        if args.save_evaluation:
            try:
                # Load data again for evaluation
                all_event_data = load_flood_events_data(config, verbose=False)

                # Categorize floods by peak for evaluation
                categories, thresholds = categorize_floods_by_peak(
                    all_event_data,
                    net_rain_key="P_eff",
                    obs_flow_key="Q_obs_eff",
                )

                # Evaluate each event (simplified evaluation)
                evaluation_results = []
                for i, event in enumerate(all_event_data):
                    category = categories[i]

                    if category in cat_uh_params:
                        category_params = cat_uh_params[category]
                        if isinstance(category_params, dict):
                            uh_params = list(category_params.values())

                            result = evaluate_single_event_from_uh(
                                event,
                                uh_params,
                                net_rain_key="P_eff",
                                obs_flow_key="Q_obs_eff",
                            )

                            if result:
                                result["所属类别"] = category
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
                        output_dir, "categorized_uh_evaluation.csv"
                    )
                    save_results_to_csv(
                        df_sorted,
                        csv_file,
                        "Categorized Unit Hydrograph Evaluation",
                    )

                    # Show preview and category statistics
                    print_report_preview(
                        df_sorted,
                        "Categorized Unit Hydrograph Evaluation",
                        top_n=5,
                    )
                    print_category_statistics(df_sorted)

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
            f"🔧 Creating categorized unit hydrograph configuration template: {args.create_template}"
        )
        config = create_categorized_uh_template(args.create_template)
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
        print("🔍 Categorized Unit Hydrograph Calibration with Unified Config")
        print("=" * 60)
        print()
        print("📋 Available options:")
        print()
        print("1️⃣ Use existing configuration file:")
        print(
            "   python run_categorized_uh_with_config.py --config my_config.yaml"
        )
        print()
        print("2️⃣ Quick setup mode (no config file needed):")
        print("   python run_categorized_uh_with_config.py --quick-setup")
        print(
            "   python run_categorized_uh_with_config.py --quick-setup --station-id songliao_21401550 --algorithm genetic_algorithm"
        )
        print()
        print("3️⃣ Create a configuration template:")
        print(
            "   python run_categorized_uh_with_config.py --create-template my_categorized_config.yaml"
        )
        print()
        print("4️⃣ Use example configurations:")

        # Check for example configs
        examples_dir = Path(repo_path) / "configs" / "examples"
        if examples_dir.exists():
            example_files = list(examples_dir.glob("*categorized*.yaml"))
            if example_files:
                print("   Available examples:")
                for example_file in example_files:
                    print(
                        f"   - python run_categorized_uh_with_config.py --config {example_file}"
                    )
            else:
                print(
                    "   - python run_categorized_uh_with_config.py --config configs/examples/categorized_uh_example.yaml"
                )
        else:
            print(
                "   - python run_categorized_uh_with_config.py --config configs/examples/categorized_uh_example.yaml"
            )

        print()
        print(
            "💡 For more options, run: python run_categorized_uh_with_config.py --help"
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
        print(f"\n🚀 Starting categorized unit hydrograph calibration...")
        results = calibrate_with_config(config, data)

        # Process results
        process_results(results, config, args)

        training_cfgs = config.training_cfgs
        output_path = os.path.join(
            training_cfgs.get("output_dir", "results"),
            training_cfgs.get("experiment_name", "experiment"),
        )

        print(f"\n🎉 Categorized unit hydrograph calibration completed!")
        print(f"✨ Used unified config-driven interface")
        print(f"💾 Results saved to: {output_path}")

    except Exception as e:
        print(f"❌ Calibration failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
