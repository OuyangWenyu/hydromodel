r"""
Author: Wenyu Ouyang
Date: 2025-08-07
LastEditTime: 2025-08-07 22:45:00
LastEditors: Wenyu Ouyang
Description: Categorized Unit Hydrograph calibration script using unified architecture
FilePath: \hydromodel\scripts\run_categorized_uh_optimization.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import sys
import argparse

from pathlib import Path
 

# Add hydromodel to path
repo_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_path)

from hydromodel.configs.config_manager import ConfigManager  # noqa: E402
from hydromodel.configs.script_utils import ScriptUtils  # noqa: E402
from hydromodel.trainers.unified_calibrate import calibrate
from hydromodel.core.results_manager import results_manager

# Optional imports - handle missing dependencies gracefully
 

 

 


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Categorized Unit Hydrograph Calibration using Latest Unified Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Categorized Unit Hydrograph Model Types Supported:
  - categorized_unit_hydrograph: Multi-category UH optimization for different flood event sizes

Algorithm Types Supported:
  - genetic_algorithm: Genetic algorithm via DEAP (recommended for categorized UH)
  - SCE_UA: Shuffled Complex Evolution via spotpy
  - scipy_minimize: SciPy optimization methods

Usage Examples:
  # Basic categorized UH calibration with default settings
  python run_categorized_uh_optimization.py --model-type categorized_unit_hydrograph --algorithm genetic_algorithm

  # Configuration file approach (recommended)
  python run_categorized_uh_optimization.py --config config.yaml
        """,
    )

    # Add common arguments
    ScriptUtils.add_common_arguments(parser)

    # Categorized UH specific arguments
    parser.add_argument(
        "--data-source-type",
        type=str,
        default="floodevent",
        choices=["floodevent", "selfmadehydrodataset"],
        help="Dataset type (default: floodevent)",
    )

    parser.add_argument(
        "--data-source-path",
        type=str,
        default="D:\\data\\waterism\\datasets-interim\\songliaorrevent",
        help="Data directory path (uses default if not specified)",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="songliaorrevents",
        help="Name of flood event dataset (default: songliaorrevents)",
    )

    parser.add_argument(
        "--basin-ids",
        nargs="+",
        default=["songliao_21401550"],
        help="Basin IDs to calibrate (default: songliao_21401550)",
    )

    parser.add_argument(
        "--variables",
        nargs="+",
        default=["P_eff", "Q_obs_eff"],
        help="Variables to calibrate (default: P_eff, Q_obs_eff)",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="categorized_unit_hydrograph",
        choices=["categorized_unit_hydrograph"],
        help="Categorized UH model type (default: categorized_unit_hydrograph)",
    )

    parser.add_argument(
        "--warmup-length",
        type=int,
        default=480,
        help="Warmup length in time steps (default: 480)",
    )

    parser.add_argument(
        "--time-unit",
        type=str,
        default="3h",
        help="Time unit for data (default: 3h)",
    )

    # Categorized UH specific parameters
    parser.add_argument(
        "--category-weights",
        type=str,
        default="default",
        choices=["default", "balanced", "aggressive"],
        help="Category weight scheme (default: default)",
    )

    parser.add_argument(
        "--uh-lengths",
        type=str,
        default='{"small":8,"medium":16,"large":24}',
        help="UH lengths for categories as JSON (default: small:8, medium:16, large:24)",
    )

    # Algorithm-specific parameters
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

    parser.add_argument(
        "--cx-prob",
        type=float,
        default=0.7,
        help="GA crossover probability (default: 0.7)",
    )

    parser.add_argument(
        "--mut-prob",
        type=float,
        default=0.2,
        help="GA mutation probability (default: 0.2)",
    )

    parser.add_argument(
        "--rep",
        type=int,
        default=1000,
        help="SCE-UA repetitions (default: 1000)",
    )

    parser.add_argument(
        "--ngs",
        type=int,
        default=200,
        help="SCE-UA number of complexes (default: 200)",
    )

    parser.add_argument(
        "--obj-func",
        type=str,
        default="multi_category_loss",
        choices=["multi_category_loss", "RMSE", "NSE", "KGE"],
        help="Objective function (default: multi_category_loss)",
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
            "data_source_type": "floodevent",
            "data_source_path": "D:\\data\\waterism\\datasets-interim\\songliaorrevent",
            "basin_ids": ["songliao_21401550"],
            "warmup_length": 480,
            "variables": ["P_eff", "Q_obs_eff"],
            "time_range": ["1960-01-01", "2024-12-31"],
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
            "loss_config": {
                "type": "event_based",
                "obj_func": "multi_category_loss",
            },
            "output_dir": "results",
            "experiment_name": "categorized_uh_experiment",
            "random_seed": 1234,
        },
        "evaluation_cfgs": {
            "metrics": [
                "RMSE",
                "NSE",
                "flood_peak_error",
                "flood_volume_error",
                "category_performance",
            ],
            "save_results": True,
            "plot_results": True,
        },
    }

    ConfigManager.save_config_to_file(config, template_file)
    return config


def create_quick_setup_config(args):
    """Deprecated placeholder: use default config + args directly.
    Note: script-level options like --uh-lengths or --category-weights will be
    applied via --override or future arg-to-config mapping.
    """
    return ConfigManager.create_calibration_config(args=args)


def process_results(results: dict, config: dict, args):
    """Process and display calibration results using unified ResultsManager"""
    # Use the unified results manager
    processed_results = results_manager.process_results(results, config, args)

    # Return processed results for potential further use
    return processed_results


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
            args.model = "categorized_unit_hydrograph"

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
            config, verbose, "Categorized Unit Hydrograph"
        ):
            return 1

        if args.dry_run:
            print("\n🔍 Dry run completed - configuration is valid")
            return 0

        # Run calibration using unified interface
        if verbose:
            print("\n🚀 Starting categorized unit hydrograph calibration with unified architecture...")
            print("📦 Using unified calibrate(config) interface")

        # The new unified calibration call - single function, single parameter!
        results = calibrate(config)

        # Process and display results using unified ResultsManager
        _ = process_results(results, config, args)

        # Save configuration file if requested
        if args.save_config:
            training_cfgs = config.get("training_cfgs", {})
            output_dir = os.path.join(
                training_cfgs.get("output_dir", "results"),
                training_cfgs.get("experiment_name", "experiment"),
            )
            config_output_path = os.path.join(
                output_dir, "categorized_uh_config.yaml"
            )
            ScriptUtils.save_config_file(config, config_output_path)

        ScriptUtils.print_completion_message(config, "Categorized Unit Hydrograph calibration")
        return 0

    except KeyboardInterrupt:
        print("\n👋 Calibration interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: Calibration failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
