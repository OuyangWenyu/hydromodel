r"""
Author: Wenyu Ouyang
Date: 2025-08-07
LastEditTime: 2025-08-08 11:23:27
LastEditors: Wenyu Ouyang
Description: Shared Unit Hydrograph calibration script using unified architecture
FilePath: \hydromodel\scripts\run_shared_uh_optimization.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import sys
import argparse
from pathlib import Path

# Add hydromodel to path
repo_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_path)

from hydromodel import SETTING
from hydromodel.configs.config_manager import ConfigManager
from hydromodel.configs.script_utils import ScriptUtils
from hydromodel.trainers.unified_calibrate import calibrate
from hydromodel.core.results_manager import results_manager


# Optional imports - handle missing dependencies gracefully
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Shared Unit Hydrograph Calibration with Unified Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Unit Hydrograph Model Types Supported:
  - unit_hydrograph: Shared unit hydrograph optimization for flood events

Algorithm Types Supported:
  - scipy_minimize: SciPy optimization methods (default)
  - SCE_UA: Shuffled Complex Evolution via spotpy
  - genetic_algorithm: Genetic algorithm via DEAP (if installed)

Usage Examples:
  # Basic UH calibration with default settings
  python run_shared_uh_optimization.py --model-type unit_hydrograph --algorithm scipy_minimize

  # Configuration file approach (recommended)
  python run_shared_uh_optimization.py --config config.yaml
        """,
    )

    # Add common arguments
    ScriptUtils.add_common_arguments(parser)

    # Unit hydrograph specific arguments
    parser.add_argument(
        "--data-source-type",
        type=str,
        default="floodevent",
        choices=["floodevent", "selfmadehydrodataset", "camels"],
        help="Dataset type (default: floodevent)",
    )

    parser.add_argument(
        "--data-source-path",
        type=str,
        default=os.path.join(
            SETTING["local_data_path"]["datasets-interim"], "songliaorrevent"
        ),
        help="Data directory path (uses default if not specified)",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="songliaorrevents",
        help="Name of songliao flood event dataset (default: songliaorrevents)",
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
        default=["net_rain", "inflow"],
        help="Variables to calibrate",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="unit_hydrograph",
        choices=["unit_hydrograph"],
        help="Unit hydrograph model type (default: unit_hydrograph)",
    )

    # Unit hydrograph specific parameters
    parser.add_argument(
        "--n-uh",
        type=int,
        default=24,
        help="Unit hydrograph length (default: 24)",
    )

    parser.add_argument(
        "--smoothing-factor",
        type=float,
        default=0.1,
        help="Smoothing factor for unit hydrograph (default: 0.1)",
    )

    parser.add_argument(
        "--peak-violation-weight",
        type=float,
        default=10000.0,
        help="Peak violation penalty weight (default: 10000.0)",
    )

    # Algorithm-specific parameters
    parser.add_argument(
        "--scipy-method",
        type=str,
        default="SLSQP",
        help="SciPy optimization method (default: SLSQP)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=500,
        help="Maximum iterations for scipy (default: 500)",
    )

    parser.add_argument(
        "--obj-func",
        type=str,
        default="RMSE",
        choices=["RMSE", "NSE", "KGE"],
        help="Objective function (default: RMSE)",
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

    args = parser.parse_args()
    return args


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
            args.model = "unit_hydrograph"

        config = ScriptUtils.setup_configuration(
            args,
        )
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
            config, verbose, "Unit Hydrograph"
        ):
            return 1

        if args.dry_run:
            print("\n🔍 Dry run completed - configuration is valid")
            return 0

        # Run calibration using unified interface
        if verbose:
            print(
                "\n🚀 Starting unit hydrograph calibration with unified architecture..."
            )
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
                output_dir, "uh_calibration_config.yaml"
            )
            ScriptUtils.save_config_file(config, config_output_path)

        ScriptUtils.print_completion_message(
            config, "Unit Hydrograph calibration"
        )
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
