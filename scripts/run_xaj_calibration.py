r"""
Author: Wenyu Ouyang
Date: 2025-08-07
LastEditTime: 2025-08-08 19:17:46
LastEditors: Wenyu Ouyang
Description: XAJ model calibration script using the latest unified architecture
FilePath: \hydromodel\scripts\run_xaj_calibration_unified.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import argparse
import sys
import os
from pathlib import Path

# Add hydromodel to path
repo_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_path)

from hydromodel import SETTING
from hydromodel.trainers.unified_calibrate import calibrate  # noqa: E402
from hydromodel.configs.script_utils import ScriptUtils  # noqa: E402
from hydromodel.core.results_manager import results_manager  # noqa: E402


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

Usage Examples:
  # Basic XAJ calibration with default settings
  python run_xaj_calibration_unified.py --model-type xaj_mz --algorithm SCE_UA

  # Configuration file approach (recommended)
  python run_xaj_calibration_unified.py --config config.yaml
        """,
    )

    # Add common arguments
    ScriptUtils.add_common_arguments(parser)

    # XAJ-specific arguments
    parser.add_argument(
        "--data-source-type",
        type=str,
        default="camels",
        choices=["camels", "selfmadehydrodataset", "owndata"],
        help="Dataset type (default: camels)",
    )

    parser.add_argument(
        "--data-source-path",
        type=str,
        default=os.path.join(
            SETTING["local_data_path"]["datasets-origin"],
            "camels",
            "camels_us",
        ),
        help="Data directory path (uses default if not specified)",
    )

    parser.add_argument(
        "--basin-ids",
        nargs="+",
        default=["01013500"],
        help="Basin IDs to calibrate (default: 01013500)",
    )

    parser.add_argument(
        "--variables",
        nargs="+",
        default=["prcp", "PET", "streamflow"],
        help="Variables to calibrate (default: prcp, PET, streamflow)",
    )

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
        help="XAJ computation method (default: HF)",
    )

    parser.add_argument(
        "--kernel-size",
        type=int,
        default=15,
        help="XAJ convolutional kernel size (default: 15)",
    )

    # Algorithm-specific parameters
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

    parser.add_argument(
        "--obj-func",
        type=str,
        default="RMSE",
        choices=["RMSE", "NSE", "KGE"],
        help="Objective function (default: RMSE)",
    )

    parser.add_argument(
        "--param-range-file",
        type=str,
        help="Parameter range file path (uses default if not specified)",
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
            args.model = "xaj_mz"

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
            config, verbose, "XAJ Model"
        ):
            return 1

        if args.dry_run:
            print("\n🔍 Dry run completed - configuration is valid")
            return 0

        # Run calibration using unified interface
        if verbose:
            print("\n🚀 Starting XAJ calibration with unified architecture...")
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
                output_dir, "calibration_config.yaml"
            )
            ScriptUtils.save_config_file(config, config_output_path)

        ScriptUtils.print_completion_message(config, "XAJ calibration")
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
