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

from hydromodel.trainers.unified_calibrate import calibrate
from hydromodel.configs.config_manager import ConfigManager
from hydromodel.configs.script_utils import ScriptUtils
from hydromodel.core.results_manager import results_manager


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
        default=["prcp", "pet", "streamflow"],
        help="Variables to calibrate (default: prcp, pet, streamflow)",
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


def create_xaj_template(template_file: str):
    """Create an XAJ configuration template"""
    config = {
        "data_cfgs": {
            "data_source_type": "camels",
            "data_source_path": None,  # Will use default from hydro_setting.yml
            "basin_ids": ["01013500"],
            "warmup_length": 365,
            "variables": ["prcp", "pet", "streamflow"],
            "time_range": ["1990-01-01", "2010-12-31"],
        },
        "model_cfgs": {
            "model_name": "xaj_mz",
            "model_params": {
                "source_type": "sources",
                "source_book": "HF",
                "kernel_size": 15,
            },
        },
        "training_cfgs": {
            "algorithm_name": "SCE_UA",
            "algorithm_params": {
                "rep": 5000,
                "ngs": 1000,
            },
            "loss_config": {
                "type": "time_series",
                "obj_func": "RMSE",
            },
            "output_dir": "results",
            "experiment_name": "xaj_calibration_experiment",
            "random_seed": 1234,
        },
        "evaluation_cfgs": {
            "metrics": ["NSE", "RMSE", "KGE", "PBIAS"],
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
            self.data_source_type = getattr(args, "data_source_type", "camels")
            self.data_source_path = getattr(args, "data_source_path", None)
            self.basin_ids = getattr(args, "basin_ids", ["01013500"])
            self.warmup_length = getattr(args, "warmup_length", 365)
            self.variables = getattr(
                args, "variables", ["prcp", "pet", "streamflow"]
            )
            self.model = getattr(args, "model_type", "xaj_mz")
            self.algorithm = getattr(args, "algorithm", "SCE_UA")
            self.output_dir = getattr(args, "output_dir", "results")
            self.experiment_name = getattr(args, "experiment_name", None)
            self.random_seed = getattr(args, "random_seed", 1234)

            # XAJ-specific parameters
            self.source_type = getattr(args, "source_type", "sources")
            self.source_book = getattr(args, "source_book", "HF")
            self.kernel_size = getattr(args, "kernel_size", 15)
            self.obj_func = getattr(args, "obj_func", "RMSE")
            self.param_range_file = getattr(args, "param_range_file", None)

            # Algorithm-specific parameters
            if self.algorithm == "scipy_minimize":
                self.scipy_method = getattr(args, "scipy_method", "L-BFGS-B")
                self.max_iterations = getattr(args, "max_iterations", 1000)
            elif self.algorithm == "SCE_UA":
                self.rep = getattr(args, "rep", 5000)
                self.ngs = getattr(args, "ngs", 1000)
            elif self.algorithm == "genetic_algorithm":
                self.pop_size = getattr(args, "pop_size", 80)
                self.n_generations = getattr(args, "n_generations", 50)

    quick_args = QuickArgs()

    # Use ConfigManager to create the configuration
    config = ConfigManager.create_calibration_config(args=quick_args)

    # Add XAJ-specific parameters
    config["model_cfgs"]["model_params"].update(
        {
            "source_type": quick_args.source_type,
            "source_book": quick_args.source_book,
            "kernel_size": quick_args.kernel_size,
        }
    )

    # Set parameter range file if specified
    if quick_args.param_range_file:
        config["training_cfgs"][
            "param_range_file"
        ] = quick_args.param_range_file

    return config


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
        # Handle template creation
        if ScriptUtils.handle_template_creation(
            args, create_xaj_template, "XAJ Model"
        ):
            return 0

        # Setup configuration using unified workflow
        config = ScriptUtils.setup_configuration(
            args,
            create_quick_setup_config,
            "run_xaj_calibration_unified.py",
            "XAJ Model",
            "*xaj*.yaml",
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
            print(
                f"\n🚀 Starting XAJ calibration with unified architecture..."
            )
            print(f"📦 Using unified calibrate(config) interface")

        # The new unified calibration call - single function, single parameter!
        results = calibrate(config)

        # Process and display results using unified ResultsManager
        processed_results = process_results(results, config, args)

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
