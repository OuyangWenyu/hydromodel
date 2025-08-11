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
        """,
    )

    # Add common arguments
    ScriptUtils.add_common_arguments(parser)

    # Add unit hydrograph specific arguments
    parser.add_argument(
        "--n-uh",
        type=int,
        default=24,
        help="Unit hydrograph length (quick setup mode)",
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


 


def process_results(results, config: dict, args):
    """Process and display calibration results using unified ResultsManager"""
    # Use the unified results manager
    _ = results_manager.process_results(results, config, args)

    # Return processed results for potential further use
    return processed_results


def main():
    """Main function"""
    args = parse_arguments()

    # Ensure correct model defaults for quick setup
    if not getattr(args, "model_type", None) and not getattr(args, "model", None):
        args.model = "unit_hydrograph"

    # Skip template creation; go straight to default+args config

    # Setup configuration using unified workflow
    config = ScriptUtils.setup_configuration(
        args,
        None,
        "run_shared_uh_optimization.py",
        "*unit_hydrograph*.yaml",
    )
    if config is None:
        return

    # Apply overrides
    ScriptUtils.apply_overrides(config, args.override)

    # Apply command line overrides for output settings
    if args.output_dir:
        config["training_cfgs"]["output_dir"] = args.output_dir
    if args.experiment_name:
        config["training_cfgs"]["experiment_name"] = args.experiment_name

    # Validate configuration
    if not ScriptUtils.validate_and_show_config(
        config, args.verbose, "Unit Hydrograph"
    ):
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

        ScriptUtils.print_completion_message(
            config, "unit hydrograph calibration"
        )

    except Exception as e:
        print(f"❌ Calibration failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
