"""
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
import json
from pathlib import Path
import pandas as pd

# Add hydromodel to path
repo_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_path)

from hydromodel.configs.config_manager import ConfigManager
from hydromodel.configs.script_utils import ScriptUtils
from hydromodel.trainers.unified_calibrate import calibrate
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
        print_category_statistics,
        categorize_floods_by_peak,
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
        description="Categorized Unit Hydrograph Calibration with Unified Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Unified Architecture Categorized Unit Hydrograph Calibration

Configuration File Mode:
  python run_categorized_uh_optimization.py --config configs/categorized_uh_config.yaml

Quick Setup Mode:
  python run_categorized_uh_optimization.py --quick-setup --station-id songliao_21401550 --algorithm genetic_algorithm

Advanced Usage:
  # Override specific parameters
  python run_categorized_uh_optimization.py --config my_config.yaml --override model_cfgs.model_params.uh_lengths='{"small":8,"medium":16,"large":24}'
        """,
    )

    # Add common arguments
    ScriptUtils.add_common_arguments(parser)
    
    # Add categorized UH specific arguments
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
    """Create configuration from quick setup arguments"""
    # Parse UH lengths JSON
    try:
        uh_lengths = json.loads(args.uh_lengths)
    except json.JSONDecodeError:
        print(f"❌ Invalid UH lengths JSON: {args.uh_lengths}")
        uh_lengths = {"small": 8, "medium": 16, "large": 24}

    # Create a minimal args namespace for ConfigManager
    class QuickArgs:
        def __init__(self):
            self.data_source_type = "floodevent"
            self.data_source_path = args.data_path
            self.basin_ids = [args.station_id]
            self.warmup_length = args.warmup_length
            self.variables = ["P_eff", "Q_obs_eff"]
            self.model = "categorized_unit_hydrograph"
            self.algorithm = args.algorithm
            self.output_dir = args.output_dir or "results"
            self.experiment_name = (
                args.experiment_name
                or f"categorized_uh_{args.station_id}_{args.algorithm}"
            )
            self.random_seed = 1234

            # Algorithm-specific parameters
            if args.algorithm == "scipy_minimize":
                self.scipy_method = "SLSQP"
                self.max_iterations = 1000
            elif args.algorithm == "SCE_UA":
                self.rep = 2000
                self.ngs = 1000
            elif args.algorithm == "genetic_algorithm":
                self.pop_size = 80
                self.n_generations = 50

    quick_args = QuickArgs()

    # Use ConfigManager to create the configuration
    config = ConfigManager.create_calibration_config(args=quick_args)

    # Add categorized unit hydrograph specific parameters
    config["model_cfgs"]["model_params"].update(
        {
            "net_rain_name": "P_eff",
            "obs_flow_name": "Q_obs_eff",
            "category_weights": get_category_weights_scheme(
                args.category_weights
            ),
            "uh_lengths": uh_lengths,
        }
    )

    return config








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
    if ScriptUtils.handle_template_creation(args, create_categorized_uh_template, "Categorized Unit Hydrograph"):
        return

    # Setup configuration using unified workflow
    config = ScriptUtils.setup_configuration(
        args, create_quick_setup_config, 
        "run_categorized_uh_optimization.py", "Categorized Unit Hydrograph",
        "*categorized*.yaml"
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
    if not ScriptUtils.validate_and_show_config(config, args.verbose, "Categorized Unit Hydrograph"):
        return

    if args.dry_run:
        print("\n🔍 Dry run completed - configuration is valid")
        return

    try:
        # Note: For categorized unit hydrograph models with flood events, we don't need to
        # load data separately as the unified calibrate() function handles it
        print(f"\n🚀 Starting categorized unit hydrograph calibration...")
        print(f"📦 Using unified calibrate(config) interface")

        # Run calibration using unified interface
        results = calibrate(config)

        # Process results using unified ResultsManager
        processed_results = process_results(results, config, args)


        ScriptUtils.print_completion_message(config, "categorized unit hydrograph calibration")

    except Exception as e:
        print(f"❌ Calibration failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
