"""
Author: Wenyu Ouyang
Date: 2025-08-07
LastEditTime: 2025-08-07 11:56:18
LastEditors: Wenyu Ouyang
Description: Configuration-driven calibration script using unified config system
FilePath: \hydromodel\scripts\run_calibration_with_config.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np

# Add hydromodel to path
repo_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_path)

from hydromodel.configs.unified_config import UnifiedConfig, load_config
from hydromodel.trainers.unified_calibrate import (
    calibrate_with_config,
    DEAP_AVAILABLE,
)
from hydromodel.datasets.data_preprocess import (
    _get_pe_q_from_ts,
    cross_val_split_tsdata,
)
from hydrodatasource.reader.floodevent import FloodEventDatasource


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Configuration-driven Hydrological Model Calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration File Structure:
  The config file should contain four main sections:
  - data_cfgs: Data settings (paths, basins, periods, etc.)
  - model_cfgs: Model settings (model type, parameters, etc.)
  - training_cfgs: Training/calibration settings (algorithm, parameters, etc.)
  - evaluation_cfgs: Evaluation settings (loss function, metrics, etc.)

Examples:
  # Run with existing config file
  python run_calibration_with_config.py --config configs/examples/xaj_sceua_example.yaml

  # Create and run with default config
  python run_calibration_with_config.py --create-default-config my_config.yaml

  # Run with config override
  python run_calibration_with_config.py --config my_config.yaml --override model_cfgs.model_name=gr4j

  # List available example configs
  python run_calibration_with_config.py --list-examples
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=False,
        help="Path to configuration file (YAML or JSON)",
    )

    parser.add_argument(
        "--create-default-config",
        type=str,
        help="Create a default configuration file and exit",
    )

    parser.add_argument(
        "--list-examples",
        action="store_true",
        help="List available example configuration files",
    )

    parser.add_argument(
        "--override",
        "-o",
        action="append",
        help="Override config values (e.g., -o model_cfgs.model_name=xaj -o training_cfgs.algorithm_name=SCE_UA)",
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

    return parser.parse_args()


def list_example_configs():
    """List available example configuration files"""
    examples_dir = Path(repo_path) / "configs" / "examples"

    print("📋 Available Example Configuration Files:")
    print("=" * 60)

    if not examples_dir.exists():
        print("❌ No example configurations found")
        return

    for config_file in sorted(examples_dir.glob("*.yaml")):
        print(f"📄 {config_file.name}")

        # Try to read the description from the file
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                first_lines = [f.readline().strip() for _ in range(3)]
                description = ""
                for line in first_lines:
                    if (
                        line.startswith("#")
                        and "configuration" in line.lower()
                    ):
                        description = line.lstrip("# ").strip()
                        break

            if description:
                print(f"   📝 {description}")
            print(f"   📂 Path: {config_file}")
            print()

        except Exception:
            print(f"   📂 Path: {config_file}")
            print()


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
            # Try to evaluate as Python literal
            import ast

            current[final_key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Keep as string
            current[final_key] = value

        print(f"   ✅ {key_path} = {value}")


def validate_config(config: UnifiedConfig, verbose: bool = True) -> bool:
    """Validate configuration and show summary"""
    if verbose:
        print("🔍 Configuration Validation:")
        print("=" * 60)

        print("📊 Data Configuration:")
        data_cfgs = config.data_cfgs
        print(f"   📁 Data type: {data_cfgs.get('data_type')}")
        print(f"   📂 Data directory: {data_cfgs.get('data_dir')}")
        print(f"   🏭 Basin IDs: {', '.join(data_cfgs.get('basin_ids', []))}")
        print(f"   ⏱️ Warmup length: {data_cfgs.get('warmup_length')} days")

        print("\n🤖 Model Configuration:")
        model_cfgs = config.model_cfgs
        print(f"   🔧 Model: {model_cfgs.get('model_name')}")

        print("\n🎯 Training Configuration:")
        training_cfgs = config.training_cfgs
        print(f"   🔬 Algorithm: {training_cfgs.get('algorithm_name')}")
        print(
            f"   📁 Output: {training_cfgs.get('output_dir')}/{training_cfgs.get('experiment_name')}"
        )

        print("\n📈 Evaluation Configuration:")
        eval_cfgs = config.evaluation_cfgs
        print(f"   📉 Objective: {eval_cfgs.get('objective_function')}")
        print(f"   📊 Loss type: {eval_cfgs.get('loss_type')}")

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


def load_data(config: UnifiedConfig, verbose: bool = True):
    """Load data based on configuration"""
    data_cfgs = config.data_cfgs
    model_cfgs = config.model_cfgs

    data_type = data_cfgs.get("data_type")
    model_name = model_cfgs.get("model_name")

    if verbose:
        print(f"\n🔄 Loading data (type: {data_type}, model: {model_name})...")

    if model_name in ["unit_hydrograph", "categorized_unit_hydrograph"]:
        # Load flood event data for unit hydrograph models
        data_dir = data_cfgs.get("data_dir")
        basin_ids = data_cfgs.get("basin_ids", [])
        warmup_length = data_cfgs.get("warmup_length", 0)

        if not basin_ids:
            raise ValueError(
                "Basin IDs must be specified for unit hydrograph models"
            )

        # Load flood events
        dataset = FloodEventDatasource(
            data_dir,
            time_unit=["3h"],
            trange4cache=["1960-01-01 02", "2024-12-31 23"],
            warmup_length=warmup_length,
        )

        all_event_data = dataset.load_1basin_flood_events(
            station_id=basin_ids[0],  # Use first basin for now
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

    else:
        # Load time series data for traditional models
        periods = data_cfgs.get("time_periods", {})
        cross_val = data_cfgs.get("cross_validation", {})

        train_and_test_data = cross_val_split_tsdata(
            data_cfgs.get("data_type"),
            data_cfgs.get("data_dir"),
            cross_val.get("folds", 1),
            periods.get("calibration"),
            periods.get("testing"),
            periods.get("overall"),
            data_cfgs.get("warmup_length", 365),
            data_cfgs.get("basin_ids", []),
        )

        # Use first fold for now (TODO: handle cross-validation)
        p_and_e, qobs = _get_pe_q_from_ts(train_and_test_data[0])

        if verbose:
            print(
                f"   ✅ Loaded time series data: {p_and_e.shape}, {qobs.shape}"
            )

        return (p_and_e, qobs)


def main():
    """Main calibration function"""
    args = parse_arguments()

    # Handle special operations
    if args.list_examples:
        list_example_configs()
        return

    if args.create_default_config:
        print(
            f"🔧 Creating default configuration: {args.create_default_config}"
        )
        from hydromodel.configs.unified_config import create_default_config

        config = create_default_config(args.create_default_config)
        print(
            f"✅ Default configuration saved to: {args.create_default_config}"
        )
        print("\n📋 Configuration preview:")
        print(config)
        return

    # Load configuration
    if not args.config:
        print(
            "❌ Configuration file is required. Use --config or --create-default-config"
        )
        print("💡 Run with --list-examples to see available examples")
        return

    try:
        config = load_config(args.config)
        print(f"✅ Loaded configuration: {args.config}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
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
    if not validate_config(config, args.verbose):
        return

    if args.dry_run:
        print("\n🔍 Dry run completed - configuration is valid")
        return

    try:
        # Load data
        data = load_data(config, args.verbose)

        # Run calibration
        print(f"\n🚀 Starting calibration with unified interface...")
        results = calibrate_with_config(config, data)

        # Process results
        print(f"\n🎉 Calibration completed!")
        print(f"📊 Results summary:")

        if isinstance(results, dict):
            for key, result in results.items():
                convergence = result.get("convergence", "unknown")
                objective_value = result.get("objective_value", float("inf"))
                print(
                    f"   🏭 {key}: {convergence}, objective={objective_value:.6f}"
                )
        else:
            print(f"   📈 Result: {results}")

    except Exception as e:
        print(f"❌ Calibration failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
