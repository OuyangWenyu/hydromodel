"""
GR Model Calibration Script

Calibrate GR-series models (GR4J, GR5J, GR6J) using the unified config API.

Usage:
    # With config file
    python scripts/calibrate_gr.py --config configs/example_gr_config.yaml

    # Dry run (validate only, no actual calibration)
    python scripts/calibrate_gr.py --config configs/example_gr_config.yaml --dry-run
"""

import argparse
import sys
from pathlib import Path

from hydromodel.configs.config_manager import (
    load_config_from_file,
    validate_config,
)
from hydromodel.trainers.unified_calibrate import calibrate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate GR-series hydrological models using unified config API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/calibrate_gr.py --config configs/example_config.yaml
  python scripts/calibrate_gr.py --config configs/example_config.yaml --dry-run
        """,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML/JSON config file (canonical schema: data_cfgs, model_cfgs, training_cfgs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running calibration",
    )
    parser.add_argument(
        "--output-dir",
        help="Override training_cfgs.output_dir",
    )
    parser.add_argument(
        "--experiment-name",
        help="Override training_cfgs.experiment_name",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config_from_file(args.config)

    # Apply CLI overrides
    if args.output_dir:
        config.setdefault("training_cfgs", {})["output_dir"] = args.output_dir
    if args.experiment_name:
        config.setdefault("training_cfgs", {})["experiment_name"] = args.experiment_name

    # Validate
    result = validate_config(config)
    if not result["valid"]:
        print("Configuration validation FAILED:")
        for err in result["errors"]:
            print(f"  - {err}")
        sys.exit(1)

    if result["warnings"]:
        for warn in result["warnings"]:
            print(f"Warning: {warn}")

    if args.dry_run:
        print("Configuration validated successfully (dry-run).")
        print(f"  Model: {config['model_cfgs']['name']}")
        print(f"  Basins: {config['data_cfgs'].get('basin_ids', [])}")
        print(f"  Algorithm: {config['training_cfgs'].get('algorithm', 'SCE_UA')}")
        return

    # Run calibration
    print(f"Starting calibration: {config['model_cfgs']['name']} model")
    results = calibrate(config)
    print("Calibration completed successfully.")
    print(f"  Results saved to: {config['training_cfgs'].get('output_dir', 'results')}")
    return results


if __name__ == "__main__":
    main()
