"""
GR Model Evaluation Script

Evaluate calibrated GR-series models using the unified config API.

Usage:
    # With config file (same as calibration config)
    python scripts/evaluate_gr.py --config configs/example_gr_config.yaml

    # Dry run (validate only, no actual evaluation)
    python scripts/evaluate_gr.py --config configs/example_gr_config.yaml --dry-run
"""

import argparse
import sys
from pathlib import Path

from hydromodel.configs.config_manager import (
    load_config_from_file,
    validate_config,
)
from hydromodel.trainers.unified_evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GR-series hydrological models using unified config API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/evaluate_gr.py --config configs/example_config.yaml
  python scripts/evaluate_gr.py --config configs/example_config.yaml --dry-run
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
        help="Validate configuration without running evaluation",
    )
    parser.add_argument(
        "--param-dir",
        help="Override param_dir for evaluation",
    )
    parser.add_argument(
        "--eval-period",
        default="test",
        choices=["train", "test"],
        help="Period to evaluate (default: test)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config_from_file(args.config)

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
        return

    # Determine param_dir
    param_dir = args.param_dir or config["training_cfgs"].get("output_dir", "results")

    # Run evaluation
    print(f"Starting evaluation: {config['model_cfgs']['name']} model")
    results = evaluate(config, param_dir=param_dir, eval_period=args.eval_period)
    print("Evaluation completed successfully.")
    return results


if __name__ == "__main__":
    main()
