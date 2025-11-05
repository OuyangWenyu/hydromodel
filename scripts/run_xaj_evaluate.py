r"""
Author: Wenyu Ouyang
Date: 2025-08-11
LastEditTime: 2025-11-01 20:30:00
LastEditors: zhuanglaihong
Description: XAJ model evaluation script using the unified architecture
FilePath: \hydromodel\scripts\run_xaj_evaluate.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import argparse
import sys
import os
from pathlib import Path
import yaml

# Add hydromodel to path
repo_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_path)

from hydromodel.trainers.unified_evaluate import evaluate  # noqa: E402


def load_config_from_calibration(calibration_dir: str) -> dict:
    """
    Load configuration from calibration directory.

    Parameters
    ----------
    calibration_dir : str
        Directory where calibration results are stored

    Returns
    -------
    dict
        Configuration dictionary
    """
    config_file = os.path.join(calibration_dir, "calibration_config.yaml")
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Configuration file not found: {config_file}\n"
            "Please make sure you are using the correct calibration directory."
        )

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def parse_arguments():
    """Parse command line arguments for evaluation script."""
    parser = argparse.ArgumentParser(
        description="XAJ model evaluation script using unified architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # Evaluate on test period
  python run_xaj_evaluate.py --calibration-dir results/xaj_experiment --eval-period test

  # Evaluate on train period
  python run_xaj_evaluate.py --calibration-dir results/xaj_experiment --eval-period train

  # Evaluate on custom period
  python run_xaj_evaluate.py --calibration-dir results/xaj_experiment \\
      --eval-period custom --custom-period 2020-01-01 2021-12-31

  # Specify output directory
  python run_xaj_evaluate.py --calibration-dir results/xaj_experiment \\
      --eval-period test --output-dir results/xaj_experiment/evaluation
        """,
    )

    parser.add_argument(
        "--calibration-dir",
        type=str,
        required=True,
        default="path/to/calibration_results",
        help="Calibration results directory (containing calibration_config.yaml and parameter files)",
    )

    parser.add_argument(
        "--eval-period",
        type=str,
        choices=["train", "test", "custom"],
        default="test",
        help="Evaluation period: train (training period), test (testing period), or custom (custom period)",
    )

    parser.add_argument(
        "--custom-period",
        type=str,
        nargs=2,
        help="Custom evaluation period, format: start_date end_date (e.g., 2020-01-01 2021-12-31)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Evaluation results output directory (default: calibration-dir/evaluation_<period>)",
    )

    parser.add_argument(
        "--param-dir",
        type=str,
        help="Parameter files directory (default: use calibration-dir)",
    )

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_arguments()

    try:
        print(f"\n {'='*60}")
        print(f" Loading configuration from: {args.calibration_dir}")
        print(f" {'='*60}\n")
        config = load_config_from_calibration(args.calibration_dir)

        # Determine evaluation period
        if args.eval_period == "train":
            eval_period = config["data_cfgs"]["train_period"]
            period_name = "train"
        elif args.eval_period == "test":
            eval_period = config["data_cfgs"]["test_period"]
            period_name = "test"
        elif args.eval_period == "custom":
            if args.custom_period is None:
                print(
                    "❌ Error: --custom-period required when --eval-period is 'custom'"
                )
                return 1
            eval_period = list(args.custom_period)
            period_name = (
                f"custom_{args.custom_period[0]}_{args.custom_period[1]}"
            )
        else:
            print(f"❌ Error: Invalid eval-period: {args.eval_period}")
            return 1

        print(f" Evaluation period ({args.eval_period}): {eval_period}")

        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.join(
                args.calibration_dir, f"evaluation_{period_name}"
            )

        # Determine parameter directory
        param_dir = args.param_dir if args.param_dir else args.calibration_dir

        # Create evaluation configuration
        print(f" Results will be saved to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Run evaluation
        results = evaluate(
            config,
            param_dir=param_dir,
            eval_period=eval_period,
            eval_output_dir=output_dir,
        )

        # Save evaluation summary
        print("\n📋 " + "=" * 77)
        print(" EVALUATION SUMMARY")
        print(f"   Calibration directory: {args.calibration_dir}")
        print(f"   Evaluation period: {eval_period}")
        print(f"   Output directory: {output_dir}")
        print(f"   Number of basins: {len(results)}")
        print(f"\n   Basin IDs:")
        for basin_id in results.keys():
            print(f"      • {basin_id}")
        print("=" * 80)

        # Save evaluation info
        eval_info = {
            "calibration_dir": args.calibration_dir,
            "param_dir": param_dir,
            "eval_period": eval_period,
            "eval_period_type": args.eval_period,
            "output_dir": output_dir,
            "basin_ids": list(results.keys()),
        }

        eval_info_file = os.path.join(output_dir, "evaluation_info.yaml")
        with open(eval_info_file, "w", encoding="utf-8") as f:
            yaml.dump(eval_info, f, allow_unicode=True)

        print(f"\n💾 Evaluation info saved to: {eval_info_file}")
        print(f"\n✅ Evaluation completed successfully! ✅\n")
        return 0

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return 1
    except KeyError as e:
        print(f"\n❌ Error: Missing configuration key: {e}")
        print("Please check that the calibration configuration is complete.")
        return 1
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
