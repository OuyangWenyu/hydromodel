"""
Run event-based XAJ calibration with unified data path resolution.
"""

import argparse
import os
import sys
from pathlib import Path


repo_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_path)

from hydromodel.configs.config_manager import (  # noqa: E402
    load_config_from_file,
    validate_and_show_config,
)
from hydromodel.trainers.unified_calibrate import calibrate  # noqa: E402


def get_default_event_config():
    return {
        "data_cfgs": {
            "dataset": "songliao_event",
            "source": "local",
            "dataset_name": "songliaorrevent",
            "time_unit": ["3h"],
            "basin_ids": ["songliao_21401550"],
            "warmup_length": 360,
            "variables": ["rain", "ES", "inflow", "flood_event"],
            "is_event_data": True,
            "train_period": ["1984-01-01", "2005-12-31"],
            "test_period": ["2006-01-01", "2023-12-31"],
        },
        "model_cfgs": {
            "name": "xaj_mz",
            "params": {
                "source_type": "sources",
                "source_book": "HF",
                "kernel_size": 15,
            },
        },
        "training_cfgs": {
            "algorithm": "SCE_UA",
            "SCE_UA": {"rep": 5000, "ngs": 1000},
            "loss": "RMSE",
            "output_dir": "results/event_calibration",
            "experiment_name": "event_xaj_calibration",
            "save_config": True,
        },
        "evaluation_cfgs": {
            "metrics": ["NSE", "KGE", "RMSE"],
        },
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run event-based XAJ calibration.",
    )
    parser.add_argument("--config", type=str, help="YAML or JSON config.")
    parser.add_argument(
        "--default",
        action="store_true",
        help="Use the built-in songliao_event example config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate configuration without calibration.",
    )
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument(
        "--no-save-config",
        dest="save_config",
        action="store_false",
        default=True,
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    try:
        if args.config:
            if not os.path.exists(args.config):
                print(f"Configuration file not found: {args.config}")
                return 1
            config = load_config_from_file(args.config)
        elif args.default:
            config = get_default_event_config()
        else:
            print("Specify --config or --default")
            return 1

        training_cfgs = config.setdefault("training_cfgs", {})
        if args.output_dir:
            training_cfgs["output_dir"] = args.output_dir
        if args.experiment_name:
            training_cfgs["experiment_name"] = args.experiment_name
        training_cfgs["save_config"] = args.save_config

        if not validate_and_show_config(config, True, "Event-based XAJ Model"):
            return 1

        if args.dry_run:
            print("Configuration resolved and validated")
            return 0

        calibrate(config)
        print("Event calibration completed")
        return 0

    except KeyboardInterrupt:
        print("Calibration interrupted")
        return 1
    except Exception as exc:
        print(f"Error: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
