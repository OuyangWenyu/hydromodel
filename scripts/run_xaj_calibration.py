"""
Run XAJ model calibration with unified data path resolution.
"""

import argparse
import os
import sys
from pathlib import Path


repo_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_path)

from hydromodel.configs.config_manager import (  # noqa: E402
    get_default_calibration_config,
    load_config_from_file,
    validate_and_show_config,
)
from hydromodel.trainers.unified_calibrate import calibrate  # noqa: E402


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run XAJ calibration with unified data configuration.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="YAML or JSON config using *_cfgs schema.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate configuration without calibration.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override training_cfgs.output_dir.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Override training_cfgs.experiment_name.",
    )
    parser.add_argument(
        "--no-save-config",
        dest="save_config",
        action="store_false",
        default=True,
        help="Disable saving resolved calibration config.",
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
        else:
            config = get_default_calibration_config()

        training_cfgs = config.setdefault("training_cfgs", {})
        if args.output_dir:
            training_cfgs["output_dir"] = args.output_dir
        if args.experiment_name:
            training_cfgs["experiment_name"] = args.experiment_name
        training_cfgs["save_config"] = args.save_config

        if not validate_and_show_config(config, True, "XAJ Model"):
            return 1

        if args.dry_run:
            print("Configuration resolved and validated")
            return 0

        calibrate(config)
        print("XAJ calibration completed")
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
