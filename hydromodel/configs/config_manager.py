r"""
Author: Wenyu Ouyang
Date: 2025-08-07
LastEditTime: 2025-08-31 10:21:28
LastEditors: Wenyu Ouyang
Description: Unified configuration management system for hydromodel
FilePath: \hydromodel\hydromodel\configs\config_manager.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

from copy import deepcopy
import os
import warnings
import yaml
import json
import argparse
from datetime import datetime
from typing import Dict, Any

from hydromodel.models.model_config import resolve_model_param_config
from hydromodel.models.model_dict import MODEL_DICT, resolve_loss_config


def get_default_calibration_config() -> Dict[str, Any]:
    """
    Get default configuration for model calibration.

    Returns
        -------
        Dict[str, Any]
            Default calibration configuration
    """
    return {
        "data_cfgs": {
            "dataset": "camels_us",
            "basin_ids": ["01013500"],
            "warmup_length": 365,
            "variables": [
                "precipitation",
                "potential_evapotranspiration",
                "streamflow",
            ],
            "train_period": ["1985-10-01", "1995-09-30"],
            "valid_period": ["1995-10-01", "2005-09-30"],
            "test_period": ["2005-10-01", "2014-09-30"],
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
            "SCE_UA": {
                "rep": 5000,
                "ngs": 1000,
            },
            "loss": "RMSE",
            "param_range_file": None,
            "output_dir": "results",
            "experiment_name": None,  # Will be auto-generated
            "random_seed": 1234,
            # Save calibration config and param_range to output directory
            "save_config": True,
        },
        "evaluation_cfgs": {
            "metrics": ["NSE", "RMSE", "KGE", "PBIAS"],
            "save_results": True,
            "plot_results": True,
            "validation_split": 0.2,
            "bootstrap_samples": None,
        },
    }


def update_config_from_args(
    base_config: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    """
    Update base configuration with command line arguments.

    Parameters
    ----------
    base_config : Dict[str, Any]
        Base configuration to update
    args : argparse.Namespace
        Command line arguments

    Returns
    -------
    Dict[str, Any]
        Updated configuration
    """
    config = deepcopy(base_config)

    # Update data configuration
    if hasattr(args, "dataset") and args.dataset is not None:
        config["data_cfgs"]["dataset"] = args.dataset

    if hasattr(args, "source") and args.source is not None:
        config["data_cfgs"]["source"] = args.source

    if hasattr(args, "basin_ids") and args.basin_ids is not None:
        config["data_cfgs"]["basin_ids"] = args.basin_ids
    elif hasattr(args, "station_id") and args.station_id is not None:
        # Fallback: map single station_id to basin_ids list
        config["data_cfgs"]["basin_ids"] = [args.station_id]

    if hasattr(args, "warmup_length") and args.warmup_length is not None:
        config["data_cfgs"]["warmup_length"] = args.warmup_length

    if hasattr(args, "variables") and args.variables is not None:
        config["data_cfgs"]["variables"] = args.variables

    if hasattr(args, "time_unit") and args.time_unit is not None:
        config["data_cfgs"]["time_unit"] = [args.time_unit]

        # Convert time_unit to time_interval_hours for models that need it
        time_unit = args.time_unit
        if isinstance(time_unit, list) and len(time_unit) > 0:
            time_unit = time_unit[0]  # Take first element if it's a list

        # Convert time unit to hours using pandas functionality
        import pandas as pd

        time_unit_str = str(time_unit).strip()

        # Handle special cases and normalize
        if time_unit_str.lower() in ["daily", "1d", "d"]:
            time_unit_str = "1D"
        # Convert deprecated 'H' to 'h' to avoid pandas warning
        time_unit_str = time_unit_str.replace("H", "h")

        # Use pandas to parse the frequency and convert to hours
        try:
            freq = pd.Timedelta(time_unit_str)
            time_interval_hours = freq.total_seconds() / 3600
            # Put time_interval_hours in model_params where it belongs
            if "model_params" not in config["model_cfgs"]:
                config["model_cfgs"]["model_params"] = {}
            config["model_cfgs"]["model_params"][
                "time_interval_hours"
            ] = time_interval_hours
        except Exception:
            # Fallback to default if parsing fails
            if "model_params" not in config["model_cfgs"]:
                config["model_cfgs"]["model_params"] = {}
            config["model_cfgs"]["model_params"]["time_interval_hours"] = 24

    if hasattr(args, "is_event") and args.is_event is not None:
        config["data_cfgs"]["is_event_data"] = args.is_event

    # Update model configuration
    if hasattr(args, "model_type") and args.model_type is not None:
        config["model_cfgs"]["name"] = args.model_type
    elif hasattr(args, "model") and args.model is not None:
        config["model_cfgs"]["name"] = args.model

    # Update model parameters
    if hasattr(args, "source_type") and args.source_type is not None:
        config["model_cfgs"]["params"]["source_type"] = args.source_type

    if hasattr(args, "source_book") and args.source_book is not None:
        config["model_cfgs"]["params"]["source_book"] = args.source_book

    if hasattr(args, "kernel_size") and args.kernel_size is not None:
        config["model_cfgs"]["params"]["kernel_size"] = args.kernel_size

    # Update training configuration (if exists)
    if "training_cfgs" in config:
        if hasattr(args, "algorithm") and args.algorithm is not None:
            config["training_cfgs"]["algorithm"] = args.algorithm

        if hasattr(args, "obj_func") and args.obj_func is not None:
            config["training_cfgs"]["loss"] = args.obj_func

        if hasattr(args, "output_dir") and args.output_dir is not None:
            config["training_cfgs"]["output_dir"] = args.output_dir

        if (
            hasattr(args, "experiment_name")
            and args.experiment_name is not None
        ):
            config["training_cfgs"]["experiment_name"] = args.experiment_name

        if hasattr(args, "random_seed") and args.random_seed is not None:
            config["training_cfgs"]["random_seed"] = args.random_seed

        if hasattr(args, "save_config") and args.save_config is not None:
            config["training_cfgs"]["save_config"] = args.save_config

        # Algorithm-specific parameters
        if hasattr(args, "rep") and args.rep is not None:
            algorithm = config["training_cfgs"]["algorithm"]
            config["training_cfgs"].setdefault(algorithm, {})["rep"] = args.rep

        if hasattr(args, "ngs") and args.ngs is not None:
            algorithm = config["training_cfgs"]["algorithm"]
            config["training_cfgs"].setdefault(algorithm, {})["ngs"] = args.ngs

    # Handle model parameters
    if hasattr(args, "model_parameters") and args.model_parameters is not None:
        config["model_cfgs"]["parameters"] = args.model_parameters

    # Generate experiment name if not provided
    if (
        "training_cfgs" in config
        and config["training_cfgs"]["experiment_name"] is None
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = config["model_cfgs"]["name"]
        algorithm = config["training_cfgs"]["algorithm"]
        config["training_cfgs"][
            "experiment_name"
        ] = f"{model_name}_{algorithm}_{timestamp}"

    return config


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.

    Parameters
    ----------
    config_path : str
        Path to configuration file (YAML or JSON)

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.endswith(".json"):
            config = json.load(f)
        else:
            config = yaml.safe_load(f)

    return config or {}


def save_config_to_file(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to file.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to save
    config_path : str
        Path where to save the configuration
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        if config_path.endswith(".json"):
            json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            yaml.dump(
                config,
                f,
                default_flow_style=False,
                indent=2,
                allow_unicode=True,
            )


def setup_configuration_from_args(args) -> Dict[str, Any]:
    """
    Setup configuration from command line arguments

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments

    Returns
    -------
    Dict[str, Any] or None
        Configuration if successful, None if failed
    """
    if hasattr(args, "config") and args.config:
        # Configuration file mode
        try:
            config = load_config_from_file(args.config)
            return config
        except Exception as e:
            print(f"Error: Failed to load configuration: {e}")
            return None
    else:
        # Use default configuration
        config = get_default_calibration_config()
        # Apply args updates
        config = update_config_from_args(config, args)
        return config


def load_simplified_config(
    config_path: str = None, simple_config: dict = None
) -> dict:
    """Load a config file that already uses the canonical *_cfgs schema."""
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            simple_config = yaml.safe_load(f)
    elif simple_config is None:
        raise ValueError(
            " Must provide config.path or simple_config parameter "
        )

    required_sections = ["data_cfgs", "model_cfgs", "training_cfgs"]
    for section in required_sections:
        if section not in simple_config:
            raise ValueError(
                "Configuration must use canonical *_cfgs schema; "
                f"missing section: {section}"
            )

    return simple_config


def load_config_from_calibration(calibration_dir: str) -> dict:
    """
    Load configuration from calibration directory.

    This function loads the saved calibration configuration from a previous
    calibration run. The configuration is stored as 'calibration_config.yaml'
    in the calibration output directory.

    Parameters
    ----------
    calibration_dir : str
        Directory where calibration results are stored

    Returns
    -------
    dict
        Configuration dictionary with data_cfgs, model_cfgs, training_cfgs
        sections

    Raises
    ------
    FileNotFoundError
        If calibration_config.yaml is not found in the specified directory

    Examples
    --------
    >>> config = load_config_from_calibration("results/xaj_experiment")
    >>> eval_period = config["data_cfgs"]["test_period"]
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


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration and resolve agent-facing contracts."""
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "resolved_loss_config": None,
        "resolved_param_range": None,
    }

    if not isinstance(config, dict):
        result["errors"].append("Config must be a dictionary")
        result["valid"] = False
        return result

    required_sections = ["data_cfgs", "model_cfgs", "training_cfgs"]
    for section in required_sections:
        if section not in config:
            result["errors"].append(
                f"Missing required config section: {section}"
            )

    if result["errors"]:
        result["valid"] = False
        return result

    data_cfgs = config["data_cfgs"]
    model_cfgs = config["model_cfgs"]
    training_cfgs = config["training_cfgs"]

    model_name = model_cfgs.get("model_name") or model_cfgs.get("name")
    if not model_name:
        result["errors"].append("model_cfgs.model_name is required")
    elif model_name not in MODEL_DICT:
        result["errors"].append(f"Unsupported model: {model_name}")

    loss_config = training_cfgs.get("loss_config")
    if loss_config is None:
        loss_config = {
            "type": "time_series",
            "obj_func": training_cfgs.get("loss", "RMSE"),
        }
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result["resolved_loss_config"] = resolve_loss_config(loss_config)
        result["warnings"].extend(str(item.message) for item in caught)
    except Exception as exc:
        result["errors"].append(f"Invalid loss_config: {exc}")

    algorithm_name = training_cfgs.get(
        "algorithm", training_cfgs.get("algorithm_name", "SCE_UA")
    )
    supported_algorithms = {
        "SCE_UA",
        "sceua",
        "GA",
        "genetic_algorithm",
        "scipy",
        "Scipy",
        "scipy_minimize",
    }
    if algorithm_name not in supported_algorithms:
        result["errors"].append(f"Unsupported algorithm: {algorithm_name}")

    train_period = data_cfgs.get("train_period")
    if train_period is not None and (
        not isinstance(train_period, list) or len(train_period) != 2
    ):
        result["errors"].append("data_cfgs.train_period must be [start, end]")

    param_range_file = training_cfgs.get("param_range_file")
    if model_name in MODEL_DICT:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result["resolved_param_range"] = resolve_model_param_config(
                    model_name,
                    param_range_file=param_range_file,
                    strict=param_range_file is not None,
                )
            result["warnings"].extend(str(item.message) for item in caught)
        except Exception as exc:
            result["errors"].append(f"Invalid param_range_file: {exc}")

    result["valid"] = not result["errors"]
    return result


def validate_and_show_config(
    config: Dict[str, Any], verbose: bool = True, model_type: str = "Model"
) -> bool:
    """
    Validate configuration and show summary

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration to validate
    verbose : bool
        Whether to show detailed output (kept for compatibility but ignored)
    model_type : str
        Type of model for display purposes

    Returns
    -------
    bool
        True if validation passed
    """
    validation = validate_config(config)
    if validation["valid"]:
        for warning in validation["warnings"]:
            print(f"Warning: {warning}")
        return True

    for error in validation["errors"]:
        print(f"Error: {error}")
    return False
