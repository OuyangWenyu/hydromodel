"""
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
import yaml
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union


def load_hydro_settings() -> Dict[str, Any]:
    """Load hydro_setting.yml from user's home directory"""
    try:
        setting_path = os.path.join(
            os.path.expanduser("~"), "hydro_setting.yml"
        )
        if os.path.exists(setting_path):
            with open(setting_path, "r", encoding="utf-8") as f:
                settings = yaml.safe_load(f)
            return settings or {}
        else:
            return {}
    except Exception as e:
        print(f"Warning: Could not load hydro_setting.yml: {e}")
        return {}


def get_default_data_path(
    data_source_type: str, hydro_settings: Dict[str, Any]
) -> str:
    """
    Get default data path based on data source type and hydro settings.

    Parameters
    ----------
    data_source_type : str
        Type of data source (camels, selfmadehydrodataset, floodevent, etc.)
    hydro_settings : Dict[str, Any]
        Hydro settings from hydro_setting.yml

    Returns
    -------
    str
        Default data path
    """
    local_data_path = hydro_settings.get("local_data_path", {})
    datasets_origin = local_data_path.get("datasets-origin")
    datasets_interim = local_data_path.get("datasets-interim")
    basins_origin = local_data_path.get("basins-origin")

    if data_source_type == "camels" and datasets_origin:
        return os.path.join(datasets_origin, "camels", "camels_us")
    elif data_source_type == "selfmadehydrodataset" and datasets_interim:
        return os.path.join(datasets_interim, "songliaorrevent")
    elif data_source_type == "floodevent" and datasets_interim:
        return os.path.join(datasets_interim, "songliaorrevent")
    elif basins_origin:
        return basins_origin
    else:
        return os.path.join(os.path.expanduser("~"), "hydro_data")


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
            "data_source_type": "camels_us",
            "data_source_path": None,  # Will be filled from hydro_setting.yml
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
            "param_range_file": None,
            "output_dir": "results",
            "experiment_name": None,  # Will be auto-generated
            "random_seed": 1234,
            "save_config": True,  # Save calibration config and param_range to output directory
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

    # Load hydro settings for default paths
    hydro_settings = load_hydro_settings()

    # Update data configuration
    if hasattr(args, "data_source_type") and args.data_source_type is not None:
        config["data_cfgs"]["data_source_type"] = args.data_source_type

    if hasattr(args, "data_source_path") and args.data_source_path is not None:
        config["data_cfgs"]["data_source_path"] = args.data_source_path
    elif hasattr(args, "data_path") and args.data_path is not None:
        config["data_cfgs"]["data_source_path"] = args.data_path
    elif config["data_cfgs"]["data_source_path"] is None:
        # Use default path from hydro settings
        data_type = config["data_cfgs"]["data_source_type"]
        config["data_cfgs"]["data_source_path"] = get_default_data_path(
            data_type, hydro_settings
        )

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
        config["model_cfgs"]["model_name"] = args.model_type
    elif hasattr(args, "model") and args.model is not None:
        config["model_cfgs"]["model_name"] = args.model

    # Update model parameters
    if hasattr(args, "source_type") and args.source_type is not None:
        config["model_cfgs"]["model_params"]["source_type"] = args.source_type

    if hasattr(args, "source_book") and args.source_book is not None:
        config["model_cfgs"]["model_params"]["source_book"] = args.source_book

    if hasattr(args, "kernel_size") and args.kernel_size is not None:
        config["model_cfgs"]["model_params"]["kernel_size"] = args.kernel_size

    # Update training configuration (if exists)
    if "training_cfgs" in config:
        if hasattr(args, "algorithm") and args.algorithm is not None:
            config["training_cfgs"]["algorithm_name"] = args.algorithm

        if hasattr(args, "obj_func") and args.obj_func is not None:
            config["training_cfgs"]["loss_config"]["obj_func"] = args.obj_func

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
            config["training_cfgs"]["algorithm_params"]["rep"] = args.rep

        if hasattr(args, "ngs") and args.ngs is not None:
            config["training_cfgs"]["algorithm_params"]["ngs"] = args.ngs

    # Handle model parameters
    if hasattr(args, "model_parameters") and args.model_parameters is not None:
        config["model_cfgs"]["parameters"] = args.model_parameters

    # Generate experiment name if not provided
    if (
        "training_cfgs" in config
        and config["training_cfgs"]["experiment_name"] is None
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = config["model_cfgs"]["model_name"]
        algorithm = config["training_cfgs"]["algorithm_name"]
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
    """
    Load simplified configuration file and convert to unified format.

    This function handles configuration files with simplified structure (data, model,
    training, evaluation sections) and converts them to the unified configuration
    format used throughout hydromodel.

    Supports all dataset types including:
    - Public datasets from hydrodataset (camels_us, camels_aus, etc.)
    - Custom datasets from hydrodatasource (selfmadehydrodataset, floodevent)

    Parameters
    ----------
    config_path : str, optional
        Path to simplified YAML configuration file
    simple_config : dict, optional
        Pre-loaded simplified configuration dictionary

    Returns
    -------
    dict
        Unified configuration dictionary with data_cfgs, model_cfgs, training_cfgs,
        and evaluation_cfgs sections

    Raises
    ------
    ValueError
        If neither config_path nor simple_config is provided, or if required
        configuration sections are missing

    Examples
    --------
    >>> # Load from file
    >>> config = load_simplified_config("configs/example_config.yaml")

    >>> # Load from dictionary
    >>> simple = {"data": {...}, "model": {...}, "training": {...}, "evaluation": {...}}
    >>> config = load_simplified_config(simple_config=simple)
    """
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            simple_config = yaml.safe_load(f)
    elif simple_config is None:
        raise ValueError(
            " Must provide config.path or simplic_config parameter "
        )

    # Validate required sections
    required_sections = ["data", "model", "training", "evaluation"]
    for section in required_sections:
        if section not in simple_config:
            raise ValueError(
                f"Configuration file is missing necessary parts: {section}"
            )

    data_cfg = simple_config["data"]
    model_cfg = simple_config["model"]
    training_cfg = simple_config["training"]
    eval_cfg = simple_config["evaluation"]

    # Convert to unified configuration format
    unified_config = {
        "data_cfgs": {
            "data_source_type": data_cfg["dataset"],
            "data_source_path": data_cfg.get("path"),
            "basin_ids": data_cfg["basin_ids"],
            "variables": data_cfg.get(
                "variables",
                [
                    "precipitation",
                    "potential_evapotranspiration",
                    "streamflow",
                ],
            ),
            "train_period": data_cfg["train_period"],
            "test_period": data_cfg["test_period"],
            "warmup_length": data_cfg.get("warmup_length", 365),
            "is_event_data": data_cfg.get("is_event_data", False),
        },
        "model_cfgs": {
            "model_name": model_cfg["name"],
            **model_cfg.get("params", {}),
        },
        "training_cfgs": {
            "algorithm_name": training_cfg["algorithm"],
            "algorithm_params": training_cfg.get(
                training_cfg["algorithm"], {}
            ),
            "loss_config": {
                "type": "time_series",
                "obj_func": training_cfg["loss"],
            },
            "output_dir": data_cfg.get("output_dir", "results"),
            "experiment_name": f"{model_cfg['name']}_{training_cfg['algorithm']}",
            "random_seed": training_cfg.get("random_seed", 1234),
            "save_config": training_cfg.get("save_config", True),
        },
        "evaluation_cfgs": {
            "metrics": eval_cfg["metrics"],
            "save_results": eval_cfg.get("save_results", True),
            "plot_results": eval_cfg.get("plot_results", True),
        },
    }

    # Add optional validation period
    if "valid_period" in data_cfg:
        unified_config["data_cfgs"]["valid_period"] = data_cfg["valid_period"]

    # Add dataset-specific parameters for custom datasets
    # These are critical for selfmadehydrodataset and floodevent
    if "dataset_name" in data_cfg:
        unified_config["data_cfgs"]["dataset_name"] = data_cfg["dataset_name"]

    if "time_unit" in data_cfg:
        unified_config["data_cfgs"]["time_unit"] = data_cfg["time_unit"]

    if "datasource_kwargs" in data_cfg:
        unified_config["data_cfgs"]["datasource_kwargs"] = data_cfg[
            "datasource_kwargs"
        ]

    return unified_config


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
        Configuration dictionary with data_cfgs, model_cfgs, training_cfgs sections

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
    # Basic validation - ensure required sections exist
    required_sections = ["data_cfgs", "model_cfgs", "training_cfgs"]
    for section in required_sections:
        if section not in config:
            print(f"Error: Missing required config section: {section}")
            return False

    return True
