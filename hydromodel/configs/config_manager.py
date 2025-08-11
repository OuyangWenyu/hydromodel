"""
Author: Wenyu Ouyang
Date: 2025-08-07
LastEditTime: 2025-08-08 18:54:26
LastEditors: Wenyu Ouyang
Description: Unified configuration management system for hydromodel
FilePath: \hydromodel\hydromodel\configs\config_manager.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import yaml
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from copy import deepcopy


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


def update_nested_dict(
    target_dict: Dict[str, Any], keys: List[str], value: Any
) -> None:
    """
    Update nested dictionary with a value using a list of keys.

    Parameters
    ----------
    target_dict : Dict[str, Any]
        Target dictionary to update
    keys : List[str]
        List of keys representing the path to the value
    value : Any
        Value to set at the specified path
    """
    if len(keys) == 1:
        target_dict[keys[0]] = value
    else:
        if keys[0] not in target_dict:
            target_dict[keys[0]] = {}
        update_nested_dict(target_dict[keys[0]], keys[1:], value)


def merge_configs(
    base_config: Dict[str, Any], update_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.

    Parameters
    ----------
    base_config : Dict[str, Any]
        Base configuration dictionary
    update_config : Dict[str, Any]
        Configuration updates to apply

    Returns
    -------
    Dict[str, Any]
        Merged configuration dictionary
    """
    result = deepcopy(base_config)

    for key, value in update_config.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


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
        return os.path.join(datasets_interim, "songliaorrevents")
    elif data_source_type == "floodevent" and datasets_interim:
        return os.path.join(datasets_interim, "songliaorrevent")
    elif basins_origin:
        return basins_origin
    else:
        return os.path.join(os.path.expanduser("~"), "hydro_data")


class ConfigManager:
    """
    Unified configuration manager for hydromodel.

    Provides default configurations and update mechanisms for both
    calibration and simulation workflows, following the torchhydro
    pattern of default + update.
    """

    @staticmethod
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
                "data_source_type": "selfmadehydrodataset",
                "data_source_path": None,  # Will be filled from hydro_setting.yml
                "basin_ids": ["basin_001"],
                "warmup_length": 365,
                "variables": ["prcp", "pet", "streamflow"],
                "time_range": None,
                "train_period": None,
                "valid_period": None,
                "test_period": None,
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
                    "obj_func": "NSE",
                },
                "param_range_file": None,
                "output_dir": "results",
                "experiment_name": None,  # Will be auto-generated
                "random_seed": 1234,
            },
            "evaluation_cfgs": {
                "metrics": ["NSE", "RMSE", "KGE", "PBIAS"],
                "save_results": True,
                "plot_results": True,
                "validation_split": 0.2,
                "bootstrap_samples": None,
            },
        }

    @staticmethod
    def get_unit_hydrograph_calibration_config() -> Dict[str, Any]:
        """
        Get default configuration for unit hydrograph model calibration.

        Returns
        -------
        Dict[str, Any]
            Default unit hydrograph calibration configuration
        """
        return {
            "data_cfgs": {
                "data_source_type": "floodevent",
                "data_source_path": None,  # Will be filled from hydro_setting.yml
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
                "algorithm_params": {
                    "method": "SLSQP",
                    "max_iterations": 500,
                },
                "loss_config": {
                    "type": "event_based",
                    "obj_func": "RMSE",
                },
                "param_range_file": None,
                "output_dir": "results",
                "experiment_name": None,  # Will be auto-generated
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

    @staticmethod
    def get_categorized_uh_calibration_config() -> Dict[str, Any]:
        """
        Get default configuration for categorized unit hydrograph model calibration.

        Returns
        -------
        Dict[str, Any]
            Default categorized unit hydrograph calibration configuration
        """
        return {
            "data_cfgs": {
                "data_source_type": "floodevent",
                "data_source_path": None,  # Will be filled from hydro_setting.yml
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
                "param_range_file": None,
                "output_dir": "results",
                "experiment_name": None,  # Will be auto-generated
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

    @staticmethod
    def get_default_simulation_config() -> Dict[str, Any]:
        """
        Get default configuration for model simulation.

        Returns
        -------
        Dict[str, Any]
            Default simulation configuration
        """
        return {
            "data_cfgs": {
                "data_source_type": "selfmadehydrodataset",
                "data_source_path": None,  # Will be filled from hydro_setting.yml
                "basin_ids": ["basin_001"],
                "warmup_length": 365,
                "variables": ["prcp", "pet", "streamflow"],
                "time_range": None,
            },
            "model_cfgs": {
                "model_name": "xaj_mz",
                "model_params": {
                    "source_type": "sources",
                    "source_book": "HF",
                    "kernel_size": 15,
                },
                "parameters": {
                    # Default XAJ parameters - user should override these
                    "K": 0.5,
                    "B": 0.3,
                    "IM": 0.01,
                    "UM": 20,
                    "LM": 80,
                    "DM": 120,
                    "C": 0.15,
                    "SM": 50,
                    "EX": 1.0,
                    "KI": 0.3,
                    "KG": 0.2,
                    "A": 0.8,
                    "THETA": 0.2,
                    "CI": 0.8,
                    "CG": 0.15,
                },
            },
            "simulation_cfgs": {
                "output_variables": ["streamflow"],
                "save_states": False,
                "calculate_components": False,
                "output_dir": "results/simulations",
                "experiment_name": None,  # Will be auto-generated
                "save_results": True,
                "plot_results": False,
            },
        }

    @staticmethod
    def get_model_default_parameters(model_name: str) -> Dict[str, Any]:
        """
        Get default parameters for specific model types.

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        Dict[str, Any]
            Default parameters for the model
        """
        model_defaults = {
            "xaj": {
                "K": 0.5,
                "B": 0.3,
                "IM": 0.01,
                "UM": 20,
                "LM": 80,
                "DM": 120,
                "C": 0.15,
                "SM": 50,
                "EX": 1.0,
                "KI": 0.3,
                "KG": 0.2,
                "A": 0.8,
                "THETA": 0.2,
                "CI": 0.8,
                "CG": 0.15,
            },
            "xaj_mz": {
                "K": 0.5,
                "B": 0.3,
                "IM": 0.01,
                "UM": 20,
                "LM": 80,
                "DM": 120,
                "C": 0.15,
                "SM": 50,
                "EX": 1.0,
                "KI": 0.3,
                "KG": 0.2,
                "A": 0.8,
                "THETA": 0.2,
                "CI": 0.8,
                "CG": 0.15,
            },
            "unit_hydrograph": {
                "uh_values": [
                    0.01,
                    0.05,
                    0.12,
                    0.18,
                    0.22,
                    0.20,
                    0.15,
                    0.10,
                    0.08,
                    0.06,
                    0.05,
                    0.04,
                    0.03,
                    0.02,
                    0.02,
                    0.01,
                    0.01,
                    0.01,
                    0.00,
                    0.00,
                    0.00,
                    0.00,
                    0.00,
                    0.00,
                ]
            },
            "categorized_unit_hydrograph": {
                "uh_categories": {
                    "small": [0.1, 0.3, 0.4, 0.15, 0.03, 0.01, 0.005, 0.005],
                    "medium": [
                        0.02,
                        0.08,
                        0.15,
                        0.20,
                        0.22,
                        0.18,
                        0.10,
                        0.06,
                        0.04,
                        0.03,
                        0.02,
                        0.01,
                        0.01,
                        0.005,
                        0.003,
                        0.002,
                    ],
                    "large": [
                        0.01,
                        0.03,
                        0.08,
                        0.12,
                        0.16,
                        0.18,
                        0.15,
                        0.12,
                        0.08,
                        0.06,
                        0.05,
                        0.04,
                        0.03,
                        0.02,
                        0.015,
                        0.01,
                        0.008,
                        0.006,
                        0.004,
                        0.003,
                        0.002,
                        0.001,
                        0.001,
                        0.001,
                    ],
                },
                "thresholds": {"small_medium": 10.0, "medium_large": 25.0},
            },
            "gr4j": {"X1": 350.0, "X2": 0.0, "X3": 90.0, "X4": 1.2},
            "gr6j": {
                "X1": 350.0,
                "X2": 0.0,
                "X3": 90.0,
                "X4": 1.2,
                "X5": 0.5,
                "X6": 3.0,
            },
        }

        return model_defaults.get(model_name, {})

    @staticmethod
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
        if (
            hasattr(args, "data_source_type")
            and args.data_source_type is not None
        ):
            config["data_cfgs"]["data_source_type"] = args.data_source_type

        if (
            hasattr(args, "data_source_path")
            and args.data_source_path is not None
        ):
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

        # Update model configuration
        if hasattr(args, "model_type") and args.model_type is not None:
            config["model_cfgs"]["model_name"] = args.model_type
        elif hasattr(args, "model") and args.model is not None:
            config["model_cfgs"]["model_name"] = args.model

        # Update model parameters
        if hasattr(args, "source_type") and args.source_type is not None:
            config["model_cfgs"]["model_params"][
                "source_type"
            ] = args.source_type

        if hasattr(args, "source_book") and args.source_book is not None:
            config["model_cfgs"]["model_params"][
                "source_book"
            ] = args.source_book

        if hasattr(args, "kernel_size") and args.kernel_size is not None:
            config["model_cfgs"]["model_params"][
                "kernel_size"
            ] = args.kernel_size

        # Update training configuration (if exists)
        if "training_cfgs" in config:
            if hasattr(args, "algorithm") and args.algorithm is not None:
                config["training_cfgs"]["algorithm_name"] = args.algorithm

            if hasattr(args, "obj_func") and args.obj_func is not None:
                config["training_cfgs"]["loss_config"][
                    "obj_func"
                ] = args.obj_func

            if hasattr(args, "output_dir") and args.output_dir is not None:
                config["training_cfgs"]["output_dir"] = args.output_dir

            if (
                hasattr(args, "experiment_name")
                and args.experiment_name is not None
            ):
                config["training_cfgs"][
                    "experiment_name"
                ] = args.experiment_name

            if hasattr(args, "random_seed") and args.random_seed is not None:
                config["training_cfgs"]["random_seed"] = args.random_seed

            # Algorithm-specific parameters
            if hasattr(args, "rep") and args.rep is not None:
                config["training_cfgs"]["algorithm_params"]["rep"] = args.rep

            if hasattr(args, "ngs") and args.ngs is not None:
                config["training_cfgs"]["algorithm_params"]["ngs"] = args.ngs

        # Update simulation configuration (if exists)
        if "simulation_cfgs" in config:
            if hasattr(args, "output_dir") and args.output_dir is not None:
                config["simulation_cfgs"]["output_dir"] = args.output_dir

            if (
                hasattr(args, "experiment_name")
                and args.experiment_name is not None
            ):
                config["simulation_cfgs"][
                    "experiment_name"
                ] = args.experiment_name

            if hasattr(args, "save_results") and args.save_results is not None:
                config["simulation_cfgs"]["save_results"] = args.save_results

            if hasattr(args, "plot_results") and args.plot_results is not None:
                config["simulation_cfgs"]["plot_results"] = args.plot_results

        # Handle simulation-specific parameters
        if (
            hasattr(args, "model_parameters")
            and args.model_parameters is not None
        ):
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

        if (
            "simulation_cfgs" in config
            and config["simulation_cfgs"]["experiment_name"] is None
        ):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = config["model_cfgs"]["model_name"]
            config["simulation_cfgs"][
                "experiment_name"
            ] = f"{model_name}_simulation_{timestamp}"

        return config

    @staticmethod
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
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.endswith(".json"):
                config = json.load(f)
            else:
                config = yaml.safe_load(f)

        return config or {}

    @staticmethod
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

    @staticmethod
    def create_calibration_config(
        config_file: Optional[str] = None,
        args: Optional[argparse.Namespace] = None,
        updates: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create calibration configuration using default + updates pattern.

        Parameters
        ----------
        config_file : Optional[str]
            Path to configuration file to load
        args : Optional[argparse.Namespace]
            Command line arguments to apply
        updates : Optional[Dict[str, Any]]
            Additional updates to apply

        Returns
        -------
        Dict[str, Any]
            Final calibration configuration
        """
        # Determine model type from args to select appropriate defaults
        model_type = None
        if args is not None:
            model_type = getattr(args, "model_type", None) or getattr(
                args, "model", None
            )

        # Start with appropriate defaults based on model type
        if model_type == "unit_hydrograph":
            config = ConfigManager.get_unit_hydrograph_calibration_config()
        elif model_type == "categorized_unit_hydrograph":
            config = ConfigManager.get_categorized_uh_calibration_config()
        else:
            config = ConfigManager.get_default_calibration_config()

        # Apply file-based config
        if config_file and os.path.exists(config_file):
            file_config = ConfigManager.load_config_from_file(config_file)
            config = merge_configs(config, file_config)

        # Apply command line arguments
        if args is not None:
            config = ConfigManager.update_config_from_args(config, args)

        # Apply additional updates
        if updates is not None:
            config = merge_configs(config, updates)

        return config

    @staticmethod
    def create_simulation_config(
        config_file: Optional[str] = None,
        args: Optional[argparse.Namespace] = None,
        updates: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create simulation configuration using default + updates pattern.

        Parameters
        ----------
        config_file : Optional[str]
            Path to configuration file to load
        args : Optional[argparse.Namespace]
            Command line arguments to apply
        updates : Optional[Dict[str, Any]]
            Additional updates to apply

        Returns
        -------
        Dict[str, Any]
            Final simulation configuration
        """
        # Start with defaults
        config = ConfigManager.get_default_simulation_config()

        # Apply file-based config
        if config_file and os.path.exists(config_file):
            file_config = ConfigManager.load_config_from_file(config_file)
            config = merge_configs(config, file_config)

        # Apply command line arguments
        if args is not None:
            config = ConfigManager.update_config_from_args(config, args)

        # Apply additional updates
        if updates is not None:
            config = merge_configs(config, updates)

        return config
