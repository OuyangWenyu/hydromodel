"""
Author: Wenyu Ouyang
Date: 2025-08-07
LastEditTime: 2025-08-07 17:28:15
LastEditors: Wenyu Ouyang
Description: Unified configuration system for hydromodel - consistent with torchhydro
FilePath: /hydromodel/hydromodel/configs/unified_config.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import yaml
import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


class UnifiedConfig:
    """
    Unified configuration class for hydromodel that mirrors torchhydro's config structure.

    Configuration is organized into four main sections:
    - data_cfgs: Data-related configurations (paths, basins, periods, etc.)
    - model_cfgs: Model-specific configurations (model type, parameters, etc.)
    - training_cfgs: Training/calibration algorithm configurations
    - evaluation_cfgs: Evaluation and loss function configurations
    """

    def __init__(
        self,
        config_file: Optional[str] = None,
        config_dict: Optional[Dict] = None,
    ):
        """
        Initialize configuration from file or dictionary.

        Parameters
        ----------
        config_file : str, optional
            Path to YAML or JSON configuration file
        config_dict : dict, optional
            Configuration dictionary
        """
        if config_file is not None:
            self.config = self._load_config_file(config_file)
        elif config_dict is not None:
            self.config = config_dict
        else:
            self.config = self._get_default_config()

        # Validate and set defaults
        self._validate_and_set_defaults()

    def _load_config_file(self, config_file: str) -> Dict:
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_file)

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_file}"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                return yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                return json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config file format: {config_path.suffix}"
                )

    def _get_default_config(self) -> Dict:
        """Get default configuration structure."""
        return {
            "data_cfgs": {},
            "model_cfgs": {},
            "training_cfgs": {},
            "evaluation_cfgs": {},
        }

    def _validate_and_set_defaults(self):
        """Validate configuration and set default values."""
        required_sections = [
            "data_cfgs",
            "model_cfgs",
            "training_cfgs",
            "evaluation_cfgs",
        ]

        for section in required_sections:
            if section not in self.config:
                self.config[section] = {}

        # Set section-specific defaults
        self._set_data_defaults()
        self._set_model_defaults()
        self._set_training_defaults()
        self._set_evaluation_defaults()

    def _set_data_defaults(self):
        """Set default values for data configuration."""
        data_defaults = {
            "data_type": "selfmade",
            "data_path": None,
            "dataset_name": "experiment",
            "basin_ids": [],
            "warmup_length": 365,
            "time_periods": {
                "overall": ["1990-10-01", "2000-09-30"],
                "calibration": ["1990-10-01", "1995-09-30"],
                "testing": ["1995-10-01", "2000-09-30"],
            },
            "variables": ["prcp", "PET", "streamflow"],
            "time_unit": ["1D"],
            "cross_validation": {"enabled": False, "folds": 1},
            "random_seed": 1234,
            # Event data specific configuration
            "rain_key": "rain",
            "net_rain_key": "P_eff",
            "obs_flow_key": "Q_obs_eff",
            # Additional datasource kwargs
            "datasource_kwargs": {},
            "read_kwargs": {},
        }

        for key, value in data_defaults.items():
            if key not in self.config["data_cfgs"]:
                self.config["data_cfgs"][key] = value

    def _set_model_defaults(self):
        """Set default values for model configuration."""
        # Set basic model defaults
        if "model_name" not in self.config["model_cfgs"]:
            self.config["model_cfgs"]["model_name"] = "xaj_mz"
            
        if "model_params" not in self.config["model_cfgs"]:
            self.config["model_cfgs"]["model_params"] = {}
            
        # Set model-specific parameter defaults based on model type
        model_name = self.config["model_cfgs"]["model_name"]
        self._set_model_specific_defaults(model_name)

    def _set_model_specific_defaults(self, model_name: str):
        """Set model-specific default parameters."""
        model_params = self.config["model_cfgs"]["model_params"]
        
        if model_name in ["xaj", "xaj_mz"]:
            # XAJ specific parameters
            xaj_defaults = {
                "source_type": "sources",
                "source_book": "HF", 
                "kernel_size": 15,
                "time_interval_hours": 24,
            }
            for key, value in xaj_defaults.items():
                if key not in model_params:
                    model_params[key] = value
                    
        elif model_name == "unit_hydrograph":
            # Unit Hydrograph specific parameters
            uh_defaults = {
                "n_uh": 24,
                "smoothing_factor": 0.1,
                "peak_violation_weight": 10000.0,
                "apply_peak_penalty": True,
                "net_rain_name": "P_eff",
                "obs_flow_name": "Q_obs_eff",
            }
            for key, value in uh_defaults.items():
                if key not in model_params:
                    model_params[key] = value
                    
        elif model_name == "categorized_unit_hydrograph":
            # Categorized UH specific parameters  
            cuh_defaults = {
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
            }
            for key, value in cuh_defaults.items():
                if key not in model_params:
                    model_params[key] = value
                    
        elif model_name == "dhf":
            # DHF model typically loads parameters from files, minimal defaults
            pass
            
        # No defaults for unknown model types - let them be specified via args or config

    def _set_training_defaults(self):
        """Set default values for training/calibration configuration."""
        # Set basic training defaults
        training_defaults = {
            "algorithm_name": "SCE_UA",
            "loss_config": {"type": "time_series", "obj_func": "RMSE"},
            "param_range_file": None,
            "output_dir": "results", 
            "experiment_name": "hydromodel_experiment",
        }
        
        for key, value in training_defaults.items():
            if key not in self.config["training_cfgs"]:
                self.config["training_cfgs"][key] = value
                
        # Set algorithm-specific parameter defaults
        if "algorithm_params" not in self.config["training_cfgs"]:
            self.config["training_cfgs"]["algorithm_params"] = {}
            
        algorithm_name = self.config["training_cfgs"]["algorithm_name"]
        self._set_algorithm_specific_defaults(algorithm_name)

    def _set_algorithm_specific_defaults(self, algorithm_name: str):
        """Set algorithm-specific default parameters."""
        algo_params = self.config["training_cfgs"]["algorithm_params"]
        
        if algorithm_name == "SCE_UA":
            # SCE-UA specific parameters
            sceua_defaults = {
                "random_seed": 1234,
                "rep": 1000,
                "ngs": 1000,
                "kstop": 50,
                "peps": 0.1,
                "pcento": 0.1,
            }
            for key, value in sceua_defaults.items():
                if key not in algo_params:
                    algo_params[key] = value
                    
        elif algorithm_name == "genetic_algorithm":
            # Genetic Algorithm specific parameters
            ga_defaults = {
                "random_seed": 1234,
                "pop_size": 80,
                "n_generations": 50,
                "cx_prob": 0.7,
                "mut_prob": 0.2,
                "save_freq": 5,
            }
            for key, value in ga_defaults.items():
                if key not in algo_params:
                    algo_params[key] = value
                    
        elif algorithm_name == "scipy_minimize":
            # SciPy optimization specific parameters
            scipy_defaults = {
                "random_seed": 1234,
                "method": "SLSQP",
                "max_iterations": 500,
            }
            for key, value in scipy_defaults.items():
                if key not in algo_params:
                    algo_params[key] = value

    def _set_evaluation_defaults(self):
        """Set default values for evaluation configuration."""
        evaluation_defaults = {
            "metrics": ["RMSE", "NSE", "KGE", "Bias"],
            "evaluation_period": "testing",
            "save_results": True,
            "plot_results": True,
            "export_format": ["json", "csv"],
        }

        for key, value in evaluation_defaults.items():
            if key not in self.config["evaluation_cfgs"]:
                self.config["evaluation_cfgs"][key] = value

    # Property accessors for each configuration section
    @property
    def data_cfgs(self) -> Dict:
        """Get data configuration."""
        return self.config["data_cfgs"]

    @property
    def model_cfgs(self) -> Dict:
        """Get model configuration."""
        return self.config["model_cfgs"]

    @property
    def training_cfgs(self) -> Dict:
        """Get training/calibration configuration."""
        return self.config["training_cfgs"]

    @property
    def evaluation_cfgs(self) -> Dict:
        """Get evaluation configuration."""
        return self.config["evaluation_cfgs"]

    # Convenient methods to get specific configurations
    def get_model_config(self) -> Dict:
        """Get model configuration in format expected by unified calibrate function."""
        model_cfg = self.model_cfgs

        config = {"name": model_cfg["model_name"]}

        # Add model-specific parameters
        if "model_params" in model_cfg:
            config.update(model_cfg["model_params"])

        return config

    def save_config(self, config_file: str):
        """Save current configuration to file."""
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                yaml.dump(
                    self.config,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                )
            elif config_path.suffix.lower() == ".json":
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(
                    f"Unsupported config file format: {config_path.suffix}"
                )

    def update_config(self, updates: Dict):
        """Update configuration with new values."""

        def deep_update(base_dict: Dict, update_dict: Dict):
            """Recursively update nested dictionaries."""
            for key, value in update_dict.items():
                if (
                    key in base_dict
                    and isinstance(base_dict[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value

        # Store old names to check if they changed
        old_model_name = self.config.get("model_cfgs", {}).get("model_name")
        old_algorithm_name = self.config.get("training_cfgs", {}).get("algorithm_name")
        
        deep_update(self.config, updates)
        
        # If model name changed, reapply model-specific defaults
        new_model_name = self.config.get("model_cfgs", {}).get("model_name")
        if old_model_name != new_model_name and new_model_name:
            # Clear existing model_params to avoid mixing
            self.config["model_cfgs"]["model_params"] = {}
            # Reapply model-specific defaults
            self._set_model_specific_defaults(new_model_name)
            # Reapply updates to model_params if they exist in updates
            if "model_cfgs" in updates and "model_params" in updates["model_cfgs"]:
                deep_update(self.config["model_cfgs"]["model_params"], updates["model_cfgs"]["model_params"])
        
        # If algorithm name changed, reapply algorithm-specific defaults
        new_algorithm_name = self.config.get("training_cfgs", {}).get("algorithm_name")
        if old_algorithm_name != new_algorithm_name and new_algorithm_name:
            # Clear existing algorithm_params to avoid mixing
            self.config["training_cfgs"]["algorithm_params"] = {}
            # Reapply algorithm-specific defaults
            self._set_algorithm_specific_defaults(new_algorithm_name)
            # Reapply updates to algorithm_params if they exist in updates
            if "training_cfgs" in updates and "algorithm_params" in updates["training_cfgs"]:
                deep_update(self.config["training_cfgs"]["algorithm_params"], updates["training_cfgs"]["algorithm_params"])

    def __str__(self) -> str:
        """String representation of configuration."""
        return yaml.dump(
            self.config, default_flow_style=False, allow_unicode=True
        )

    def __repr__(self) -> str:
        """Detailed representation of configuration."""
        return f"UnifiedConfig(\n{self.__str__()})"


def load_config(config_file: str) -> UnifiedConfig:
    """Convenient function to load configuration from file."""
    return UnifiedConfig(config_file=config_file)


def create_default_config(config_file: str = None) -> UnifiedConfig:
    """Create a default configuration, optionally saving to file."""
    config = UnifiedConfig()
    if config_file:
        config.save_config(config_file)
    return config
