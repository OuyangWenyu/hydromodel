"""
Simple example demonstrating the unified calibrate() function.

This example shows both usage patterns:
1. Using UnifiedConfig object
2. Using individual config dictionaries

Author: Wenyu Ouyang
Date: 2025-01-22
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydromodel.configs.unified_config import UnifiedConfig
from hydromodel.trainers.unified_calibrate import calibrate


def example_with_unified_config():
    """Example using UnifiedConfig object."""
    print("=== Example 1: Using UnifiedConfig ===")
    
    # Create configuration programmatically
    config_dict = {
        "data_cfgs": {
            "data_type": "camels",
            "data_path": "path/to/camels/data",
            "basin_ids": ["01013500"],
            "variables": ["prcp", "PET", "streamflow"],
            "warmup_length": 365,
            "time_periods": {
                "calibration": ["2010-01-01", "2015-12-31"],
                "testing": ["2016-01-01", "2020-12-31"],
            }
        },
        "model_cfgs": {
            "model_name": "xaj",
            "model_params": {
                "source_type": "sources",
                "time_interval_hours": 24,
            }
        },
        "training_cfgs": {
            "algorithm_name": "SCE_UA",
            "algorithm_params": {
                "rep": 100,
                "ngs": 100,
            },
            "loss_config": {
                "type": "time_series",
                "obj_func": "RMSE"
            },
            "param_range_file": "param_range.yaml",
            "output_dir": "results/unified_config_example",
            "experiment_name": "xaj_example"
        },
        "evaluation_cfgs": {
            "metrics": ["RMSE", "NSE", "KGE"],
            "evaluation_period": "testing"
        }
    }
    
    config = UnifiedConfig(config_dict=config_dict)
    
    print("Config created:")
    print(f"Data type: {config.data_cfgs['data_type']}")
    print(f"Model: {config.model_cfgs['model_name']}")
    print(f"Algorithm: {config.training_cfgs['algorithm_name']}")
    
    try:
        # Method 1: Pass config object
        results = calibrate(config)
        print("Calibration completed successfully!")
        return results
    except Exception as e:
        print(f"Calibration failed (expected if data not available): {e}")
        return None


def example_with_config_dict():
    """Example using config dictionary directly."""
    print("\n=== Example 2: Using Config Dictionary ===")
    
    # Create a complete config dictionary with the expected structure
    config_dict = {
        "data_cfgs": {
            "data_type": "floodevent",
            "data_path": "path/to/flood/events",
            "basin_ids": ["basin1"],
            "variables": ["P_eff", "Q_obs_eff"],
            "warmup_length": 0,
            "time_periods": {
                "calibration": ["2020-01-01", "2022-12-31"],
            },
            "net_rain_key": "P_eff",
            "obs_flow_key": "Q_obs_eff"
        },
        "model_cfgs": {
            "model_name": "xaj",
            "model_params": {
                "source_type": "sources",
                "time_interval_hours": 3
            }
        },
        "training_cfgs": {
            "algorithm_name": "SCE_UA",
            "algorithm_params": {
                "rep": 50,
                "ngs": 50,
            },
            "loss_config": {
                "type": "time_series",
                "obj_func": "NSE"
            },
            "param_range_file": "param_range.yaml",
            "output_dir": "results/config_dict_example",
            "experiment_name": "flood_event_xaj"
        },
        "evaluation_cfgs": {
            "metrics": ["RMSE", "NSE", "KGE"],
            "evaluation_period": "testing"
        }
    }
    
    print("Config dictionary created:")
    print(f"Data type: {config_dict['data_cfgs']['data_type']}")
    print(f"Model: {config_dict['model_cfgs']['model_name']}")
    print(f"Algorithm: {config_dict['training_cfgs']['algorithm_name']}")
    print(f"Objective: {config_dict['training_cfgs']['loss_config']['obj_func']}")
    
    try:
        # Method 2: Pass config dictionary directly
        results = calibrate(config_dict)
        print("Calibration completed successfully!")
        return results
    except Exception as e:
        print(f"Calibration failed (expected if data not available): {e}")
        return None


def demonstrate_error_handling():
    """Demonstrate error handling for invalid configurations."""
    print("\n=== Example 3: Error Handling ===")
    
    # Try calling with incomplete config dictionary
    try:
        incomplete_config = {"data_cfgs": {"data_type": "camels"}}
        results = calibrate(incomplete_config)
        print("This should not print - error expected")
    except ValueError as e:
        print(f"✓ Expected error for incomplete config: {e}")
    
    # Try calling with invalid config type
    try:
        invalid_config = "invalid_config_string"
        results = calibrate(invalid_config)
        print("This should not print - error expected")
    except ValueError as e:
        print(f"✓ Expected error for invalid config type: {e}")
    
    print("Error handling demonstration completed.")


if __name__ == "__main__":
    print("Unified Calibrate Function Examples")
    print("===================================")
    print()
    
    print("The calibrate() function now accepts a single config parameter:")
    print("1. calibrate(unified_config_object)")
    print("2. calibrate(config_dictionary)")
    print()
    
    # Run examples
    example_with_unified_config()
    example_with_config_dict()
    demonstrate_error_handling()
    
    print("\n" + "="*50)
    print("Key Benefits of Single Parameter Design:")
    print("- Extremely simple: calibrate(config)")
    print("- No parameter confusion: only one config argument")
    print("- Flexible: supports UnifiedConfig objects or dictionaries")
    print("- Clean and intuitive: all configuration in one place")
    print("="*50)
