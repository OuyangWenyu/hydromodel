#!/usr/bin/env python3
"""
Simple test script to verify configuration loading and data path handling
"""

import yaml
from pathlib import Path

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_config():
    """Test configuration loading"""
    config_path = "configs/camels_xaj_config.yaml"
    print(f"Loading config from: {config_path}")
    
    try:
        config = load_config(config_path)
        print("Config loaded successfully")
        
        # Check data configuration
        data_cfg = config['data_cfgs']
        print(f"Data source type: {data_cfg['data_source_type']}")
        print(f"Data source path: {data_cfg['data_source_path']}")
        print(f"Basin IDs: {data_cfg['basin_ids']}")
        
        # Check if path is None
        if data_cfg['data_source_path'] is None:
            print("ERROR: data_source_path is None!")
        else:
            print(f"Data path is set to: {data_cfg['data_source_path']}")
            
        return config
        
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def test_unified_data_loader(config):
    """Test UnifiedDataLoader with the config"""
    try:
        from hydromodel.datasets.unified_data_loader import UnifiedDataLoader
        print("\nTesting UnifiedDataLoader...")
        
        data_config = config['data_cfgs']
        print(f"Passing data_config: {data_config}")
        
        data_loader = UnifiedDataLoader(data_config)
        print("UnifiedDataLoader created successfully")
        
    except Exception as e:
        print(f"Error creating UnifiedDataLoader: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing configuration loading...")
    config = test_config()
    
    if config:
        test_unified_data_loader(config)