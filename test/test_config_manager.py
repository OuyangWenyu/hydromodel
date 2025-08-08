"""
Test cases for ConfigManager and configuration handling

Consolidated tests for configuration loading, validation, and data path handling.
Replaces the standalone test_config.py from project root.
"""

import pytest
import yaml
import tempfile
import os
from pathlib import Path

from hydromodel.configs.config_manager import ConfigManager
from hydromodel.datasets.unified_data_loader import UnifiedDataLoader


class TestConfigManager:
    """Test ConfigManager functionality."""

    def test_load_config_from_file(self):
        """Test loading configuration from YAML file."""
        # Create a temporary config file
        config_data = {
            "data_cfgs": {
                "data_source_type": "camels",
                "data_source_path": "/test/path",
                "basin_ids": ["01013500"],
                "warmup_length": 365,
                "variables": ["prcp", "pet", "streamflow"]
            },
            "model_cfgs": {
                "model_name": "xaj_mz",
                "model_params": {
                    "source_type": "sources",
                    "source_book": "HF",
                    "kernel_size": 15
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            loaded_config = ConfigManager.load_config_from_file(config_path)
            
            assert loaded_config is not None
            assert "data_cfgs" in loaded_config
            assert "model_cfgs" in loaded_config
            assert loaded_config["data_cfgs"]["data_source_type"] == "camels"
            assert loaded_config["model_cfgs"]["model_name"] == "xaj_mz"
        finally:
            os.unlink(config_path)

    def test_save_config_to_file(self):
        """Test saving configuration to file."""
        config_data = {
            "data_cfgs": {
                "data_source_type": "selfmadehydrodataset",
                "basin_ids": ["basin_001"]
            },
            "model_cfgs": {
                "model_name": "unit_hydrograph"
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name

        try:
            ConfigManager.save_config_to_file(config_data, config_path)
            
            # Verify file was created and content is correct
            assert os.path.exists(config_path)
            
            with open(config_path, 'r') as f:
                loaded_data = yaml.safe_load(f)
            
            assert loaded_data["data_cfgs"]["data_source_type"] == "selfmadehydrodataset"
            assert loaded_data["model_cfgs"]["model_name"] == "unit_hydrograph"
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)

    def test_get_default_calibration_config(self):
        """Test default calibration configuration."""
        config = ConfigManager.get_default_calibration_config()
        
        assert "data_cfgs" in config
        assert "model_cfgs" in config
        assert "training_cfgs" in config
        assert "evaluation_cfgs" in config
        
        # Check data config
        data_cfg = config["data_cfgs"]
        assert "data_source_type" in data_cfg
        assert "basin_ids" in data_cfg
        assert "warmup_length" in data_cfg
        
        # Check model config
        model_cfg = config["model_cfgs"]
        assert "model_name" in model_cfg
        assert "model_params" in model_cfg
        
        # Check training config
        training_cfg = config["training_cfgs"]
        assert "algorithm_name" in training_cfg
        assert "algorithm_params" in training_cfg
        assert "loss_config" in training_cfg

    def test_get_unit_hydrograph_calibration_config(self):
        """Test unit hydrograph calibration configuration."""
        config = ConfigManager.get_unit_hydrograph_calibration_config()
        
        assert config["data_cfgs"]["data_source_type"] == "floodevent"
        assert config["model_cfgs"]["model_name"] == "unit_hydrograph"
        assert "n_uh" in config["model_cfgs"]["model_params"]
        assert "net_rain_name" in config["model_cfgs"]["model_params"]
        assert "obs_flow_name" in config["model_cfgs"]["model_params"]

    def test_get_categorized_uh_calibration_config(self):
        """Test categorized unit hydrograph calibration configuration."""
        config = ConfigManager.get_categorized_uh_calibration_config()
        
        assert config["model_cfgs"]["model_name"] == "categorized_unit_hydrograph"
        assert "category_weights" in config["model_cfgs"]["model_params"]
        assert "uh_lengths" in config["model_cfgs"]["model_params"]
        
        # Check category structure
        category_weights = config["model_cfgs"]["model_params"]["category_weights"]
        assert "small" in category_weights
        assert "medium" in category_weights
        assert "large" in category_weights

    def test_create_calibration_config(self):
        """Test creating calibration config with args."""
        # Mock args object
        class MockArgs:
            def __init__(self):
                self.model = "unit_hydrograph"
                self.data_source_type = "floodevent"
                self.basin_ids = ["test_basin"]
                self.algorithm = "scipy_minimize"
                self.output_dir = "test_results"

        args = MockArgs()
        config = ConfigManager.create_calibration_config(args=args)
        
        assert config["model_cfgs"]["model_name"] == "unit_hydrograph"
        assert config["data_cfgs"]["data_source_type"] == "floodevent"
        assert config["data_cfgs"]["basin_ids"] == ["test_basin"]
        assert config["training_cfgs"]["algorithm_name"] == "scipy_minimize"
        assert config["training_cfgs"]["output_dir"] == "test_results"

    def test_get_default_simulation_config(self):
        """Test default simulation configuration."""
        config = ConfigManager.get_default_simulation_config()
        
        assert "data_cfgs" in config
        assert "model_cfgs" in config
        assert "simulation_cfgs" in config
        
        # Check simulation-specific config
        sim_cfg = config["simulation_cfgs"]
        assert "output_variables" in sim_cfg
        assert "save_results" in sim_cfg
        assert "output_dir" in sim_cfg

    def test_merge_configs_functionality(self):
        """Test configuration merging functionality."""
        base_config = {
            "data_cfgs": {
                "data_source_type": "camels",
                "basin_ids": ["01013500"]
            },
            "model_cfgs": {
                "model_name": "xaj"
            }
        }
        
        update_config = {
            "data_cfgs": {
                "warmup_length": 365
            },
            "training_cfgs": {
                "algorithm_name": "SCE_UA"
            }
        }
        
        from hydromodel.configs.config_manager import merge_configs
        merged = merge_configs(base_config, update_config)
        
        # Check that original data is preserved
        assert merged["data_cfgs"]["data_source_type"] == "camels"
        assert merged["data_cfgs"]["basin_ids"] == ["01013500"]
        assert merged["model_cfgs"]["model_name"] == "xaj"
        
        # Check that new data is added
        assert merged["data_cfgs"]["warmup_length"] == 365
        assert merged["training_cfgs"]["algorithm_name"] == "SCE_UA"


class TestUnifiedDataLoaderIntegration:
    """Test UnifiedDataLoader with ConfigManager."""

    def test_data_loader_with_config(self):
        """Test UnifiedDataLoader integration with ConfigManager config."""
        config = ConfigManager.get_default_calibration_config()
        data_config = config['data_cfgs']
        
        # Modify for testing (avoid actual data loading)
        data_config['data_source_path'] = '/mock/path'
        
        # Test that UnifiedDataLoader can be created with config
        # (We won't actually load data to avoid dependencies)
        try:
            data_loader = UnifiedDataLoader(data_config)
            # If we get here, the config structure is compatible
            assert data_loader is not None
        except ImportError:
            # Skip if dependencies not available
            pytest.skip("UnifiedDataLoader dependencies not available")
        except Exception as e:
            # Other exceptions are acceptable for this test
            # We're mainly testing config structure compatibility
            assert "data_source_path" in str(e) or "not found" in str(e).lower()