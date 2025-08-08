"""
Test cases for unified calibration interface.

This module consolidates tests for the unified calibration system,
including unit hydrograph models, genetic algorithm integration,
and interface compatibility.
"""

import pytest
import numpy as np
import os
import shutil
import tempfile
from pathlib import Path

# Add local dependency paths for testing
import sys
from pathlib import Path
current_dir = Path(__file__).parent
hydromodel_root = current_dir.parent
workspace_root = hydromodel_root.parent

# Add local packages to path
for local_pkg in ['hydroutils', 'hydrodatasource', 'hydrodataset']:
    local_path = workspace_root / local_pkg
    if local_path.exists():
        sys.path.insert(0, str(local_path))

from hydromodel.trainers.unified_calibrate import (
    calibrate,
    UnifiedModelSetup,  # Updated class name
    DEAP_AVAILABLE,
)
from hydromodel.core.unified_simulate import UnifiedSimulator


class TestBasicFunctionality:
    """Test basic data processing and parameter handling."""

    def test_warmup_processing_logic(self):
        """Test warmup period removal logic."""
        warmup_length = 8
        event_length = 20
        total_length = warmup_length + event_length

        # Create mock data
        mock_data = np.random.rand(total_length) * 10

        # Test warmup period removal logic
        processed_data = mock_data[warmup_length:]

        assert len(processed_data) == event_length

    def test_parameter_generation(self):
        """Test parameter name generation and bounds."""
        n_uh = 12
        param_names = [f"uh_{i+1}" for i in range(n_uh)]
        assert len(param_names) == n_uh

        # Test parameter bounds
        bounds = [(0.0, 1.0) for _ in range(n_uh)]
        assert len(bounds) == n_uh

    def test_parameter_normalization(self):
        """Test parameter normalization for unit hydrograph."""
        n_uh = 12
        test_params = np.random.rand(n_uh)
        test_params_normalized = test_params / test_params.sum()
        param_sum = test_params_normalized.sum()
        assert abs(param_sum - 1.0) < 1e-6


class TestUnifiedModelSetup:
    """Test unified model setup and processing (replaces UnitHydrographSetup)."""

    @pytest.fixture
    def mock_event_data(self):
        """Create mock event data for testing."""
        warmup_length = 8
        event_length = 20
        total_length = warmup_length + event_length

        return [
            {
                "P_eff": np.random.rand(total_length) * 10,
                "Q_obs_eff": np.random.rand(total_length) * 5,
                "filepath": "mock_event_1.csv",
            },
            {
                "P_eff": np.random.rand(total_length) * 8,
                "Q_obs_eff": np.random.rand(total_length) * 4,
                "filepath": "mock_event_2.csv",
            },
        ]

    @pytest.fixture
    def model_config(self):
        """Standard model configuration for testing."""
        return {
            "name": "unit_hydrograph",
            "n_uh": 12,
            "smoothing_factor": 0.1,
            "peak_violation_weight": 1000.0,
            "apply_peak_penalty": True,
        }

    @pytest.fixture
    def loss_config(self):
        """Standard loss configuration for testing."""
        return {"type": "time_series", "obj_func": "RMSE"}

    def test_setup_creation(self, mock_event_data, model_config, loss_config):
        """Test UnitHydrographSetup creation and parameter handling."""
        warmup_length = 8

        # Create data and model configuration
        data_config = {
            "data_source_type": "mock_event",
            "warmup_length": warmup_length
        }
        
        setup = UnifiedModelSetup(
            data_config=data_config,
            model_config=model_config,
            loss_config=loss_config,
        )

        # Verify parameter names
        param_names = setup.get_parameter_names()
        assert len(param_names) == 12

        # Verify parameter boundaries
        bounds = setup.get_parameter_bounds()
        assert len(bounds) == 12

        # Verify warmup period processing
        processed_data = setup.processed_event_data
        assert len(processed_data) == 2

        # Check warmup period removal
        for i, event in enumerate(processed_data):
            original_length = len(mock_event_data[i]["P_eff"])
            processed_length = len(event["P_eff"])
            expected_length = original_length - warmup_length
            assert processed_length == expected_length

    def test_objective_function(
        self, mock_event_data, model_config, loss_config
    ):
        """Test objective function calculation."""
        warmup_length = 8

        # Create data and model configuration
        data_config = {
            "data_source_type": "mock_event",
            "warmup_length": warmup_length
        }
        
        setup = UnifiedModelSetup(
            data_config=data_config,
            model_config=model_config,
            loss_config=loss_config,
        )

        # Test objective function calculation
        test_params = np.random.rand(12)
        test_params = test_params / test_params.sum()  # Normalize

        obj_value = setup.calculate_objective(test_params)
        assert isinstance(obj_value, (int, float))
        assert obj_value >= 0  # RMSE should be non-negative


class TestUnifiedInterface:
    """Test unified calibration interface."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def simple_mock_data(self):
        """Create simple mock data for interface testing."""
        warmup_length = 5
        return [
            {
                "P_eff": np.array(
                    [
                        0.1,
                        0.2,
                        0.3,
                        0.4,
                        0.5,  # warmup
                        1.0,
                        2.5,
                        1.8,
                        0.8,
                        0.3,
                        0.1,
                        0.0,
                        0.0,
                        0.0,
                        0.0,  # event
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
                "Q_obs_eff": np.array(
                    [
                        0.05,
                        0.1,
                        0.15,
                        0.2,
                        0.25,  # warmup
                        0.5,
                        1.2,
                        2.0,
                        1.5,
                        1.0,
                        0.6,
                        0.3,
                        0.1,
                        0.05,
                        0.02,  # event
                        0.01,
                        0.01,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
                "filepath": "test_event_1.csv",
            }
        ]

    def test_calibrate_interface(self, simple_mock_data, temp_output_dir):
        """Test unified calibrate function."""
        model_config = {
            "name": "unit_hydrograph",
            "n_uh": 8,
            "smoothing_factor": 0.05,
            "peak_violation_weight": 5000.0,
            "apply_peak_penalty": True,
        }

        # Test scipy algorithm
        algorithm_config = {
            "name": "scipy_minimize",
            "method": "SLSQP",
            "max_iterations": 100,
        }

        loss_config = {"type": "time_series", "obj_func": "RMSE"}

        results = calibrate(
            data=simple_mock_data,
            model_config=model_config,
            algorithm_config=algorithm_config,
            loss_config=loss_config,
            output_dir=temp_output_dir,
            warmup_length=5,
        )

        # Verify results structure
        assert "best_params" in results
        assert "objective_value" in results
        assert "convergence" in results

        # Verify parameters
        if results["best_params"]:
            uh_params = results["best_params"]["unit_hydrograph"]
            param_sum = sum(uh_params.values())
            assert abs(param_sum - 1.0) < 0.1  # Allow some tolerance


class TestBackwardCompatibility:
    """Test backward compatibility with existing interfaces."""

    def test_import_compatibility(self):
        """Test that all expected imports work."""
        from hydromodel.trainers import (
            calibrate_by_sceua,
            SpotSetup,
            calibrate,
            ModelSetupBase,
            UnitHydrographSetup,
            TraditionalModelSetup,
        )

        # Verify functions are callable
        assert callable(calibrate_by_sceua)
        assert callable(calibrate)

        # Verify classes can be instantiated (we'll test with mock data) 
        from hydromodel.trainers.unified_calibrate import ModelSetupBase
        assert issubclass(UnifiedModelSetup, ModelSetupBase)


@pytest.mark.skipif(not DEAP_AVAILABLE, reason="DEAP not available")
class TestGeneticAlgorithmIntegration:
    """Test genetic algorithm integration (requires DEAP)."""

    def test_ga_availability(self):
        """Test if GA functionality is properly available."""
        assert DEAP_AVAILABLE

        # Test DEAP imports
        from deap import base, creator, tools
        from hydromodel.trainers.unified_calibrate import _calibrate_with_ga

        assert callable(_calibrate_with_ga)

    def test_ga_unit_hydrograph(self):
        """Test GA with unit hydrograph model."""
        from hydromodel.trainers.unified_calibrate import (
            UnifiedModelSetup,
            _calibrate_with_ga,
        )

        # Create mock data
        warmup_length = 5
        mock_event_data = [
            {
                "P_eff": np.array(
                    [0.1, 0.2, 0.3, 0.4, 0.5]
                    + [1.0, 2.5, 1.8, 0.8, 0.3, 0.1]
                    + [0.0] * 9
                ),
                "Q_obs_eff": np.array(
                    [0.05, 0.1, 0.15, 0.2, 0.25]
                    + [0.5, 1.2, 2.0, 1.5, 1.0, 0.6]
                    + [0.1] * 9
                ),
                "filepath": "test_event_1.csv",
            }
        ]

        model_config = {
            "name": "unit_hydrograph",
            "n_uh": 8,
            "smoothing_factor": 0.05,
            "peak_violation_weight": 5000.0,
            "apply_peak_penalty": True,
        }

        algorithm_config = {
            "name": "genetic_algorithm",
            "random_seed": 42,
            "pop_size": 20,
            "n_generations": 3,  # Small for testing
            "cx_prob": 0.7,
            "mut_prob": 0.3,
            "save_freq": 2,
        }

        loss_config = {"type": "time_series", "obj_func": "RMSE"}

        # Create data configuration
        data_config = {
            "data_source_type": "mock_event",
            "warmup_length": warmup_length
        }
        
        # Create model setup
        model_setup = UnifiedModelSetup(
            data_config=data_config,
            model_config=model_config,
            loss_config=loss_config,
        )

        # Test GA calibration with temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            best_params, objective_value, ga_results = _calibrate_with_ga(
                model_setup, algorithm_config, temp_dir
            )

            assert isinstance(best_params, dict)
            assert "unit_hydrograph" in best_params
            assert isinstance(objective_value, (int, float))
            assert objective_value >= 0

            # Verify parameter normalization
            param_values = list(best_params["unit_hydrograph"].values())
            param_sum = sum(param_values)
            assert abs(param_sum - 1.0) < 0.1  # Allow some tolerance

    def test_ga_unified_interface(self):
        """Test GA through unified calibrate interface."""
        # Mock data
        mock_event_data = [
            {
                "P_eff": np.array(
                    [0.1, 0.2, 0.3] + [1.5, 1.0, 0.5, 0.2, 0.1] + [0.0] * 5
                ),
                "Q_obs_eff": np.array(
                    [0.05, 0.1, 0.15] + [0.8, 1.5, 1.0, 0.6, 0.3] + [0.1] * 5
                ),
                "filepath": "test_event_unified.csv",
            }
        ]

        model_config = {
            "name": "unit_hydrograph",
            "n_uh": 6,
            "smoothing_factor": 0.1,
            "peak_violation_weight": 1000.0,
        }

        algorithm_config = {
            "name": "genetic_algorithm",
            "random_seed": 123,
            "pop_size": 10,  # Small for testing
            "n_generations": 2,  # Few generations for testing
            "cx_prob": 0.6,
            "mut_prob": 0.2,
            "save_freq": 1,
        }

        loss_config = {"type": "time_series", "obj_func": "RMSE"}

        with tempfile.TemporaryDirectory() as temp_dir:
            results = calibrate(
                data=mock_event_data,
                model_config=model_config,
                algorithm_config=algorithm_config,
                loss_config=loss_config,
                output_dir=temp_dir,
                warmup_length=3,
            )

            assert "convergence" in results
            assert "objective_value" in results
            assert isinstance(results["objective_value"], (int, float))


class TestConfigurationStructure:
    """Test configuration structure and interface design."""

    def test_model_config_structure(self):
        """Test model configuration structure."""
        model_config = {
            "name": "unit_hydrograph",
            "n_uh": 24,
            "smoothing_factor": 0.1,
            "peak_violation_weight": 10000.0,
            "apply_peak_penalty": True,
            "net_rain_name": "P_eff",
            "obs_flow_name": "Q_obs_eff",
        }

        # Verify required fields
        assert "name" in model_config
        assert model_config["name"] == "unit_hydrograph"
        assert "n_uh" in model_config
        assert isinstance(model_config["n_uh"], int)

    def test_algorithm_config_structure(self):
        """Test algorithm configuration structure."""
        scipy_config = {
            "name": "scipy_minimize",
            "method": "SLSQP",
            "max_iterations": 500,
        }

        ga_config = {
            "name": "genetic_algorithm",
            "random_seed": 42,
            "pop_size": 80,
            "n_generations": 50,
            "cx_prob": 0.7,
            "mut_prob": 0.2,
            "save_freq": 5,
        }

        # Verify required fields
        for config in [scipy_config, ga_config]:
            assert "name" in config
            assert config["name"] in [
                "scipy_minimize",
                "genetic_algorithm",
                "SCE_UA",
            ]

    def test_interface_parameters(self):
        """Test unified interface parameter requirements."""
        interface_params = {
            "all_event_data": "mock_data",
            "model_config": {"name": "unit_hydrograph"},
            "algorithm_config": {"name": "scipy_minimize"},
            "output_dir": "test_output",
            "warmup_length": 8,
        }

        required_params = [
            "all_event_data",
            "model_config",
            "algorithm_config",
            "output_dir",
        ]
        for param in required_params:
            assert param in interface_params
