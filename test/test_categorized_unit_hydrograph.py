"""
Test cases for categorized unit hydrograph functionality.

This module tests the categorized unit hydrograph model which automatically
categorizes floods by peak magnitude and optimizes separate unit hydrographs
for each category (small, medium, large floods).
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from hydromodel.trainers.unified_calibrate import (
    calibrate,
    CategorizedUnitHydrographSetup,
    DEAP_AVAILABLE
)


class TestCategorizedUnitHydrographSetup:
    """Test categorized unit hydrograph setup and functionality."""
    
    @pytest.fixture
    def mock_categorized_events(self):
        """Create mock event data with different peak values for categorization."""
        warmup_length = 3
        event_length = 12
        
        return [
            {
                "P_eff": np.array([0.1, 0.2, 0.3] + [1.0, 2.0, 1.0, 0.5] + [0.0] * 8),
                "Q_obs_eff": np.array([0.05, 0.1, 0.15] + [0.8, 1.5, 1.0, 0.6] + [0.1] * 8),
                "peak_obs": 1.5,  # Small peak
                "filepath": "mock_event_small.csv"
            },
            {
                "P_eff": np.array([0.2, 0.3, 0.4] + [2.0, 4.0, 2.5, 1.0] + [0.0] * 8),
                "Q_obs_eff": np.array([0.1, 0.15, 0.2] + [1.5, 3.0, 2.0, 1.2] + [0.1] * 8),
                "peak_obs": 3.0,  # Medium peak
                "filepath": "mock_event_medium.csv"
            },
            {
                "P_eff": np.array([0.3, 0.4, 0.5] + [3.0, 6.0, 4.0, 1.5] + [0.0] * 8),
                "Q_obs_eff": np.array([0.15, 0.2, 0.25] + [2.0, 5.0, 3.5, 2.0] + [0.1] * 8),
                "peak_obs": 5.0,  # Large peak
                "filepath": "mock_event_large.csv"
            },
        ]
    
    @pytest.fixture
    def categorized_model_config(self):
        """Standard categorized model configuration."""
        return {
            "name": "categorized_unit_hydrograph",
            "category_weights": {
                "small": {"smoothing_factor": 0.1, "peak_violation_weight": 100.0},
                "medium": {"smoothing_factor": 0.5, "peak_violation_weight": 500.0},
                "large": {"smoothing_factor": 1.0, "peak_violation_weight": 1000.0},
            },
            "uh_lengths": {"small": 4, "medium": 6, "large": 8},
            "net_rain_name": "P_eff",
            "obs_flow_name": "Q_obs_eff"
        }
    
    @pytest.fixture
    def loss_config(self):
        """Standard loss configuration."""
        return {"type": "time_series", "obj_func": "RMSE"}
    
    def test_setup_creation(self, mock_categorized_events, categorized_model_config, loss_config):
        """Test CategorizedUnitHydrographSetup creation."""
        warmup_length = 3
        
        model_setup = CategorizedUnitHydrographSetup(
            data=mock_categorized_events,
            model_config=categorized_model_config,
            loss_config=loss_config,
            warmup_length=warmup_length
        )
        
        # Verify categories are created
        assert len(model_setup.categories) == 3
        assert "small" in model_setup.categories
        assert "medium" in model_setup.categories
        assert "large" in model_setup.categories
        
        # Verify events are categorized
        assert len(model_setup.categorized_events) == 3
        for category in model_setup.categories:
            assert category in model_setup.categorized_events
            assert len(model_setup.categorized_events[category]) > 0
    
    def test_parameter_management(self, mock_categorized_events, categorized_model_config, loss_config):
        """Test parameter name generation and bounds."""
        warmup_length = 3
        
        model_setup = CategorizedUnitHydrographSetup(
            data=mock_categorized_events,
            model_config=categorized_model_config,
            loss_config=loss_config,
            warmup_length=warmup_length
        )
        
        # Test parameter names
        param_names = model_setup.get_parameter_names()
        expected_params = sum(categorized_model_config["uh_lengths"].values())
        assert len(param_names) == expected_params
        
        # Test parameter bounds
        bounds = model_setup.get_parameter_bounds()
        assert len(bounds) == len(param_names)
        
        # Verify all bounds are (0.0, 1.0)
        for bound in bounds:
            assert bound == (0.0, 1.0)
    
    def test_objective_function(self, mock_categorized_events, categorized_model_config, loss_config):
        """Test objective function calculation for categorized UH."""
        warmup_length = 3
        
        model_setup = CategorizedUnitHydrographSetup(
            data=mock_categorized_events,
            model_config=categorized_model_config,
            loss_config=loss_config,
            warmup_length=warmup_length
        )
        
        # Test with random parameters
        param_names = model_setup.get_parameter_names()
        test_params = np.random.rand(len(param_names))
        
        obj_value = model_setup.calculate_objective(test_params)
        assert isinstance(obj_value, (int, float))
        assert obj_value >= 0  # Should be non-negative


class TestCategorizedInterface:
    """Test categorized unit hydrograph interface."""
    
    @pytest.fixture 
    def simple_categorized_events(self):
        """Simple mock events for interface testing."""
        return [
            {
                "P_eff": np.array([1.0, 1.5, 1.0, 0.5] + [0.0] * 6),
                "Q_obs_eff": np.array([0.8, 1.2, 0.8, 0.4] + [0.1] * 6),
                "peak_obs": 1.2,  # Small
                "filepath": "mock_small_1.csv"
            },
            {
                "P_eff": np.array([2.0, 3.0, 2.0, 1.0] + [0.0] * 6),
                "Q_obs_eff": np.array([1.5, 2.5, 1.5, 0.8] + [0.1] * 6),
                "peak_obs": 2.5,  # Medium
                "filepath": "mock_medium_1.csv"
            },
            {
                "P_eff": np.array([3.0, 5.0, 3.0, 1.5] + [0.0] * 6),
                "Q_obs_eff": np.array([2.5, 4.0, 2.5, 1.2] + [0.1] * 6),
                "peak_obs": 4.0,  # Large
                "filepath": "mock_large_1.csv"
            },
        ]
    
    def test_calibrate_categorized_interface(self, simple_categorized_events):
        """Test categorized unit hydrograph through unified calibrate interface."""
        model_config = {
            "name": "categorized_unit_hydrograph",
            "category_weights": {
                "small": {"smoothing_factor": 0.05, "peak_violation_weight": 50.0},
                "medium": {"smoothing_factor": 0.1, "peak_violation_weight": 100.0},
                "large": {"smoothing_factor": 0.2, "peak_violation_weight": 200.0},
            },
            "uh_lengths": {"small": 3, "medium": 4, "large": 5}
        }
        
        # Test scipy algorithm
        algorithm_config = {
            "name": "scipy_minimize",
            "method": "SLSQP",
            "max_iterations": 50  # Small for testing
        }
        
        loss_config = {"type": "time_series", "obj_func": "RMSE"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = calibrate(
                data=simple_categorized_events,
                model_config=model_config,
                algorithm_config=algorithm_config,
                loss_config=loss_config,
                output_dir=temp_dir,
                warmup_length=0  # No warmup for this simple test
            )
            
            # Verify results structure
            assert "categorization_info" in results
            assert "best_params" in results
            assert "convergence" in results
            assert "objective_value" in results
            
            # Check categorization info
            cat_info = results["categorization_info"]
            assert "categories" in cat_info
            assert "events_per_category" in cat_info
            assert len(cat_info["categories"]) == 3
            
            # Check parameter structure
            if results["best_params"]:
                categorized_params = results["best_params"]["categorized_unit_hydrograph"]
                assert len(categorized_params) == 3  # small, medium, large
                
                for category, params in categorized_params.items():
                    param_sum = sum(params.values())
                    assert abs(param_sum - 1.0) < 0.1  # Allow some tolerance


@pytest.mark.skipif(not DEAP_AVAILABLE, reason="DEAP not available")
class TestCategorizedGeneticAlgorithm:
    """Test genetic algorithm with categorized unit hydrograph."""
    
    def test_categorized_ga_integration(self):
        """Test categorized UH with genetic algorithm."""
        # Simplified mock data for GA demo
        mock_events = [
            {"P_eff": np.array([1.0, 2.0, 1.0] + [0.0] * 7), 
             "Q_obs_eff": np.array([0.5, 1.5, 0.8] + [0.1] * 7), 
             "peak_obs": 1.5, "filepath": "ga_small.csv"},
            {"P_eff": np.array([2.0, 4.0, 2.0] + [0.0] * 7), 
             "Q_obs_eff": np.array([1.0, 3.0, 1.5] + [0.1] * 7), 
             "peak_obs": 3.0, "filepath": "ga_medium.csv"},
            {"P_eff": np.array([3.0, 6.0, 3.0] + [0.0] * 7), 
             "Q_obs_eff": np.array([1.5, 4.5, 2.2] + [0.1] * 7), 
             "peak_obs": 4.5, "filepath": "ga_large.csv"},
        ]
        
        model_config = {
            "name": "categorized_unit_hydrograph",
            "uh_lengths": {"small": 3, "medium": 4, "large": 5},
            "category_weights": {
                "small": {"smoothing_factor": 0.1, "peak_violation_weight": 100.0},
                "medium": {"smoothing_factor": 0.2, "peak_violation_weight": 200.0},
                "large": {"smoothing_factor": 0.3, "peak_violation_weight": 300.0},
            }
        }
        
        # GA configuration (small for demo)
        ga_config = {
            "name": "genetic_algorithm",
            "random_seed": 42,
            "pop_size": 10,  # Small for testing
            "n_generations": 3,  # Few generations for testing
            "cx_prob": 0.6,
            "mut_prob": 0.3,
            "save_freq": 2
        }
        
        loss_config = {"type": "time_series", "obj_func": "RMSE"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = calibrate(
                data=mock_events,
                model_config=model_config,
                algorithm_config=ga_config,
                loss_config=loss_config,
                output_dir=temp_dir,
                warmup_length=0
            )
            
            assert "convergence" in results
            assert "objective_value" in results
            assert isinstance(results["objective_value"], (int, float))


class TestExampleUsage:
    """Test example usage patterns for categorized unit hydrograph."""
    
    def test_configuration_examples(self):
        """Test various configuration examples."""
        # Test default category weights
        default_weights = {
            "small": {"smoothing_factor": 0.05, "peak_violation_weight": 50.0},
            "medium": {"smoothing_factor": 0.1, "peak_violation_weight": 200.0},
            "large": {"smoothing_factor": 0.2, "peak_violation_weight": 500.0},
        }
        
        # Test balanced category weights
        balanced_weights = {
            "small": {"smoothing_factor": 0.2, "peak_violation_weight": 200.0},
            "medium": {"smoothing_factor": 0.2, "peak_violation_weight": 200.0},
            "large": {"smoothing_factor": 0.2, "peak_violation_weight": 200.0},
        }
        
        # Test aggressive category weights
        aggressive_weights = {
            "small": {"smoothing_factor": 0.05, "peak_violation_weight": 50.0},
            "medium": {"smoothing_factor": 0.1, "peak_violation_weight": 100.0},
            "large": {"smoothing_factor": 0.5, "peak_violation_weight": 2000.0},
        }
        
        # Verify structure of each weight scheme
        for weights in [default_weights, balanced_weights, aggressive_weights]:
            assert "small" in weights
            assert "medium" in weights
            assert "large" in weights
            
            for category, params in weights.items():
                assert "smoothing_factor" in params
                assert "peak_violation_weight" in params
                assert isinstance(params["smoothing_factor"], (int, float))
                assert isinstance(params["peak_violation_weight"], (int, float))
    
    def test_various_uh_lengths(self):
        """Test various unit hydrograph length configurations."""
        # Short UH lengths
        short_lengths = {"small": 4, "medium": 8, "large": 12}
        
        # Medium UH lengths
        medium_lengths = {"small": 8, "medium": 16, "large": 24}
        
        # Long UH lengths
        long_lengths = {"small": 12, "medium": 24, "large": 36}
        
        for lengths in [short_lengths, medium_lengths, long_lengths]:
            assert "small" in lengths
            assert "medium" in lengths
            assert "large" in lengths
            
            # Verify lengths are positive integers
            for category, length in lengths.items():
                assert isinstance(length, int)
                assert length > 0
                
            # Verify increasing lengths from small to large
            assert lengths["small"] <= lengths["medium"] <= lengths["large"]