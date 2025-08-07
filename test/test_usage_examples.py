"""
Test cases for usage examples and example patterns.

This module tests example usage patterns and ensures that the example
code would work correctly when adapted to real use cases.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from hydromodel.trainers.unified_calibrate import (
    calibrate,
    DEAP_AVAILABLE
)


class TestBasicUsageExamples:
    """Test basic usage examples for unit hydrograph calibration."""
    
    def test_scipy_unit_hydrograph_example(self):
        """Test basic scipy unit hydrograph example pattern."""
        # Example mock data pattern
        mock_event_data = [
            {
                "P_eff": np.array([1.0, 2.5, 1.8, 0.8, 0.3, 0.1] + [0.0] * 14),
                "Q_obs_eff": np.array([0.5, 1.8, 2.2, 1.5, 1.0, 0.6] + [0.1] * 14),
                "filepath": "example_event_1.csv"
            },
            {
                "P_eff": np.array([0.8, 2.0, 1.5, 0.6, 0.2, 0.05] + [0.0] * 14),
                "Q_obs_eff": np.array([0.4, 1.5, 1.8, 1.2, 0.8, 0.4] + [0.1] * 14),
                "filepath": "example_event_2.csv"
            }
        ]
        
        # Example model configuration
        model_config = {
            "name": "unit_hydrograph",
            "n_uh": 12,
            "smoothing_factor": 0.1,
            "peak_violation_weight": 1000.0,
            "apply_peak_penalty": True,
            "net_rain_name": "P_eff",
            "obs_flow_name": "Q_obs_eff"
        }
        
        # Example algorithm configuration
        algorithm_config = {
            "name": "scipy_minimize",
            "method": "SLSQP",
            "max_iterations": 200
        }
        
        loss_config = {"type": "time_series", "obj_func": "RMSE"}
        
        with tempfile.TemporaryDirectory() as output_dir:
            # This pattern should work
            results = calibrate(
                data=mock_event_data,
                model_config=model_config,
                algorithm_config=algorithm_config,
                loss_config=loss_config,
                output_dir=output_dir,
                warmup_length=0
            )
            
            assert "convergence" in results
            assert "best_params" in results
            assert "objective_value" in results
    
    @pytest.mark.skipif(not DEAP_AVAILABLE, reason="DEAP not available")
    def test_ga_unit_hydrograph_example(self):
        """Test genetic algorithm unit hydrograph example pattern."""
        # Simplified mock data for GA
        mock_event_data = [
            {
                "P_eff": np.array([1.5, 3.0, 2.0, 1.0, 0.3] + [0.0] * 10),
                "Q_obs_eff": np.array([0.8, 2.2, 2.5, 1.8, 1.0] + [0.1] * 10),
                "filepath": "ga_example_1.csv"
            }
        ]
        
        model_config = {
            "name": "unit_hydrograph",
            "n_uh": 8,
            "smoothing_factor": 0.05,
            "peak_violation_weight": 5000.0
        }
        
        # Example GA configuration
        ga_config = {
            "name": "genetic_algorithm",
            "random_seed": 42,
            "pop_size": 20,
            "n_generations": 10,
            "cx_prob": 0.7,
            "mut_prob": 0.2,
            "save_freq": 3
        }
        
        loss_config = {"type": "time_series", "obj_func": "RMSE"}
        
        with tempfile.TemporaryDirectory() as output_dir:
            results = calibrate(
                data=mock_event_data,
                model_config=model_config,
                algorithm_config=ga_config,
                loss_config=loss_config,
                output_dir=output_dir,
                warmup_length=0
            )
            
            assert "convergence" in results
            assert "objective_value" in results
            assert isinstance(results["objective_value"], (int, float))


class TestCategorizedUsageExamples:
    """Test categorized unit hydrograph usage examples."""
    
    def test_categorized_basic_example(self):
        """Test basic categorized unit hydrograph example pattern."""
        # Example data with different peak magnitudes
        mock_events = [
            # Small floods
            {
                "P_eff": np.array([0.0, 0.5, 1.2, 0.8, 0.3, 0.1] + [0.0] * 14),
                "Q_obs_eff": np.array([0.1, 0.3, 0.8, 1.2, 0.8, 0.4] + [0.1] * 14),
                "peak_obs": 1.2,
                "filepath": "small_flood_1.csv"
            },
            {
                "P_eff": np.array([0.0, 0.3, 1.0, 0.6, 0.2, 0.05] + [0.0] * 14),
                "Q_obs_eff": np.array([0.05, 0.2, 0.6, 1.0, 0.6, 0.3] + [0.05] * 14),
                "peak_obs": 1.0,
                "filepath": "small_flood_2.csv"
            },
            
            # Medium floods
            {
                "P_eff": np.array([0.0, 1.0, 2.5, 1.8, 0.8, 0.3] + [0.0] * 14),
                "Q_obs_eff": np.array([0.2, 0.8, 2.0, 2.8, 2.0, 1.0] + [0.2] * 14),
                "peak_obs": 2.8,
                "filepath": "medium_flood_1.csv"
            },
            {
                "P_eff": np.array([0.0, 1.2, 3.0, 2.0, 1.0, 0.4] + [0.0] * 14),
                "Q_obs_eff": np.array([0.3, 1.0, 2.5, 3.2, 2.2, 1.2] + [0.3] * 14),
                "peak_obs": 3.2,
                "filepath": "medium_flood_2.csv"
            },
            
            # Large floods  
            {
                "P_eff": np.array([0.0, 2.0, 5.0, 3.5, 1.8, 0.8] + [0.0] * 14),
                "Q_obs_eff": np.array([0.5, 2.0, 4.5, 5.8, 4.0, 2.5] + [0.5] * 14),
                "peak_obs": 5.8,
                "filepath": "large_flood_1.csv"
            },
            {
                "P_eff": np.array([0.0, 1.8, 4.5, 3.2, 1.5, 0.6] + [0.0] * 14),
                "Q_obs_eff": np.array([0.4, 1.8, 4.0, 5.2, 3.5, 2.0] + [0.4] * 14),
                "peak_obs": 5.2,
                "filepath": "large_flood_2.csv"
            },
        ]
        
        # Example model configuration for categorized UH
        model_config = {
            "name": "categorized_unit_hydrograph",
            "category_weights": {
                "small": {"smoothing_factor": 0.05, "peak_violation_weight": 50.0},
                "medium": {"smoothing_factor": 0.1, "peak_violation_weight": 200.0},
                "large": {"smoothing_factor": 0.2, "peak_violation_weight": 500.0},
            },
            "uh_lengths": {
                "small": 6,    # Shorter UH for small floods
                "medium": 10,  # Medium length for medium floods  
                "large": 14    # Longer UH for large floods
            },
            "net_rain_name": "P_eff",
            "obs_flow_name": "Q_obs_eff"
        }
        
        # Example algorithm configuration
        algorithm_config = {
            "name": "scipy_minimize",
            "method": "SLSQP",
            "max_iterations": 200
        }
        
        loss_config = {"type": "time_series", "obj_func": "RMSE"}
        
        with tempfile.TemporaryDirectory() as output_dir:
            results = calibrate(
                data=mock_events,
                model_config=model_config,
                algorithm_config=algorithm_config,
                loss_config=loss_config,
                output_dir=output_dir,
                warmup_length=0
            )
            
            assert "convergence" in results
            assert "objective_value" in results
            
            # Check categorization info
            assert "categorization_info" in results
            cat_info = results["categorization_info"]
            assert "categories" in cat_info
            assert "thresholds" in cat_info
            assert "events_per_category" in cat_info
            
            # Should have all three categories
            assert len(cat_info["categories"]) == 3
            assert "small" in cat_info["categories"]
            assert "medium" in cat_info["categories"] 
            assert "large" in cat_info["categories"]
            
            # Check that events are distributed among categories
            total_events = sum(cat_info["events_per_category"].values())
            assert total_events == len(mock_events)
    
    @pytest.mark.skipif(not DEAP_AVAILABLE, reason="DEAP not available")
    def test_categorized_ga_example(self):
        """Test categorized UH with genetic algorithm example."""
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
            "pop_size": 20,
            "n_generations": 10,
            "cx_prob": 0.6,
            "mut_prob": 0.3,
            "save_freq": 5
        }
        
        loss_config = {"type": "time_series", "obj_func": "RMSE"}
        
        with tempfile.TemporaryDirectory() as output_dir:
            results = calibrate(
                data=mock_events,
                model_config=model_config,
                algorithm_config=ga_config,
                loss_config=loss_config,
                output_dir=output_dir,
                warmup_length=0
            )
            
            assert "convergence" in results
            assert "objective_value" in results


class TestCommandLinePatterns:
    """Test command line usage patterns (simulated)."""
    
    def test_scipy_command_line_pattern(self):
        """Test pattern that would be used from command line with scipy."""
        # This simulates what would happen with command line arguments
        args_simulation = {
            "algorithm": "scipy_minimize",
            "method": "SLSQP",
            "max_iterations": 500,
            "uh_lengths": '{"small":8,"medium":16,"large":24}',
            "category_weights": "default",
            "warmup_length": 160,
            "station_id": "test_station",
            "output_dir": "test_results/"
        }
        
        # Test that we can create configurations from these args
        import json
        
        # Parse UH lengths
        try:
            uh_lengths = json.loads(args_simulation["uh_lengths"])
            assert isinstance(uh_lengths, dict)
            assert "small" in uh_lengths
            assert "medium" in uh_lengths 
            assert "large" in uh_lengths
        except json.JSONDecodeError:
            pytest.fail("UH lengths should be valid JSON")
        
        # Test algorithm config creation
        if args_simulation["algorithm"] == "scipy_minimize":
            algorithm_config = {
                "name": "scipy_minimize",
                "method": args_simulation["method"],
                "max_iterations": args_simulation["max_iterations"],
            }
            assert algorithm_config["name"] == "scipy_minimize"
        
        # Test category weights
        category_weight_schemes = {
            "default": {
                "small": {"smoothing_factor": 0.1, "peak_violation_weight": 100.0},
                "medium": {"smoothing_factor": 0.5, "peak_violation_weight": 500.0},
                "large": {"smoothing_factor": 1.0, "peak_violation_weight": 1000.0},
            }
        }
        
        weights = category_weight_schemes.get(args_simulation["category_weights"])
        assert weights is not None
        assert len(weights) == 3
    
    @pytest.mark.skipif(not DEAP_AVAILABLE, reason="DEAP not available")
    def test_ga_command_line_pattern(self):
        """Test pattern for GA command line usage."""
        args_simulation = {
            "algorithm": "genetic_algorithm",
            "pop_size": 100,
            "n_generations": 50,
            "cx_prob": 0.7,
            "mut_prob": 0.2,
            "save_freq": 5,
            "random_seed": 1234
        }
        
        # Test GA config creation
        if args_simulation["algorithm"] == "genetic_algorithm":
            ga_config = {
                "name": "genetic_algorithm",
                "random_seed": args_simulation["random_seed"],
                "pop_size": args_simulation["pop_size"],
                "n_generations": args_simulation["n_generations"],
                "cx_prob": args_simulation["cx_prob"],
                "mut_prob": args_simulation["mut_prob"],
                "save_freq": args_simulation["save_freq"],
            }
            
            assert ga_config["name"] == "genetic_algorithm"
            assert ga_config["pop_size"] == 100
            assert ga_config["n_generations"] == 50
            assert 0.0 <= ga_config["cx_prob"] <= 1.0
            assert 0.0 <= ga_config["mut_prob"] <= 1.0


class TestConfigurationVariations:
    """Test various configuration variations and edge cases."""
    
    def test_different_smoothing_factors(self):
        """Test various smoothing factor configurations."""
        smoothing_variations = [
            {"small": 0.01, "medium": 0.05, "large": 0.1},   # Low smoothing
            {"small": 0.1, "medium": 0.2, "large": 0.3},     # Medium smoothing
            {"small": 0.5, "medium": 1.0, "large": 2.0},     # High smoothing
        ]
        
        for smoothing in smoothing_variations:
            # Create category weights with these smoothing factors
            category_weights = {}
            for category, factor in smoothing.items():
                category_weights[category] = {
                    "smoothing_factor": factor,
                    "peak_violation_weight": 100.0 * (1 if category == "small" else 
                                                      5 if category == "medium" else 10)
                }
            
            # Verify structure
            assert len(category_weights) == 3
            for category in ["small", "medium", "large"]:
                assert category in category_weights
                assert "smoothing_factor" in category_weights[category]
                assert "peak_violation_weight" in category_weights[category]
    
    def test_different_uh_length_ratios(self):
        """Test various UH length ratio configurations."""
        length_variations = [
            {"small": 4, "medium": 8, "large": 12},      # 1:2:3 ratio
            {"small": 6, "medium": 12, "large": 24},     # 1:2:4 ratio
            {"small": 8, "medium": 16, "large": 32},     # 1:2:4 ratio (longer)
            {"small": 5, "medium": 10, "large": 15},     # 1:2:3 ratio (different base)
        ]
        
        for lengths in length_variations:
            # Verify increasing lengths
            assert lengths["small"] <= lengths["medium"] <= lengths["large"]
            
            # Verify reasonable ratios
            medium_ratio = lengths["medium"] / lengths["small"]
            large_ratio = lengths["large"] / lengths["small"]
            
            assert medium_ratio >= 1.5  # Medium should be at least 1.5x small
            assert large_ratio >= 2.0   # Large should be at least 2x small