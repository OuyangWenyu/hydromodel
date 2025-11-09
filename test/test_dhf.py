"""
Author: Wenyu Ouyang
Date: 2025-01-27
LastEditTime: 2025-01-27
LastEditors: Wenyu Ouyang
Description: Test module for DHF (Dahuofang) hydrological model
FilePath: \hydromodel\test\test_dhf.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import pytest
import numpy as np
import warnings
from hydromodel.models.dhf import dhf


@pytest.fixture
def setup_dhf_data():
    """Prepare test data for DHF model"""

    time_length = 30
    basin_num = 1
    var_num = 2
    warmup_length = 10

    # Create input data [time, basin, variable]
    p_and_e = np.ones((time_length, basin_num, var_num))
    # Set precipitation data (mm)
    p_and_e[:, :, 0] = 8.0
    # Set PET data (mm)
    p_and_e[:, :, 1] = 3.0

    # Create parameter data [basin, parameter] - normalized values
    parameters = np.array(
        [
            [
                0.5,
                0.6,
                0.4,
                0.3,
                0.2,
                0.1,
                0.8,
                0.15,
                0.7,
                0.9,
                0.5,
                0.4,
                0.6,
                0.3,
                0.7,
                0.5,
                0.4,
                0.6,
            ]
        ]
    )

    # Required kwargs for DHF model
    kwargs = {
        "main_river_length": 155.763,  # km
        "basin_area": 5482.0,  # km²
        "time_interval_hours": 3.0,  # hours
    }

    return {
        "p_and_e": p_and_e,
        "parameters": parameters,
        "warmup_length": warmup_length,
        "kwargs": kwargs,
        "time_length": time_length,
        "basin_num": basin_num,
    }


def test_dhf_basic_functionality(setup_dhf_data):
    """Test basic DHF model functionality"""
    data = setup_dhf_data

    # Suppress runtime warnings for cleaner test output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        q_sim = dhf(
            p_and_e=data["p_and_e"],
            parameters=data["parameters"],
            warmup_length=data["warmup_length"],
            return_state=False,
            normalized_params=True,
            **data["kwargs"],
        )

    # Check output shape
    expected_time_steps = data["time_length"] - data["warmup_length"]
    assert q_sim.shape == (expected_time_steps, data["basin_num"], 1)

    # Check that discharge values are non-negative
    assert np.all(q_sim >= 0)


def test_dhf_with_return_state(setup_dhf_data):
    """Test DHF model with return_state=True"""
    data = setup_dhf_data

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        result = dhf(
            p_and_e=data["p_and_e"],
            parameters=data["parameters"],
            warmup_length=data["warmup_length"],
            return_state=True,
            normalized_params=True,
            **data["kwargs"],
        )

    # Should return tuple with 9 elements
    assert len(result) == 9
    q_sim, runoff_sim, y0, yu, yl, y, sa, ua, ya = result

    # Check shapes
    expected_time_steps = data["time_length"] - data["warmup_length"]
    expected_shape = (expected_time_steps, data["basin_num"], 1)

    for var in [q_sim, runoff_sim, y0, yu, yl, y, sa, ua, ya]:
        assert var.shape == expected_shape


def test_dhf_initial_states_vs_default(setup_dhf_data):
    """Test DHF model with and without initial_states parameter"""
    data = setup_dhf_data

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        # Run 1: Default initial states (after warmup)
        (
            q_sim_default,
            runoff_default,
            y0_default,
            yu_default,
            yl_default,
            y_default,
            sa_default,
            ua_default,
            ya_default,
        ) = dhf(
            p_and_e=data["p_and_e"],
            parameters=data["parameters"],
            warmup_length=data["warmup_length"],
            return_state=True,
            normalized_params=True,
            **data["kwargs"],
        )

        # Run 2: Custom initial states
        custom_initial_states = {
            "sa0": 15.0,  # Surface storage (mm)
            "ua0": 12.0,  # Subsurface storage (mm)
            "ya0": 20.0,  # Precedent precipitation (mm)
        }

        (
            q_sim_custom,
            runoff_custom,
            y0_custom,
            yu_custom,
            yl_custom,
            y_custom,
            sa_custom,
            ua_custom,
            ya_custom,
        ) = dhf(
            p_and_e=data["p_and_e"],
            parameters=data["parameters"],
            warmup_length=data["warmup_length"],
            return_state=True,
            normalized_params=True,
            initial_states=custom_initial_states,
            **data["kwargs"],
        )

    # Check that results are different when using custom initial states
    assert not np.allclose(
        q_sim_default, q_sim_custom
    ), "Discharge should be different with custom initial states"
    assert not np.allclose(
        runoff_default, runoff_custom
    ), "Runoff should be different with custom initial states"
    assert not np.allclose(
        sa_default[0, :, 0], sa_custom[0, :, 0]
    ), "Initial surface storage should be different"
    assert not np.allclose(
        ya_default[0, :, 0], ya_custom[0, :, 0]
    ), "Initial precedent precipitation should be different"

    # Check that shapes remain the same
    assert q_sim_default.shape == q_sim_custom.shape
    assert runoff_default.shape == runoff_custom.shape


def test_dhf_partial_initial_states(setup_dhf_data):
    """Test DHF model with partial initial states (only some variables specified)"""
    data = setup_dhf_data

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        # Run with only sa0 specified
        partial_initial_states = {"sa0": 10.0}

        q_sim_partial = dhf(
            p_and_e=data["p_and_e"],
            parameters=data["parameters"],
            warmup_length=data["warmup_length"],
            return_state=False,
            normalized_params=True,
            initial_states=partial_initial_states,
            **data["kwargs"],
        )

        # Run with default states
        q_sim_default = dhf(
            p_and_e=data["p_and_e"],
            parameters=data["parameters"],
            warmup_length=data["warmup_length"],
            return_state=False,
            normalized_params=True,
            **data["kwargs"],
        )

    # Results should be different even with partial override
    assert not np.allclose(
        q_sim_default, q_sim_partial
    ), "Results should differ with partial initial states"


def test_dhf_initial_states_impact(setup_dhf_data):
    """Test that initial_states parameter has measurable impact on simulation"""
    data = setup_dhf_data

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        # Default run
        q_sim_default = dhf(
            p_and_e=data["p_and_e"],
            parameters=data["parameters"],
            warmup_length=data["warmup_length"],
            return_state=False,
            normalized_params=True,
            **data["kwargs"],
        )

        # Run with significantly different initial states
        extreme_initial_states = {"sa0": 25.0, "ua0": 20.0, "ya0": 30.0}

        q_sim_extreme = dhf(
            p_and_e=data["p_and_e"],
            parameters=data["parameters"],
            warmup_length=data["warmup_length"],
            return_state=False,
            normalized_params=True,
            initial_states=extreme_initial_states,
            **data["kwargs"],
        )

    # Calculate difference
    discharge_diff = np.mean(np.abs(q_sim_extreme - q_sim_default))

    # Should have significant impact
    assert (
        discharge_diff > 0.01
    ), f"Initial states should have significant impact, got difference: {discharge_diff}"


def test_dhf_required_parameters():
    """Test that DHF model raises proper errors for missing required parameters"""

    # Simple test data
    p_and_e = np.ones((10, 1, 2))
    parameters = np.array([[0.5] * 18])

    # Test missing main_river_length
    with pytest.raises(
        ValueError, match="main_river_length.*must be provided"
    ):
        dhf(p_and_e=p_and_e, parameters=parameters, basin_area=5482.0)

    # Test missing basin_area
    with pytest.raises(ValueError, match="basin_area.*must be provided"):
        dhf(p_and_e=p_and_e, parameters=parameters, main_river_length=155.763)
