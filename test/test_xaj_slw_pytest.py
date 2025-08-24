"""
Test module for XAJ-SLW (Xinanjiang-Songliaowei) hydrological model
"""

import pytest
import numpy as np
import warnings
from hydromodel.models.xaj_slw import xaj_slw


@pytest.fixture
def setup_xaj_slw_data():
    """Prepare test data for XAJ-SLW model"""
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

    # Create parameter data [basin, parameter]
    parameters = np.array(
        [
            [
                6.257,   # WUP: Initial upper layer tension water
                12.874,  # WLP: Initial lower layer tension water
                25.0,    # WDP: Initial deep layer tension water
                0.854,   # SP: Initial free water storage
                0.106,   # FRP: Initial runoff area ratio
                128.82,  # WM: Total tension water capacity
                0.197,   # WUMx: Upper layer capacity ratio
                0.885,   # WLMx: Lower layer capacity ratio
                0.987,   # KC: Evaporation coefficient
                0.173,   # B: Exponent of tension water capacity curve
                0.189,   # C: Deep evapotranspiration coefficient
                0.019,   # IM: Impervious area ratio
                39.37,   # SM: Average free water capacity
                1.5,     # EX: Exponent of free water capacity curve
                0.317,   # KG: Groundwater outflow coefficient
                0.440,   # KI: Interflow outflow coefficient
                0.288,   # CS: Channel system recession constant
                0.759,   # CI: Lower interflow recession constant
                0.951,   # CG: Groundwater storage recession constant
                4.0,     # LAG: Lag time
                6.0,     # KK: Muskingum K parameter
                0.125,   # X: Muskingum X parameter
                2.0,     # MP: Number of Muskingum reaches
                0.0,     # QSP: Initial surface flow
                1.265,   # QIP: Initial interflow
                0.172,   # QGP: Initial groundwater flow
            ]
        ]
    )

    # Required kwargs for XAJ-SLW model
    kwargs = {
        "time_interval_hours": 6.0,  # hours
        "area": 2163.0,  # km²
    }

    return {
        "p_and_e": p_and_e,
        "parameters": parameters,
        "warmup_length": warmup_length,
        "kwargs": kwargs,
        "time_length": time_length,
        "basin_num": basin_num,
    }


def test_xaj_slw_basic_functionality(setup_xaj_slw_data):
    """Test basic XAJ-SLW model functionality"""
    data = setup_xaj_slw_data

    # Suppress runtime warnings for cleaner test output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        q_sim = xaj_slw(
            p_and_e=data["p_and_e"],
            parameters=data["parameters"],
            warmup_length=data["warmup_length"],
            return_state=False,
            normalized_params=False,
            **data["kwargs"],
        )

    # Check output shape
    expected_time_steps = data["time_length"] - data["warmup_length"]
    assert q_sim.shape == (expected_time_steps, data["basin_num"], 1)

    # Check that discharge values are non-negative
    assert np.all(q_sim >= 0)


def test_xaj_slw_with_return_state(setup_xaj_slw_data):
    """Test XAJ-SLW model with return_state=True"""
    data = setup_xaj_slw_data

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        result = xaj_slw(
            p_and_e=data["p_and_e"],
            parameters=data["parameters"],
            warmup_length=data["warmup_length"],
            return_state=True,
            normalized_params=False,
            **data["kwargs"],
        )

    # Should return tuple with 9 elements
    assert len(result) == 9
    q_sim, runoff_sim, rs, ri, rg, pe, wu, wl, wd = result

    # Check shapes
    expected_time_steps = data["time_length"] - data["warmup_length"]
    expected_shape = (expected_time_steps, data["basin_num"], 1)

    for var in [q_sim, runoff_sim, rs, ri, rg, pe, wu, wl, wd]:
        assert var.shape == expected_shape


def test_xaj_slw_initial_states_vs_default(setup_xaj_slw_data):
    """Test XAJ-SLW model with and without initial_states parameter"""
    data = setup_xaj_slw_data

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        # Run 1: Default initial states (after warmup)
        (
            q_sim_default,
            runoff_default,
            rs_default,
            ri_default,
            rg_default,
            pe_default,
            wu_default,
            wl_default,
            wd_default,
        ) = xaj_slw(
            p_and_e=data["p_and_e"],
            parameters=data["parameters"],
            warmup_length=data["warmup_length"],
            return_state=True,
            normalized_params=False,
            **data["kwargs"],
        )

        # Run 2: Custom initial states
        custom_initial_states = {
            "wu0": 15.0,  # Upper layer tension water (mm)
            "wl0": 12.0,  # Lower layer tension water (mm)
            "wd0": 20.0,  # Deep layer tension water (mm)
            "s0": 5.0,    # Free water storage (mm)
            "fr0": 0.2,   # Runoff area ratio
        }

        (
            q_sim_custom,
            runoff_custom,
            rs_custom,
            ri_custom,
            rg_custom,
            pe_custom,
            wu_custom,
            wl_custom,
            wd_custom,
        ) = xaj_slw(
            p_and_e=data["p_and_e"],
            parameters=data["parameters"],
            warmup_length=data["warmup_length"],
            return_state=True,
            normalized_params=False,
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
        wu_default[0, :, 0], wu_custom[0, :, 0]
    ), "Initial upper layer tension water should be different"
    assert not np.allclose(
        wl_default[0, :, 0], wl_custom[0, :, 0]
    ), "Initial lower layer tension water should be different"

    # Check that shapes remain the same
    assert q_sim_default.shape == q_sim_custom.shape
    assert runoff_default.shape == runoff_custom.shape


def test_xaj_slw_partial_initial_states(setup_xaj_slw_data):
    """Test XAJ-SLW model with partial initial states (only some variables specified)"""
    data = setup_xaj_slw_data

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        # Run with only wu0 specified
        partial_initial_states = {"wu0": 10.0}

        q_sim_partial = xaj_slw(
            p_and_e=data["p_and_e"],
            parameters=data["parameters"],
            warmup_length=data["warmup_length"],
            return_state=False,
            normalized_params=False,
            initial_states=partial_initial_states,
            **data["kwargs"],
        )

        # Run with default states
        q_sim_default = xaj_slw(
            p_and_e=data["p_and_e"],
            parameters=data["parameters"],
            warmup_length=data["warmup_length"],
            return_state=False,
            normalized_params=False,
            **data["kwargs"],
        )

    # Results should be different even with partial override
    assert not np.allclose(
        q_sim_default, q_sim_partial
    ), "Results should differ with partial initial states"


def test_xaj_slw_initial_states_impact(setup_xaj_slw_data):
    """Test that initial_states parameter has measurable impact on simulation"""
    data = setup_xaj_slw_data

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        # Default run
        q_sim_default = xaj_slw(
            p_and_e=data["p_and_e"],
            parameters=data["parameters"],
            warmup_length=data["warmup_length"],
            return_state=False,
            normalized_params=False,
            **data["kwargs"],
        )

        # Run with significantly different initial states
        extreme_initial_states = {
            "wu0": 25.0,
            "wl0": 20.0,
            "wd0": 30.0,
            "s0": 10.0,
            "fr0": 0.3,
        }

        q_sim_extreme = xaj_slw(
            p_and_e=data["p_and_e"],
            parameters=data["parameters"],
            warmup_length=data["warmup_length"],
            return_state=False,
            normalized_params=False,
            initial_states=extreme_initial_states,
            **data["kwargs"],
        )

    # Calculate difference
    discharge_diff = np.mean(np.abs(q_sim_extreme - q_sim_default))

    # Should have significant impact
    assert (
        discharge_diff > 0.01
    ), f"Initial states should have significant impact, got difference: {discharge_diff}"


def test_xaj_slw_lag_initial_states(setup_xaj_slw_data):
    """Test XAJ-SLW model with LAG3 initial states"""
    data = setup_xaj_slw_data

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        # Run with LAG3 initial states
        lag_initial_states = {
            "qsig_initial": np.array([0.032, 0.034, 0.039, 0.044, 0.055, 0.061]),
            "qx_initial": np.array([0.071, 0.069, 0.067, 0.065, 0.061, 0.055, 0.05]),
        }

        q_sim_with_lag = xaj_slw(
            p_and_e=data["p_and_e"],
            parameters=data["parameters"],
            warmup_length=data["warmup_length"],
            return_state=False,
            normalized_params=False,
            lag_initial_states=lag_initial_states,
            **data["kwargs"],
        )

        # Run without LAG3 initial states
        q_sim_without_lag = xaj_slw(
            p_and_e=data["p_and_e"],
            parameters=data["parameters"],
            warmup_length=data["warmup_length"],
            return_state=False,
            normalized_params=False,
            **data["kwargs"],
        )

    # Results should be different with LAG3 initial states
    assert not np.allclose(
        q_sim_with_lag, q_sim_without_lag
    ), "Results should differ with LAG3 initial states"


def test_xaj_slw_required_parameters():
    """Test that XAJ-SLW model raises proper errors for missing required parameters"""

    # Simple test data
    p_and_e = np.ones((10, 1, 2))
    parameters = np.array([[0.5] * 26])  # 26 parameters for XAJ-SLW

    # Test missing time_interval_hours
    with pytest.raises(KeyError, match="time_interval_hours"):
        xaj_slw(p_and_e=p_and_e, parameters=parameters, area=2163.0)

    # Test missing area
    with pytest.raises(KeyError, match="area"):
        xaj_slw(p_and_e=p_and_e, parameters=parameters, time_interval_hours=6.0)
