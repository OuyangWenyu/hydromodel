"""
Author: Wenyu Ouyang
Date: 2025-07-08 19:01:27
LastEditTime: 2025-07-08 19:57:21
LastEditors: Wenyu Ouyang
Description: Test unit hydrograph functions
FilePath: \hydromodel\test\test_unit_hydrograph.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import numpy as np
from hydromodel.models.unit_hydrograph import uh_conv


def test_uh():
    uh_from_gamma = np.tile(1, (5, 3, 1))
    # uh_from_gamma = np.arange(15).reshape(5, 3, 1)
    rf = np.arange(30).reshape(10, 3, 1) / 100
    qs = uh_conv(rf, uh_from_gamma)
    np.testing.assert_almost_equal(
        np.array(
            [
                [0.0000, 0.0100, 0.0200],
                [0.0300, 0.0500, 0.0700],
                [0.0900, 0.1200, 0.1500],
                [0.1800, 0.2200, 0.2600],
                [0.3000, 0.3500, 0.4000],
                [0.4500, 0.5000, 0.5500],
                [0.6000, 0.6500, 0.7000],
                [0.7500, 0.8000, 0.8500],
                [0.9000, 0.9500, 1.0000],
                [1.0500, 1.1000, 1.1500],
            ]
        ),
        qs[:, :, 0],
        decimal=3,
    )


def test_uh_conv_1d():
    """Test 1D array convolution"""
    # Simple 1D case
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    uh = np.array([0.5, 0.3, 0.2])

    result = uh_conv(x, uh)

    # Expected result from manual calculation
    # conv([1,2,3,4,5], [0.5,0.3,0.2]) truncated
    expected = np.array([0.5, 1.3, 2.3, 3.3, 4.3])
    np.testing.assert_almost_equal(result, expected, decimal=3)

    # Check output shape matches input
    assert result.shape == x.shape


def test_uh_conv_2d():
    """Test 2D array convolution [seq, batch]"""
    # 2D case with multiple batches
    x = np.array(
        [
            [1.0, 2.0],  # seq=0
            [3.0, 4.0],  # seq=1
            [5.0, 6.0],  # seq=2
        ]
    )  # shape: [3, 2]

    uh = np.array(
        [
            [0.5, 0.4],  # uh=0
            [0.3, 0.6],  # uh=1
        ]
    )  # shape: [2, 2]

    result = uh_conv(x, uh)

    # Manual calculation for verification
    # batch 0: conv([1,3,5], [0.5,0.3]) = [0.5, 1.8, 3.4]
    # batch 1: conv([2,4,6], [0.4,0.6]) = [0.8, 2.8, 4.8]
    expected = np.array(
        [
            [0.5, 0.8],
            [1.8, 2.8],
            [3.4, 4.8],
        ]
    )

    np.testing.assert_almost_equal(result, expected, decimal=3)
    assert result.shape == x.shape


def test_uh_conv_3d():
    """Test 3D array convolution [seq, batch, feature]"""
    # 3D case with batch and feature dimensions
    x = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],  # seq=0
            [[5.0, 6.0], [7.0, 8.0]],  # seq=1
            [[9.0, 10.0], [11.0, 12.0]],  # seq=2
        ]
    )  # shape: [3, 2, 2]

    uh = np.array(
        [
            [[0.5, 0.4], [0.3, 0.2]],  # uh=0
            [[0.3, 0.6], [0.7, 0.8]],  # uh=1
        ]
    )  # shape: [2, 2, 2]

    result = uh_conv(x, uh)

    # Check shape preservation
    assert result.shape == x.shape

    # Check that convolution is applied correctly along sequence dimension
    # For batch=0, feature=0: conv([1,5,9], [0.5,0.3]) = [0.5, 2.8, 6.0]
    expected_batch0_feat0 = np.array([0.5, 2.8, 6.0])
    np.testing.assert_almost_equal(result[:, 0, 0], expected_batch0_feat0, decimal=3)


def test_uh_conv_dimension_mismatch_1d():
    """Test error handling for 1D dimension mismatch"""
    x = np.array([1.0, 2.0, 3.0])
    uh = np.array([[0.5, 0.3], [0.2, 0.1]])  # 2D instead of 1D

    result = uh_conv(x, uh)
    # Should return zeros due to dimension mismatch
    np.testing.assert_array_equal(result, np.zeros_like(x))


def test_uh_conv_dimension_mismatch_2d():
    """Test error handling for 2D dimension mismatch"""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])  # [2, 2]
    uh = np.array([[0.5, 0.3, 0.1]])  # Wrong batch size

    result = uh_conv(x, uh)
    # Should return zeros due to dimension mismatch
    np.testing.assert_array_equal(result, np.zeros_like(x))


def test_uh_conv_dimension_mismatch_3d():
    """Test error handling for 3D dimension mismatch"""
    x = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])  # [2, 1, 2]
    uh = np.array([[[0.5, 0.3, 0.1]]])  # Wrong feature size

    result = uh_conv(x, uh)
    # Should return zeros due to dimension mismatch
    np.testing.assert_array_equal(result, np.zeros_like(x))


def test_uh_conv_unsupported_dimension():
    """Test error handling for unsupported dimensions (4D+)"""
    x = np.ones((2, 2, 2, 2))  # 4D array
    uh = np.ones((2, 2, 2, 2))

    result = uh_conv(x, uh)
    # Should return zeros for unsupported dimensions
    np.testing.assert_array_equal(result, np.zeros_like(x))


def test_uh_conv_edge_cases():
    """Test edge cases"""
    # Empty arrays
    x_empty = np.array([])
    uh_empty = np.array([])
    result = uh_conv(x_empty, uh_empty)
    assert result.shape == x_empty.shape

    # Single element
    x_single = np.array([5.0])
    uh_single = np.array([0.5])
    result = uh_conv(x_single, uh_single)
    expected = np.array([2.5])
    np.testing.assert_almost_equal(result, expected, decimal=3)


def test_uh_conv_zero_input():
    """Test with zero inputs"""
    x = np.zeros(5)
    uh = np.array([0.5, 0.3, 0.2])

    result = uh_conv(x, uh)
    expected = np.zeros(5)
    np.testing.assert_array_equal(result, expected)
