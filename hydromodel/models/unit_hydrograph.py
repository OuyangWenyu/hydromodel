"""
Author: Wenyu Ouyang
Date: 2025-08-07
LastEditTime: 2025-08-07 14:49:55
LastEditors: Wenyu Ouyang
Description: Unit hydrograph model with unified interface - CORE MODEL FUNCTIONS ONLY
FilePath: /hydromodel/hydromodel/models/unit_hydrograph.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import logging
import numpy as np
from typing import Union, Dict, Any
from scipy.stats import gamma


# =============================================================================
# CORE CONVOLUTION FUNCTIONS
# =============================================================================


def uh_conv(x, uh, truncate=True):
    """
    Function for convolution calculation supporting different array dimensions

    Parameters
    ----------
    x
        input array for convolution:
        - 1D: [seq] - sequence data
        - 2D: [seq, batch] - sequence data with batch dimension
        - 3D: [seq, batch, feature] - sequence data with batch and feature dims
    uh
        unit hydrograph array:
        - 1D: [len_uh] - for 1D input
        - 2D: [len_uh, batch] - for 2D input
        - 3D: [len_uh, batch, feature] - for 3D input
    truncate : bool, optional
        If True (default), truncate convolution result to original sequence length.
        If False, return full convolution result (changes output shape).

    Returns
    -------
    np.array
        convolution result. If truncate=True, same shape as x.
        If truncate=False, sequence dimension length = len(x) + len(uh) - 1.
    """
    x = np.asarray(x)
    uh = np.asarray(uh)

    if x.ndim == 1:
        # 1D case: [seq]
        if uh.ndim != 1:
            logging.error("For 1D input x, uh should also be 1D")
            return np.zeros_like(x)
        # Handle empty arrays
        if len(x) == 0 or len(uh) == 0:
            return np.zeros_like(x)

        conv_result = np.convolve(x, uh)
        return conv_result[: len(x)] if truncate else conv_result

    elif x.ndim == 2:
        return _uh_conv_2d(x, uh, truncate)
    elif x.ndim == 3:
        return _uh_conv_3d(x, uh, truncate)
    else:
        logging.error(
            f"Unsupported array dimension: {x.ndim}D. "
            f"Only 1D, 2D, 3D are supported."
        )
        return np.zeros_like(x)


def _uh_conv_2d(x, uh, truncate=True):
    """2D case: [seq, batch]"""
    seq_length, batch_size = x.shape
    if uh.ndim != 2 or uh.shape[1] != batch_size:
        logging.error(
            "For 2D input x [seq, batch], uh should be [len_uh, batch]"
        )
        return np.zeros_like(x)

    # Handle empty arrays
    if seq_length == 0 or uh.shape[0] == 0:
        return np.zeros_like(x)

    # Calculate output shape
    if truncate:
        output_shape = x.shape
        outputs = np.zeros_like(x)
    else:
        conv_length = seq_length + uh.shape[0] - 1
        output_shape = (conv_length, batch_size)
        outputs = np.zeros(output_shape, dtype=x.dtype)

    for i in range(batch_size):
        conv_result = np.convolve(x[:, i], uh[:, i])
        outputs[:, i] = conv_result[:seq_length] if truncate else conv_result
    return outputs


def _uh_conv_3d(x, uh, truncate=True):
    """3D case: [seq, batch, feature]"""
    seq_length, batch_size, feature_size = x.shape
    if uh.ndim != 3 or uh.shape[1:] != (batch_size, feature_size):
        logging.error(
            "For 3D input x [seq, batch, feature], "
            "uh should be [len_uh, batch, feature]"
        )
        return np.zeros_like(x)

    # Handle empty arrays
    if seq_length == 0 or uh.shape[0] == 0:
        return np.zeros_like(x)

    # Calculate output shape
    if truncate:
        output_shape = x.shape
        outputs = np.zeros_like(x)
    else:
        conv_length = seq_length + uh.shape[0] - 1
        output_shape = (conv_length, batch_size, feature_size)
        outputs = np.zeros(output_shape, dtype=x.dtype)

    for i in range(batch_size):
        for j in range(feature_size):
            conv_result = np.convolve(x[:, i, j], uh[:, i, j])
            outputs[:, i, j] = (
                conv_result[:seq_length] if truncate else conv_result
            )
    return outputs


def init_unit_hydrograph(length, method="gamma", **kwargs):
    """
    初始化一个单峰且归一化的单位线。

    Parameters
    ----------
    length : int
        单位线长度。
    method : str, optional
        初始化方法，'gamma'（默认，偏态）或'gaussian'（对称）。
    **kwargs :
        gamma分布参数：shape, scale
        gaussian分布参数：peak_pos, std_ratio

    Returns
    -------
    uh : np.ndarray
        归一化的单位线数组。
    """
    if method == "gaussian":
        peak_pos = kwargs.get("peak_pos", length // 2)
        std_ratio = kwargs.get("std_ratio", 0.15)
        std = length * std_ratio
        x = np.arange(length)
        uh = np.exp(-0.5 * ((x - peak_pos) / std) ** 2)
    else:  # 默认gamma
        shape = kwargs.get("shape", 2.0)
        scale = kwargs.get("scale", 2.0)
        x = np.arange(length)
        uh = gamma.pdf(x, a=shape, scale=scale)
    uh = np.maximum(uh, 0)
    uh /= uh.sum()
    return uh


# =============================================================================
# UNIFIED MODEL INTERFACES
# =============================================================================


def unit_hydrograph(
    inputs: np.ndarray,
    params: np.ndarray,
    warmup_length: int = 0,
    return_state: bool = False,
    **kwargs,
) -> Union[np.ndarray, tuple]:
    """
    Unit hydrograph model with unified interface.

    This function provides a unified interface consistent with other hydrological
    models (XAJ, GR series), where the core computation is a convolution between
    net rainfall and unit hydrograph parameters.

    Parameters
    ----------
    inputs : np.ndarray
        Input data array with shape [time, basin, features]:
        - time: sequence length
        - basin: number of basins (usually 1 for single basin)
        - features: input variables (net_rain, observed_flow, etc.)
        For single basin, inputs can be [time, features] or [time] for net_rain only
    params : np.ndarray
        Unit hydrograph parameters with shape [basin, n_uh]:
        - basin: number of basins
        - n_uh: length of unit hydrograph
        For single basin, params can be 1D array [n_uh]
    warmup_length : int, optional
        Length of warmup period to exclude from output (default: 0)
    return_state : bool, optional
        If True, return state information (default: False)
    **kwargs
        Additional parameters:
        - net_rain_idx: index of net rainfall in input features (default: 0)
        - truncate: whether to truncate convolution result (default: True)

    Returns
    -------
    np.ndarray or tuple
        If return_state=False: simulated flow with shape same as inputs
        If return_state=True: (simulated_flow, state_dict)

    Examples
    --------
    >>> # Single basin example
    >>> net_rain = np.array([0, 5, 10, 8, 3, 1, 0])  # [time]
    >>> uh_params = np.array([0.3, 0.5, 0.2])  # [n_uh=3]
    >>> flow = unit_hydrograph(net_rain, uh_params)

    >>> # Multi-basin example
    >>> inputs = np.random.rand(100, 2, 1)  # [time=100, basin=2, feature=1]
    >>> params = np.random.rand(2, 24)  # [basin=2, n_uh=24]
    >>> flows = unit_hydrograph(inputs, params, warmup_length=10)
    """
    # Ensure inputs and params are numpy arrays
    inputs = np.asarray(inputs)
    params = np.asarray(params)

    # Handle different input dimensions
    if inputs.ndim == 1:
        # Single time series: [time] -> [time, 1, 1]
        inputs = inputs.reshape(-1, 1, 1)
        single_series = True
    elif inputs.ndim == 2:
        # Two cases: [time, basin] or [time, features]
        if params.ndim == 1:
            # Assume [time, features], single basin
            inputs = inputs.reshape(inputs.shape[0], 1, -1)
            params = params.reshape(1, -1)
        else:
            # Assume [time, basin], add feature dim
            inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], 1)
        single_series = False
    elif inputs.ndim == 3:
        # Standard format: [time, basin, features]
        single_series = False
    else:
        raise ValueError(f"Unsupported input dimension: {inputs.ndim}")

    # Handle parameter dimensions
    if params.ndim == 1:
        # Single UH: [n_uh] -> [1, n_uh]
        params = params.reshape(1, -1)
    elif params.ndim != 2:
        raise ValueError(f"Parameters must be 1D or 2D, got {params.ndim}D")

    time_steps, n_basins, n_features = inputs.shape
    n_basins_params, n_uh = params.shape

    # Check dimension consistency
    if n_basins != n_basins_params:
        raise ValueError(
            f"Basin dimension mismatch: inputs has {n_basins}, params has {n_basins_params}"
        )

    # Extract net rainfall (default: first feature)
    net_rain_idx = kwargs.get("net_rain_idx", 0)
    if net_rain_idx >= n_features:
        raise ValueError(
            f"net_rain_idx {net_rain_idx} >= n_features {n_features}"
        )

    net_rain = inputs[:, :, net_rain_idx]  # [time, basin]

    # Normalize unit hydrograph parameters (ensure sum to 1.0)
    params_normalized = params / params.sum(axis=1, keepdims=True)

    # Perform convolution for each basin
    truncate = kwargs.get("truncate", True)
    simulated_flows = np.zeros((time_steps, n_basins))

    for basin_idx in range(n_basins):
        basin_net_rain = net_rain[:, basin_idx]  # [time]
        basin_uh = params_normalized[basin_idx, :]  # [n_uh]

        # Convolution
        flow_conv = uh_conv(basin_net_rain, basin_uh, truncate=truncate)
        simulated_flows[:, basin_idx] = flow_conv

    # Apply warmup period
    if warmup_length > 0:
        simulated_flows = simulated_flows[warmup_length:]

    # Prepare output
    if single_series and n_basins == 1:
        # Return to original format for single series
        simulated_flows = simulated_flows.flatten()

    # Prepare state information if requested
    if return_state:
        state_dict = {
            "uh_params_normalized": params_normalized,
            "warmup_length": warmup_length,
            "n_uh": n_uh,
            "model_type": "unit_hydrograph",
        }
        return simulated_flows, state_dict

    return simulated_flows


def categorized_unit_hydrograph(
    inputs: np.ndarray,
    params: Dict[str, np.ndarray],
    warmup_length: int = 0,
    return_state: bool = False,
    **kwargs,
) -> Union[np.ndarray, tuple]:
    """
    Categorized unit hydrograph model with unified interface.

    This model uses different unit hydrographs for different flood magnitude categories
    (e.g., small, medium, large floods). Events are categorized based on peak flow
    and appropriate UH is applied.

    Parameters
    ----------
    inputs : np.ndarray
        Input data array with shape [time, basin, features]:
        - Must include net_rain and observed_flow for categorization
        - For flood events, typically [event_length, 1, features]
    params : dict
        Dictionary of unit hydrograph parameters by category:
        {
            'small': np.ndarray [basin, n_uh_small],
            'medium': np.ndarray [basin, n_uh_medium],
            'large': np.ndarray [basin, n_uh_large],
            'thresholds': dict with categorization thresholds,
            'category_weights': dict with optimization weights (optional)
        }
    warmup_length : int, optional
        Length of warmup period to exclude from output (default: 0)
    return_state : bool, optional
        If True, return state information (default: False)
    **kwargs
        Additional parameters:
        - net_rain_idx: index of net rainfall (default: 0)
        - obs_flow_idx: index of observed flow for categorization (default: 1)
        - categorization_method: 'peak_magnitude' (default)

    Returns
    -------
    np.ndarray or tuple
        If return_state=False: simulated flow
        If return_state=True: (simulated_flow, state_dict)

    Examples
    --------
    >>> # Setup categorized parameters
    >>> params = {
    ...     'small': np.array([[0.4, 0.4, 0.2]]),  # [1 basin, 3 params]
    ...     'medium': np.array([[0.2, 0.3, 0.3, 0.2]]),  # [1 basin, 4 params]
    ...     'large': np.array([[0.1, 0.2, 0.3, 0.2, 0.2]]),  # [1 basin, 5 params]
    ...     'thresholds': {'small_medium': 10.0, 'medium_large': 25.0}
    ... }
    >>> inputs = np.random.rand(50, 1, 2)  # [time=50, basin=1, features=2]
    >>> flow = categorized_unit_hydrograph(inputs, params)
    """
    # Ensure inputs is numpy array
    inputs = np.asarray(inputs)

    if inputs.ndim != 3:
        raise ValueError(
            f"Categorized UH requires 3D inputs [time, basin, features], got {inputs.ndim}D"
        )

    time_steps, n_basins, n_features = inputs.shape

    # Validate required parameters
    required_categories = ["small", "medium", "large"]
    for cat in required_categories:
        if cat not in params:
            raise ValueError(f"Missing UH parameters for category: {cat}")

    # Extract input data
    net_rain_idx = kwargs.get("net_rain_idx", 0)
    obs_flow_idx = kwargs.get("obs_flow_idx", 1)

    if net_rain_idx >= n_features:
        raise ValueError(
            f"net_rain_idx {net_rain_idx} >= n_features {n_features}"
        )
    if obs_flow_idx >= n_features:
        raise ValueError(
            f"obs_flow_idx {obs_flow_idx} >= n_features {n_features}"
        )

    net_rain = inputs[:, :, net_rain_idx]  # [time, basin]
    obs_flow = inputs[:, :, obs_flow_idx]  # [time, basin]

    # Apply warmup period to categorization data (but not to simulation)
    if warmup_length > 0:
        obs_flow_for_categorization = obs_flow[warmup_length:]
    else:
        obs_flow_for_categorization = obs_flow

    # Determine flood category based on peak flow
    categorization_method = kwargs.get(
        "categorization_method", "peak_magnitude"
    )
    simulated_flows = np.zeros((time_steps, n_basins))

    for basin_idx in range(n_basins):
        basin_obs_flow = obs_flow_for_categorization[:, basin_idx]
        basin_net_rain = net_rain[:, basin_idx]

        # Categorize based on peak flow
        peak_flow = np.max(basin_obs_flow)

        # Default thresholds if not provided
        thresholds = params.get(
            "thresholds", {"small_medium": 10.0, "medium_large": 25.0}
        )

        if peak_flow < thresholds.get("small_medium", 10.0):
            category = "small"
        elif peak_flow < thresholds.get("medium_large", 25.0):
            category = "medium"
        else:
            category = "large"

        # Get UH parameters for this category
        category_uh_params = params[category]
        if category_uh_params.ndim == 1:
            category_uh_params = category_uh_params.reshape(1, -1)

        # Normalize UH parameters
        category_uh_normalized = (
            category_uh_params[basin_idx] / category_uh_params[basin_idx].sum()
        )

        # Apply unit hydrograph convolution
        basin_flow = uh_conv(
            basin_net_rain, category_uh_normalized, truncate=True
        )
        simulated_flows[:, basin_idx] = basin_flow

    # Apply warmup period to output
    if warmup_length > 0:
        simulated_flows = simulated_flows[warmup_length:]

    # Prepare state information if requested
    if return_state:
        state_dict = {
            "categories_used": {
                basin_idx: category for basin_idx in range(n_basins)
            },
            "thresholds": thresholds,
            "warmup_length": warmup_length,
            "model_type": "categorized_unit_hydrograph",
            "uh_params_by_category": {
                cat: params[cat] for cat in required_categories
            },
        }
        return simulated_flows, state_dict

    return simulated_flows
