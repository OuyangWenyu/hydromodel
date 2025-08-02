"""
Author: Wenyu Ouyang
Date: 2025-07-08 19:01:27
LastEditTime: 2025-07-31 15:43:08
LastEditors: Wenyu Ouyang
Description: Unit hydrograph functions
FilePath: /hydromodel/hydromodel/models/unit_hydrograph.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import itertools
import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import gamma

# Use string constants directly instead of importing from consts
NET_RAIN = "P_eff"
OBS_FLOW = "Q_obs_eff"


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

    for i, j in itertools.product(range(batch_size), range(feature_size)):
        conv_result = np.convolve(x[:, i, j], uh[:, i, j])
        outputs[:, i, j] = (
            conv_result[:seq_length] if truncate else conv_result
        )
    return outputs


# --- 核心计算函数 ---
def objective_function_multi_event(
    U_params,
    list_of_event_data_for_opt,
    lambda_smooth,
    lambda_peak_violation,
    apply_peak_penalty_flag,
    common_n_uh,
):
    """
    Objective function for multi-event unit hydrograph optimization.

    Parameters
    ----------
    U_params : np.ndarray
        Unit hydrograph parameters (array of length common_n_uh).
    list_of_event_data_for_opt : list of dict
        List of event data dictionaries for optimization. Each dict should contain
        NET_RAIN (effective rainfall) and OBS_FLOW (observed flow).
    lambda_smooth : float
        Weight for the smoothness penalty term.
    lambda_peak_violation : float
        Weight for the unimodality (single-peak) violation penalty.
    apply_peak_penalty_flag : bool
        Whether to apply the unimodality penalty.
    common_n_uh : int
        Length of the unit hydrograph.

    Returns
    -------
    float
        Value of the objective function (to be minimized).
    """
    total_fit_loss = 0  # 总拟合损失
    if len(U_params) != common_n_uh:
        return 1e18
    for event_data in list_of_event_data_for_opt:
        P_event, Q_event_obs = (
            event_data[NET_RAIN],
            event_data[OBS_FLOW],
        )  # 场次降雨和观测径流
        Q_sim_full_event = uh_conv(
            P_event, U_params, truncate=False
        )  # 模拟径流
        Q_sim_compare_event = Q_sim_full_event[
            : len(Q_event_obs)
        ]  # 用于比较的模拟径流
        total_fit_loss += np.sum(
            (Q_sim_compare_event - Q_event_obs) ** 2
        )  # 累加均方误差
    loss_smooth_val = (
        np.sum(np.diff(U_params) ** 2) if len(U_params) > 1 else 0
    )  # 平滑性惩罚项
    peak_violation_penalty_val = 0  # 单峰违反惩罚项
    if apply_peak_penalty_flag and len(U_params) > 2:
        actual_k_peak = np.argmax(U_params)  # 单位线峰值位置
        for j in range(actual_k_peak):
            if U_params[j + 1] < U_params[j] - 1e-6:
                peak_violation_penalty_val += (
                    U_params[j] - U_params[j + 1]
                ) ** 2
        for j in range(actual_k_peak, len(U_params) - 1):
            if U_params[j + 1] > U_params[j] + 1e-6:
                peak_violation_penalty_val += (
                    U_params[j + 1] - U_params[j]
                ) ** 2
    return (
        total_fit_loss
        + lambda_smooth * loss_smooth_val
        + lambda_peak_violation * peak_violation_penalty_val
    )


def optimize_shared_unit_hydrograph(
    all_event_data,
    common_n_uh,
    smoothing_factor,
    peak_violation_weight,
    apply_peak_penalty,
    max_iterations=500,
    verbose=True,
):
    """
    Optimize shared unit hydrograph parameters for multi-event data.

    Parameters
    ----------
    all_event_data : list
        List of event data dictionaries.
    common_n_uh : int
        Length of the unit hydrograph.
    smoothing_factor : float
        Smoothing factor for regularization.
    peak_violation_weight : float
        Weight for peak violation penalty.
    apply_peak_penalty : bool
        Whether to apply peak penalty.
    max_iterations : int, optional
        Maximum number of optimization iterations (default is 500).
    verbose : bool, optional
        Whether to display optimization progress (default is True).

    Returns
    -------
    np.ndarray or None
        Optimized unit hydrograph parameters array, or None if optimization fails.
    """
    U_initial_guess = init_unit_hydrograph(common_n_uh)
    bounds = [(0, 1) for _ in range(common_n_uh)]
    constraints = {"type": "eq", "fun": lambda U: np.sum(U) - 1}
    result = minimize(
        objective_function_multi_event,
        U_initial_guess,
        args=(
            all_event_data,
            smoothing_factor,
            peak_violation_weight,
            apply_peak_penalty,
            common_n_uh,
        ),
        # method="L-BFGS-B",
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"disp": verbose, "maxiter": max_iterations},
    )
    if result.success or result.status in [0, 2]:
        return result.x
    else:
        return None


def optimize_uh_for_group(events_in_group, group_name, weights, n_uh_group):
    """
    Optimize unit hydrograph for a specific event group.

    Parameters
    ----------
    events_in_group : list
        List of events in the group.
    group_name : str
        Name of the group.
    weights : dict
        Weight dictionary containing 'smoothing_factor' and 'peak_violation_weight'.
    n_uh_group : int
        Length of the unit hydrograph.

    Returns
    -------
    np.ndarray or None
        Optimized unit hydrograph parameters, or None if optimization fails.
    """
    print(
        f"\n--- 正在为组 '{group_name}' 优化特征单位线 ({len(events_in_group)} 场) ---"
    )
    if len(events_in_group) < 3:
        print("事件数量过少，跳过优化。")
        return None

    smoothing, peak_penalty = (
        weights["smoothing_factor"],
        weights["peak_violation_weight"],
    )
    apply_penalty = n_uh_group > 2  # 是否应用单峰惩罚
    print(
        f"  使用权重: 平滑={smoothing}, 单峰罚={peak_penalty if apply_penalty else 'N/A'}"
    )
    # Direct optimization logic (previously in _internal_optimize_unit_hydrograph)
    U_initial_guess = init_unit_hydrograph(n_uh_group)
    bounds = [(0, 1) for _ in range(n_uh_group)]
    constraints = {"type": "eq", "fun": lambda U: np.sum(U) - 1}
    result = minimize(
        objective_function_multi_event,
        U_initial_guess,
        args=(
            events_in_group,
            smoothing,
            peak_penalty,
            apply_penalty,
            n_uh_group,
        ),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"disp": False, "maxiter": 500},
    )
    U_optimized = (
        result.x if (result.success or result.status in [0, 2]) else None
    )
    status_message = "成功" if U_optimized is not None else "可能未收敛"
    print(f"  优化{status_message}")
    return U_optimized


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
