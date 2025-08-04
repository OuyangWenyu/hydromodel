"""
Author: Wenyu Ouyang
Date: 2025-07-08 19:01:27
LastEditTime: 2025-08-04 08:59:00
LastEditors: Wenyu Ouyang
Description: Unit hydrograph functions
FilePath: /hydromodel/hydromodel/models/unit_hydrograph.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import itertools
import logging
import numpy as np
import os
from typing import Optional, List
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import gamma
from hydroutils.hydro_stat import nse, flood_peak_error, flood_volume_error


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
    net_rain_name="P_eff",
    obs_flow_name="Q_obs_eff",
):
    """
    Objective function for multi-event unit hydrograph optimization.

    Parameters
    ----------
    U_params : np.ndarray
        Unit hydrograph parameters (array of length common_n_uh).
    list_of_event_data_for_opt : list of dict
        List of event data dictionaries for optimization. Each dict should contain
    lambda_smooth : float
        Weight for the smoothness penalty term.
    lambda_peak_violation : float
        Weight for the unimodality (single-peak) violation penalty.
    apply_peak_penalty_flag : bool
        Whether to apply the unimodality penalty.
    common_n_uh : int
        Length of the unit hydrograph.
    net_rain_name : str, optional
        Name of the effective rainfall column in event data.
    obs_flow_name : str, optional
        Name of the observed flow column in event data.

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
            event_data[net_rain_name],
            event_data[obs_flow_name],
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


def evaluate_single_event_from_uh(
    event_data,
    U_optimized,
    category_name=None,
    net_rain_name="P_eff",
    obs_flow_name="Q_obs_eff",
):
    """
    评估单个洪水事件的性能指标

    Parameters:
    ----------
        event_data: 事件数据字典
        U_optimized: 优化的单位线参数
        category_name: 类别名称（可选）

    Returns
    -------
        dict: 包含评估结果的字典
    """
    P_event = event_data[net_rain_name]
    Q_obs_event_full = event_data[obs_flow_name]
    event_filename = os.path.basename(event_data["filepath"])

    # 初始化指标
    result = {
        "文件名": event_filename,
        "NSE": np.nan,
        "洪量相误(%)": np.nan,
        "洪峰相误(%)": np.nan,
    }

    # 如果指定了类别，添加到结果中
    if category_name is not None:
        result["所属类别"] = category_name

    if U_optimized is not None:
        Q_sim_event_full = uh_conv(P_event, U_optimized, truncate=False)
        Q_sim_event_compare = Q_sim_event_full[: len(Q_obs_event_full)]

        if len(Q_obs_event_full) > 0 and len(Q_sim_event_compare) == len(
            Q_obs_event_full
        ):
            result["NSE"] = nse(Q_obs_event_full, Q_sim_event_compare)
            result["洪量相误(%)"] = flood_volume_error(
                Q_obs_event_full, Q_sim_event_compare
            )
            result["洪峰相误(%)"] = flood_peak_error(
                Q_obs_event_full, Q_sim_event_compare
            )

    return result


def categorize_floods_by_peak(all_events_data):
    """
    根据洪峰将洪水事件分为三类

    Args:
        all_events_data: 包含peak_obs的事件数据列表

    Returns:
        dict: 分类后的事件字典
        tuple: (threshold_low, threshold_high) 分类阈值
    """
    event_peaks = [
        data["peak_obs"] for data in all_events_data if data["peak_obs"] > 0
    ]

    if not event_peaks:
        print("❌ 没有有效的洪峰数据")
        return None, (None, None)

    threshold_low = np.percentile(event_peaks, 33.3)  # 33.3%分位数
    threshold_high = np.percentile(event_peaks, 66.6)  # 66.6%分位数

    categorized_events = {"small": [], "medium": [], "large": []}

    for event_data in all_events_data:
        peak = event_data["peak_obs"]
        if peak <= threshold_low:
            categorized_events["small"].append(event_data)
        elif peak <= threshold_high:
            categorized_events["medium"].append(event_data)
        else:
            categorized_events["large"].append(event_data)

    return categorized_events, (threshold_low, threshold_high)


# --- 结果保存和输出功能 ---
def save_results_to_csv(report_data, output_filename, sort_columns=None):
    """
    保存结果到CSV文件

    Args:
        report_data: 报告数据列表
        output_filename: 输出文件名
        sort_columns: 排序列名列表

    Returns:
        pd.DataFrame: 排序后的DataFrame
    """
    if not report_data:
        print("❌ 没有数据可以保存")
        return None

    report_df = pd.DataFrame(report_data)

    # 排序
    if sort_columns:
        ascending = [True] * len(sort_columns)  # 默认升序
        if "NSE" in sort_columns:
            # NSE列按降序排列
            nse_idx = sort_columns.index("NSE")
            ascending[nse_idx] = False
        report_df_sorted = report_df.sort_values(
            by=sort_columns, ascending=ascending
        ).reset_index(drop=True)
    else:
        report_df_sorted = report_df.copy()

    # 保存文件
    try:
        save_dataframe_to_csv(
            report_df_sorted,
            output_filename,
            encoding="utf-8-sig",
            float_format="%.4f",
        )
        print(f"\n✅ 评估报告已成功保存到文件: {output_filename}")
    except Exception as e:
        print(f"\n❌ 保存报告到文件失败: {e}")

    return report_df_sorted


def print_report_preview(report_df_sorted, title="评估报告预览"):
    """
    打印报告预览

    Args:
        report_df_sorted: 排序后的DataFrame
        title: 预览标题
    """
    print(f"\n📊 --- {title} ---")
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.width", 120)
    print(report_df_sorted)
    pd.reset_option("display.max_rows")
    pd.reset_option("display.width")


def print_category_statistics(report_df_sorted):
    """
    打印各类别性能统计信息

    Args:
        report_df_sorted: 包含类别信息的DataFrame
    """
    if "所属类别" not in report_df_sorted.columns:
        return

    print("\n📈 --- 各类别性能统计 ---")
    for category in ["small", "medium", "large"]:
        cat_data = report_df_sorted[report_df_sorted["所属类别"] == category]
        if len(cat_data) > 0:
            mean_nse = cat_data["NSE"].mean()
            mean_vol_err = cat_data["洪量相误(%)"].mean()
            mean_peak_err = cat_data["洪峰相误(%)"].mean()
            print(f"🏷️ {category.capitalize()} 类别 ({len(cat_data)} 场):")
            print(f"   平均NSE: {mean_nse:.4f}")
            print(f"   平均洪量误差: {mean_vol_err:.2f}%")
            print(f"   平均洪峰误差: {mean_peak_err:.2f}%")


def save_dataframe_to_csv(
    df: pd.DataFrame,
    filepath: str,
    metadata_lines: Optional[List[str]] = None,
    encoding: str = "utf-8",
    float_format: str = "%.6f",
    **kwargs,
) -> None:
    """
    Save DataFrame to CSV file with optional metadata header.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    filepath : str
        Output file path.
    metadata_lines : list of str, optional
        Optional metadata lines to write before CSV data.
    encoding : str, optional
        File encoding (default is "utf-8").
    float_format : str, optional
        Float formatting string (default is "%.6f").
    **kwargs
        Additional arguments passed to DataFrame.to_csv().
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Default CSV parameters
    csv_kwargs = {
        "index": False,
        "encoding": encoding,
        "float_format": float_format,
        "header": True,
    }
    csv_kwargs.update(kwargs)

    if metadata_lines:
        with open(filepath, "w", encoding=encoding, newline="") as f:
            f.write("\n".join(metadata_lines) + "\n")
            df.to_csv(f, **csv_kwargs)
    else:
        df.to_csv(filepath, **csv_kwargs)
