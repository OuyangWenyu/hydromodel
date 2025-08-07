"""
Author: Wenyu Ouyang
Date: 2025-08-07
LastEditTime: 2025-08-07
LastEditors: Wenyu Ouyang
Description: Unit hydrograph model with unified interface
FilePath: /hydromodel/hydromodel/models/unit_hydrograph.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import logging
import numpy as np
from typing import Union, Dict, Any
from scipy.stats import gamma


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


def print_report_preview(report_df_sorted, title="评估报告预览", top_n=None):
    """
    打印报告预览

    Args:
        report_df_sorted: 排序后的DataFrame
        title: 预览标题
        top_n: 显示前n个最佳事件，如果为None则显示所有事件
    """
    print(f"\n📊 --- {title} ---")
    
    # 根据top_n参数决定显示的数据
    if top_n is not None and top_n > 0:
        display_df = report_df_sorted.head(top_n)
        print(f"显示前 {min(top_n, len(report_df_sorted))} 个最佳事件：")
    else:
        display_df = report_df_sorted
        print(f"显示全部 {len(report_df_sorted)} 个事件：")
    
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.width", 120)
    print(display_df)
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


# =============================================================================
# UNIFIED MODEL INTERFACES
# =============================================================================


def unit_hydrograph(
    inputs: np.ndarray,
    params: np.ndarray,
    warmup_length: int = 0,
    return_state: bool = False,
    **kwargs
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
        raise ValueError(f"Basin dimension mismatch: inputs has {n_basins}, params has {n_basins_params}")
    
    # Extract net rainfall (default: first feature)
    net_rain_idx = kwargs.get('net_rain_idx', 0)
    if net_rain_idx >= n_features:
        raise ValueError(f"net_rain_idx {net_rain_idx} >= n_features {n_features}")
    
    net_rain = inputs[:, :, net_rain_idx]  # [time, basin]
    
    # Normalize unit hydrograph parameters (ensure sum to 1.0)
    params_normalized = params / params.sum(axis=1, keepdims=True)
    
    # Perform convolution for each basin
    truncate = kwargs.get('truncate', True)
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
            'uh_params_normalized': params_normalized,
            'warmup_length': warmup_length,
            'n_uh': n_uh,
            'model_type': 'unit_hydrograph'
        }
        return simulated_flows, state_dict
    
    return simulated_flows


def categorized_unit_hydrograph(
    inputs: np.ndarray,
    params: Dict[str, np.ndarray],
    warmup_length: int = 0,
    return_state: bool = False,
    **kwargs
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
        raise ValueError(f"Categorized UH requires 3D inputs [time, basin, features], got {inputs.ndim}D")
    
    time_steps, n_basins, n_features = inputs.shape
    
    # Validate required parameters
    required_categories = ['small', 'medium', 'large']
    for cat in required_categories:
        if cat not in params:
            raise ValueError(f"Missing UH parameters for category: {cat}")
    
    # Extract input data
    net_rain_idx = kwargs.get('net_rain_idx', 0) 
    obs_flow_idx = kwargs.get('obs_flow_idx', 1)
    
    if net_rain_idx >= n_features:
        raise ValueError(f"net_rain_idx {net_rain_idx} >= n_features {n_features}")
    if obs_flow_idx >= n_features:
        raise ValueError(f"obs_flow_idx {obs_flow_idx} >= n_features {n_features}")
    
    net_rain = inputs[:, :, net_rain_idx]  # [time, basin]
    obs_flow = inputs[:, :, obs_flow_idx]  # [time, basin]
    
    # Apply warmup period to categorization data (but not to simulation)
    if warmup_length > 0:
        obs_flow_for_categorization = obs_flow[warmup_length:]
    else:
        obs_flow_for_categorization = obs_flow
    
    # Determine flood category based on peak flow
    categorization_method = kwargs.get('categorization_method', 'peak_magnitude')
    simulated_flows = np.zeros((time_steps, n_basins))
    
    for basin_idx in range(n_basins):
        basin_obs_flow = obs_flow_for_categorization[:, basin_idx]
        basin_net_rain = net_rain[:, basin_idx]
        
        # Categorize based on peak flow
        peak_flow = np.max(basin_obs_flow)
        
        # Default thresholds if not provided
        thresholds = params.get('thresholds', {'small_medium': 10.0, 'medium_large': 25.0})
        
        if peak_flow < thresholds.get('small_medium', 10.0):
            category = 'small'
        elif peak_flow < thresholds.get('medium_large', 25.0):
            category = 'medium'
        else:
            category = 'large'
        
        # Get UH parameters for this category
        category_uh_params = params[category]
        if category_uh_params.ndim == 1:
            category_uh_params = category_uh_params.reshape(1, -1)
        
        # Normalize UH parameters
        category_uh_normalized = category_uh_params[basin_idx] / category_uh_params[basin_idx].sum()
        
        # Apply unit hydrograph convolution
        basin_flow = uh_conv(basin_net_rain, category_uh_normalized, truncate=True)
        simulated_flows[:, basin_idx] = basin_flow
    
    # Apply warmup period to output
    if warmup_length > 0:
        simulated_flows = simulated_flows[warmup_length:]
    
    # Prepare state information if requested
    if return_state:
        state_dict = {
            'categories_used': {basin_idx: category for basin_idx in range(n_basins)},
            'thresholds': thresholds,
            'warmup_length': warmup_length,
            'model_type': 'categorized_unit_hydrograph',
            'uh_params_by_category': {cat: params[cat] for cat in required_categories}
        }
        return simulated_flows, state_dict
    
    return simulated_flows
