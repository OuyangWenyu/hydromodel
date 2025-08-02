"""
Author: Zheng Zhang, supervised by Heng Lv
Date: 2025-07-08 18:30:00
LastEditTime: 2025-08-01 10:46:57
LastEditors: Wenyu Ouyang
Description: 水文模型工具模块 -- 包含脚本中公共功能的工具函数
FilePath: /hydromodel/hydromodel/models/uh_utils.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hydromodel.models.unit_hydrograph import uh_conv
# Use string constants directly instead of importing from consts
NET_RAIN = "P_eff"
OBS_FLOW = "Q_obs_eff"
DELTA_T_SECONDS = 10800.0  # 3.0 * 3600.0


# --- 图表配置 ---
from hydromodel.models.common_utils import setup_matplotlib_chinese


def setup_matplotlib():
    """设置matplotlib的中文字体和样式"""
    setup_matplotlib_chinese()
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "sans-serif"


# --- 评估指标计算功能 ---
def calculate_nse(Q_obs, Q_sim):
    """
    计算Nash-Sutcliffe效率系数

    Args:
        Q_obs: 观测径流
        Q_sim: 模拟径流

    Returns:
        float: NSE值
    """
    if len(Q_obs) == 0 or len(Q_sim) == 0 or len(Q_obs) != len(Q_sim):
        return np.nan

    mean_obs = np.mean(Q_obs)
    den_nse = np.sum((Q_obs - mean_obs) ** 2)

    if den_nse == 0:
        return 1.0 if np.allclose(Q_sim, Q_obs) else -np.inf
    else:
        return 1 - (np.sum((Q_obs - Q_sim) ** 2) / den_nse)


def calculate_volume_error(Q_obs, Q_sim):
    """
    计算洪量相对误差

    Args:
        Q_obs: 观测径流
        Q_sim: 模拟径流

    Returns:
        float: 洪量相对误差(%)
    """
    vol_obs = np.sum(Q_obs) * DELTA_T_SECONDS
    vol_sim = np.sum(Q_sim) * DELTA_T_SECONDS

    if vol_obs > 1e-6:
        return ((vol_sim - vol_obs) / vol_obs) * 100.0
    else:
        return np.nan


def calculate_peak_error(Q_obs, Q_sim):
    """
    计算洪峰相对误差

    Args:
        Q_obs: 观测径流
        Q_sim: 模拟径流

    Returns:
        float: 洪峰相对误差(%)
    """
    peak_obs = np.max(Q_obs)
    peak_sim = np.max(Q_sim)

    if peak_obs > 1e-6:
        return ((peak_sim - peak_obs) / peak_obs) * 100.0
    else:
        return np.nan


def evaluate_single_event(event_data, U_optimized, category_name=None):
    """
    评估单个洪水事件的性能指标

    Args:
        event_data: 事件数据字典
        U_optimized: 优化的单位线参数
        category_name: 类别名称（可选）

    Returns:
        dict: 包含评估结果的字典
    """
    P_event = event_data[NET_RAIN]
    Q_obs_event_full = event_data[OBS_FLOW]
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
            result["NSE"] = calculate_nse(
                Q_obs_event_full, Q_sim_event_compare
            )
            result["洪量相误(%)"] = calculate_volume_error(
                Q_obs_event_full, Q_sim_event_compare
            )
            result["洪峰相误(%)"] = calculate_peak_error(
                Q_obs_event_full, Q_sim_event_compare
            )

    return result


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
        # Use common utility for CSV saving
        from hydromodel.models.common_utils import save_dataframe_to_csv

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


# --- 洪水分类功能 ---
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
