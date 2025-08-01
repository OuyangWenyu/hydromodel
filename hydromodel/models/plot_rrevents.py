from typing import Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
from hydromodel.models.consts import OBS_FLOW, NET_RAIN, DELTA_T_HOURS
from hydrodatasource.configs.config import SETTING
from hydromodel.models.floodevent import (
    calculate_events_characteristics,
    load_and_preprocess_events_unified,
)
from hydromodel.models.common_utils import setup_matplotlib_chinese

setup_matplotlib_chinese()

# --- 新增：为LaTeX数学公式渲染配置字体 ---
# 这可以帮助确保数学符号和中文字体看起来更和谐
plt.rcParams["mathtext.fontset"] = (
    "stix"  # 'stix' 是一种与Times New Roman相似的科学字体
)
plt.rcParams["font.family"] = "sans-serif"  # 保持其他文本为无衬线字体


def plot_event_characteristics(
    event_analysis: Dict,
    output_folder: str,
    delta_t_hours: float = 3.0,
):
    """为单个洪水事件绘制特征图并保存，确保径流曲线在柱状图上层。@author: Zheng Zhang"""
    net_rain = event_analysis[NET_RAIN]
    direct_runoff = event_analysis[OBS_FLOW]
    event_filename = os.path.basename(event_analysis["filepath"])

    fig, ax1 = plt.subplots(figsize=(15, 7))
    fig.suptitle(f"洪水事件特征分析 - {event_filename}", fontsize=16)

    # 绘制径流曲线 (左Y轴)
    x_axis = np.arange(len(direct_runoff))
    # --- 核心修改：为曲线设置一个较高的 zorder ---
    ax1.plot(
        x_axis,
        direct_runoff,
        color="orangered",
        label=r"径流 ($m^3/s$)",
        zorder=10,
        linewidth=2,
    )  # zorder=10

    ax1.set_xlabel(f"时段 (Δt = {delta_t_hours}h)", fontsize=12)
    ax1.set_ylabel(r"径流流量 ($m^3/s$)", color="orangered", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="orangered")
    ax1.set_ylim(bottom=0)
    ax1.grid(
        True, which="major", linestyle="--", linewidth="0.5", color="gray"
    )

    # 创建共享X轴的第二个Y轴
    ax2 = ax1.twinx()
    # 绘制净雨柱状图 (右Y轴，向下)
    # --- 核心修改：为柱状图设置一个较低的 zorder (可选，但好习惯) ---
    ax2.bar(
        x_axis,
        net_rain,
        color="steelblue",
        label="净雨 (mm)",
        width=0.8,
        zorder=5,
    )  # zorder=5

    ax2.set_ylabel("净雨量 (mm)", color="steelblue", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax2.invert_yaxis()
    ax2.set_ylim(top=0)

    # 准备并标注文本框 (与之前对齐版本相同)
    labels = [
        "洪峰流量:",
        "洪   量:",
        "洪水历时:",
        "总净雨量:",
        "洪峰雨峰延迟:",
    ]
    values = [
        f"{event_analysis['peak_obs']:.2f} " + r"$m^3/s$",
        f"{event_analysis['runoff_volume_m3'] / 1e6:.2f} "
        + r"$\times 10^6\ m^3$",
        f"{event_analysis['runoff_duration_hours']:.1f} 小时",
        f"{event_analysis['total_net_rain']:.2f} mm",
        f"{event_analysis['lag_time_hours']:.1f} 小时",
    ]
    label_text = "\n".join(labels)
    value_text = "\n".join(values)
    bbox_props = dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8)
    ax1.text(
        0.85,
        0.95,
        "--- 水文特征 ---",
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="center",
        bbox=bbox_props,
    )
    ax1.text(
        0.80,
        0.85,
        label_text,
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        family="SimSun",
    )
    ax1.text(
        0.82,
        0.85,
        value_text,
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="left",
    )

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.95),
    )

    # 保存图形
    output_filename = os.path.join(
        output_folder, f"{os.path.splitext(event_filename)[0]}.png"
    )
    try:
        plt.savefig(output_filename, dpi=150, bbox_inches="tight")
        print(f"  已生成图表: {output_filename}")
    except Exception as e:
        print(f"  保存图表失败: {output_filename}, 错误: {e}")

    plt.close(fig)


def plot_unit_hydrograph(
    U_optimized, title, smoothing_factor=None, peak_violation_weight=None
):
    """
    绘制单位线图

    Args:
        U_optimized: 优化的单位线参数
        title: 图表标题
        smoothing_factor: 平滑因子（可选）
        peak_violation_weight: 单峰惩罚权重（可选）
    """
    if U_optimized is None:
        print(f"⚠️ 无法绘制单位线：{title} - 优化失败")
        return

    time_axis_uh = np.arange(1, len(U_optimized) + 1) * DELTA_T_HOURS

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis_uh, U_optimized, marker="o", linestyle="-")

    # 构建完整标题
    full_title = title
    if smoothing_factor is not None and peak_violation_weight is not None:
        full_title += (
            f" (平滑={smoothing_factor}, 单峰罚={peak_violation_weight})"
        )

    plt.title(full_title)
    plt.xlabel(f"时间 (小时, Δt={DELTA_T_HOURS}h)")
    plt.ylabel("1mm净雨单位线纵坐标 (mm/3h)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


# --- 主程序 ---
if __name__ == "__main__":
    data_folder = os.path.join(
        SETTING["local_data_path"]["datasets-interim"],
        "songliaorrevent",
    )
    events = load_and_preprocess_events_unified(
        data_dir=data_folder,
        station_id="songliao_21401550",
        include_peak_obs=True,
        verbose=True,
        flow_unit="m^3/s",
    )
    enhanced_events = calculate_events_characteristics(events)
    output_plot_folder = "results/event_characteristic_plots"
    if not os.path.exists(output_plot_folder):
        os.makedirs(output_plot_folder)
        print(f"已创建文件夹: {output_plot_folder}")

    for event in enhanced_events:
        plot_event_characteristics(event, output_plot_folder)

    print(f"\n处理完成！共为 {len(events)} 场有效洪水事件生成了特征分析图。")
