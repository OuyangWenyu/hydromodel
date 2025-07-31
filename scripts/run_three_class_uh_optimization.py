"""
Author: Zheng Zhang, supervised by Heng Lv
Date: 2025-07-08 17:56:32
LastEditTime: 2025-07-16 16:31:53
LastEditors: Wenyu Ouyang
Description: 三类别单位线优化脚本（支持CSV和Excel数据源）-- 将洪水数据根据其洪峰大小分为三类（小、中、大），分别推求特征单位线
FilePath: \hydromodel_dev\scripts\run_three_class_uh_optimization.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import sys
import os
import argparse
import json
from hydrodatasource.configs.config import SETTING
from plot_rrevents import plot_unit_hydrograph

# 添加项目根目录到Python路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from hydromodel_dev import (
    optimize_uh_for_group,
    setup_matplotlib,
    load_and_preprocess_events_unified,
    categorize_floods_by_peak,
    evaluate_single_event,
    save_results_to_csv,
    print_report_preview,
    print_category_statistics,
)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="三类别单位线优化工具 - 支持CSV和Excel数据源",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""songliaorrevent数据集加载数据""",
    )

    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default=os.path.join(
            SETTING["local_data_path"]["datasets-interim"], "songliaorrevent"
        ),
        help="松辽场次数据集文件夹路径",
    )

    parser.add_argument(
        "--station-id",
        type=str,
        default="songliao_21401550",
        help="松辽场次数据集站点ID (如: songliao_21401550)，仅对松辽场次数据集有效",
    )

    parser.add_argument(
        "--output-dir", "-o", type=str, default="results/", help="输出结果目录"
    )

    parser.add_argument(
        "--category-weights",
        type=str,
        default="default",
        help="分类权重方案: default, balanced, aggressive",
    )

    parser.add_argument(
        "--uh-lengths",
        type=str,
        default='{"small":8,"medium":16,"large":24}',
        help='各类别单位线长度，JSON格式，如: \'{"small":8,"medium":16,"large":24}\'',
    )

    # parser.add_argument(
    #     "--common-n-uh",
    #     type=int,
    #     default=24,
    #     help="共享单位线长度 (默认: 24)",
    # )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="静默模式，减少输出信息"
    )

    return parser.parse_args()


def validate_data_path(data_path):
    """验证松辽河数据路径的有效性"""
    if not os.path.exists(data_path):
        print(f"❌ 数据路径不存在: {data_path}")
        return False

    if not os.path.isdir(data_path):
        print(f"❌ 松辽河数据源需要文件夹路径: {data_path}")
        return False

    # 检查是否包含松辽河数据文件
    try:
        csv_files = [
            f
            for f in os.listdir(data_path)
            if f.startswith("songliao_") and f.endswith(".csv")
        ]
        if not csv_files:
            print(f"❌ 文件夹中未找到松辽河数据文件: {data_path}")
            return False
    except Exception as e:
        print(f"❌ 无法访问数据文件夹: {str(e)}")
        return False

    return True


def get_category_weights(scheme_name):
    """获取分类权重方案"""
    schemes = {
        "default": {
            "small": {"smoothing_factor": 0.1, "peak_violation_weight": 100.0},
            "medium": {
                "smoothing_factor": 0.5,
                "peak_violation_weight": 500.0,
            },
            "large": {
                "smoothing_factor": 1.0,
                "peak_violation_weight": 1000.0,
            },
        },
        "balanced": {
            "small": {"smoothing_factor": 0.2, "peak_violation_weight": 200.0},
            "medium": {
                "smoothing_factor": 0.2,
                "peak_violation_weight": 200.0,
            },
            "large": {"smoothing_factor": 0.2, "peak_violation_weight": 200.0},
        },
        "aggressive": {
            "small": {"smoothing_factor": 0.05, "peak_violation_weight": 50.0},
            "medium": {
                "smoothing_factor": 0.1,
                "peak_violation_weight": 100.0,
            },
            "large": {
                "smoothing_factor": 0.5,
                "peak_violation_weight": 2000.0,
            },
        },
    }
    return schemes.get(scheme_name, schemes["default"])


def main():
    """三类别单位线优化主函数"""
    # 解析命令行参数
    args = parse_arguments()

    # 初始化图表设置
    setup_matplotlib()
    # 1. 数据加载和预处理
    verbose = not args.quiet
    if verbose:
        print("=" * 60)
        print("🚀 三类别单位线优化工具启动")
        print("=" * 60)
        print(f"📁 数据路径: {args.data_path}")
        print(f"📤 输出目录: {args.output_dir}")
        print(f"⚙️ 权重方案: {args.category_weights}")
        print("-" * 60)
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    all_events_data = load_and_preprocess_events_unified(
        data_dir=args.data_path,
        station_id=args.station_id,
        include_peak_obs=True,  # 三类别分析需要洪峰观测值
        verbose=verbose,
    )
    if all_events_data is None:
        return

    # 2. 洪水事件分类（基于洪峰）
    categorized_events, (threshold_low, threshold_high) = (
        categorize_floods_by_peak(all_events_data)
    )
    if categorized_events is None:
        return

    if verbose:
        print(
            f"\n📊 洪峰分类阈值：小洪水 ≤ {threshold_low:.2f} mm/3h < "
            f"中洪水 ≤ {threshold_high:.2f} mm/3h < 大洪水"
        )
        print("📈 各类别洪水事件数量:")
        for category, events in categorized_events.items():
            print(f"  🏷️ {category.capitalize()} 洪水: {len(events)} 场")

    # 3. 为每个类别推求特征单位线
    category_weights = get_category_weights(args.category_weights)

    # 解析单位线长度参数
    try:
        uh_length_by_category = json.loads(args.uh_lengths)
    except Exception as e:
        print("❌ 单位线长度参数解析失败: {}".format(e))
        return

    optimized_uhs = {}
    if verbose:
        print("\n🚀 开始为各类别推求特征单位线...")

    for category_name, events in categorized_events.items():
        weights = category_weights.get(category_name, {})
        n_uh = uh_length_by_category.get(category_name, 24)  # 默认24
        optimized_uhs[category_name] = optimize_uh_for_group(
            events, category_name, weights, n_uh
        )

    # 4. 绘制每个类别的特征单位线
    if verbose:
        print("\n📊 绘制各类别特征单位线...")
        for category_name, U_optimized_cat in optimized_uhs.items():
            if U_optimized_cat is not None:
                plot_unit_hydrograph(
                    U_optimized_cat,
                    f"特征单位线 - 类别: {category_name.capitalize()}",
                )
            else:
                print(f"⚠️ 类别 '{category_name}' 的单位线优化失败，跳过绘图")

    # 5. 评估并整合所有结果
    if verbose:
        print("\n📈 开始评估各类别单位线性能...")
    final_report_data = []

    for category_name, events_in_category in categorized_events.items():
        U_optimized_cat = optimized_uhs.get(category_name)

        # 使用该类别的特征单位线评估其内部所有事件
        for event_data in events_in_category:
            result = evaluate_single_event(
                event_data, U_optimized_cat, category_name
            )
            final_report_data.append(result)

    # 6. 保存和显示结果
    if final_report_data:
        # 生成松辽河数据输出文件名
        station_suffix = f"_{args.station_id}" if args.station_id else ""
        data_source_suffix = f"songliao{station_suffix}"
        output_filename = os.path.join(
            args.output_dir,
            f"UH_categorized_eva_output_{data_source_suffix}.csv",
        )

        report_df_sorted = save_results_to_csv(
            final_report_data,
            output_filename,
            sort_columns=["所属类别", "NSE"],
        )

        if report_df_sorted is not None and verbose:
            # 打印报告预览
            print_report_preview(
                report_df_sorted, "分类评估报告预览 (按类别和NSE排序)"
            )

            # 打印各类别统计信息
            print_category_statistics(report_df_sorted)

            # 打印最终统计信息
            best_nse_by_category = report_df_sorted.groupby("所属类别")[
                "NSE"
            ].max()
            print("\n🎯 优化完成！")
            print(f"   数据源: 松辽河流域数据")
            if args.station_id:
                print(f"   处理站点: {args.station_id}")
            print(f"   权重方案: {args.category_weights}")
            print("   各类别最优NSE:")
            for category, nse in best_nse_by_category.items():
                print(f"     {category}: {nse:.4f}")
            print(f"   结果保存至: {output_filename}")
    else:
        print("\n❌ 没有生成任何评估结果。")

    if not verbose:
        # 静默模式下也输出关键信息
        best_nse = (
            report_df_sorted["NSE"].max()
            if report_df_sorted is not None
            else 0
        )
        print(f"优化完成 - 最优NSE: {best_nse:.4f}, 输出: {output_filename}")


if __name__ == "__main__":
    main()
