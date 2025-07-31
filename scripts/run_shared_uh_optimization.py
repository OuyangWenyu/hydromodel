"""
Author: Zheng Zhang, supervised by Heng Lv
Date: 2025-07-08 18:05:00
LastEditTime: 2025-07-16 16:36:39
LastEditors: Wenyu Ouyang
Description: 使用洪水事件数据生成唯一的共享单位线的执行脚本（支持CSV和Excel数据源）
FilePath: \hydromodel_dev\scripts\run_shared_uh_optimization.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import sys
import os
import argparse
from hydrodatasource.configs.config import SETTING
from hydromodel_dev.floodevent import check_event_data_nan
from hydromodel_dev.unit_hydrograph import optimize_shared_unit_hydrograph
from plot_rrevents import plot_unit_hydrograph

# 添加项目根目录到Python路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from hydromodel_dev import (
    setup_matplotlib,
    load_and_preprocess_events_unified,
    evaluate_single_event,
    save_results_to_csv,
    print_report_preview,
)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="共享单位线优化工具 - 松辽河流域数据专用",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理碧流河站点数据
  python run_shared_uh_optimization.py --station-id songliao_21401550
        """,
    )

    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default=os.path.join(
            SETTING["local_data_path"]["datasets-interim"], "songliaorrevent"
        ),
        help="场次数据文件夹路径",
    )

    parser.add_argument(
        "--station-id",
        type=str,
        default="songliao_21401550",
        help="站点ID (如: songliao_21401550)",
    )

    parser.add_argument(
        "--output-dir", "-o", type=str, default="results/", help="输出结果目录"
    )

    parser.add_argument(
        "--common-n-uh",
        type=int,
        default=24,
        help="共享单位线长度 (默认: 24)",
    )

    parser.add_argument(
        "--smoothing-factor",
        type=float,
        default=0.1,
        help="平滑性惩罚权重因子 (默认: 0.1)",
    )

    parser.add_argument(
        "--peak-violation-weight",
        type=float,
        default=10000.0,
        help="单峰违反惩罚权重因子 (默认: 10000.0)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=500,
        help="优化最大迭代次数 (默认: 500)",
    )

    parser.add_argument(
        "--no-peak-obs", action="store_true", help="不包含洪峰观测值"
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="静默模式，减少输出信息"
    )

    return parser.parse_args()


def main():
    """共享单位线优化主函数"""
    # 解析命令行参数
    args = parse_arguments()

    # 初始化图表设置
    setup_matplotlib()

    # 1. 数据加载和预处理
    verbose = not args.quiet
    include_peak_obs = not args.no_peak_obs
    if verbose:
        print("=" * 60)
        print("🚀 松辽流域单位线优化工具启动")
        print("=" * 60)
        print(f"📁 数据路径: {args.data_path}")
        if args.station_id:
            print(f"🏭 指定站点: {args.station_id}")
        print(f"📤 输出目录: {args.output_dir}")
        print(f"⚙️ 平滑因子: {args.smoothing_factor}")
        print(f"⚙️ 单峰惩罚因子: {args.peak_violation_weight}")
        print(f"🔄 最大迭代次数: {args.max_iterations}")
        print(f"📈 包含洪峰观测值: {include_peak_obs}")
        print("-" * 60)
    all_event_data = load_and_preprocess_events_unified(
        data_dir=args.data_path,
        station_id=args.station_id,
        include_peak_obs=include_peak_obs,
        verbose=verbose,
    )
    check_event_data_nan(all_event_data)

    # 2. 优化参数
    common_n_uh = args.common_n_uh
    smoothing_factor = args.smoothing_factor
    peak_violation_weight = args.peak_violation_weight
    apply_peak_penalty = common_n_uh > 2  # 是否应用单峰惩罚（长度>2时）

    if verbose:
        print(
            f"\n🚀 开始使用 {len(all_event_data)} 场洪水数据优化共享单位线..."
        )
        print(
            f"⚙️ 平滑因子: {smoothing_factor}, 单峰惩罚因子: {peak_violation_weight if apply_peak_penalty else 'N/A'}"
        )

    # 执行优化（调用公用函数）
    U_optimized_shared = optimize_shared_unit_hydrograph(
        all_event_data,
        common_n_uh,
        smoothing_factor,
        peak_violation_weight,
        apply_peak_penalty,
        max_iterations=args.max_iterations,
        verbose=verbose,
    )

    if U_optimized_shared is None:
        print("❌ 共享单位线优化失败，程序终止。")
        return
    if verbose:
        print("\n✅ 共享单位线优化完成！")

    # 3. 绘制共享单位线图
    if verbose:
        plot_unit_hydrograph(
            U_optimized_shared,
            "共享优化单位线",
            smoothing_factor,
            peak_violation_weight if apply_peak_penalty else None,
        )

    # 4. 评估共享单位线在所有事件上的表现
    if verbose:
        print("\n📈 正在评估共享单位线性能...")
    final_report_data = []

    for event_data in all_event_data:
        result = evaluate_single_event(event_data, U_optimized_shared)
        final_report_data.append(result)

    # 5. 保存和显示结果
    # 生成松辽河数据输出文件名
    station_suffix = f"_{args.station_id}" if args.station_id else ""
    output_filename = os.path.join(
        args.output_dir,
        f"UH_shared_eva_output_songliao{station_suffix}.csv",
    )
    data_source_label = "松辽河数据源"

    report_df_sorted = save_results_to_csv(
        final_report_data, output_filename, sort_columns=["NSE"]
    )

    if report_df_sorted is not None and verbose:
        # 打印报告预览
        print_report_preview(
            report_df_sorted,
            f"共享单位线评估报告预览 ({data_source_label}, 按NSE排序)",
        )

        # 打印最终统计信息
        best_nse = report_df_sorted["NSE"].max()
        print(f"\n🎯 优化完成！")
        print(f"   数据源: {data_source_label}")
        if args.station_id:
            print(f"   处理站点: {args.station_id}")
        print(f"   共享单位线长度: {common_n_uh}")
        print(f"   最优NSE: {best_nse:.4f}")
        print(f"   结果保存至: {output_filename}")

    if not verbose:
        # 静默模式下也输出关键信息
        best_nse = (
            report_df_sorted["NSE"].max()
            if report_df_sorted is not None
            else 0
        )
        print(
            f"优化完成 - 单位线长度: {common_n_uh}, 最优NSE: {best_nse:.4f}, 输出: {output_filename}"
        )


if __name__ == "__main__":
    main()
