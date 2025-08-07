"""
Author: Wenyu Ouyang
Date: 2025-08-06
LastEditTime: 2025-08-07 11:01:36
LastEditors: Wenyu Ouyang
Description: 使用统一calibrate()接口的分类单位线优化脚本 - 展示统一接口的灵活性
FilePath: \hydromodel\scripts\run_categorized_uh_optimization_unified.py

This script demonstrates the unified calibration interface flexibility by
using the same calibrate() function for categorized unit hydrograph models.
The unified interface provides consistent behavior across all model types.
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import argparse
import json
from hydroutils.hydro_plot import (
    plot_unit_hydrograph,
    setup_matplotlib_chinese,
)
from hydrodatasource.configs.config import SETTING
from hydrodatasource.reader.floodevent import (
    FloodEventDatasource,
)
from hydromodel.models.unit_hydrograph import (
    evaluate_single_event_from_uh,
    print_report_preview,
    save_results_to_csv,
    print_category_statistics,
    categorize_floods_by_peak,
)
from hydromodel.trainers.unified_calibrate import calibrate


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="统一接口分类单位线优化工具 - 松辽河流域数据专用",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用scipy优化处理碧流河站点数据
  python run_categorized_uh_optimization_unified.py --station-id songliao_21401550 --algorithm scipy_minimize
  
  # 使用遗传算法
  python run_categorized_uh_optimization_unified.py --station-id songliao_21401550 --algorithm genetic_algorithm --pop-size 100
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

    # 分类相关参数
    parser.add_argument(
        "--category-weights",
        type=str,
        default="default",
        choices=["default", "balanced", "aggressive"],
        help="分类权重方案 (默认: default)",
    )

    parser.add_argument(
        "--uh-lengths",
        type=str,
        default='{"small":8,"medium":16,"large":24}',
        help='各类别单位线长度，JSON格式 (默认: {"small":8,"medium":16,"large":24})',
    )

    parser.add_argument(
        "--warmup-length",
        type=int,
        default=8
        * 60,  # 8 hours * 60 minutes / 3 hours = 160 steps for 3h data
        help="预热期长度（步数）(默认: 160步，对应8小时)",
    )

    # 算法选择参数
    parser.add_argument(
        "--algorithm",
        type=str,
        default="scipy_minimize",
        choices=["scipy_minimize", "SCE_UA", "genetic_algorithm"],
        help="优化算法选择 (默认: scipy_minimize)",
    )

    # scipy优化参数
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=500,
        help="scipy优化最大迭代次数 (默认: 500)",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="SLSQP",
        help="scipy优化方法 (默认: SLSQP)",
    )

    # SCE-UA参数
    parser.add_argument(
        "--rep",
        type=int,
        default=1000,
        help="SCE-UA算法repetitions (默认: 1000)",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=1234,
        help="随机种子 (默认: 1234)",
    )

    # 遗传算法参数
    parser.add_argument(
        "--pop-size",
        type=int,
        default=80,
        help="遗传算法种群大小 (默认: 80)",
    )

    parser.add_argument(
        "--n-generations",
        type=int,
        default=50,
        help="遗传算法进化代数 (默认: 50)",
    )

    parser.add_argument(
        "--cx-prob",
        type=float,
        default=0.7,
        help="遗传算法交叉概率 (默认: 0.7)",
    )

    parser.add_argument(
        "--mut-prob",
        type=float,
        default=0.2,
        help="遗传算法变异概率 (默认: 0.2)",
    )

    parser.add_argument(
        "--save-freq",
        type=int,
        default=5,
        help="遗传算法保存频率（每几代保存一次） (默认: 5)",
    )

    parser.add_argument(
        "--no-peak-obs", action="store_true", help="不包含洪峰观测值"
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="静默模式，减少输出信息"
    )

    return parser.parse_args()


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


def create_model_config(args):
    """创建模型配置"""
    try:
        uh_lengths = json.loads(args.uh_lengths)
    except Exception as e:
        print(f"❌ 单位线长度参数解析失败: {e}")
        raise

    return {
        "name": "categorized_unit_hydrograph",
        "category_weights": get_category_weights(args.category_weights),
        "uh_lengths": uh_lengths,
        "net_rain_name": "P_eff",
        "obs_flow_name": "Q_obs_eff",
    }


def create_algorithm_config(args):
    """创建算法配置"""
    if args.algorithm == "scipy_minimize":
        return {
            "name": "scipy_minimize",
            "method": args.method,
            "max_iterations": args.max_iterations,
        }
    elif args.algorithm == "SCE_UA":
        return {
            "name": "SCE_UA",
            "rep": args.rep,
            "random_seed": args.random_seed,
        }
    elif args.algorithm == "genetic_algorithm":
        return {
            "name": "genetic_algorithm",
            "random_seed": args.random_seed,
            "pop_size": args.pop_size,
            "n_generations": args.n_generations,
            "cx_prob": args.cx_prob,
            "mut_prob": args.mut_prob,
            "save_freq": args.save_freq,
        }
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")


def main():
    """统一接口分类单位线优化主函数"""
    # 解析命令行参数
    args = parse_arguments()

    # 初始化图表设置
    setup_matplotlib_chinese()

    # 1. 数据加载和预处理
    verbose = not args.quiet
    include_peak_obs = not args.no_peak_obs

    if verbose:
        print("=" * 60)
        print("🚀 统一接口分类单位线优化工具启动")
        print("=" * 60)
        print(f"📁 数据路径: {args.data_path}")
        if args.station_id:
            print(f"🏭 指定站点: {args.station_id}")
        print(f"📤 输出目录: {args.output_dir}")
        print(f"🔧 优化算法: {args.algorithm}")
        print(f"⏱️ 预热期长度: {args.warmup_length} 步")
        print(f"⚙️ 分类权重方案: {args.category_weights}")
        print(f"📏 单位线长度配置: {args.uh_lengths}")
        print(f"📈 包含洪峰观测值: {include_peak_obs}")
        print("-" * 60)

    # 创建数据源，加载带预热期的数据
    dataset = FloodEventDatasource(
        args.data_path,
        time_unit=["3h"],
        trange4cache=["1960-01-01 02", "2024-12-31 23"],
        warmup_length=args.warmup_length,  # 数据源提供带预热期的数据
    )

    all_event_data = dataset.load_1basin_flood_events(
        station_id=args.station_id,
        flow_unit="mm/3h",
        include_peak_obs=include_peak_obs,  # 分类需要洪峰观测值
        verbose=verbose,
    )

    dataset.check_event_data_nan(all_event_data)

    if verbose:
        print(f"✅ 成功加载 {len(all_event_data)} 场洪水数据（含预热期）")

    # 2. 创建配置
    model_config = create_model_config(args)
    algorithm_config = create_algorithm_config(args)

    if verbose:
        print(f"\n🚀 开始使用统一接口优化分类单位线...")
        print(f"📊 模型配置: {model_config['name']}")
        print(f"🔧 算法配置: {algorithm_config}")
        print(f"📏 单位线长度: {model_config['uh_lengths']}")

    # 3. 执行优化（使用统一接口）
    results = calibrate(
        data=all_event_data,
        model_config=model_config,
        algorithm_config=algorithm_config,
        loss_config={"type": "time_series", "obj_func": "RMSE"},
        output_dir=args.output_dir,
        warmup_length=args.warmup_length,  # 统一接口会处理预热期
    )

    # 4. 检查优化结果
    if results["convergence"] != "success" or results["best_params"] is None:
        print("❌ 分类单位线优化失败，程序终止。")
        print(f"优化结果: {results}")
        return

    if verbose:
        print("\n✅ 分类单位线优化完成！")
        print(f"🎯 最优目标函数值: {results['objective_value']:.6f}")

        # 显示分类信息
        cat_info = results.get("categorization_info", {})
        print(f"📊 分类信息:")
        print(f"   分类阈值: {cat_info.get('thresholds', 'N/A')}")
        for category, count in cat_info.get("events_per_category", {}).items():
            uh_length = cat_info.get("uh_lengths", {}).get(category, 0)
            print(
                f"   {category.capitalize()}: {count} 场事件, UH长度: {uh_length}"
            )

    # 5. 提取各类别优化的单位线参数
    best_uh_by_category = {}
    if results["best_params"]:
        categorized_params = results["best_params"][
            "categorized_unit_hydrograph"
        ]
        for category, category_params in categorized_params.items():
            # 转换为列表形式
            param_values = []
            for i in range(len(category_params)):
                param_name = f"uh_{category}_{i+1}"
                if param_name in category_params:
                    param_values.append(category_params[param_name])
            best_uh_by_category[category] = param_values

            if verbose:
                print(
                    f"✅ {category.capitalize()}类单位线: {len(param_values)} 个参数"
                )

    # 6. 绘制各类别单位线图
    if verbose and best_uh_by_category:
        print("\n📊 绘制各类别单位线...")
        for category, uh_params in best_uh_by_category.items():
            if uh_params:
                plot_unit_hydrograph(
                    uh_params,
                    f"统一接口优化 - {category.capitalize()}类单位线",
                )

    # 7. 评估各类别单位线性能
    # 我们需要手动进行分类和评估，因为评估函数需要分类信息
    if verbose:
        print("\n📈 正在评估各类别单位线性能...")

    # 获取分类信息

    # 处理事件数据（移除预热期用于评估）
    processed_events_for_eval = []
    for event_data in all_event_data:
        eval_event_data = {}
        for key, value in event_data.items():
            if key in [
                "P_eff",
                "net_rain",
                "Q_obs_eff",
                "obs_discharge",
            ] and hasattr(value, "__len__"):
                eval_event_data[key] = (
                    value[args.warmup_length :]
                    if args.warmup_length > 0
                    else value
                )
            else:
                eval_event_data[key] = value
        processed_events_for_eval.append(eval_event_data)

    # 分类事件
    categorized_events, _ = categorize_floods_by_peak(
        processed_events_for_eval
    )

    final_report_data = []
    for category_name, events_in_category in categorized_events.items():
        uh_params = best_uh_by_category.get(category_name)

        if uh_params:
            # 使用该类别的特征单位线评估其内部所有事件
            for event_data in events_in_category:
                result = evaluate_single_event_from_uh(
                    event_data, uh_params, category_name
                )
                final_report_data.append(result)

    # 8. 保存和显示结果
    if final_report_data:
        # 生成输出文件名
        station_suffix = f"_{args.station_id}" if args.station_id else ""
        algorithm_suffix = f"_{args.algorithm}"
        output_filename = os.path.join(
            args.output_dir,
            f"UH_categorized_unified_eva_output_songliao{station_suffix}{algorithm_suffix}.csv",
        )

        report_df_sorted = save_results_to_csv(
            final_report_data,
            output_filename,
            sort_columns=["所属类别", "NSE"],
        )

        if verbose:
            print(f"\n💾 详细结果已保存至: {output_filename}")
            print(
                f"📊 JSON结果已保存至: {os.path.join(args.output_dir, 'categorized_unit_hydrograph_calibration_results.json')}"
            )

            # 显示性能统计
            print_report_preview(
                report_df_sorted, "分类评估报告预览 (按类别和NSE排序)"
            )
            print_category_statistics(report_df_sorted)

            # 显示各类别最优NSE
            best_nse_by_category = report_df_sorted.groupby("所属类别")[
                "NSE"
            ].max()
            print("\n🎯 各类别优化完成！")
            print(f"   算法: {args.algorithm}")
            print(f"   权重方案: {args.category_weights}")
            print("   各类别最优NSE:")
            for category, nse in best_nse_by_category.items():
                print(f"     {category}: {nse:.4f}")

    print("\n🎉 统一接口分类单位线优化完成！")


if __name__ == "__main__":
    main()
