"""
Author: Wenyu Ouyang
Date: 2025-08-06
LastEditTime: 2025-08-07 08:42:43
LastEditors: Wenyu Ouyang
Description: This script demonstrates the power of the unified calibration interface by using the general calibrate() function instead of model-specific functions. The same interface works for: Unit hydrograph models, Categorized unit hydrograph models, Traditional hydrological models (XAJ, GR series, etc.), All optimization algorithms (scipy, SCE-UA, genetic algorithms)
FilePath: \hydromodel\scripts\run_shared_uh_optimization_unified.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os
import argparse
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
)
from hydromodel.trainers.unified_calibrate import calibrate


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="统一接口单位线优化工具 - 松辽河流域数据专用",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用scipy优化处理碧流河站点数据
  python run_shared_uh_optimization_unified.py --station-id songliao_21401550 --algorithm scipy_minimize
  
  # 使用SCE-UA算法
  python run_shared_uh_optimization_unified.py --station-id songliao_21401550 --algorithm SCE_UA --rep 2000
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
        default=50,
        help="遗传算法种群大小 (默认: 50)",
    )

    parser.add_argument(
        "--n-generations",
        type=int,
        default=40,
        help="遗传算法进化代数 (默认: 40)",
    )

    parser.add_argument(
        "--cx-prob",
        type=float,
        default=0.5,
        help="遗传算法交叉概率 (默认: 0.5)",
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


def create_model_config(args):
    """创建模型配置"""
    return {
        "name": "unit_hydrograph",
        "n_uh": args.common_n_uh,
        "smoothing_factor": args.smoothing_factor,
        "peak_violation_weight": args.peak_violation_weight,
        "apply_peak_penalty": args.common_n_uh > 2,
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
    """统一接口单位线优化主函数"""
    # 解析命令行参数
    args = parse_arguments()

    # 初始化图表设置
    setup_matplotlib_chinese()

    # 1. 数据加载和预处理
    verbose = not args.quiet
    include_peak_obs = not args.no_peak_obs

    if verbose:
        print("=" * 60)
        print("🚀 统一接口单位线优化工具启动")
        print("=" * 60)
        print(f"📁 数据路径: {args.data_path}")
        if args.station_id:
            print(f"🏭 指定站点: {args.station_id}")
        print(f"📤 输出目录: {args.output_dir}")
        print(f"🔧 优化算法: {args.algorithm}")
        print(f"⏱️ 预热期长度: {args.warmup_length} 步")
        print(f"⚙️ 平滑因子: {args.smoothing_factor}")
        print(f"⚙️ 单峰惩罚因子: {args.peak_violation_weight}")
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
        include_peak_obs=include_peak_obs,
        verbose=verbose,
    )

    dataset.check_event_data_nan(all_event_data)

    if verbose:
        print(f"✅ 成功加载 {len(all_event_data)} 场洪水数据（含预热期）")

    # 2. 创建配置
    model_config = create_model_config(args)
    algorithm_config = create_algorithm_config(args)

    if verbose:
        print(f"\n🚀 开始使用统一接口优化单位线...")
        print(f"✨ 使用统一的 calibrate() 函数 - 一个接口支持所有模型和算法!")
        print(f"📊 模型类型: {model_config['name']}")
        print(f"🔧 算法类型: {algorithm_config['name']}")
        print(f"📈 目标函数: RMSE")
        print(f"🎯 统一接口的优势: 相同的调用方式，一致的返回格式")

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
        print("❌ 单位线优化失败，程序终止。")
        print(f"优化结果: {results}")
        return

    # 提取优化的单位线参数
    uh_params_dict = results["best_params"]["unit_hydrograph"]
    U_optimized_shared = [
        uh_params_dict[f"uh_{i+1}"] for i in range(args.common_n_uh)
    ]

    if verbose:
        print("\n✅ 单位线优化完成！")
        print(f"🎯 最优目标函数值: {results['objective_value']:.6f}")
        print(f"📋 优化参数数量: {len(U_optimized_shared)}")

    # 5. 绘制共享单位线图
    if verbose:
        apply_peak_penalty = args.common_n_uh > 2
        plot_unit_hydrograph(
            U_optimized_shared,
            "统一接口优化单位线",
            args.smoothing_factor,
            args.peak_violation_weight if apply_peak_penalty else None,
        )

    # 6. 评估单位线性能
    # 注意：evaluate_single_event_from_uh 需要使用没有预热期的数据进行评估
    if verbose:
        print("\n📈 正在评估单位线性能...")

    final_report_data = []
    for event_data in all_event_data:
        # 对于评估，我们需要从原始事件数据中移除预热期
        # 因为单位线模型本身不需要预热期
        eval_event_data = {}
        for key, value in event_data.items():
            if key in [
                "P_eff",
                "net_rain",
                "Q_obs_eff",
                "obs_discharge",
            ] and hasattr(value, "__len__"):
                # 移除预热期用于评估
                eval_event_data[key] = (
                    value[args.warmup_length :]
                    if args.warmup_length > 0
                    else value
                )
            else:
                eval_event_data[key] = value

        result = evaluate_single_event_from_uh(
            eval_event_data, U_optimized_shared
        )
        final_report_data.append(result)

    # 7. 保存和显示结果
    # 生成输出文件名
    station_suffix = f"_{args.station_id}" if args.station_id else ""
    algorithm_suffix = f"_{args.algorithm}"
    output_filename = os.path.join(
        args.output_dir,
        f"UH_unified_eva_output_songliao{station_suffix}{algorithm_suffix}.csv",
    )

    report_df_sorted = save_results_to_csv(
        final_report_data, output_filename, sort_columns=["NSE"]
    )

    if verbose:
        print(f"\n💾 详细结果已保存至: {output_filename}")
        print(
            f"📊 JSON结果已保存至: {os.path.join(args.output_dir, 'unit_hydrograph_calibration_results.json')}"
        )

        # 显示性能统计
        print("\n📊 单位线性能统计:")
        print(f"   平均NSE: {report_df_sorted['NSE'].mean():.4f}")
        print(
            f"   平均洪量相误: {report_df_sorted['洪量相误(%)'].mean():.2f}%"
        )
        print(
            f"   平均洪峰相误: {report_df_sorted['洪峰相误(%)'].mean():.2f}%"
        )

        # 显示前几个最佳事件
        print_report_preview(report_df_sorted, "统一接口优化单位线", top_n=5)

    print("\n🎉 统一接口单位线优化完成！")
    print("✅ 成功使用统一的 calibrate() 函数完成优化")
    print("🌟 统一接口的优势:")
    print("   - 一个函数支持所有模型类型")
    print("   - 一致的参数结构和返回格式")
    print("   - 方便的算法切换和比较")
    print("   - 易于扩展新模型和算法")


if __name__ == "__main__":
    main()
