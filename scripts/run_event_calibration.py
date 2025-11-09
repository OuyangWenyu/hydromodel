r"""
Author: Wenyu Ouyang
Date: 2025-08-30
LastEditTime: 2025-08-30 16:00:00
LastEditors: Wenyu Ouyang
Description: Event-based XAJ model calibration script using songliaorrevent dataset
FilePath: \hydromodel\scripts\run_event_calibration.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import argparse
import sys
import os
from pathlib import Path

# Add hydromodel to path
repo_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_path)

from hydromodel import SETTING
from hydromodel.trainers.unified_calibrate import calibrate  # noqa: E402
from hydromodel.configs.config_manager import (  # noqa: E402
    setup_configuration_from_args,
    load_simplified_config,
    validate_and_show_config,
    save_config_to_file,
)

def parse_arguments():
    """解析命令行参数 - 专门针对事件率定"""
    parser = argparse.ArgumentParser(
        description="事件率定脚本 - 使用songliaorrevent数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
事件率定配置文件格式（四个要素）:
  data:     # 数据配置
    dataset: "floodevent"              # 数据集类型（洪水事件数据）
    dataset_name: "songliaorrevent"    # 数据集名称
    path: null                         # 数据路径（null则从hydro_setting.yml读取）
    basin_ids: ["songliao_21401550"]   # 流域ID列表
    variables: ["rain", "ES", "inflow", "flood_event"]  # 变量（含flood_event）
    time_unit: ["3h"]                  # 时间单位
    is_event_data: true                # 是否为事件数据
    warmup_length: 360                 # 预热期
    train_period: ["1984-01-01", "2005-12-31"]  # 训练期
    test_period: ["2006-01-01", "2023-12-31"]   # 测试期
    output_dir: "results/event_calibration"     # 结果目录
  
  model:    # 模型配置
    name: "xaj_mz"                 # 模型类型
    params:                        # 模型参数
      source_type: "sources"
      source_book: "HF"
      kernel_size: 15
  
  training: # 训练配置
    algorithm: "SCE_UA"            # 算法类型（SCE_UA/GA/scipy）
    SCE_UA:                        # 对应算法的参数
      rep: 5000
      ngs: 1000
    loss: "RMSE"                   # 损失函数
  
  evaluation: # 评估配置  
    metrics: ["NSE", "KGE", "RMSE"]  # 评估指标

使用示例:
  # 使用默认配置运行songliao_21401550
  python run_event_calibration.py --default
  
  # 使用自定义配置文件
  python run_event_calibration.py --config event_config.yaml
  
  # 只验证配置
  python run_event_calibration.py --default --dry-run
        """,
    )

    # 核心参数
    parser.add_argument(
        "--config",
        type=str,
        help="事件率定配置文件路径（YAML格式）",
    )

    parser.add_argument(
        "--default",
        action="store_true",
        help="使用默认配置运行songliao_21401550流域的事件率定",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只验证配置，不执行率定",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="覆盖配置中的输出目录",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        help="覆盖配置中的实验名称",
    )

    parser.add_argument(
        "--save-config",
        action="store_true",
        help="运行后保存配置文件",
    )

    # 数据相关参数 - 使用setup_configuration_from_args期望的参数名
    parser.add_argument(
        "--data-source-type",
        type=str,
        default="floodevent",
        help="数据源类型 (默认: floodevent - 洪水事件数据)",
    )

    parser.add_argument(
        "--basin-ids",
        type=str,
        nargs="+",
        default=["songliao_21401550"],
        help="流域ID列表 (默认: songliao_21401550)",
    )

    parser.add_argument(
        "--warmup-length",
        type=int,
        default=360,
        help="预热期长度 (默认: 360天)",
    )

    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=["rain", "ES", "inflow", "flood_event"],
        help="变量列表",
    )

    parser.add_argument(
        "--time-unit",
        type=str,
        default="3h",
        help="时间单位 (默认: 3h)",
    )

    parser.add_argument(
        "--is-event",
        action="store_true",
        default=True,
        help="是否为场次计算 (默认: True)",
    )

    # 模型参数 - 使用setup_configuration_from_args期望的参数名
    parser.add_argument(
        "--model-type",
        type=str,
        default="xaj_mz",
        help="模型类型 (默认: xaj_mz)",
    )

    parser.add_argument(
        "--source-type",
        type=str,
        default="sources",
        help="模型source_type (默认: sources)",
    )

    parser.add_argument(
        "--source-book",
        type=str,
        default="HF",
        help="模型source_book (默认: HF)",
    )

    parser.add_argument(
        "--kernel-size",
        type=int,
        default=15,
        help="模型kernel_size (默认: 15)",
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="SCE_UA",
        choices=["SCE_UA", "GA", "scipy"],
        help="率定算法 (默认: SCE_UA)",
    )

    parser.add_argument(
        "--obj-func",
        type=str,
        default="RMSE",
        help="目标函数 (默认: RMSE)",
    )

    parser.add_argument(
        "--rep",
        type=int,
        default=5000,
        help="SCE_UA rep参数 (默认: 5000)",
    )

    parser.add_argument(
        "--ngs",
        type=int,
        default=1000,
        help="SCE_UA ngs参数 (默认: 1000)",
    )

    return parser.parse_args()


def main():
    """主执行函数 - 事件率定版本"""
    args = parse_arguments()

    try:
        # Support two modes: config file or command-line args
        if args.config:
            # Mode 1: Load from config file
            if not os.path.exists(args.config):
                print(f"❌ 配置文件不存在: {args.config}")
                return 1
            print("📄 Loading from configuration file...")
            config = load_simplified_config(args.config)
        elif args.default:
            # Mode 2: Use default command-line configuration
            print("📋 Using default configuration for event calibration...")
            config = setup_configuration_from_args(args)
        else:
            print("❌ 请指定 --config 或 --default 参数")
            print("   示例: python run_event_calibration.py --default")
            print("   示例: python run_event_calibration.py --config event_config.yaml")
            return 1

        if config is None:
            print("❌ 配置创建失败")
            return 1

        # Override config with command-line arguments (if provided)
        if args.output_dir:
            config["training_cfgs"]["output_dir"] = args.output_dir
        if args.experiment_name:
            config["training_cfgs"]["experiment_name"] = args.experiment_name

        # Set save_config flag
        config["training_cfgs"]["save_config"] = args.save_config

        # 验证配置
        if not validate_and_show_config(config, True, "Event-based XAJ Model"):
            return 1

        if args.dry_run:
            print("✅ 事件率定配置验证完成")
            return 0

        # 执行率定
        print("🚀 开始事件率定...")
        results = calibrate(config)

        print("✅ 事件率定完成")
        return 0

    except KeyboardInterrupt:
        print("❌ 率定被用户中断")
        return 1
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
