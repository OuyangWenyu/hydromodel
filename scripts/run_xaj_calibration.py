r"""
Author: Wenyu Ouyang
Date: 2025-08-07
LastEditTime: 2025-08-08 19:17:46
LastEditors: Wenyu Ouyang
Description: XAJ model calibration script using the latest unified architecture
FilePath: \hydromodel\scripts\run_xaj_calibration_unified.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import argparse
import sys
import os
from pathlib import Path
import shutil
import yaml

# Add hydromodel to path
repo_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_path)

from hydromodel import SETTING
from hydromodel.trainers.unified_calibrate import calibrate  # noqa: E402
from hydromodel.configs.config_manager import (  # noqa: E402
    setup_configuration_from_args,
    validate_and_show_config,
    save_config_to_file,
)
from hydromodel.models.model_config import MODEL_PARAM_DICT


def load_simplified_config(
    config_path: str = None, simple_config: dict = None
) -> dict:
    """加载简化的配置文件并转换为统一格式"""
    import yaml

    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            simple_config = yaml.safe_load(f)
    elif simple_config is None:
        raise ValueError("必须提供config_path或simple_config参数")

    # 验证简化配置的完整性
    required_sections = ["data", "model", "training", "evaluation"]
    for section in required_sections:
        if section not in simple_config:
            raise ValueError(f"配置缺少必需部分: {section}")

    data_cfg = simple_config["data"]
    model_cfg = simple_config["model"]
    training_cfg = simple_config["training"]
    eval_cfg = simple_config["evaluation"]

    # 转换为统一配置格式
    unified_config = {
        "data_cfgs": {
            "data_source_type": data_cfg["dataset"],
            "data_source_path": data_cfg["path"],
            "dataset_name": data_cfg["dataset"],
            "basin_ids": data_cfg["basin_ids"],
            "variables": data_cfg.get(
                "variables", ["prcp", "PET", "streamflow"]
            ),
            "train_period": data_cfg["train_period"],
            "test_period": data_cfg["test_period"],
            "warmup_length": data_cfg.get("warmup_length", 360),
        },
        "model_cfgs": {
            "model_name": model_cfg["name"],
            **model_cfg.get("params", {}),
        },
        "training_cfgs": {
            "algorithm_name": training_cfg["algorithm"],
            "algorithm_params": training_cfg.get(training_cfg["algorithm"], {}),
            "loss_config": {
                "type": "time_series",
                "obj_func": training_cfg["loss"],
            },
            "output_dir": data_cfg.get("output_dir", "results"),
            "experiment_name": f"{model_cfg['name']}_{training_cfg['algorithm']}",
            "random_seed": 1234,
        },
        "evaluation_cfgs": {
            "metrics": eval_cfg["metrics"],
        },
    }

    # 添加验证期（如果有）
    if "valid_period" in data_cfg:
        unified_config["data_cfgs"]["valid_period"] = data_cfg["valid_period"]

    return unified_config


def parse_arguments():
    """解析命令行参数 - 彻底简化，只保留必要参数"""
    parser = argparse.ArgumentParser(
        description="简化的XAJ模型率定脚本 - 支持四要素配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
配置文件格式（四个要素）:
  data:     # 数据配置
    dataset: "camels"          # 数据集类型
    path: "/path/to/data"      # 数据路径 
    basin_ids: ["01013500"]    # 流域ID列表
    warmup_length: 360         # 预热期
    train_period: ["1990-10-01", "1995-09-30"]  # 训练期
    test_period: ["1995-10-01", "2000-09-30"]   # 测试期
    output_dir: "results"      # 结果目录
  
  model:    # 模型配置
    name: "xaj_mz"             # 模型类型
    params:                    # 模型参数
      source_type: "sources"
      source_book: "HF"
      kernel_size: 15
  
  training: # 训练配置（每次只允许一个算法！）
    algorithm: "SCE_UA"        # 算法类型（SCE_UA/GA/scipy）
    SCE_UA:                    # 对应算法的参数
      rep: 5000
      ngs: 1000
    loss: "RMSE"               # 损失函数
  
  evaluation: # 评估配置  
    metrics: ["NSE", "KGE", "RMSE"]  # 评估指标

使用示例:
  # 使用简化配置文件（推荐）
  python run_xaj_calibration.py --config example_config.yaml
  
  # 快速测试
  python run_xaj_calibration.py --quick-test
        """,
    )

    # 核心参数 - 只保留最必要的
    parser.add_argument(
        "--config",
        type=str,
        help="简化配置文件路径（YAML格式，包含四要素配置）",
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
        default=True,
        help="运行后保存配置文件",
    )

    return parser.parse_args()


def main():
    """主执行函数 - 简化版"""
    args = parse_arguments()

    try:
        # 只支持两种方式：配置文件 或 解析器默认值
        if args.config:
            # 方式1：从简化配置文件加载
            if not os.path.exists(args.config):
                print(f"❌ 配置文件不存在: {args.config}")
                return 1

            config = load_simplified_config(args.config)

        else:
            # 方式2：使用解析器默认值
            config = setup_configuration_from_args(args)

        if config is None:
            print("❌ 配置创建失败")
            return 1

        # 应用命令行覆盖
        if args.output_dir:
            config["training_cfgs"]["output_dir"] = args.output_dir
        if args.experiment_name:
            config["training_cfgs"]["experiment_name"] = args.experiment_name

        # 验证配置
        if not validate_and_show_config(config, True, "XAJ Model"):
            return 1

        if args.dry_run:
            print("配置验证完成")
            return 0

        # 执行率定
        results = calibrate(config)

        # 保存配置文件（如果需要）
        if args.save_config:
            training_cfgs = config.get("training_cfgs", {})
            output_dir = os.path.join(
                training_cfgs.get("output_dir", "results"),
                training_cfgs.get("experiment_name", "experiment"),
            )
            os.makedirs(output_dir, exist_ok=True)

            # 保存配置文件
            config_output_path = os.path.join(
                output_dir, "calibration_config.yaml"
            )

            # 保存 param_range 文件
            param_range_file = training_cfgs.get("param_range_file")
            param_range_saved = False

            if param_range_file and os.path.exists(param_range_file):
                # 如果指定了参数文件且存在，复制它
                param_range_target = os.path.join(
                    output_dir, os.path.basename(param_range_file)
                )
                shutil.copy(param_range_file, param_range_target)
                # 更新配置中的路径为文件名（相对于输出目录）
                config["training_cfgs"]["param_range_file"] = os.path.basename(
                    param_range_file
                )
                param_range_saved = True
                print(f"Saved param_range file to: {param_range_target}")
            elif param_range_file is None or not os.path.exists(
                param_range_file
            ):
                # 如果没有指定或文件不存在，保存默认的 MODEL_PARAM_DICT
                param_range_target = os.path.join(
                    output_dir, "param_range.yaml"
                )
                with open(param_range_target, "w", encoding="utf-8") as f:
                    yaml.dump(
                        MODEL_PARAM_DICT,
                        f,
                        default_flow_style=False,
                        allow_unicode=True,
                    )
                # 更新配置中的路径
                config["training_cfgs"][
                    "param_range_file"
                ] = "param_range.yaml"
                param_range_saved = True
                print(f"Saved default param_range to: {param_range_target}")

            save_config_to_file(config, config_output_path)
            print(f"Saved calibration config to: {config_output_path}")

        print("XAJ率定完成")
        return 0

    except KeyboardInterrupt:
        print("率定被用户中断")
        return 1
    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
