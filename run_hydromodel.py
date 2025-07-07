import httpx
from mcp.server.fastmcp import FastMCP
import sys
import os
import yaml
from pathlib import Path
import pandas as pd

# 添加hydromodel到Python路径
repo_path = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(repo_path)

import logging
from hydromodel.models.model_dict import MODEL_DICT
from hydromodel.models.model_config import MODEL_PARAM_DICT
from hydromodel.datasets.data_preprocess import (
    cross_val_split_tsdata,
    _get_pe_q_from_ts,
    check_tsdata_format,
    check_basin_attr_format,
    check_folder_contents,
    process_and_save_data_as_nc,
)
from hydromodel.trainers.calibrate_sceua import calibrate_by_sceua
import numpy as np
from hydromodel.datasets import *
from hydromodel.trainers.evaluate import Evaluator
from hydromodel.trainers.evaluate import read_yaml_config

mcp = FastMCP("run_hydromodel_dev", description="率定水文模型")

logging.basicConfig(level=logging.DEBUG)

def _evaluate(cali_dir, param_dir, train_data, test_data):
    eval_train_dir = os.path.join(param_dir, "train")
    eval_test_dir = os.path.join(param_dir, "test")
    train_eval = Evaluator(cali_dir, param_dir, eval_train_dir)
    test_eval = Evaluator(cali_dir, param_dir, eval_test_dir)
    qsim_train, qobs_train, etsim_train = train_eval.predict(train_data)
    qsim_test, qobs_test, etsim_test = test_eval.predict(test_data)
    train_eval.save_results(train_data, qsim_train, qobs_train, etsim_train)
    test_eval.save_results(test_data, qsim_test, qobs_test, etsim_test)


def _read_r2_score(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return df.loc[0, "R2"]
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError) as e:
        logging.error(f"读取指标文件失败: {str(e)}")
        return None


@mcp.tool()
async def get_model_params(model_name: str):
    """
    获取模型参数信息

    Args:
        model_name: 模型名称
    """
    if model_name not in MODEL_PARAM_DICT:
        return f"错误：不支持的模型 {model_name}"

    param_info = MODEL_PARAM_DICT[model_name]
    return {
        "param_names": param_info["param_name"],
        "param_ranges": param_info["param_range"],
    }


@mcp.tool()
async def prepare_data(
    data_dir: str = "D:\Project\MCP\hydromodel\data\camels_11532500",
    target_data_scale: str = "D",
):
    """
    准备水文数据

    Args:
        data_dir: 数据目录路径
        target_data_scale: 数据时间尺度，默认为日尺度("D")，可选"M"(月尺度)或"Y"(年尺度)
    """
    try:
        # 检查数据目录是否存在
        if not os.path.exists(data_dir):
            return {"status": "error", "message": f"数据目录不存在: {data_dir}"}

        # 处理并保存数据为nc格式
        if process_and_save_data_as_nc(
            data_dir, target_data_scale, save_folder=data_dir
        ):
            return {"status": "success", "message": "数据准备完成，已转换为nc格式"}
        else:
            return {"status": "error", "message": "数据准备失败，请检查数据格式"}

    except Exception as e:
        return {"status": "error", "message": f"数据准备失败: {str(e)}"}


def generate_calibration_config(
    basin_id: str = "11532500",
) -> dict:
    """
    生成水文模型率定所需的所有配置参数

    Args:
        basin_id (str): 流域ID

    Returns:
        dict: 包含所有率定参数的配置字典
    """
    # 构建基础路径
    base_dir = Path(os.path.dirname(Path(os.path.abspath(__file__)).parent))
    data_dir = os.path.join(base_dir, "hydromodel", "data", f"camels_{basin_id}")
    result_dir = os.path.join(base_dir, "result")
    param_range_file = os.path.join(base_dir, "hydromodel", "hydromodel", "models", "param.yaml")
    
    # 默认配置
    default_config = {
        "data_type": "owndata",
        "data_dir": data_dir,
        "result_dir": result_dir,
        "exp_name": f"exp_{basin_id}",
        "model": {
            "name": "gr4j",
            "source_type": "sources",
            "source_book": "HF",
            "kernel_size": 15,
            "time_interval_hours": 24,
        },
        "basin_ids": [basin_id],
        "periods": ["2000-01-01", "2023-12-31"],
        "calibrate_period": ["2000-01-01", "2018-12-31"],
        "test_period": ["2019-01-01", "2023-12-31"],
        "warmup": 720,
        "cv_fold": 1,
        "algorithm": {
            "name": "SCE_UA",
            "random_seed": 1234,
            "rep": 200,
            "ngs": 50,
            "kstop": 10,
            "peps": 0.01,
            "pcento": 0.01,
        },
        "loss": {
            "type": "time_series",
            "obj_func": "RMSE",
            "events": None,
        },
        "param_range_file": param_range_file,
    }
    
    return default_config


@mcp.tool()
async def calibrate_and_evaluate(r2_threshold: float = 0.7):
    """
    率定水文模型并进行评估

    Args:
        r2_threshold: R²达标阈值，默认0.7
    """
    try:
        # 生成配置参数
        config = generate_calibration_config()
        
        # 如果未指定结果目录，使用默认值
        if config["result_dir"] is None:
            config["result_dir"] = os.path.join(
                os.path.dirname(Path(os.path.abspath(__file__)).parent), "result"
            )

        # 确保result_dir最后一级目录与exp_name一致
        if config["result_dir"].endswith(config["exp_name"]):
            config["result_dir"] = os.path.normpath(config["result_dir"])
        else:
            config["result_dir"] = os.path.join(config["result_dir"], config["exp_name"])

        # 创建结果目录（如果不存在）
        where_save = Path(config["result_dir"])
        if not where_save.exists():
            os.makedirs(where_save, exist_ok=True)
            
        # 处理参数范围文件
        param_range_file = os.path.join(where_save, "param_range.yaml")
        if os.path.exists(param_range_file):
            # 如果实验目录下已有参数文件，优先使用已有的
            logging.info(f"使用已有参数文件: {param_range_file}")
        else:
            # 如果没有参数文件，则从配置中复制或创建新的
            if config["param_range_file"] is None:
                yaml.dump(MODEL_PARAM_DICT, open(param_range_file, "w"))
            else:
                import shutil
                shutil.copy(config["param_range_file"], param_range_file)

        # 准备数据
        train_and_test_data = cross_val_split_tsdata(
            config["data_type"],
            config["data_dir"],
            config["cv_fold"],
            config["calibrate_period"],
            config["test_period"],
            config["periods"],
            config["warmup"],
            config["basin_ids"],
        )
        
        logging.info("Start to calibrate the model")
        print(f"[DEBUG] 接收 Start to calibrate the model ")
        # 率定模型
        if config["cv_fold"] <= 1:
            p_and_e, qobs = _get_pe_q_from_ts(train_and_test_data[0])

            # 添加类型检查
            if isinstance(p_and_e, str) or isinstance(qobs, str):
                raise ValueError("输入数据应为numpy数组格式")

            # 使用处理后的参数文件路径进行率定
            calibrate_by_sceua(
                config["basin_ids"],
                p_and_e.astype(np.float64),
                qobs.astype(np.float64),
                os.path.join(where_save, "sceua_gr_model"),
                config["warmup"],
                model=config["model"],
                algorithm=config["algorithm"],
                loss=config["loss"],
                param_file=param_range_file,  # 使用新的参数文件路径
            )
            
            # 保存配置文件
            with open(os.path.join(where_save, "config.yaml"), "w") as f:
                yaml.dump(config, f)

            # 在率定完成后直接进行评估
            logging.info("Start to evaluate the model")

            cali_dir = where_save
            param_dir = os.path.join(cali_dir, "sceua_gr_model")
            train_data = train_and_test_data[0]
            test_data = train_and_test_data[1]
            _evaluate(cali_dir, param_dir, train_data, test_data)

        else:
            for i in range(config["cv_fold"]):
                train_data, test_data = train_and_test_data[i]
                p_and_e_cv, qobs_cv = _get_pe_q_from_ts(train_data)

                # 率定
                model_save_dir = os.path.join(where_save, f"sceua_gr_model_cv{i+1}")
                calibrate_by_sceua(
                    config["basin_ids"],
                    p_and_e_cv,
                    qobs_cv,
                    model_save_dir,
                    config["warmup"],
                    model=config["model"],
                    algorithm=config["algorithm"],
                    loss=config["loss"],
                    param_file=param_range_file,
                )

                # 评估
                logging.info(f"Start to evaluate the {i+1}-th fold")
                cali_dir = where_save
                fold_dir = os.path.join(cali_dir, f"sceua_gr_model_cv{i+1}")

                train_data = train_and_test_data[i][0]
                test_data = train_and_test_data[i][1]
                _evaluate(cali_dir, fold_dir, train_data, test_data)

        # 读取评估结果
        train_r2 = _read_r2_score(os.path.join(param_dir, "train", "basins_metrics.csv"))
        test_r2 = _read_r2_score(os.path.join(param_dir, "test", "basins_metrics.csv"))

        return {
            "status": "success",
            "message": "模型率定与评估完成",
            "result_dir": str(where_save),
            "evl_info": {
                "训练期R2": f"{round(train_r2, 4)}" if train_r2 is not None else "N/A",
                "测试期R2": f"{round(test_r2, 4)}" if test_r2 is not None else "N/A",
                "需要继续率定": (
                    True
                    if (train_r2 is not None and train_r2 < r2_threshold) or
                       (test_r2 is not None and test_r2 < r2_threshold)
                    else False
                )
            },
        }

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return {
            "status": "error",
            "message": f"模型率定失败: {str(e)}",
            "error_stack": error_trace,
        }


@mcp.tool()
async def adjust_param_ranges(
    model_name: str, 
    current_params: dict, 
    r2_scores: dict, 
    result_dir: str,
    r2_threshold: float
):
    """
    根据模型表现自动调整参数范围

    Args:
        model_name (str): 模型名称
        current_params (dict): 当前最优参数值
        r2_scores (dict): 训练集和测试集的R²值
        result_dir (str): 实验结果目录
        r2_threshold (float): R²达标阈值
    """
    # 从实验目录读取参数范围文件
    param_range_file = os.path.join(result_dir, "param_range.yaml")
    if not os.path.exists(param_range_file):
        return {"status": "error", "message": "参数范围文件不存在"}
        
    # 读取当前参数范围
    with open(param_range_file, 'r') as f:
        param_dict = yaml.safe_load(f)
    current_ranges = param_dict[model_name]["param_range"]
    
    # 根据模型表现调整范围
    new_ranges = {}
    for param_name, param_value in current_params.items():
        current_range = current_ranges[param_name]
        range_width = current_range[1] - current_range[0]
        
        # 如果模型表现良好，缩小范围到最优值周围
        if r2_scores["train"] > r2_threshold and r2_scores["test"] > r2_threshold:
            new_min = max(current_range[0], param_value - range_width * 0.2)
            new_max = min(current_range[1], param_value + range_width * 0.2)
        # 如果模型表现不佳，扩大范围
        else:
            new_min = max(0, param_value - range_width * 0.5)
            new_max = param_value + range_width * 0.5
            
        new_ranges[param_name] = [new_min, new_max]
    
    # 更新参数范围文件
    param_dict[model_name]["param_range"] = new_ranges
    with open(param_range_file, 'w') as f:
        yaml.dump(param_dict, f)
    
    return {
        "status": "success",
        "message": "参数范围已更新",
        "new_ranges": new_ranges
    }


@mcp.tool()
async def auto_calibrate_with_range_adjustment(
    r2_threshold: float = 0.7,
    max_attempts: int = 5,
    model_name="gr4j",
):
    """
    自动率定并根据结果调整参数范围重新率定

    Args:
        r2_threshold: R²达标阈值
        max_attempts: 最大尝试次数
    """
    # 生成配置参数
    config = generate_calibration_config()
    
    # 创建R²变化记录文件
    r2_history = []
    attempt = 1
    
    while attempt <= max_attempts:
        logging.info(f"开始第{attempt}次率定尝试")
        
        # 执行率定和评估
        result = await calibrate_and_evaluate(r2_threshold)
        
        if result["status"] == "error":
            return result
            
        evl_info = result["evl_info"]
        
        # 记录本次率定的R²值
        r2_record = {
            "attempt": attempt,
            "train_r2": float(evl_info["训练期R2"]) if evl_info["训练期R2"] != "N/A" else 0,
            "test_r2": float(evl_info["测试期R2"]) if evl_info["测试期R2"] != "N/A" else 0
        }
        r2_history.append(r2_record)
        
        # 保存R²历史记录到CSV文件
        r2_history_df = pd.DataFrame(r2_history)
        r2_history_df.to_csv(os.path.join(result["result_dir"], "r2_history.csv"), index=False)
        
        if not evl_info["需要继续率定"]:
            return {
                "status": "success",
                "message": f"在第{attempt}次尝试后达到目标性能",
                "final_result": result
            }
            
        # 如果需要继续率定，调整参数范围
        if attempt < max_attempts:
            # 获取当前最优参数
            config_file = os.path.join(result["result_dir"], "config.yaml")
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            param_file = os.path.join(result["result_dir"], "param.yaml")
            with open(param_file, 'r') as f:
                current_params = yaml.safe_load(f)
                
            r2_scores = {
                "train": float(evl_info["训练期R2"]) if evl_info["训练期R2"] != "N/A" else 0,
                "test": float(evl_info["测试期R2"]) if evl_info["测试期R2"] != "N/A" else 0
            }
            
            # 调整参数范围
            adjust_result = await adjust_param_ranges(
                model_name,
                current_params,
                r2_scores,
                result["result_dir"],
                r2_threshold
            )
            
            if adjust_result["status"] == "error":
                return adjust_result
                
            logging.info(f"参数范围已调整: {adjust_result['new_ranges']}")
        
        attempt += 1
    
    return {
        "status": "warning",
        "message": f"达到最大尝试次数{max_attempts}，未能达到目标性能",
        "final_result": result
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
