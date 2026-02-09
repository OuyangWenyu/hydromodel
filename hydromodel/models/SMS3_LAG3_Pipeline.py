#!/usr/bin/env python3
"""
SMS_3 + LAG_3 Pipeline - 三水源产流汇流串联模型
SMS_3 (产流模型) → LAG_3 (汇流模型)

工作流程：
1. 运行 SMS_3 产流模型，计算三水源产流（RS, RI, RG）
2. 将 SMS_3 的输出转换为 LAG_3 的输入格式
3. 运行 LAG_3 汇流模型，计算河道流量过程
4. 输出最终流量结果

Author: Converted from Java version
Date: 2026-02-08
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import SMS_3
import LAG_3


def extract_lag3_parameters(pipeline_input: Dict) -> Dict:
    """
    从pipeline输入中提取LAG_3参数

    如果输入中没有LAG参数，使用默认值

    Args:
        pipeline_input: Pipeline输入数据字典

    Returns:
        LAG_3参数字典
    """
    lag3_params = {}

    # LAG_3 参数及其默认值（来自LAG-input.json的标准值）
    lag3_params['KK'] = pipeline_input.get('KK', 3.0)
    lag3_params['MP'] = str(pipeline_input.get('MP', 4))
    lag3_params['CG'] = str(pipeline_input.get('CG', 0.968))
    lag3_params['CI'] = str(pipeline_input.get('CI', 0.571))
    lag3_params['CS'] = str(pipeline_input.get('CS', 0.103))
    lag3_params['LAG'] = pipeline_input.get('LAG', 0.0)
    lag3_params['X'] = str(pipeline_input.get('X', -0.803))
    lag3_params['clen'] = pipeline_input.get('clen', 3)

    # 流域面积（必需参数）
    # 优先从 parameters 对象中获取
    f_value = None
    if 'parameters' in pipeline_input:
        parameters = pipeline_input['parameters']
        if isinstance(parameters, dict) and 'F' in parameters:
            f_value = parameters['F']

    # 如果 parameters 中没有，尝试从顶层获取
    if f_value is None and 'F' in pipeline_input:
        f_value = pipeline_input['F']

    if f_value is None:
        raise ValueError("缺少必需字段: F (流域面积)，请在 parameters 对象或顶层提供")

    lag3_params['F'] = f_value

    # 初始状态参数
    lag3_params['QSP'] = str(pipeline_input.get('QSP', 0))
    lag3_params['QIP'] = str(pipeline_input.get('QIP', 0))
    lag3_params['QGP'] = str(pipeline_input.get('QGP', 0))

    # 初始河道流量序列
    lag3_params['QSIG'] = pipeline_input.get('QSIG', [0, 0, 0, 0, 0, 0])

    # 初始河段流量
    lag3_params['QXSIG'] = pipeline_input.get('QXSIG', [0])

    # 起始时间
    if 'start' in pipeline_input:
        lag3_params['start'] = pipeline_input['start']

    return lag3_params


def convert_sms3_output_to_flowseries(sms3_output: Dict) -> List[Dict]:
    """
    将 SMS_3 的输出转换为 LAG_3 的 flowSeries 输入格式

    flowSeries 格式：
    {
        "time": "2025-06-05 08:00:00",
        "rain": 10.5,
        "runoff": 5.2,
        "rSim": [RS, RI, RG]
    }

    Args:
        sms3_output: SMS_3模型的输出字典

    Returns:
        flowSeries列表
    """
    time_list = sms3_output["time_list"]

    # 提取SMS_3的输出结果（现在包含完整时间序列）
    raw = sms3_output["raw"]
    RunoffSim = raw["RunoffSim"]  # [time, basin, 1]
    RS = raw["RS"]  # [time, basin, 1]
    RI = raw["RI"]  # [time, basin, 1]
    RG = raw["RG"]  # [time, basin, 1]

    # 提取降雨数据（从p_and_e中）
    p_and_e = sms3_output["p_and_e"]  # [time, basin, 2] - [:,:,0]是降雨

    # 构建flowSeries（完整时间序列）
    flow_series = []
    for i in range(len(time_list)):
        point = {
            "time": time_list[i],
            "rain": float(p_and_e[i, 0, 0]),
            "runoff": float(RunoffSim[i, 0, 0]),
            "rSim": [
                float(RS[i, 0, 0]),
                float(RI[i, 0, 0]),
                float(RG[i, 0, 0])
            ]
        }
        flow_series.append(point)

    return flow_series


def run_pipeline(
    pipeline_input: Dict,
    warmup_length: int = 0
) -> Dict:
    """
    运行 SMS_3 + LAG_3 串联模型

    Args:
        pipeline_input: Pipeline输入数据字典
        warmup_length: 预热期长度（如果为0，将自动计算）

    Returns:
        LAG_3模型的输出字典
    """
    print("=" * 60)
    print("开始 SMS_3 → LAG_3 串联计算")
    print("=" * 60)

    # ===== 步骤1: 运行 SMS_3 产流模型 =====
    print("\n步骤1: 运行 SMS_3 产流模型...")

    # 将pipeline输入转换为SMS_3的JSON格式
    sms3_input_json = json.dumps(pipeline_input, ensure_ascii=False)

    # 预处理SMS_3输入
    sms3_preprocessed = SMS_3.preprocess_json(sms3_input_json)

    # 计算forecast_start_index用于LAG_3（但SMS_3不使用warmup_length）
    forecast_start_index = 0
    if 'forecast_start_time' in sms3_preprocessed and sms3_preprocessed['forecast_start_time']:
        forecast_start_time = sms3_preprocessed['forecast_start_time']
        time_list = sms3_preprocessed['time_list']

        start_time = datetime.strptime(forecast_start_time, "%Y-%m-%d %H:%M:%S")
        for i, t in enumerate(time_list):
            t_obj = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
            if t_obj >= start_time:
                forecast_start_index = i
                print(f"  Forecast起始索引: {forecast_start_index} (warmup: {forecast_start_index} 时段)")
                break

    # 运行SMS_3模型 - 不使用warmup_length，输出完整时间序列
    # 这样可以获取warmup期间的产流值用于LAG_3
    sms3_output = SMS_3.run_model(
        sms3_preprocessed,
        warmup_length=0,  # 不使用warmup，输出完整结果
        return_state=True
    )

    # 获取SMS_3的输出统计
    raw = sms3_output["raw"]
    time_steps = raw["RunoffSim"].shape[0]
    max_runoff = float(np.max(raw["RunoffSim"]))

    print(f"  SMS_3 产流计算完成:")
    print(f"    - 时段数: {time_steps}")
    print(f"    - 最大产流: {max_runoff:.6f} mm")
    print(f"    - RS (地表径流): {raw['RS'].shape}")
    print(f"    - RI (壤中流): {raw['RI'].shape}")
    print(f"    - RG (地下径流): {raw['RG'].shape}")

    # ===== 步骤2: 构建 LAG_3 输入数据 =====
    print("\n步骤2: 构建 LAG_3 输入数据...")

    # 提取LAG_3参数
    lag3_params = extract_lag3_parameters(pipeline_input)

    # 将SMS_3的输出转换为LAG_3的flowSeries格式
    flow_series = convert_sms3_output_to_flowseries(sms3_output)

    # 构建LAG_3输入数据结构
    lag3_input = lag3_params.copy()
    lag3_input['flowSeries'] = flow_series

    print(f"  LAG_3 输入数据构建完成:")
    print(f"    - flowSeries 时段数: {len(flow_series)}")
    print(f"    - KK: {lag3_params['KK']}")
    print(f"    - CS: {lag3_params['CS']}")
    print(f"    - CI: {lag3_params['CI']}")
    print(f"    - CG: {lag3_params['CG']}")
    print(f"    - F: {lag3_params['F']}")

    # ===== 步骤3: 运行 LAG_3 汇流模型 =====
    print("\n步骤3: 运行 LAG_3 汇流模型...")

    # 将LAG_3输入转换为JSON
    lag3_input_json = json.dumps(lag3_input, ensure_ascii=False)

    # 预处理LAG_3输入
    lag3_preprocessed = LAG_3.preprocess_json(lag3_input_json)

    # LAG_3不使用warmup_length，因为flowSeries中warmup期间的产流已经是0
    # 这样LAG_3会输出完整的时间序列（与输入长度一致）
    lag3_warmup_length = 0

    # 运行LAG_3模型
    lag3_output = LAG_3.run_model(
        lag3_preprocessed,
        warmup_length=lag3_warmup_length,
        return_state=True
    )

    # 获取LAG_3的输出统计
    raw_lag = lag3_output["raw"]
    qsim = raw_lag["QSim"]
    max_q = float(np.max(qsim))
    max_q_idx = int(np.argmax(qsim))

    print(f"  LAG_3 汇流计算完成:")
    print(f"    - 流量过程 QSim: {qsim.shape[0]} 个时段")
    print(f"    - 峰值流量: {max_q:.3f} m3/s (第 {max_q_idx + 1} 时段)")

    # 显示前5个流量值
    print(f"    - 前5个流量值:")
    for i in range(min(5, qsim.shape[0])):
        print(f"      时段 {i + 1}: {qsim[i, 0, 0]:.6f} m3/s")

    print("\n" + "=" * 60)
    print("SMS_3 → LAG_3 串联计算完成")
    print("=" * 60)

    return lag3_output


def main(
    input_json_path: str,
    output_json_path: str,
    warmup_length: int = 0,
    model_version: str = "1.0"
) -> Path:
    """
    主函数：读取输入 JSON → 运行Pipeline → 输出 JSON

    Args:
        input_json_path: 输入JSON文件路径（SMS_3格式的pipeline输入）
        output_json_path: 输出JSON文件路径（LAG_3格式的输出）
        warmup_length: 预热期长度（0表示自动计算）
        model_version: 模型版本号

    Returns:
        输出文件路径
    """
    input_json_path = Path(input_json_path)
    output_json_path = Path(output_json_path)

    print("=" * 60)
    print("SMS_3 + LAG_3 Pipeline 模型运行")
    print(f"输入文件：{input_json_path}")
    print(f"输出文件：{output_json_path}")
    print("=" * 60)
    print()

    # ---- 1. 读取 JSON 文件 ----
    with open(input_json_path, "r", encoding="utf-8") as f:
        pipeline_input = json.load(f)

    # ---- 2. 运行Pipeline ----
    lag3_output = run_pipeline(pipeline_input, warmup_length=warmup_length)

    # ---- 3. 后处理（生成 JSON 字符串）----
    output_json_str = LAG_3.postprocess(
        lag3_output,
        model_version=model_version
    )

    # ---- 4. 保存输出 JSON 文件 ----
    with open(output_json_path, "w", encoding="utf-8") as f:
        f.write(output_json_str)

    print()
    print("=" * 60)
    print("Pipeline 运行完成")
    print(f"输入文件：{input_json_path}")
    print(f"输出文件：{output_json_path}")
    print("=" * 60)

    return output_json_path


if __name__ == "__main__":
    # 测试示例
    input_file = r"D:\project\songliao\hydromodeljava\hydro-model-core\src\main\resources\json\SMS3_LAG3_Pipeline-input.json"
    output_file = "SMS3_LAG3_Pipeline-output.json"

    main(input_file, output_file, warmup_length=0)
