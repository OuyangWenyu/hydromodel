#!/usr/bin/env python3
"""
MSK Model - Muskingum Flood Routing Method
马斯京根河道洪水演算法 - Python实现

马斯京根法是一种经典的河道洪水演算方法，用于模拟洪水波在河道中的传播过程。
该方法基于水量平衡原理和河道蓄水量与流量的关系，通过上游流量推算下游流量。

Author: Converted from Java version
Date: 2026-02-06
"""

import json
import numpy as np
import math
from datetime import datetime
from typing import Dict, Tuple, Union
from numba import jit
from pathlib import Path


# ======================================================
#                核心模型函数 (Numba加速)
# ======================================================

@jit(nopython=True)
def muskingum_routing(
    Q: np.ndarray,
    N: int,
    KE: float,
    XE: float,
    time_interval: float,
    start_index: int,
    QX_initial: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    分段马斯京根河道演进算法（核心计算方法）

    算法原理：
    马斯京根法基于水量平衡方程和河段蓄水量方程推导而来：
    1. 水量平衡：dS/dt = I - Q
    2. 蓄水量方程：S = K[X·I + (1-X)·Q]
    3. 差分离散化后得到演算公式：
       Q₂ = C₀·I₂ + C₁·I₁ + C₂·Q₁

    演算系数计算：
    CD = 0.5·Δt + K - K·X
    C₀ = (0.5·Δt - K·X) / CD
    C₁ = (0.5·Δt + K·X) / CD
    C₂ = 1 - C₀ - C₁

    稳定性条件：
    C₀ ≥ 0 且 C₂ ≥ 0

    Args:
        Q: 上游断面入流流量过程 (m³/s)
        N: 河段分段数
        KE: 子河段洪水波传播时间 (h)
        XE: 流量比重因子 (0-0.5)
        time_interval: 时段长度 Δt (h)
        start_index: 预报开始时间索引
        QX_initial: 河段下断面初始流量数组

    Returns:
        (QRouting, QX0): 下游断面出流流量、河段流量状态矩阵
    """
    # 初始化数组
    QC = np.zeros(N)  # 各子河段下断面的当前流量
    QRouting = np.zeros(len(Q))  # 河段出口流量
    QX0 = np.zeros((len(Q) + 1, N))  # 流量状态矩阵

    # 初始化各子河段下断面的初始流量
    for j in range(N):
        if j < len(QX_initial):
            QC[j] = QX_initial[j]
        else:
            QC[j] = QC[j - 1]
        QX0[0, j] = QC[j]

    # 计算马斯京根演算系数
    CD = 0.5 * time_interval + KE - KE * XE
    C0 = (0.5 * time_interval - KE * XE) / CD
    C1 = (0.5 * time_interval + KE * XE) / CD
    C2 = 1 - C0 - C1

    # 检查稳定性条件
    if C0 >= 0.0 and C2 >= 0.0:
        # 逐时段演算
        for i in range(start_index, len(Q)):
            # 确定入流流量
            if i == start_index:
                QI1 = Q[i]
            else:
                QI1 = Q[i - 1]
            QI2 = Q[i]

            # 分段演算
            if N > 0:
                for j in range(N):
                    QX0[i, j] = QC[j]
                    QO1 = QC[j]

                    # 马斯京根演算公式
                    QO2 = C0 * QI2 + C1 * QI1 + C2 * QO1

                    # 传递流量
                    QI1 = QO1
                    QI2 = QO2

                    # 更新当前流量
                    QC[j] = QO2
            else:
                # N=0：不演算，直接输出
                QO2 = Q[i]

            # 处理极小值
            if QO2 < 0.0001:
                QO2 = 0.0

            QRouting[i] = QO2

        # 记录最终状态
        for j in range(N):
            QX0[len(Q), j] = QC[j]
    else:
        # 不满足稳定性条件，直接输出上游流量
        QRouting = Q.copy()

    return QRouting, QX0


def msk(
    flow_input: np.ndarray,
    parameters: np.ndarray,
    warmup_length: int = 0,
    return_state: bool = False,
    **kwargs
) -> Union[np.ndarray, Tuple]:
    """
    马斯京根河道演算模型主函数

    核心原理：
    - 水量平衡方程：dS/dt = I - Q
    - 蓄水量方程：S = K[X·I + (1-X)·Q]

    Args:
        flow_input (np.ndarray): 上游流量，3维: [time, basin, feature=1]
        parameters (np.ndarray): 模型参数，2维: [basin, parameter]
            参数: [KE, XE, N, time_interval]
        warmup_length (int, optional): 预热期长度（默认: 0）
        return_state (bool, optional): 是否返回内部状态变量（默认: False）
        **kwargs: 其他参数，包括:
            - initial_states (default: None): 初始状态字典
            - start_index (default: 0): 预报开始索引

    Returns:
        Union[np.ndarray, Tuple]: 取决于return_state参数:
        - return_state=False: QSim数组 [time, basin, 1]
        - return_state=True: (QSim, QX0)
    """
    # 获取数据维度
    time_steps, num_basins, _ = flow_input.shape

    # 提取参数
    KE = parameters[:, 0]           # 洪水波传播时间 (h)
    XE = parameters[:, 1]           # 流量比重因子
    N = parameters[:, 2].astype(int)  # 河段分段数
    time_interval = parameters[:, 3]  # 时段长度 (h)

    # 获取预报开始索引
    start_index = kwargs.get("start_index", 0)

    # 处理预热期
    if warmup_length > 0:
        flow_warmup = flow_input[0:warmup_length, :, :]
        warmup_kwargs = {k: v for k, v in kwargs.items() if k != "initial_states"}
        warmup_kwargs["start_index"] = 0
        QSim_warmup, QX0_warmup = msk(
            flow_warmup,
            parameters,
            warmup_length=0,
            return_state=True,
            **warmup_kwargs
        )
        # 从预热期获取初始状态
        QX_initial_from_warmup = QX0_warmup[-1, :, :]
    else:
        QX_initial_from_warmup = None

    inputs = flow_input[warmup_length:, :, :]
    actual_time_steps = inputs.shape[0]

    # 初始化输出
    QSim_out = np.zeros((actual_time_steps, num_basins))
    QX0_all = []

    # 逐流域计算
    for b in range(num_basins):
        Q = inputs[:, b, 0]

        # 获取初始状态
        initial_states = kwargs.get("initial_states", None)
        if initial_states is not None and "QX" in initial_states:
            QX_initial = np.array(initial_states["QX"])
        elif QX_initial_from_warmup is not None:
            QX_initial = QX_initial_from_warmup[b, :]
        else:
            QX_initial = np.zeros(int(N[b]))

        # 修正初始流量
        if len(QX_initial) > 0 and len(Q) > 0 and start_index < len(Q):
            if QX_initial[0] != 0:
                rate = Q[start_index] / QX_initial[0]
                QX_initial = QX_initial * rate
            else:
                difference = Q[start_index]
                QX_initial = QX_initial + difference

        # 调用核心演算函数
        QRouting, QX0 = muskingum_routing(
            Q, int(N[b]), KE[b], XE[b], time_interval[b], start_index, QX_initial
        )

        QSim_out[:, b] = QRouting
        QX0_all.append(QX0)

    # 扩展维度
    QSim_result = np.expand_dims(QSim_out, axis=2)

    if return_state:
        # 转换QX0_all为numpy数组
        QX0_result = np.array(QX0_all)
        QX0_result = np.transpose(QX0_result, (1, 0, 2))  # [time, basin, N]
        return QSim_result, QX0_result
    else:
        return QSim_result


# ======================================================
#                数据处理函数
# ======================================================

def preprocess_json(input_json_str: str) -> Dict:
    """
    预处理JSON输入数据

    Args:
        input_json_str: JSON字符串

    Returns:
        预处理后的数据字典
    """
    data = json.loads(input_json_str)

    # ---------- initial states ----------
    initial_state = data.get('initialState', {})
    initial_states = {}
    if 'QX' in initial_state:
        initial_states['QX'] = initial_state['QX']
    else:
        initial_states['QX'] = [0]

    # ---------- parameters ----------
    params = data.get('parameters', {})
    KE = params.get('KK', 24)
    XE = params.get('X', 0.25)
    N = params.get('MP', 1)

    # ---------- time series ----------
    time_series = data.get('timeSeries', data.get('flowSeries', []))
    time_list = [e['time'] for e in time_series]

    # 提取流量
    Q_values = []
    for point in time_series:
        flow_value = point.get('QSim', point.get('Q', point.get('q', 0)))
        Q_values.append(float(flow_value))

    # 自动检测时间间隔（小时）
    if len(time_list) >= 2:
        t1 = datetime.strptime(time_list[0], "%Y-%m-%d %H:%M:%S")
        t2 = datetime.strptime(time_list[1], "%Y-%m-%d %H:%M:%S")
        dt_hours = (t2 - t1).total_seconds() / 3600.0
    else:
        dt_hours = data.get('metadata', {}).get('timestep_hours', 1.0)

    # 构建参数数组
    parameters = np.array([[KE, XE, N, dt_hours]])

    # ---------- flow_input ----------
    T = len(time_list)
    flow_input = np.zeros((T, 1, 1))
    flow_input[:, 0, 0] = Q_values

    # ---------- 预见期起始时间 ----------
    forecast_start_time = data.get('forecastStartTime', data.get('start', None))

    # 计算start_index
    start_index = 0
    if forecast_start_time:
        for i, t in enumerate(time_list):
            if t == forecast_start_time:
                start_index = i
                break

    # ---------- 返回结构 ----------
    return {
        "flow_input": flow_input,
        "parameters": parameters,
        "initial_states": initial_states,
        "kwargs": {"start_index": start_index},
        "time_list": time_list,
        "forecast_start_time": forecast_start_time
    }


def run_model(
    preprocessed: Dict,
    warmup_length: int = 0,
    return_state: bool = True
) -> Dict:
    """
    运行MSK模型

    Args:
        preprocessed: 预处理后的数据
        warmup_length: 预热期长度
        return_state: 是否返回状态变量

    Returns:
        模型输出字典
    """
    flow_input = preprocessed["flow_input"]
    parameters = preprocessed["parameters"]
    initial_states = preprocessed["initial_states"]
    kwargs = preprocessed["kwargs"]
    time_list = preprocessed["time_list"]

    # msk() 需要 initial_states 作为命名参数
    all_kwargs = kwargs.copy()
    all_kwargs["initial_states"] = initial_states

    # =============== 调用模型 ===============
    QSim, QX0 = msk(
        flow_input=flow_input,
        parameters=parameters,
        warmup_length=warmup_length,
        return_state=return_state,
        **all_kwargs
    )

    # 返回数据
    return {
        "raw": {
            "QSim": QSim,
            "QX0": QX0,
            "Q": flow_input
        },
        "time_list": time_list,
        "parameters": parameters,
        "kwargs": kwargs,
        "warmup_length": warmup_length,
        "forecast_start_time": preprocessed.get("forecast_start_time")
    }


def postprocess(model_output_dict: Dict, model_version: str = "1.0") -> str:
    """
    后处理模型输出，生成JSON字符串

    Args:
        model_output_dict: 模型输出字典
        model_version: 模型版本号

    Returns:
        JSON字符串
    """
    raw = model_output_dict["raw"]

    QSim = raw["QSim"]
    Q_input = raw["Q"]

    time_list = model_output_dict["time_list"]
    warmup_length = model_output_dict["warmup_length"]

    # 上游流量
    Q_upstream = Q_input[:, 0, 0]

    total_steps = len(time_list)
    forecast_periods = total_steps - warmup_length

    # 计算预见期起始时间
    forecast_start_time = model_output_dict.get("forecast_start_time")
    if forecast_start_time is None:
        forecast_start_time = (
            time_list[warmup_length] if warmup_length < total_steps else time_list[-1]
        )

    # =========== timeSeries ===========
    time_series = []
    for i in range(total_steps):
        is_warmup = i < warmup_length

        if is_warmup:
            entry = {
                "time": time_list[i],
                "isWarmUp": True,
                "Q_upstream": float(Q_upstream[i]),
                "Q_downstream": 0.0,
            }
        else:
            idx = i - warmup_length
            entry = {
                "time": time_list[i],
                "isWarmUp": False,
                "Q_upstream": float(Q_upstream[i]),
                "Q_downstream": float(QSim[idx, 0, 0]),
            }

        time_series.append(entry)

    # =========== 最终状态 ===========
    QX0 = raw["QX0"]
    final_state = {}
    if len(QX0) > 0 and QX0.shape[2] > 0:
        final_QX = QX0[-1, 0, :].tolist()
        final_state["QX"] = final_QX

    # =========== 输出结构 ===========
    output = {
        "modelName": "MSK",
        "modelVersion": model_version,
        "computeTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "warmUpPeriods": warmup_length,
        "forecastPeriods": forecast_periods,
        "forecastStartTime": forecast_start_time,
        "timeSeries": time_series,
        "finalState": final_state,
    }

    # 返回 JSON 字符串
    return json.dumps(output, indent=2, ensure_ascii=False)


def main(
    input_json_path: str,
    output_json_path: str,
    warmup_length: int = 0,
    model_version: str = "1.0"
) -> Path:
    """
    主函数：读取输入 JSON → 预处理 → 运行模型 → 后处理 → 输出 JSON

    Args:
        input_json_path: 输入JSON文件路径
        output_json_path: 输出JSON文件路径
        warmup_length: 预热期长度
        model_version: 模型版本号

    Returns:
        输出文件路径
    """
    input_json_path = Path(input_json_path)
    output_json_path = Path(output_json_path)

    # ---- 1. 读取 JSON 文件 ----
    with open(input_json_path, "r", encoding="utf-8") as f:
        input_json_str = f.read()

    # ---- 2. 预处理 ----
    preprocessed = preprocess_json(input_json_str)

    # ---- 3. 运行模型 ----
    model_output = run_model(
        preprocessed,
        warmup_length=warmup_length,
        return_state=True
    )

    # ---- 4. 后处理（生成 JSON 字符串）----
    output_json_str = postprocess(
        model_output,
        model_version=model_version
    )

    # ---- 5. 保存输出 JSON 文件 ----
    with open(output_json_path, "w", encoding="utf-8") as f:
        f.write(output_json_str)

    print("=" * 60)
    print("MSK Model 运行完成")
    print(f"输入文件：{input_json_path}")
    print(f"输出文件：{output_json_path}")
    print("=" * 60)
    return output_json_path


if __name__ == "__main__":
    # 测试示例
    input_file = "MSK-input.json"
    output_file = "MSK-output.json"

    main(input_file, output_file, warmup_length=0)
