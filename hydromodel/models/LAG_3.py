#!/usr/bin/env python3
"""
LAG_3 Model - Three Water Source Lag Routing Model
三水源滞后演算模型 - Python实现

该模型用于三水源（地表径流、壤中流、地下水）的河道汇流计算，
采用马斯京根法和滞后演算法相结合的方式进行河道演进。

Author: Converted from Java version
Date: 2026-02-06
"""

import json
import numpy as np
import math
from datetime import datetime
from typing import Dict, Tuple, Union, List
from numba import jit
from pathlib import Path


# ======================================================
#                核心模型函数 (Numba加速)
# ======================================================

@jit(nopython=True)
def calculate_muskingum_coefficients(KK: float, X: float, T: float) -> Tuple[float, float, float]:
    """
    计算马斯京根法系数 C0, C1, C2

    马斯京根法原理：
    基于河段蓄水量与上下断面流量的线性关系：
    S = K[X·I + (1-X)·Q]

    差分方程：
    Q₂ = C0·I₂ + C1·I₁ + C2·Q₁

    系数计算公式：
    FKT = KK - KK×X + 0.5×T
    C0 = (0.5×T - KK×X) / FKT
    C1 = (KK×X + 0.5×T) / FKT
    C2 = (KK - KK×X - 0.5×T) / FKT

    Args:
        KK: 河道演算系数/马斯京根参数K (h)
        X: 马斯京根权重系数（通常取0.2-0.3）
        T: 计算时段长度 (h)

    Returns:
        (C0, C1, C2): 马斯京根系数
    """
    FKT = KK - KK * X + 0.5 * T
    C0 = (0.5 * T - KK * X) / FKT
    C1 = (KK * X + 0.5 * T) / FKT
    C2 = (KK - KK * X - 0.5 * T) / FKT
    return C0, C1, C2


@jit(nopython=True)
def lchco_routing(MP: int, RQ: float, QX: np.ndarray, C0: float, C1: float, C2: float) -> float:
    """
    马斯京根河道演进算法 - 多河段串联计算（LCHCO）

    将河道分成MP段，每段应用马斯京根法，
    上一段的出流作为下一段的入流，依次演算

    Args:
        MP: 河段数量
        RQ: 当前时段入流流量 (m³/s)
        QX: 河段流量状态数组（长度MP+1）
        C0: 马斯京根系数0
        C1: 马斯京根系数1
        C2: 马斯京根系数2

    Returns:
        河道出口流量 (m³/s)
    """
    IM = MP + 1
    if IM == 1:
        QX[int(IM - 1)] = RQ
    else:
        for J in range(1, int(IM)):
            Q1 = RQ
            Q2 = QX[J - 1]
            Q3 = QX[J]
            QX[J - 1] = RQ
            RQ = C0 * Q1 + C1 * Q2 + C2 * Q3
        QX[int(IM - 1)] = RQ
    return RQ


def lag_3(
    runoff_sources: np.ndarray,
    parameters: np.ndarray,
    warmup_length: int = 0,
    return_state: bool = False,
    **kwargs
) -> Union[np.ndarray, Tuple]:
    """
    三水源滞后演算模型主函数

    物理过程：
    产流输入（RS, RI, RG）→ 消退演算 → 滞后演算 → 马斯京根河道演进 → 河道流量

    Args:
        runoff_sources (np.ndarray): 三水源产流，3维: [time, basin, source=3]
            其中 source=0 是地表径流RS, source=1 是壤中流RI, source=2 是地下水RG
        parameters (np.ndarray): 模型参数，2维: [basin, parameter]
            参数: [CG, CI, X, KK, LAG, CS, MP, F, T]
        warmup_length (int, optional): 预热期长度（默认: 0）
        return_state (bool, optional): 是否返回内部状态变量（默认: False）
        **kwargs: 其他参数，包括:
            - initial_states (default: None): 初始状态字典

    Returns:
        Union[np.ndarray, Tuple]: 取决于return_state参数:
        - return_state=False: QSim数组 [time, basin, 1]
        - return_state=True: (QSim, QS, QI, QG, QSS, QIS, QGS)
    """
    # 获取数据维度
    time_steps, num_basins, _ = runoff_sources.shape

    # 提取参数 - 所有都是 [basin] 数组
    CG = parameters[:, 0]      # 地下水消退系数
    CI = parameters[:, 1]      # 壤中流消退系数
    X = parameters[:, 2]       # 马斯京根权重系数
    KK = parameters[:, 3]      # 河道演算系数 (h)
    LAG = parameters[:, 4]     # 滞时（时段数）
    CS = parameters[:, 5]      # 河道演算消退系数
    MP = parameters[:, 6].astype(int)  # 河段数量
    F = parameters[:, 7]       # 流域面积 (km²)
    T = parameters[:, 8]       # 计算时段长度 (h)

    # 处理预热期
    if warmup_length > 0:
        runoff_warmup = runoff_sources[0:warmup_length, :, :]
        warmup_kwargs = {k: v for k, v in kwargs.items() if k != "initial_states"}
        *_, QS_w, QI_w, QG_w = lag_3(
            runoff_warmup,
            parameters,
            warmup_length=0,
            return_state=True,
            **warmup_kwargs
        )
        QSP = QS_w[-1, :, 0].copy()
        QIP = QI_w[-1, :, 0].copy()
        QGP = QG_w[-1, :, 0].copy()
    else:
        # 默认初始状态
        QSP = np.zeros(num_basins)
        QIP = np.zeros(num_basins)
        QGP = np.zeros(num_basins)

    # 应用初始状态覆盖（如果提供）
    initial_states = kwargs.get("initial_states", None)
    if initial_states is not None:
        if "QSP" in initial_states:
            QSP = np.full(num_basins, initial_states["QSP"])
        if "QIP" in initial_states:
            QIP = np.full(num_basins, initial_states["QIP"])
        if "QGP" in initial_states:
            QGP = np.full(num_basins, initial_states["QGP"])

    inputs = runoff_sources[warmup_length:, :, :]
    actual_time_steps = inputs.shape[0]

    # 初始化输出数组
    QS_out = np.zeros((actual_time_steps, num_basins))  # 地表径流流量
    QI_out = np.zeros((actual_time_steps, num_basins))  # 壤中流流量
    QG_out = np.zeros((actual_time_steps, num_basins))  # 地下水流量
    QSim = np.zeros((actual_time_steps, num_basins))    # 总流量

    # 状态序列（包含初始状态）
    QSS = np.zeros((actual_time_steps + 1, num_basins))
    QIS = np.zeros((actual_time_steps + 1, num_basins))
    QGS = np.zeros((actual_time_steps + 1, num_basins))

    # 设置初始状态
    QSS[0] = QSP
    QIS[0] = QIP
    QGS[0] = QGP

    # 逐流域计算
    for b in range(num_basins):
        # 提取该流域的三水源径流
        RS = inputs[:, b, 0]  # 地表径流 (mm)
        RI = inputs[:, b, 1]  # 壤中流 (mm)
        RG = inputs[:, b, 2]  # 地下水 (mm)

        Pnum = len(RS)
        lag_periods = int(LAG[b])

        # 初始化QSIG数组
        QSIG = np.zeros(Pnum + lag_periods)

        # 计算单位转换系数
        CP = F[b] / T[b] / 3.6  # 径流深度(mm) -> 流量(m³/s)

        # 时段消退系数转换
        CG_adj = math.pow(CG[b], T[b] / 24.0)
        CI_adj = math.pow(CI[b], T[b] / 24.0)

        # 计算马斯京根系数
        C0, C1, C2 = calculate_muskingum_coefficients(KK[b], X[b], T[b])

        # 初始化河段流量
        QXSIG = np.zeros(int(MP[b]) + 1)
        QXSIGSS = np.zeros(int(MP[b]) + 1)

        # 初始流量
        QSIG1 = 0.0 if lag_periods <= 1 else QSIG[lag_periods - 1]

        QSP_val = QSP[b]
        QIP_val = QIP[b]
        QGP_val = QGP[b]

        # 逐时段计算
        for J in range(Pnum):
            # 三水源消退演算
            QGP_val = QGP_val * CG_adj + RG[J] * (1.0 - CG_adj) * CP
            QIP_val = QIP_val * CI_adj + RI[J] * (1.0 - CI_adj) * CP
            QSP_val = RS[J] * CP

            QG_out[J, b] = QGP_val
            QI_out[J, b] = QIP_val
            QS_out[J, b] = QSP_val

            QGS[J + 1, b] = QGP_val
            QIS[J + 1, b] = QIP_val
            QSS[J + 1, b] = QSP_val

            # 河道滞后演算
            QSIG1 = QSIG1 * CS[b] + (QGP_val + QIP_val + QSP_val) * (1.0 - CS[b])
            QTSIG = QSIG1

            # 马斯京根河道演进
            QSIG[J + lag_periods] = lchco_routing(int(MP[b]), QTSIG, QXSIGSS, C0, C1, C2)

        # 保存总流量
        QSim[:, b] = QSIG[lag_periods:Pnum + lag_periods]

    # 扩展维度
    QSim_out = np.expand_dims(QSim, axis=2)
    QS_out_3d = np.expand_dims(QS_out, axis=2)
    QI_out_3d = np.expand_dims(QI_out, axis=2)
    QG_out_3d = np.expand_dims(QG_out, axis=2)
    QSS_out = np.expand_dims(QSS, axis=2)
    QIS_out = np.expand_dims(QIS, axis=2)
    QGS_out = np.expand_dims(QGS, axis=2)

    if return_state:
        return QSim_out, QS_out_3d, QI_out_3d, QG_out_3d, QSS_out, QIS_out, QGS_out
    else:
        return QSim_out


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
    # 注意：输入JSON中初始状态直接在根级别，不在initialState对象中
    initial_states = {
        'QSP': float(data.get('QSP', 0)),
        'QIP': float(data.get('QIP', 0)),
        'QGP': float(data.get('QGP', 0))
    }

    # ---------- parameters ----------
    # 注意：输入JSON中参数直接在根级别，不在parameters对象中
    param_order_map = {
        'CG': float(data.get('CG', 0.3)),
        'CI': float(data.get('CI', 0.3)),
        'X': float(data.get('X', 0.25)),
        'KK': float(data.get('KK', 24)),
        'LAG': float(data.get('LAG', 0)),
        'CS': float(data.get('CS', 0.5)),
        'MP': int(data.get('MP', 1)),
        'F': float(data.get('F', 1000)),
    }

    param_names = ['CG', 'CI', 'X', 'KK', 'LAG', 'CS', 'MP', 'F']

    # ---------- time series ----------
    flow_series = data.get('flowSeries', [])
    time_list = [e['time'] for e in flow_series]

    # 提取三水源径流
    RS_values = []
    RI_values = []
    RG_values = []

    for point in flow_series:
        if 'rSim' in point and isinstance(point['rSim'], list) and len(point['rSim']) >= 3:
            RS_values.append(point['rSim'][0])
            RI_values.append(point['rSim'][1])
            RG_values.append(point['rSim'][2])
        else:
            RS_values.append(0)
            RI_values.append(0)
            RG_values.append(0)

    # 自动检测时间间隔（小时）
    if len(time_list) >= 2:
        t1 = datetime.strptime(time_list[0], "%Y-%m-%d %H:%M:%S")
        t2 = datetime.strptime(time_list[1], "%Y-%m-%d %H:%M:%S")
        dt_hours = (t2 - t1).total_seconds() / 3600.0
    else:
        dt_hours = 1.0

    # 添加时段长度到参数
    param_order_map['T'] = dt_hours
    param_names.append('T')

    parameters = np.array([[param_order_map[n] for n in param_names]])

    # ---------- runoff_sources ----------
    T = len(time_list)
    runoff_sources = np.zeros((T, 1, 3))
    runoff_sources[:, 0, 0] = RS_values
    runoff_sources[:, 0, 1] = RI_values
    runoff_sources[:, 0, 2] = RG_values

    # ---------- kwargs ----------
    kwargs = {}

    # ---------- 预见期起始时间 ----------
    forecast_start_time = data.get('start', None)

    # ---------- 返回结构 ----------
    return {
        "runoff_sources": runoff_sources,
        "parameters": parameters,
        "initial_states": initial_states,
        "kwargs": kwargs,
        "time_list": time_list,
        "forecast_start_time": forecast_start_time
    }


def run_model(
    preprocessed: Dict,
    warmup_length: int = 0,
    return_state: bool = True
) -> Dict:
    """
    运行LAG_3模型

    Args:
        preprocessed: 预处理后的数据
        warmup_length: 预热期长度
        return_state: 是否返回状态变量

    Returns:
        模型输出字典
    """
    runoff_sources = preprocessed["runoff_sources"]
    parameters = preprocessed["parameters"]
    initial_states = preprocessed["initial_states"]
    kwargs = preprocessed["kwargs"]
    time_list = preprocessed["time_list"]

    # lag_3() 需要 initial_states 作为命名参数
    all_kwargs = kwargs.copy()
    all_kwargs["initial_states"] = initial_states

    # =============== 调用模型 ===============
    QSim, QS, QI, QG, QSS, QIS, QGS = lag_3(
        runoff_sources=runoff_sources,
        parameters=parameters,
        warmup_length=warmup_length,
        return_state=return_state,
        **all_kwargs
    )

    # 返回数据
    return {
        "raw": {
            "QSim": QSim,
            "QS": QS,
            "QI": QI,
            "QG": QG,
            "QSS": QSS,
            "QIS": QIS,
            "QGS": QGS
        },
        "time_list": time_list,
        "parameters": parameters,
        "runoff_sources": runoff_sources,
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
    QS = raw["QS"]
    QI = raw["QI"]
    QG = raw["QG"]
    QSS = raw["QSS"]
    QIS = raw["QIS"]
    QGS = raw["QGS"]

    time_list = model_output_dict["time_list"]
    warmup_length = model_output_dict["warmup_length"]
    runoff_sources = model_output_dict["runoff_sources"]

    # 三水源径流深
    RS = runoff_sources[:, 0, 0]
    RI = runoff_sources[:, 0, 1]
    RG = runoff_sources[:, 0, 2]

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
                "RS": float(RS[i]),
                "RI": float(RI[i]),
                "RG": float(RG[i]),
                "QS": 0.0,
                "QI": 0.0,
                "QG": 0.0,
                "QSim": 0.0,
            }
        else:
            idx = i - warmup_length
            entry = {
                "time": time_list[i],
                "isWarmUp": False,
                "RS": float(RS[i]),
                "RI": float(RI[i]),
                "RG": float(RG[i]),
                "QS": float(QS[idx, 0, 0]),
                "QI": float(QI[idx, 0, 0]),
                "QG": float(QG[idx, 0, 0]),
                "QSim": float(QSim[idx, 0, 0]),
            }

        time_series.append(entry)

    # =========== 最终状态 ===========
    final_idx = len(QSim) - 1
    final_state = {
        "QSP": float(QSS[final_idx + 1, 0, 0]),
        "QIP": float(QIS[final_idx + 1, 0, 0]),
        "QGP": float(QGS[final_idx + 1, 0, 0]),
    }

    # =========== 输出结构 ===========
    output = {
        "modelName": "LAG_3",
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
    print("LAG_3 Model 运行完成")
    print(f"输入文件：{input_json_path}")
    print(f"输出文件：{output_json_path}")
    print("=" * 60)
    return output_json_path


if __name__ == "__main__":
    # 测试示例
    input_file = "LAG_3-input.json"
    output_file = "LAG_3-output.json"

    main(input_file, output_file, warmup_length=0)
