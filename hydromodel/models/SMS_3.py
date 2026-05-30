#!/usr/bin/env python3
"""
SMS_3 Model - Three Water Source Saturation Excess Runoff Model (Xinanjiang Model)
三水源蓄满产流模型（新安江模型）- Python实现

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
def calculate_tension_water_storage(
    EK: float, W: float, WU: float, WL: float, WD: float,
    RD: np.ndarray, PED: np.ndarray, ND: int,
    PE: float, WM: float, WUM: float, WLM: float, WDM: float,
    WMM: float, B: float, C: float
) -> Tuple[float, float, float, float]:
    """
    张力水蓄水量计算 - 新安江模型核心算法之一

    张力水概念：
    张力水是土壤中被毛管力吸附的水分，不能自由流动产流。
    只有当张力水接近饱和时，多余的水分才能转化为自由水产流。

    Args:
        EK: 时段蒸散发能力 (mm)
        W: 当前总张力水含量 (mm)
        WU: 当前上层张力水含量 (mm)
        WL: 当前下层张力水含量 (mm)
        WD: 当前深层张力水含量 (mm)
        RD: 各层产流量数组（输出参数）(mm)
        PED: 各层净雨量数组 (mm)
        ND: 分层数
        PE: 时段净雨 (mm)
        WM: 流域平均张力水容量 (mm)
        WUM: 上层张力水容量 (mm)
        WLM: 下层张力水容量 (mm)
        WDM: 深层张力水容量 (mm)
        WMM: 流域最大张力水容量 (mm)
        B: 蓄水容量曲线方次
        C: 深层蒸散发系数

    Returns:
        (W, WU, WL, WD): 各层张力水含量
    """
    if PE <= 0.0:
        # 蒸发情况
        if WU + PE >= 0.0:
            WU = WU + PE
        else:
            EU = WU + EK + PE
            WU = 0.0
            EL = (EK - EU) * WL / WLM

            if WL < C * WLM:
                EL = C * (EK - EU)

            if WL - EL < 0.0:
                ED = EL - WL
                WL = 0.0
                WD = WD - ED
            else:
                WL = WL - EL

        W = WU + WL + WD
    else:
        # 产流情况
        if WM - W < 0.0001:
            A = WMM
        else:
            A = WMM * (1.0 - math.pow(1.0 - W / WM, 1.0 / (1.0 + B)))

        R = 0.0
        PEDS = 0.0

        for I in range(ND):
            A = A + PED[I]
            PEDS = PEDS + PED[I]
            RI = R
            R = PEDS - WM + W

            if A < WMM:
                R = R + WM * math.pow(1.0 - A / WMM, 1.0 + B)

            RD[I] = R - RI

        # 分配到各层
        if WU + PE - R <= WUM:
            WU = WU + PE - R
        else:
            if WU + WL + PE - R - WUM >= WLM:
                WU = WUM
                WL = WLM
                WD = W + PEDS - R - WU - WL
                if WD > WDM:
                    WD = WDM
            else:
                WL = WU + WL + PE - R - WUM
                WU = WUM

        W = WU + WL + WD

    return W, WU, WL, WD


@jit(nopython=True)
def calculate_free_water_and_runoff(
    FR: float, S: float, RD: np.ndarray, PED: np.ndarray, ND: int,
    PE: float, IM: float, SM: float, SMM: float, EX: float,
    KG: float, KI: float
) -> Tuple[float, float, float, float, float]:
    """
    自由水蓄水库及分水源计算 - 新安江模型核心算法之二

    三水源分离机制：
    自由水 → 地表径流（RS）：快速响应，流速快
           → 壤中流（RI）：中等响应，在土壤中侧向流动
           → 地下径流（RG）：慢速响应，地下水补给

    Args:
        FR: 当前产流面积系数
        S: 当前自由水蓄量 (mm)
        RD: 各层产流量数组 (mm)
        PED: 各层净雨量数组 (mm)
        ND: 分层数
        PE: 时段净雨 (mm)
        IM: 不透水面积比例
        SM: 自由水蓄水库容量 (mm)
        SMM: 自由水蓄水库容曲线的方次
        EX: 自由水蓄水库容曲线方次
        KG: 地下水消退系数
        KI: 壤中流消退系数

    Returns:
        (FR, S, RS, RI, RG): 产流面积系数、自由水蓄量、三水源径流
    """
    if PE <= 0.0:
        # 无降雨，退水过程
        RS = 0.0
        RG = S * KG * FR
        RI = S * KI * FR
        S = S * (1.0 - KG - KI)

        return FR, S, RS, RI, RG
    else:
        # 有降雨，产流过程
        RB = IM * PE  # 不透水面积产流

        KID = (1.0 - math.pow(1.0 - (KG + KI), 1.0 / ND)) / (KG + KI)
        KGD = KID * KG
        KID = KID * KI

        RS = 0.0
        RI = 0.0
        RG = 0.0

        for I in range(ND):
            TD = RD[I] - IM * PED[I]
            X = FR
            FR = TD / PED[I]
            S = X * S / FR

            if S >= SM:
                RR = (PED[I] + S - SM) * FR
            else:
                AU = SMM * (1.0 - math.pow(1.0 - S / SM, 1.0 / (1.0 + EX)))
                if AU + PED[I] < SMM:
                    RR = (PED[I] - SM + S + SM *
                          math.pow(1.0 - (PED[I] + AU) / SMM, 1.0 + EX)) * FR
                else:
                    RR = (PED[I] + S - SM) * FR

            RS = RR + RS
            S = PED[I] - RR / FR + S
            RG = S * KGD * FR + RG
            RI = S * KID * FR + RI
            S = S * (1.0 - KID - KGD)

        RS = RS + RB  # 加上不透水面积产流

        return FR, S, RS, RI, RG


@jit(nopython=True)
def get_days_in_month(month: int) -> int:
    """
    获取月份天数（简化计算，不考虑闰年）

    Args:
        month: 月份 (1-12)

    Returns:
        该月天数
    """
    if month in (4, 6, 9, 11):
        return 30
    elif month == 2:
        return 28
    else:
        return 31


def sms_3(
    p_and_e: np.ndarray,
    parameters: np.ndarray,
    warmup_length: int = 72,
    return_state: bool = False,
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                              np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    三水源蓄满产流模型（新安江模型）主函数

    该模型是新安江模型的核心产流模块，采用蓄满产流机制。
    将流域划分为上层、下层和深层三层，分别计算地表径流、壤中流和地下径流。

    Args:
        p_and_e (np.ndarray): 降水和潜在蒸散发，3维: [time, basin, feature=2]
            其中 feature=0 是降水，feature=1 是潜在蒸散发
        parameters (np.ndarray): 模型参数，2维: [basin, parameter]
            参数: [WM, B, IM, K, C, WUMx, WLMx, SM, EX, KG, KI]
        warmup_length (int, optional): 预热期长度（默认: 72）
        return_state (bool, optional): 是否返回内部状态变量（默认: False）
        **kwargs: 其他参数，包括:
            - time_interval_hours (default: 1.0): 时段长度（小时）
            - initial_states (default: None): 初始状态字典

    Returns:
        Union[np.ndarray, Tuple]: 取决于return_state参数:
        - return_state=False: RunoffSim数组 [time, basin, 1]
        - return_state=True: (RunoffSim, RS, RI, RG, WU, WL, WD, S, FR)
    """
    # 获取数据维度
    time_steps, num_basins, _ = p_and_e.shape
    T = kwargs.get("time_interval_hours", 1.0)  # 时段长度（小时）

    # 提取参数 - 所有都是 [basin] 数组
    WM = parameters[:, 0]      # 流域平均张力水容量 (mm)
    B = parameters[:, 1]       # 蓄水容量曲线方次
    IM = parameters[:, 2]      # 不透水面积比例
    K = parameters[:, 3]       # 流域蒸发折算系数
    C = parameters[:, 4]       # 深层蒸散发系数
    WUMx = parameters[:, 5]    # 上层张力水容量占总容量的比例
    WLMx = parameters[:, 6]    # 下层张力水容量占总容量的比例
    SM = parameters[:, 7]      # 自由水蓄水库容量 (mm)
    EX = parameters[:, 8]      # 自由水蓄水库容曲线方次
    KG = parameters[:, 9]      # 地下水消退系数
    KI = parameters[:, 10]     # 壤中流消退系数

    # 计算各层张力水容量
    WUM = WUMx * WM
    WLM = (1 - WUMx) * WLMx * WM
    WDM = WM - WUM - WLM

    # 分层计算参数
    WMM = (1.0 + B) * WM / (1.0 - IM)
    SMM = (1.0 + EX) * SM

    # 处理预热期
    if warmup_length > 0:
        p_and_e_warmup = p_and_e[0:warmup_length, :, :]
        warmup_kwargs = {k: v for k, v in kwargs.items() if k != "initial_states"}
        *_, WU_w, WL_w, WD_w, S_w, FR_w = sms_3(
            p_and_e_warmup,
            parameters,
            warmup_length=0,
            return_state=True,
            **warmup_kwargs
        )
        WUP = WU_w[-1, :, 0].copy()
        WLP = WL_w[-1, :, 0].copy()
        WDP = WD_w[-1, :, 0].copy()
        SP = S_w[-1, :, 0].copy()
        FRP = FR_w[-1, :, 0].copy()
    else:
        # 默认初始状态
        WUP = WUM * 0.5
        WLP = WLM * 0.5
        WDP = WDM * 0.5
        SP = SM * 0.3
        FRP = np.full(num_basins, 0.3)

    # 应用初始状态覆盖（如果提供）
    initial_states = kwargs.get("initial_states", None)
    if initial_states is not None:
        if "WUP" in initial_states:
            WUP = np.full(num_basins, initial_states["WUP"])
        if "WLP" in initial_states:
            WLP = np.full(num_basins, initial_states["WLP"])
        if "WDP" in initial_states:
            WDP = np.full(num_basins, initial_states["WDP"])
        if "SP" in initial_states:
            SP = np.full(num_basins, initial_states["SP"])
        if "FRP" in initial_states:
            FRP = np.full(num_basins, initial_states["FRP"])

    inputs = p_and_e[warmup_length:, :, :]
    actual_time_steps = inputs.shape[0]

    # 初始化状态和输出数组 - [time, basin]
    WU = np.zeros((actual_time_steps, num_basins))
    WL = np.zeros((actual_time_steps, num_basins))
    WD = np.zeros((actual_time_steps, num_basins))
    W = np.zeros((actual_time_steps, num_basins))
    S = np.zeros((actual_time_steps, num_basins))
    FR = np.zeros((actual_time_steps, num_basins))

    RSP = np.zeros((actual_time_steps, num_basins))  # 地表径流
    RIP = np.zeros((actual_time_steps, num_basins))  # 壤中流
    RGP = np.zeros((actual_time_steps, num_basins))  # 地下径流
    RP = np.zeros((actual_time_steps, num_basins))   # 总径流

    # 调整消退系数确保稳定性
    KG_adj = KG.copy()
    KI_adj = KI.copy()
    for b in range(num_basins):
        if KG_adj[b] + KI_adj[b] > 0.9:
            tmp = (KG_adj[b] + KI_adj[b] - 0.9) / (KG_adj[b] + KI_adj[b])
            KG_adj[b] = KG_adj[b] - KG_adj[b] * tmp
            KI_adj[b] = KI_adj[b] - KI_adj[b] * tmp

    # 时段消退系数转换
    HGI = (1.0 - np.power(1.0 - KG_adj - KI_adj, T / 24.0)) / (KG_adj + KI_adj)
    KG_adj = HGI * KG_adj
    KI_adj = HGI * KI_adj

    # 分层计算阈值
    DIV = 5.0

    # 主时间循环
    for i in range(actual_time_steps):
        prcp = inputs[i, :, 0]  # 降水
        pet = inputs[i, :, 1]   # 潜在蒸散发

        # 获取前一时段状态
        if i == 0:
            WU0 = WUP
            WL0 = WLP
            WD0 = WDP
            W0 = WUP + WLP + WDP
            S0 = SP
            FR0 = FRP
        else:
            WU0 = WU[i - 1, :]
            WL0 = WL[i - 1, :]
            WD0 = WD[i - 1, :]
            W0 = W[i - 1, :]
            S0 = S[i - 1, :]
            FR0 = FR[i - 1, :]

        # 逐流域计算
        for b in range(num_basins):
            # 计算时段蒸散发（PET已在预处理中转换为时段值）
            EM = pet[b]  # 已经是时段PET，无需再次转换
            EK = K[b] * EM

            # 计算净雨
            PE = prcp[b] - EK

            # 根据净雨大小选择计算方法
            if PE >= 2 * DIV:
                # 大雨，分层计算
                ND = int(math.floor(PE / DIV))
                PED = np.zeros(ND)
                RD = np.zeros(ND)

                for II in range(ND - 1):
                    PED[II] = DIV
                PED[ND - 1] = PE - (ND - 1) * DIV
            else:
                # 小雨，不分层
                ND = 1
                PED = np.array([PE])
                RD = np.zeros(ND)

            # 张力水蓄水量计算
            W_new, WU_new, WL_new, WD_new = calculate_tension_water_storage(
                EK, W0[b], WU0[b], WL0[b], WD0[b],
                RD, PED, ND,
                PE, WM[b], WUM[b], WLM[b], WDM[b],
                WMM[b], B[b], C[b]
            )

            # 自由水蓄水库及分水源计算
            FR_new, S_new, RS, RI, RG = calculate_free_water_and_runoff(
                FR0[b], S0[b], RD, PED, ND,
                PE, IM[b], SM[b], SMM[b], EX[b],
                KG_adj[b], KI_adj[b]
            )

            # 确保产流非负
            RS = max(0.0, RS)
            RI = max(0.0, RI)
            RG = max(0.0, RG)

            # 保存结果
            W[i, b] = W_new
            WU[i, b] = WU_new
            WL[i, b] = WL_new
            WD[i, b] = WD_new
            S[i, b] = S_new
            FR[i, b] = FR_new
            RSP[i, b] = RS
            RIP[i, b] = RI
            RGP[i, b] = RG
            RP[i, b] = RS + RI + RG

    # 扩展维度 [time, basin, 1]
    RunoffSim = np.expand_dims(RP, axis=2)
    RS_out = np.expand_dims(RSP, axis=2)
    RI_out = np.expand_dims(RIP, axis=2)
    RG_out = np.expand_dims(RGP, axis=2)
    WU_out = np.expand_dims(WU, axis=2)
    WL_out = np.expand_dims(WL, axis=2)
    WD_out = np.expand_dims(WD, axis=2)
    S_out = np.expand_dims(S, axis=2)
    FR_out = np.expand_dims(FR, axis=2)

    if return_state:
        return RunoffSim, RS_out, RI_out, RG_out, WU_out, WL_out, WD_out, S_out, FR_out
    else:
        return RunoffSim


# ======================================================
#                数据处理函数
# ======================================================

def preprocess_json(input_json_str: str) -> Dict:
    """
    预处理JSON输入数据

    输入：JSON 字符串（SMS_3-input.json 结构）
    输出：dict（模型需要的数据）

    Args:
        input_json_str: JSON字符串

    Returns:
        预处理后的数据字典
    """
    data = json.loads(input_json_str)

    # ---------- initial states ----------
    initial_state = data.get('initialState', {})
    initial_states = {
        'WUP': initial_state.get('WUP', 0),
        'WLP': initial_state.get('WLP', 0),
        'WDP': initial_state.get('WDP', 0),
        'SP': initial_state.get('SP', 0),
        'FRP': initial_state.get('FRP', 0)
    }

    # ---------- parameters ----------
    params = data.get('parameters', {})
    param_order_map = {
        'WM': params.get('WM', 120),
        'B': params.get('B', 0.4),
        'IM': params.get('IM', 0.01),
        'K': params.get('K', 0.7),
        'C': params.get('C', 0.15),
        'WUMx': params.get('WUMx', 0.2),
        'WLMx': params.get('WLMx', 0.6),
        'SM': params.get('SM', 30),
        'EX': params.get('EX', 1.5),
        'KG': params.get('KG', 0.3),
        'KI': params.get('KI', 0.3)
    }

    param_names = ['WM', 'B', 'IM', 'K', 'C', 'WUMx', 'WLMx', 'SM', 'EX', 'KG', 'KI']
    parameters = np.array([[param_order_map[n] for n in param_names]])

    # ---------- monthly PET ----------
    monthly_pet_data = params.get('monthlyPET', [])
    monthly_pet = [0.0] * 12
    for mp in monthly_pet_data:
        monthly_pet[mp['month'] - 1] = mp['pet']

    # ---------- rainfall time series ----------
    rainfall_series = data.get('rainfallSeries', [])
    time_list = [e['time'] for e in rainfall_series]
    rain_values = [e['rain'] for e in rainfall_series]

    # 自动检测时间间隔（小时）
    if len(time_list) >= 2:
        t1 = datetime.strptime(time_list[0], "%Y-%m-%d %H:%M:%S")
        t2 = datetime.strptime(time_list[1], "%Y-%m-%d %H:%M:%S")
        dt_hours = (t2 - t1).total_seconds() / 3600.0
    else:
        dt_hours = 1.0  # fallback

    # ---------- 月 PET → 时段 PET ----------
    pet_values = []
    for dt_str in time_list:
        dt_obj = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        month = dt_obj.month
        monthly_et = monthly_pet[month - 1]

        # 使用实际月天数
        year = dt_obj.year
        days_in_month = 28  # 简化处理
        if month in [1, 3, 5, 7, 8, 10, 12]:
            days_in_month = 31
        elif month in [4, 6, 9, 11]:
            days_in_month = 30
        elif month == 2:
            days_in_month = 29 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 28

        daily_et = monthly_et / days_in_month
        hourly_et = daily_et * dt_hours / 24
        pet_values.append(hourly_et)

    # ---------- p_and_e ----------
    T = len(time_list)
    p_and_e = np.zeros((T, 1, 2))
    p_and_e[:, 0, 0] = rain_values
    p_and_e[:, 0, 1] = pet_values

    # ---------- kwargs ----------
    kwargs = {
        "time_interval_hours": dt_hours
    }

    # ---------- 预见期起始时间 ----------
    forecast_start_time = data.get('start', None)

    # ---------- 返回结构 ----------
    return {
        "p_and_e": p_and_e,
        "parameters": parameters,
        "initial_states": initial_states,
        "kwargs": kwargs,
        "time_list": time_list,
        "forecast_start_time": forecast_start_time
    }


def run_model(
    preprocessed: Dict,
    warmup_length: int = 72,
    return_state: bool = True
) -> Dict:
    """
    运行SMS_3模型

    输入：preprocess_json() 的输出
    输出：原始模型计算结果

    Args:
        preprocessed: 预处理后的数据
        warmup_length: 预热期长度
        return_state: 是否返回状态变量

    Returns:
        模型输出字典
    """
    p_and_e = preprocessed["p_and_e"]
    parameters = preprocessed["parameters"]
    initial_states = preprocessed["initial_states"]
    kwargs = preprocessed["kwargs"]
    time_list = preprocessed["time_list"]

    # sms_3() 需要 initial_states 作为命名参数
    all_kwargs = kwargs.copy()
    all_kwargs["initial_states"] = initial_states

    # =============== 调用模型 ===============
    RunoffSim, RS, RI, RG, WU, WL, WD, S, FR = sms_3(
        p_and_e=p_and_e,
        parameters=parameters,
        warmup_length=warmup_length,
        return_state=return_state,
        **all_kwargs
    )

    # 返回数据
    return {
        "raw": {
            "RunoffSim": RunoffSim,
            "RS": RS,
            "RI": RI,
            "RG": RG,
            "WU": WU,
            "WL": WL,
            "WD": WD,
            "S": S,
            "FR": FR
        },
        "time_list": time_list,
        "parameters": parameters,
        "p_and_e": p_and_e,
        "kwargs": kwargs,
        "warmup_length": warmup_length,
        "forecast_start_time": preprocessed.get("forecast_start_time")
    }


def postprocess(model_output_dict: Dict, model_version: str = "1.0") -> str:
    """
    后处理模型输出，生成JSON字符串

    输入：run_model() 的输出 dict
    输出：最终 JSON 字符串

    Args:
        model_output_dict: 模型输出字典
        model_version: 模型版本号

    Returns:
        JSON字符串
    """
    raw = model_output_dict["raw"]

    RunoffSim = raw["RunoffSim"]
    RS = raw["RS"]
    RI = raw["RI"]
    RG = raw["RG"]
    WU = raw["WU"]
    WL = raw["WL"]
    WD = raw["WD"]
    S = raw["S"]
    FR = raw["FR"]

    time_list = model_output_dict["time_list"]
    warmup_length = model_output_dict["warmup_length"]
    p_and_e = model_output_dict["p_and_e"]

    # 降水和净雨
    prcp = p_and_e[:, 0, 0]
    pet = p_and_e[:, 0, 1]

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
                "rain": float(prcp[i]),
                "PET": float(pet[i]),
                "RunoffSim": 0.0,
                "RS": 0.0,
                "RI": 0.0,
                "RG": 0.0,
            }
        else:
            idx = i - warmup_length
            entry = {
                "time": time_list[i],
                "isWarmUp": False,
                "rain": float(prcp[i]),
                "PET": float(pet[i]),
                "RunoffSim": float(RunoffSim[idx, 0, 0]),
                "RS": float(RS[idx, 0, 0]),
                "RI": float(RI[idx, 0, 0]),
                "RG": float(RG[idx, 0, 0]),
            }

        time_series.append(entry)

    # =========== 最终状态 ===========
    final_idx = len(RunoffSim) - 1
    final_state = {
        "WUP": float(WU[final_idx, 0, 0]),
        "WLP": float(WL[final_idx, 0, 0]),
        "WDP": float(WD[final_idx, 0, 0]),
        "SP": float(S[final_idx, 0, 0]),
        "FRP": float(FR[final_idx, 0, 0]),
    }

    # =========== 输出结构 ===========
    output = {
        "modelName": "SMS_3",
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
    warmup_length: int = 72,
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
    print("SMS_3 Model 运行完成")
    print(f"输入文件：{input_json_path}")
    print(f"输出文件：{output_json_path}")
    print("=" * 60)
    return output_json_path


if __name__ == "__main__":
    # 测试示例
    input_file = "SMS_3-input.json"
    output_file = "SMS_3-output.json"

    main(input_file, output_file, warmup_length=0)
