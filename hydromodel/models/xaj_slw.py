"""
Author: zhuanglaihong
Date: 2025-08-23 23:01:02
LastEditTime: 2025-08-23 19:35:20
LastEditors: zhuanglaihong
Description: XinAnJiang Model - Python implementation using SMS_3 and LAG_3 algorithms with vectorized operations
FilePath: /hydromodel/hydromodel/models/xaj_slw.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import json
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import traceback
from numba import jit

from hydromodel.models.model_config import MODEL_PARAM_DICT
from hydromodel.models.param_utils import process_parameters


@jit(nopython=True)
def calculate_net_precipitation(
    precipitation: np.ndarray,
    potential_evapotranspiration: np.ndarray,
    kc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算净雨和实际蒸散发

    Parameters
    ----------
    precipitation : np.ndarray
        降水量
    potential_evapotranspiration : np.ndarray
        潜在蒸散发
    kc : np.ndarray
        蒸散发系数

    Returns
    -------
    tuple
        (净雨, 实际蒸散发)
    """
    edt = kc * potential_evapotranspiration  # 实际蒸散发
    pe = precipitation - edt  # 净雨
    return pe, edt


def sms3_runoff_generation_vectorized(
    precipitation: np.ndarray,
    evapotranspiration: np.ndarray,
    wu: np.ndarray,
    wl: np.ndarray,
    wd: np.ndarray,
    s: np.ndarray,
    fr: np.ndarray,
    wm: float,
    wumx: float,
    wlmx: float,
    kc: float,
    b: float,
    c: float,
    im: float,
    sm: float,
    ex: float,
    kg: float,
    ki: float,
    time_interval: float,
    time_steps: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    向量化SMS3产流模型

    Parameters
    ----------
    precipitation : np.ndarray
        降水量时间序列
    evapotranspiration : np.ndarray
        蒸发量时间序列
    wu, wl, wd : np.ndarray
        初始张力水含量
    s : np.ndarray
        初始自由水蓄量
    fr : np.ndarray
        初始产流面积系数
    wm, wumx, wlmx, kc, b, c, im, sm, ex, kg, ki : float
        模型参数
    time_interval : float
        时间间隔
    es : np.ndarray
        月蒸发量
    time_steps : int
        时间步数

    Returns
    -------
    tuple
        产流结果和状态变量
    """
    # 计算衍生参数
    wum = wumx * wm
    wlm = (1.0 - wumx) * wlmx * wm
    wdm = wm - wum - wlm
    wmm = (1.0 + b) * wm / (1.0 - im)
    smm = (1.0 + ex) * sm

    # 调整KG和KI
    if kg + ki > 0.9:
        tmp = (kg + ki - 0.9) / (kg + ki)
        kg = kg - kg * tmp
        ki = ki - ki * tmp

    # 计算HGI并调整KG和KI
    hgi = (1.0 - np.power((1.0 - kg - ki), (time_interval / 24.0))) / (kg + ki)
    kg = hgi * kg
    ki = hgi * ki

    # 初始化输出数组
    wu_out = np.zeros(time_steps)
    wl_out = np.zeros(time_steps)
    wd_out = np.zeros(time_steps)
    s_out = np.zeros(time_steps)
    fr_out = np.zeros(time_steps)
    rs = np.zeros(time_steps)  # 产流数组长度不变
    ri = np.zeros(time_steps)
    rg = np.zeros(time_steps)
    runoff_total = np.zeros(time_steps)

    # 设置初始值
    wu_curr = wu[0]
    wl_curr = wl[0]
    wd_curr = wd[0]
    s_curr = s[0]
    fr_curr = fr[0]

    # 存储初始值
    wu_out[0] = wu_curr
    wl_out[0] = wl_curr
    wd_out[0] = wd_curr
    s_out[0] = s_curr
    fr_out[0] = fr_curr

    div = 5.0

    # 主循环
    for i in range(time_steps - 1):
        # 直接使用输入的蒸散发数据
        ek = evapotranspiration[i]
        pe = precipitation[i] - ek

        # 产流计算
        w_curr = wu_curr + wl_curr + wd_curr

        if pe >= 2 * div:
            nd = int(np.floor(pe / div))
            ped = np.full(nd, div)
            ped[-1] = pe - (nd - 1) * div
        else:
            nd = 1
            ped = np.array([pe])

        rd = np.zeros(nd)

        # YIELD1计算
        if pe <= 0.0:
            r = 0.0
            if wu_curr + pe >= 0.0:
                wu_curr = wu_curr + pe
            else:
                eu = wu_curr + ek + pe
                wu_curr = 0.0
                el = (ek - eu) * wl_curr / wlm
                if wl_curr < c * wlm:
                    el = c * (ek - eu)
                if wl_curr - el < 0.0:
                    ed = el - wl_curr
                    el = wl_curr
                    wl_curr = 0.0
                    wd_curr = wd_curr - ed
                else:
                    wl_curr = wl_curr - el
            w_curr = wu_curr + wl_curr + wd_curr
        else:
            a = 0.0
            if wm - w_curr < 0.0001:
                a = wmm
            else:
                a = wmm * (
                    1.0 - np.power((1.0 - w_curr / wm), (1.0 / (1.0 + b)))
                )

            r = 0.0
            peds = 0.0
            for j in range(nd):
                a = a + ped[j]
                peds = peds + ped[j]
                ri_temp = r
                r = peds - wm + w_curr
                if a < wmm:
                    r = r + wm * np.power((1.0 - a / wmm), (1.0 + b))
                rd[j] = r - ri_temp

            if wu_curr + pe - r <= wum:
                wu_curr = wu_curr + pe - r
            else:
                if wu_curr + wl_curr + pe - r - wum >= wlm:
                    wu_curr = wum
                    wl_curr = wlm
                    wd_curr = w_curr + peds - r - wu_curr - wl_curr
                    if wd_curr > wdm:
                        wd_curr = wdm
                else:
                    wl_curr = wu_curr + wl_curr + pe - r - wum
                    wu_curr = wum
            w_curr = wu_curr + wl_curr + wd_curr

        # DIVI31计算
        if pe <= 0.0:
            rs_curr = 0.0
            rg_curr = s_curr * kg * fr_curr
            ri_curr = s_curr * ki * fr_curr
            s_curr = s_curr * (1.0 - kg - ki)
        else:
            rb = im * pe
            kid = (1.0 - np.power((1.0 - (kg + ki)), (1.0 / nd))) / (kg + ki)
            kgd = kid * kg
            kid = kid * ki

            rs_curr = 0.0
            ri_curr = 0.0
            rg_curr = 0.0

            for j in range(nd):
                td = rd[j] - im * ped[j]
                x = fr_curr
                if ped[j] > 0:
                    fr_curr = td / ped[j]
                    s_curr = x * s_curr / max(fr_curr, 0.0001)

                rr = 0
                if s_curr >= sm:
                    rr = (ped[j] + s_curr - sm) * fr_curr
                else:
                    au = smm * (
                        1.0 - np.power((1.0 - s_curr / sm), (1.0 / (1.0 + ex)))
                    )
                    if au + ped[j] < smm:
                        rr = (
                            ped[j]
                            - sm
                            + s_curr
                            + sm
                            * np.power((1.0 - (ped[j] + au) / smm), (1.0 + ex))
                        ) * fr_curr
                    else:
                        rr = (ped[j] + s_curr - sm) * fr_curr

                rs_curr = rr + rs_curr
                s_curr = ped[j] - rr / max(fr_curr, 0.0001) + s_curr
                rg_curr = s_curr * kgd * fr_curr + rg_curr
                ri_curr = s_curr * kid * fr_curr + ri_curr
                s_curr = s_curr * (1.0 - kid - kgd)

            rs_curr = rs_curr + rb

        # 确保非负值
        rs_curr = max(0.0, rs_curr)
        ri_curr = max(0.0, ri_curr)
        rg_curr = max(0.0, rg_curr)

        # 存储结果
        wu_out[i] = wu_curr  # 状态变量存储到i+1位置
        wl_out[i] = wl_curr
        wd_out[i] = wd_curr
        s_out[i] = s_curr
        fr_out[i] = fr_curr
        rs[i] = rs_curr  # 产流存储到i位置
        ri[i] = ri_curr
        rg[i] = rg_curr
        runoff_total[i] = rs_curr + ri_curr + rg_curr

    return wu_out, wl_out, wd_out, s_out, fr_out, rs, ri, rg, runoff_total


def lag3_routing_vectorized(
    rs: np.ndarray,
    ri: np.ndarray,
    rg: np.ndarray,
    time_interval: float,
    area: float,
    ci: float,
    cg: float,
    lag: float,
    cs: float,
    kk: float,
    x: float,
    mp: int,
    qsp: float = 0.0,
    qip: float = 0.0,
    qgp: float = 0.0,
    qsig_initial: np.ndarray = None,
    qx_initial: np.ndarray = None,
) -> np.ndarray:
    """
    向量化LAG3汇流模型

    Parameters
    ----------
    rs, ri, rg : np.ndarray
        地表水、壤中水、地下水产流量
    time_interval : float
        时间间隔
    area : float
        流域面积
    ci, cg : float
        消退系数
    lag : float
        滞时
    cs : float
        地面径流消退系数
    kk : float
        马斯京根K参数
    x : float
        马斯京根X参数
    mp : int
        河段数
    qsp, qip, qgp : float
        初始流量
    qsig_initial, qx_initial : np.ndarray
        初始状态数组

    Returns
    -------
    np.ndarray
        最终流量
    """
    time_steps = len(rs)

    # 单位转换系数
    cp = area / time_interval / 3.6

    # 参数时段转换
    d = 24 / time_interval
    cid = np.power(ci, 1.0 / d)
    cgd = np.power(cg, 1.0 / d)

    # 线性水库汇流
    qs = np.zeros(time_steps)
    qi = np.zeros(time_steps)
    qg = np.zeros(time_steps)

    # 第一个时间步
    qs[0] = rs[0] * cp
    qi[0] = qip + ri[0] * (1 - cid) * cp
    qg[0] = qgp + rg[0] * (1 - cgd) * cp

    # 后续时间步
    for i in range(1, time_steps):
        qs[i] = rs[i] * cp
        qi[i] = qi[i - 1] * cid + ri[i] * (1 - cid) * cp
        qg[i] = qg[i - 1] * cgd + rg[i] * (1 - cgd) * cp

    # 总入流
    qtmp = qs + qi + qg
    qtmp = np.maximum(qtmp, 0.0)

    # 河网汇流-滞时法
    qf = np.zeros(time_steps)
    t = int(lag)

    if qsig_initial is None:
        qsig_initial = np.zeros(max(t, 3))

    if t <= 0:
        t = 0
        for i in range(time_steps):
            if i == 0:
                qf[i] = qsig_initial[0]
            else:
                qf[i] = cs * qf[i - 1] + (1 - cs) * qtmp[i]
    else:
        for i in range(time_steps):
            if i < t:
                if i < len(qsig_initial):
                    qf[i] = qsig_initial[i]
                else:
                    qf[i] = qsig_initial[-1] if len(qsig_initial) > 0 else 0.0
            else:
                qf[i] = cs * qf[i - 1] + (1 - cs) * qtmp[i - t]

    # 河道演进-马斯京根法
    if qx_initial is None:
        qx_initial = np.zeros(mp + 1)

    qc = np.zeros(mp + 1)
    for j in range(mp + 1):
        if j < len(qx_initial):
            qc[j] = qx_initial[j]
        else:
            qc[j] = qc[j - 1] if j > 0 else 0.0

    # 马斯京根系数（与Java版本一致）
    fkt = kk - kk * x + 0.5 * time_interval
    c0 = (0.5 * time_interval - kk * x) / fkt
    c1 = (kk * x + 0.5 * time_interval) / fkt
    c2 = (kk - kk * x - 0.5 * time_interval) / fkt

    q_routing = np.zeros(time_steps)

    if c0 >= 0.0 and c2 >= 0.0:  # 满足马斯京根法的适用条件
        for i in range(time_steps):
            if i == 0:
                qi1 = qf[i]
            else:
                qi1 = qf[i - 1]
            qi2 = qf[i]

            # 分段计算
            if mp > 0:
                for j in range(mp):
                    qo1 = qc[j]
                    qo2 = c0 * qi2 + c1 * qi1 + c2 * qo1
                    qi1 = qo1
                    qi2 = qo2
                    qc[j] = qo2
            else:
                qo2 = qf[i]

            if qo2 < 0.0001:
                qo2 = 0.0
            q_routing[i] = qo2
    else:
        # 不考虑河道演进
        q_routing = qf.copy()

    return q_routing


def xaj_slw(
    p_and_e: np.ndarray,
    parameters: np.ndarray,
    warmup_length: int = 365,
    return_state: bool = False,
    normalized_params: Union[bool, str] = "auto",
    **kwargs,
) -> Union[
    np.ndarray,
    Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
]:
    """
    向量化新安江松辽水文模型（使用SMS3和LAG3算法）

    该函数实现了新安江模型的完全NumPy向量化，
    使用[时间, 流域, 特征]张量操作同时处理所有流域。

    Parameters
    ----------
    p_and_e : np.ndarray
        降雨和潜在蒸散发数据，3维: [时间, 流域, 特征=2]
        其中特征=0为降雨量，特征=1为潜在蒸散发
    parameters : np.ndarray
        模型参数，2维: [流域, 参数]
        参数顺序: [WUP, WLP, WDP, SP, FRP, WM, WUMx, WLMx, KC, B, C, IM,
                 SM, EX, KG, KI, CS, CI, CG, LAG, KK, X, MP, QSP, QIP, QGP]
    warmup_length : int, optional
        预热期长度 (默认: 365)
    return_state : bool, optional
        如果为True，返回内部状态变量 (默认: False)
    normalized_params : Union[bool, str], optional
        参数格式说明:
        - "auto": 自动检测参数格式 (默认)
        - True: 参数已归一化 (0-1范围)，转换为原始尺度
        - False: 参数已在原始尺度，直接使用
    **kwargs
        其他关键字参数，包括time_interval_hours (默认: 1.0)

    Returns
    -------
    result : np.ndarray or tuple
        如果return_state为False: QSim数组 [时间, 流域, 1]
        如果return_state为True: (QSim, runoffSim, rs, ri, rg, pe, wu, wl, wd)元组
    """
    # Get data dimensions
    time_steps, num_basins, _ = p_and_e.shape
    time_interval = kwargs.get("time_interval_hours", 1.0)
    area = kwargs.get("area", 100.0)

    # Process parameters using unified parameter handling
    processed_parameters = parameters.copy()
    if normalized_params != False:
        model_param_dict = MODEL_PARAM_DICT.get("xaj_slw")
        if model_param_dict is not None:
            param_ranges = model_param_dict["param_range"]
            processed_parameters = process_parameters(
                parameters, param_ranges, normalized=normalized_params
            )

    # Extract parameters - all are [basin] arrays
    wup = processed_parameters[:, 0]  # Initial upper layer tension water
    wlp = processed_parameters[:, 1]  # Initial lower layer tension water
    wdp = processed_parameters[:, 2]  # Initial deep layer tension water
    sp = processed_parameters[:, 3]  # Initial free water storage
    frp = processed_parameters[:, 4]  # Initial runoff area ratio
    wm = processed_parameters[:, 5]  # Total tension water capacity
    wumx = processed_parameters[:, 6]  # Upper layer capacity ratio
    wlmx = processed_parameters[:, 7]  # Lower layer capacity ratio
    kc = processed_parameters[:, 8]  # Evaporation coefficient
    b = processed_parameters[:, 9]  # Exponent of tension water capacity curve
    c = processed_parameters[:, 10]  # Deep evapotranspiration coefficient
    im = processed_parameters[:, 11]  # Impervious area ratio
    sm = processed_parameters[:, 12]  # Average free water capacity
    ex = processed_parameters[:, 13]  # Exponent of free water capacity curve
    kg = processed_parameters[:, 14]  # Groundwater outflow coefficient
    ki = processed_parameters[:, 15]  # Interflow outflow coefficient
    cs = processed_parameters[:, 16]  # Channel system recession constant
    ci = processed_parameters[:, 17]  # Lower interflow recession constant
    cg = processed_parameters[:, 18]  # Groundwater storage recession constant
    lag = processed_parameters[:, 19]  # Lag time
    kk = processed_parameters[:, 20]  # Muskingum K parameter
    x = processed_parameters[:, 21]  # Muskingum X parameter
    mp = processed_parameters[:, 22]  # Number of Muskingum reaches
    qsp = processed_parameters[:, 23]  # Initial surface flow
    qip = processed_parameters[:, 24]  # Initial interflow
    qgp = processed_parameters[:, 25]  # Initial groundwater flow

    # Default monthly evaporation (can be customized)
    es = np.array([100, 90, 80, 70, 60, 50, 40, 50, 60, 70, 80, 90])

    # Handle warmup period
    if warmup_length > 0:
        p_and_e_warmup = p_and_e[0:warmup_length, :, :]
        *_, wu_init, wl_init, wd_init, s_init, fr_init = xaj_slw(
            p_and_e_warmup,
            parameters,
            warmup_length=0,
            return_state=True,
            normalized_params=False,  # Already processed
            **kwargs,
        )
        # Use final states as initial conditions
        wu0 = wu_init[-1, :, 0].copy()
        wl0 = wl_init[-1, :, 0].copy()
        wd0 = wd_init[-1, :, 0].copy()
        s0 = s_init[-1, :, 0].copy()
        fr0 = fr_init[-1, :, 0].copy()
    else:
        # Default initial states
        wu0 = wup.copy()
        wl0 = wlp.copy()
        wd0 = wdp.copy()
        s0 = sp.copy()
        fr0 = frp.copy()

    inputs = p_and_e[warmup_length:, :, :]
    actual_time_steps = inputs.shape[0]

    # Initialize output arrays
    q_sim = np.zeros((actual_time_steps, num_basins))
    runoff_sim = np.zeros((actual_time_steps, num_basins))
    rs_out = np.zeros((actual_time_steps, num_basins))
    ri_out = np.zeros((actual_time_steps, num_basins))
    rg_out = np.zeros((actual_time_steps, num_basins))
    pe_out = np.zeros((actual_time_steps, num_basins))
    wu_out = np.zeros(
        (actual_time_steps, num_basins)
    )  # 改回与SMS函数返回值一致的长度
    wl_out = np.zeros((actual_time_steps, num_basins))
    wd_out = np.zeros((actual_time_steps, num_basins))

    # Process each basin
    for basin_idx in range(num_basins):
        # Extract time series for this basin
        prcp = inputs[:, basin_idx, 0]
        pet = inputs[:, basin_idx, 1]

        # Calculate net precipitation
        pe, edt = calculate_net_precipitation(
            prcp, pet, np.full_like(prcp, kc[basin_idx])
        )

        # Initial states for this basin
        wu_init = np.array([wu0[basin_idx]])
        wl_init = np.array([wl0[basin_idx]])
        wd_init = np.array([wd0[basin_idx]])
        s_init = np.array([s0[basin_idx]])
        fr_init = np.array([fr0[basin_idx]])

        # Run SMS3 runoff generation
        (
            wu_new,
            wl_new,
            wd_new,
            s_new,
            fr_new,
            rs_basin,
            ri_basin,
            rg_basin,
            runoff_basin,
        ) = sms3_runoff_generation_vectorized(
            prcp,
            pet,  # 直接传递蒸散发数据
            wu_init,
            wl_init,
            wd_init,
            s_init,
            fr_init,
            wm[basin_idx],
            wumx[basin_idx],
            wlmx[basin_idx],
            kc[basin_idx],
            b[basin_idx],
            c[basin_idx],
            im[basin_idx],
            sm[basin_idx],
            ex[basin_idx],
            kg[basin_idx],
            ki[basin_idx],
            time_interval,
            actual_time_steps,
        )

        # Run LAG3 routing
        qsig_initial = np.zeros(max(int(lag[basin_idx]), 3))
        qx_initial = np.zeros(int(mp[basin_idx]) + 1)

        q_basin = lag3_routing_vectorized(
            rs_basin,
            ri_basin,
            rg_basin,
            time_interval,
            area,
            ci[basin_idx],
            cg[basin_idx],
            lag[basin_idx],
            cs[basin_idx],
            kk[basin_idx],
            x[basin_idx],
            int(mp[basin_idx]),
            qsp[basin_idx],
            qip[basin_idx],
            qgp[basin_idx],
            qsig_initial,
            qx_initial,
        )

        # Store results
        q_sim[:, basin_idx] = q_basin
        runoff_sim[:, basin_idx] = runoff_basin
        rs_out[:, basin_idx] = rs_basin
        ri_out[:, basin_idx] = ri_basin
        rg_out[:, basin_idx] = rg_basin
        pe_out[:, basin_idx] = pe
        wu_out[:, basin_idx] = wu_new
        wl_out[:, basin_idx] = wl_new
        wd_out[:, basin_idx] = wd_new

    # Ensure non-negative discharge
    q_sim = np.maximum(q_sim, 0.0)

    # Format outputs to match DHF interface: [seq, batch, feature]
    q_sim = np.expand_dims(q_sim, axis=2)
    runoff_sim = np.expand_dims(runoff_sim, axis=2)
    rs_out = np.expand_dims(rs_out, axis=2)
    ri_out = np.expand_dims(ri_out, axis=2)
    rg_out = np.expand_dims(rg_out, axis=2)
    pe_out = np.expand_dims(pe_out, axis=2)
    wu_out = np.expand_dims(wu_out, axis=2)
    wl_out = np.expand_dims(wl_out, axis=2)
    wd_out = np.expand_dims(wd_out, axis=2)

    if return_state:
        return (
            q_sim,
            runoff_sim,
            rs_out,
            ri_out,
            rg_out,
            pe_out,
            wu_out,
            wl_out,
            wd_out,
        )
    else:
        return q_sim


def load_xaj_data_from_json(
    json_file_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从JSON文件加载XAJ模型所需的时序数据和参数

    Parameters
    ----------
    json_file_path : str
        JSON文件路径，包含时间序列、降雨数据和模型参数

    Returns
    -------
    p_and_e : np.ndarray
        降雨和蒸发数据 [time, basin=1, feature=2]
    parameters : np.ndarray
        模型参数 [basin=1, parameter=26]
    """
    # 读取JSON文件
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 解析时间序列、降雨和蒸散发数据
    dt = (
        json.loads(data["dt"]) if isinstance(data["dt"], str) else data["dt"]
    )  # 时间序列
    rain = (
        json.loads(data["rain"])
        if isinstance(data["rain"], str)
        else data["rain"]
    )  # 降雨数据
    evap = (
        json.loads(data["evaporation"])
        if isinstance(data["evaporation"], str)
        else data["evaporation"]
    )  # 蒸散发数据

    # 构建p_and_e数组 [time, basin=1, feature=2]
    time_steps = len(rain)
    p_and_e = np.zeros((time_steps, 1, 2))
    p_and_e[:, 0, 0] = rain  # 降雨数据
    p_and_e[:, 0, 1] = evap  # 蒸散发数据

    # 构建参数数组 [basin=1, parameter=26]
    # 参数顺序: [WUP, WLP, WDP, SP, FRP, WM, WUMx, WLMx, K, B, C, IM,
    #          SM, EX, KG, KI, CS, CI, CG, LAG, KK, X, MP, QSP, QIP, QGP]
    parameters = np.array(
        [
            [
                float(data["WUP"]),
                float(data["WLP"]),
                float(data["WDP"]),
                float(data["SP"]),
                float(data["FRP"]),
                float(data["WM"]),
                float(data["WUMx"]),
                float(data["WLMx"]),
                float(data["K"]),
                float(data["B"]),
                float(data["C"]),
                float(data["IM"]),
                float(data["SM"]),
                float(data["EX"]),
                float(data["KG"]),
                float(data["KI"]),
                float(data["CS"]),
                float(data["CI"]),
                float(data["CG"]),
                float(data["LAG"]),
                float(data["KK"]),
                float(data["X"]),
                float(data["MP"]),
                float(data["QSP"]),
                float(data["QIP"]),
                float(data["QGP"]),
            ]
        ]
    )

    return p_and_e, parameters


def load_sms_lag_data_from_json(
    sms_json_path: str,
    lag_json_path: str,
    default_evap: float,
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """
    从SMS和LAG的JSON文件加载XAJ模型所需的数据

    Parameters
    ----------
    sms_json_path : str
        SMS_3模型的JSON文件路径
    lag_json_path : str
        LAG_3模型的JSON文件路径
    default_evap : float, optional
        默认蒸散发值 (默认: 2.0)

    Returns
    -------
    p_and_e : np.ndarray
        降雨和蒸发数据 [time, basin=1, feature=2]
    parameters : np.ndarray
        模型参数 [basin=1, parameter=26]
    time_dates : List[str]
        时间日期列表
    start_time : str
        开始时间
    """
    # 读取SMS JSON文件
    with open(sms_json_path, "r", encoding="utf-8") as f:
        sms_data = json.load(f)

    # 读取LAG JSON文件
    with open(lag_json_path, "r", encoding="utf-8") as f:
        lag_data = json.load(f)

    # 解析时间序列和降雨数据
    dt = sms_data["dt"]
    rain = sms_data["rain"]
    start_time = lag_data.get("start", dt[0] if dt else "")

    # 解析月蒸发量数组
    es = np.array(
        sms_data.get(
            "ES",
            [
                23.8,
                23.2,
                33.7,
                53,
                66.6,
                78.9,
                122.3,
                113.9,
                90.8,
                62.9,
                46.2,
                34.1,
            ],
        )
    )

    # 构建p_and_e数组 [time, basin=1, feature=2]
    time_steps = len(rain)
    p_and_e = np.zeros((time_steps, 1, 2))
    p_and_e[:, 0, 0] = rain  # 降雨数据
    p_and_e[:, 0, 1] = default_evap  # 使用默认蒸散发值

    # 构建参数数组 [basin=1, parameter=26]
    # 参数顺序: [WUP, WLP, WDP, SP, FRP, WM, WUMx, WLMx, KC, B, C, IM,
    #          SM, EX, KG, KI, CS, CI, CG, LAG, KK, X, MP, QSP, QIP, QGP]
    parameters = np.array(
        [
            [
                float(sms_data["WUP"]),  # 0: Initial upper layer tension water
                float(sms_data["WLP"]),  # 1: Initial lower layer tension water
                float(sms_data["WDP"]),  # 2: Initial deep layer tension water
                float(sms_data["SP"]),  # 3: Initial free water storage
                float(sms_data["FRP"]),  # 4: Initial runoff area ratio
                float(sms_data["WM"]),  # 5: Total tension water capacity
                float(sms_data["WUMx"]),  # 6: Upper layer capacity ratio
                float(sms_data["WLMx"]),  # 7: Lower layer capacity ratio
                float(sms_data["K"]),  # 8: Evaporation coefficient (KC)
                float(
                    sms_data["B"]
                ),  # 9: Exponent of tension water capacity curve
                float(
                    sms_data["C"]
                ),  # 10: Deep evapotranspiration coefficient
                float(sms_data["IM"]),  # 11: Impervious area ratio
                float(sms_data["SM"]),  # 12: Average free water capacity
                float(
                    sms_data["EX"]
                ),  # 13: Exponent of free water capacity curve
                float(sms_data["KG"]),  # 14: Groundwater outflow coefficient
                float(sms_data["KI"]),  # 15: Interflow outflow coefficient
                float(lag_data["CS"]),  # 16: Channel system recession constant
                float(
                    lag_data["CI"]
                ),  # 17: Lower interflow recession constant
                float(
                    lag_data["CG"]
                ),  # 18: Groundwater storage recession constant
                float(lag_data["LAG"]),  # 19: Lag time
                float(lag_data["KK"]),  # 20: Muskingum K parameter
                float(lag_data["X"]),  # 21: Muskingum X parameter
                float(lag_data["MP"]),  # 22: Number of Muskingum reaches
                float(lag_data["QSP"]),  # 23: Initial surface flow
                float(lag_data["QIP"]),  # 24: Initial interflow
                float(lag_data["QGP"]),  # 25: Initial groundwater flow
            ]
        ]
    )

    return p_and_e, parameters, dt, start_time, es


def test_sms_lag_integration():
    """
    测试SMS和LAG数据集成的示例函数
    """
    try:
        # 加载数据
        p_and_e, parameters, time_dates, start_time = (
            load_sms_lag_data_from_json(
                "src/main/resources/sms_3_data.json",
                "src/main/resources/lag_3_data.json",
            )
        )

        print("数据加载成功:")
        print(f"时间序列长度: {p_and_e.shape[0]}")
        print(f"流域数量: {p_and_e.shape[1]}")
        print(f"特征数量: {p_and_e.shape[2]}")
        print(f"参数数量: {parameters.shape[1]}")
        print(f"降雨数据前5个值: {p_and_e[:5, 0, 0]}")
        print(f"蒸散发数据前5个值: {p_and_e[:5, 0, 1]}")

        # 运行模型
        print("\n运行XAJ模型...")
        result = xaj_slw(
            p_and_e,
            parameters,
            warmup_length=0,
            return_state=False,
            normalized_params=False,
            time_interval_hours=6.0,  # 根据JSON数据中的clen参数
            area=2163.0,  # 根据JSON数据中的F参数
        )

        print(f"模型运行成功，输出形状: {result.shape}")
        print(f"流量模拟前10个值: {result[:10, 0, 0]}")

        return p_and_e, parameters, result

    except Exception as e:
        print(f"测试失败: {e}")
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    # 运行测试
    test_sms_lag_integration()
