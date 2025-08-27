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
        ek = kc * evapotranspiration[i]
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
        wu_out[i] = wu_curr
        wl_out[i] = wl_curr
        wd_out[i] = wd_curr
        s_out[i] = s_curr
        fr_out[i] = fr_curr
        rs[i] = rs_curr
        ri[i] = ri_curr
        rg[i] = rg_curr
        runoff_total[i] = rs_curr + ri_curr + rg_curr

    return wu_out, wl_out, wd_out, s_out, fr_out, rs, ri, rg, runoff_total


def lchco_vectorized(
    mp: int, rq: float, qx: np.ndarray, c0: float, c1: float, c2: float
) -> float:
    """
    向量化的LCHCO计算
    """
    im = mp + 1
    if im == 1:
        qx[int(im - 1)] = rq
    else:
        for j in range(1, int(im)):
            q1 = rq
            q2 = qx[j - 1]
            q3 = qx[j]
            qx[j - 1] = rq
            rq = c0 * q1 + c1 * q2 + c2 * q3
        qx[int(im - 1)] = rq
    return rq


def lag3_routing_vectorized(
    rs: np.ndarray,
    ri: np.ndarray,
    rg: np.ndarray,
    time_interval: float,
    basin_area: float,
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
    basin_area : float
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
    cp = basin_area / time_interval / 3.6

    # 参数时段转换
    ci = np.power(ci, time_interval / 24.0)
    cg = np.power(cg, time_interval / 24.0)

    # 初始化输出数组
    qs = np.zeros(time_steps)
    qi = np.zeros(time_steps)
    qg = np.zeros(time_steps)
    qsig = np.zeros(time_steps + int(lag))

    # 初始化QSIG数组
    t = int(lag)
    if qsig_initial is None:
        qsig_initial = np.zeros(max(t, 3))

    if lag <= 1:
        qsig[0] = qsig_initial[0]
    else:
        for i in range(t):
            if len(qsig_initial) >= t:
                qsig[i] = qsig_initial[i]
            else:
                if i < len(qsig_initial):
                    qsig[i] = qsig_initial[i]
                else:
                    qsig[i] = qsig[i - 1]

    # 初始化马斯京根参数
    fkt = kk - kk * x + 0.5 * time_interval
    c0 = (0.5 * time_interval - kk * x) / fkt
    c1 = (kk * x + 0.5 * time_interval) / fkt
    c2 = (kk - kk * x - 0.5 * time_interval) / fkt

    # 初始化QX数组
    qx = np.zeros(mp + 1)
    if qx_initial is not None:
        for i in range(mp + 1):
            if i < len(qx_initial):
                qx[i] = qx_initial[i]
            else:
                qx[i] = qx[i - 1] if i > 0 else 0.0

    # 主循环计算
    qip_curr = qip
    qgp_curr = qgp
    qsig1 = qsig[t - 1] if lag > 1 else qsig[0]

    for i in range(time_steps):
        # 计算三水源汇流
        qgp_curr = qgp_curr * cg + rg[i] * (1.0 - cg) * cp
        qip_curr = qip_curr * ci + ri[i] * (1.0 - ci) * cp
        qsp_curr = rs[i] * cp

        # 存储结果
        qg[i] = qgp_curr
        qi[i] = qip_curr
        qs[i] = qsp_curr

        # 计算总入流并更新QSIG
        qsig1 = qsig1 * cs + (qgp_curr + qip_curr + qsp_curr) * (1.0 - cs)
        qtsig = qsig1

        # 使用LCHCO进行马斯京根演算
        qsig[i + t] = lchco_vectorized(mp, qtsig, qx, c0, c1, c2)

    # 提取最终结果
    q_routing = qsig[:time_steps]

    # 确保非负值
    q_routing = np.maximum(q_routing, 0.0)

    return q_routing


def xaj_slw(
    p_and_e: np.ndarray,
    parameters: np.ndarray,
    warmup_length: int = 365,
    return_state: bool = False,
    return_warmup_states: bool = False,
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
    Tuple[
        np.ndarray,
        Dict[str, np.ndarray],
    ],
]:
    """
    向量化新安江松辽水文模型（使用SMS3和LAG3算法）

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
    return_warmup_states : bool, optional
        如果为True，返回预热期后的初始状态 (默认: False)
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
    if "basin_area" not in kwargs:
        raise KeyError("basin_area must be provided")

    time_steps, num_basins, _ = p_and_e.shape
    time_interval = kwargs.get("time_interval_hours", 1.0)
    basin_area = kwargs.get("basin_area", None)  # km^2

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
    frp = processed_parameters[:, 4]  # Initial runoff basin_area ratio
    wm = processed_parameters[:, 5]  # Total tension water capacity
    wumx = processed_parameters[:, 6]  # Upper layer capacity ratio
    wlmx = processed_parameters[:, 7]  # Lower layer capacity ratio
    kc = processed_parameters[:, 8]  # Evaporation coefficient
    b = processed_parameters[:, 9]  # Exponent of tension water capacity curve
    c = processed_parameters[:, 10]  # Deep evapotranspiration coefficient
    im = processed_parameters[:, 11]  # Impervious basin_area ratio
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

    # Handle warmup period
    if warmup_length > 0:
        p_and_e_warmup = p_and_e[0:warmup_length, :, :]
        # Remove initial_states from kwargs for warmup period to avoid applying override during warmup
        warmup_kwargs = {
            k: v for k, v in kwargs.items() if k != "initial_states"
        }
        *_, wu_init, wl_init, wd_init, s_init, fr_init = xaj_slw(
            p_and_e_warmup,
            parameters,
            warmup_length=0,
            return_state=True,
            normalized_params=False,
            **warmup_kwargs,
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

    # Apply initial state overrides if provided (only after warmup in main call)
    initial_states = kwargs.get("initial_states", None)
    if initial_states is not None:
        if "wu0" in initial_states:
            wu0.fill(initial_states["wu0"])
        if "wl0" in initial_states:
            wl0.fill(initial_states["wl0"])
        if "wd0" in initial_states:
            wd0.fill(initial_states["wd0"])
        if "s0" in initial_states:
            s0.fill(initial_states["s0"])
        if "fr0" in initial_states:
            fr0.fill(initial_states["fr0"])

    # Save warmup states before applying overrides (for return_warmup_states)
    warmup_states = None
    if return_warmup_states:
        warmup_states = {
            "wu0": wu0.copy(),  # [basin] array
            "wl0": wl0.copy(),  # [basin] array
            "wd0": wd0.copy(),  # [basin] array
            "s0": s0.copy(),  # [basin] array
            "fr0": fr0.copy(),  # [basin] array
        }

    inputs = p_and_e[warmup_length:, :, :]
    actual_time_steps = inputs.shape[0]

    # Initialize output arrays
    q_sim = np.zeros((actual_time_steps, num_basins))
    runoff_sim = np.zeros((actual_time_steps, num_basins))
    rs_out = np.zeros((actual_time_steps, num_basins))
    ri_out = np.zeros((actual_time_steps, num_basins))
    rg_out = np.zeros((actual_time_steps, num_basins))
    pe_out = np.zeros((actual_time_steps, num_basins))
    wu_out = np.zeros((actual_time_steps, num_basins))
    wl_out = np.zeros((actual_time_steps, num_basins))
    wd_out = np.zeros((actual_time_steps, num_basins))

    # Process each basin
    for basin_idx in range(num_basins):
        # Extract time series for this basin
        prcp = inputs[:, basin_idx, 0]
        pet = inputs[:, basin_idx, 1]

        # Calculate net precipitation
        pe = prcp - pet

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
            pet,
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
        # 获取初始状态（从kwargs中获取，如果没有则使用零数组）
        lag_initial_states = kwargs.get("lag_initial_states", None)
        if lag_initial_states is not None:
            qsig_initial = lag_initial_states.get(
                "qsig_initial", np.zeros(max(int(lag[basin_idx]), 3))
            )
            qx_initial = lag_initial_states.get(
                "qx_initial", np.zeros(int(mp[basin_idx]) + 1)
            )
        else:
            qsig_initial = np.zeros(max(int(lag[basin_idx]), 3))
            qx_initial = np.zeros(int(mp[basin_idx]) + 1)

        q_basin = lag3_routing_vectorized(
            rs_basin,
            ri_basin,
            rg_basin,
            time_interval,
            basin_area,
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
        result = (
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
        if return_warmup_states and warmup_states is not None:
            return result + (warmup_states,)
        else:
            return result
    else:
        if return_warmup_states and warmup_states is not None:
            return q_sim, warmup_states
        else:
            return q_sim


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

    # 根据ES数组计算蒸散发值
    if "ES" in sms_data:
        # 获取时间间隔参数
        time_interval = float(sms_data.get("clen", 1.0))

        # 计算每个时间步的蒸散发值
        evap_values = np.zeros(time_steps)
        for i in range(time_steps):
            # 从时间字符串中提取月份，默认为8月
            try:
                if dt and i < len(dt):
                    time_str = dt[i]
                    # 尝试解析时间字符串获取月份
                    if ":" in time_str:  # 包含时间的格式
                        month = int(time_str.split("-")[1])  # 提取月份
                    else:  # 只有日期的格式
                        month = int(time_str.split("-")[1])  # 提取月份
                else:
                    month = 8  # 默认月份为8
            except:
                month = 8  # 解析失败时使用默认月份

            # 根据月份确定天数
            if month in [4, 6, 9, 11]:
                iday = 30
            elif month == 2:
                iday = 28
            else:
                iday = 31

            # 计算蒸散发值：ES[month-1] / (IDAY * 24.0 / T)
            em = es[month - 1] / (iday * 24.0 / time_interval)
            evap_values[i] = em
    else:
        # 如果没有ES数组，使用输入的蒸散发数值
        evap_values = np.full(time_steps, default_evap)

    p_and_e[:, 0, 1] = evap_values

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
                float(sms_data["FRP"]),  # 4: Initial runoff basin_area ratio
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
                float(sms_data["IM"]),  # 11: Impervious basin_area ratio
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
