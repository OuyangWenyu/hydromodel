"""
Author: Wenyu Ouyang
Date: 2025-07-30 16:44:15
LastEditTime: 2025-08-19 17:19:04
LastEditors: Wenyu Ouyang
Description: Dahuofang Model - Python implementation based on Java version
FilePath: \hydromodel\hydromodel\models\dhf.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union
import traceback
from numba import jit

from hydromodel.models.model_config import MODEL_PARAM_DICT
from hydromodel.models.param_utils import process_parameters


@jit(nopython=True)
def calculate_dhf_evapotranspiration(
    precipitation: np.ndarray,
    potential_evapotranspiration: np.ndarray,
    kc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate evapotranspiration and net precipitation for DHF model"""
    edt = kc * potential_evapotranspiration
    pe = precipitation - edt  # net precipitation
    return pe, edt


@jit(nopython=True)
def calculate_dhf_surface_runoff(
    pe: np.ndarray,
    sa: np.ndarray,
    s0: np.ndarray,
    a: np.ndarray,
    g: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate surface runoff and impervious area runoff"""
    y0 = g * pe  # impervious area runoff
    pc = pe - y0  # net infiltration

    # Calculate surface water storage contribution (vectorized)
    temp = np.where(sa > 0, (1 - sa / s0) ** (1 / a), 0.0)
    sm = a * s0 * (1 - temp)

    # Calculate runoff from surface storage
    rr = np.where(
        pc > 0.0,
        np.where(
            sm + pc < a * s0,
            pc + sa - s0 + s0 * (1 - (sm + pc) / (a * s0)) ** a,
            pc - (s0 - sa),
        ),
        0.0,
    )

    return y0, pc, rr


@jit(nopython=True)
def calculate_dhf_subsurface_flow(
    rr: np.ndarray,
    ua: np.ndarray,
    u0: np.ndarray,
    d0: np.ndarray,
    b: np.ndarray,
    k2: np.ndarray,
    kw: np.ndarray,
    time_interval: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate subsurface flow components"""
    # Calculate subsurface flow parameters
    temp = np.where(ua > 0, (1 - ua / u0) ** (1 / b), 0.0)
    un = b * u0 * (1 - temp)
    temp = np.where(ua > 0, (1 - ua / u0) ** (u0 / (b * d0)), 0.0)
    dn = b * d0 * (1 - temp)

    z1 = 1 - np.exp(-k2 * time_interval * u0 / d0)
    z2 = 1 - np.exp(-k2 * time_interval)

    # Calculate total flow
    y = np.where(
        rr + z2 * un < z2 * b * u0,
        rr
        + z2 * (ua - u0)
        + z2 * u0 * (1 - (z2 * un + rr) / (z2 * b * u0)) ** b,
        rr + z2 * (ua - u0),
    )

    # Calculate interflow
    temp = np.where(ua > 0, (1 - ua / u0) ** (u0 / d0), 0.0)
    yu = np.where(
        z1 * dn + rr < z1 * b * d0,
        rr
        - z1 * d0 * temp
        + z1 * d0 * (1 - (z1 * dn + rr) / (z1 * b * d0)) ** b,
        rr - z1 * d0 * temp,
    )

    # Calculate groundwater runoff
    yl = (y - yu) * kw

    return y, yu, yl


@jit(nopython=True)
def calculate_dhf_storage_update(
    sa: np.ndarray,
    ua: np.ndarray,
    pc: np.ndarray,
    rr: np.ndarray,
    y: np.ndarray,
    s0: np.ndarray,
    u0: np.ndarray,
    a: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Update water storage states"""
    # Calculate surface water storage parameters
    temp = np.where(sa > 0, (1 - sa / s0) ** (1 / a), 0.0)
    sm = a * s0 * (1 - temp)

    # Update surface storage
    sa_new = np.where(
        pc > 0.0,
        np.where(
            sm + pc < a * s0,
            s0 * (1 - (1 - (sm + pc) / (a * s0)) ** a),
            sa + pc - rr,
        ),
        sa,
    )
    sa_new = np.clip(sa_new, 0.0, s0)

    # Update subsurface storage
    ua_new = ua + rr - y
    ua_new = np.clip(ua_new, 0.0, u0)

    return sa_new, ua_new


@jit(nopython=True)
def calculate_dhf_evaporation_deficit(
    precipitation: np.ndarray,
    edt: np.ndarray,
    sa: np.ndarray,
    ua: np.ndarray,
    s0: np.ndarray,
    u0: np.ndarray,
    a: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate evaporation when precipitation is insufficient"""
    ec = edt - precipitation
    eb = ec  # accumulated deficit

    # Calculate surface evaporation
    temp1 = (1 - (eb - ec) / (a * s0)) ** a
    temp2 = (1 - eb / (a * s0)) ** a

    eu = np.where(
        (eb / (a * s0) <= 0.999999) & ((eb - ec) / (a * s0) <= 0.999999),
        s0 * (temp1 - temp2),
        np.where(
            (eb / (a * s0) >= 1.00001) & ((eb - ec) / (a * s0) <= 0.999999),
            s0 * temp1,
            0.00001,
        ),
    )

    # Update storages after evaporation
    el = np.where(sa - eu < 0.0, (ec - sa) * ua / u0, (ec - eu) * ua / u0)
    sa_new = np.where(sa - eu < 0.0, 0.0, sa - eu)
    ua_new = ua - el
    ua_new = np.maximum(ua_new, 0.0)

    return sa_new, ua_new


@jit(nopython=True)
def calculate_dhf_routing_params(
    ya: np.ndarray,
    runoff_sim: np.ndarray,
    l: np.ndarray,
    b0: np.ndarray,
    k0: np.ndarray,
    n: np.ndarray,
    coe: np.ndarray,
    dd: np.ndarray,
    cc: np.ndarray,
    ddl: np.ndarray,
    ccl: np.ndarray,
    time_interval: float,
    pai: float,
):
    """Calculate routing parameters for DHF model"""
    # Ensure ya >= 0.5 for stability
    ya = np.maximum(ya, 0.5)

    # Calculate timing parameters
    temp_tm = (ya + runoff_sim) ** (-k0)
    lb = l / b0
    tm = lb * temp_tm

    tt = (n * tm).astype(np.int32)
    ts = (coe * tm).astype(np.int32)

    # Surface flow routing coefficient
    w0 = 1.0 / time_interval

    # Calculate surface routing coefficient K3
    k3 = np.zeros_like(tm)
    aa = np.zeros_like(tm)

    mask = tm > 0
    if np.any(mask):
        temp_aa = (pai * coe[mask]) ** (dd[mask] - 1)
        aa[mask] = cc[mask] / (dd[mask] * temp_aa * np.tan(pai * coe[mask]))

        for j in range(int(np.max(tm[mask])) + 1):
            j_mask = mask & (j < tm)
            if np.any(j_mask):
                temp = (pai * j / tm[j_mask]) ** dd[j_mask]
                temp1 = (np.sin(pai * j / tm[j_mask])) ** cc[j_mask]
                k3[j_mask] += np.exp(-aa[j_mask] * temp) * temp1

        k3[mask] = tm[mask] * w0 / k3[mask]

    # Calculate subsurface routing coefficient K3L
    k3l = np.zeros_like(tm)
    aal = np.zeros_like(tm)

    tt_mask = tt > 0
    if np.any(tt_mask):
        temp_aal = (pai * coe[tt_mask] / n[tt_mask]) ** (ddl[tt_mask] - 1)
        aal[tt_mask] = ccl[tt_mask] / (
            ddl[tt_mask] * temp_aal * np.tan(pai * coe[tt_mask] / n[tt_mask])
        )

        for j in range(int(np.max(tt[tt_mask])) + 1):
            j_mask = tt_mask & (j < tt)
            if np.any(j_mask):
                temp = (pai * j / tt[j_mask]) ** ddl[j_mask]
                temp1 = (np.sin(pai * j / tt[j_mask])) ** ccl[j_mask]
                k3l[j_mask] += np.exp(-aal[j_mask] * temp) * temp1

        k3l[tt_mask] = tt[tt_mask] * w0 / k3l[tt_mask]

    return tm, k3, k3l, aa, aal, tt, ts


@jit(nopython=True)
def calculate_dhf_routing(
    runoff_sim: np.ndarray,
    rl: np.ndarray,
    tm: np.ndarray,
    k3: np.ndarray,
    k3l: np.ndarray,
    aa: np.ndarray,
    aal: np.ndarray,
    tt: np.ndarray,
    ts: np.ndarray,
    dd: np.ndarray,
    cc: np.ndarray,
    ddl: np.ndarray,
    ccl: np.ndarray,
    time_steps: int,
    pai: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate routing for DHF model"""
    qs = np.zeros_like(runoff_sim)
    ql = np.zeros_like(runoff_sim)

    for i in range(time_steps):
        tl = tt[i] + ts[i] - 1
        tl = max(tl, 0)

        for j in range(int(tl) + 1):
            if i + j >= time_steps:
                break

            # Surface routing
            if tm[i] > 0:
                temp0 = pai * j / tm[i]
                temp1 = temp0 ** dd[i]
                temp2 = np.exp(-aa[i] * temp1)
                temp3 = (np.sin(temp0)) ** cc[i]
                qs_contrib = (
                    (runoff_sim[i] - rl[i]) * k3[i] / tm[i] * temp2 * temp3
                )
            else:
                qs_contrib = 0.0

            # Subsurface routing
            if tt[i] > 0 and j >= ts[i]:
                temp00 = pai * (j - ts[i]) / tt[i]
                temp10 = temp00 ** ddl[i]
                temp20 = np.exp(-aal[i] * temp10)
                temp30 = (np.sin(temp00)) ** ccl[i]
                ql_contrib = rl[i] * k3l[i] / tt[i] * temp20 * temp30
            else:
                ql_contrib = 0.0

            # Add contributions based on timing conditions
            if j <= tm[i]:
                if j <= ts[i]:
                    qs[i + j] += qs_contrib
                else:
                    qs[i + j] += qs_contrib
                    ql[i + j] += ql_contrib
            else:
                ql[i + j] += ql_contrib

    return qs, ql


def run_dhf_single_basin(
    precipitation: np.ndarray,
    potential_evapotranspiration: np.ndarray,
    parameters: np.ndarray,
    warmup_length: int = 30,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    DHF模型运行函数，用于单个流域的径流计算

    Parameters
    ----------
    precipitation : np.ndarray
        降雨数据
    potential_evapotranspiration : np.ndarray
        潜在蒸散发数据
    parameters : np.ndarray
        模型参数数组 [S0, U0, D0, KC, KW, K2, KA, G, A, B, B0, K0, N, L, DD, CC, COE, DDL, CCL]
    warmup_length : int, optional
        预热期长度，默认30
    **kwargs
        Additional keyword arguments, including time_interval_hours (default: 1.0)

    Returns
    -------
    Dict[str, np.ndarray]
        包含模拟结果和状态变量的字典

    Notes
    -----
    输入输出单位已统一，不需要流域面积转换
    """
    # 提取参数（只包含模型参数，不包含状态变量）
    s0, u0, d0, kc, kw, k2, ka = parameters[0:7]
    g, a, b, b0, k0, n, l = parameters[7:14]
    dd, cc, coe, ddl, ccl = parameters[14:19]

    # 预热期处理 - 递归调用获取合适的初始状态
    if warmup_length > 0:
        warmup_precipitation = precipitation[:warmup_length]
        warmup_pet = potential_evapotranspiration[:warmup_length]

        # 递归调用预热期（无预热期）
        warmup_results = run_dhf_single_basin(
            warmup_precipitation,
            warmup_pet,
            parameters,
            warmup_length=0,
            **kwargs,
        )

        # 从预热结果获取最终状态作为正式计算的初始状态
        sa0 = warmup_results["sa"][-1]
        ua0 = warmup_results["ua"][-1]
        ya0 = warmup_results["ya"][-1]

        # 使用预热期后的数据进行正式计算
        precipitation = precipitation[warmup_length:]
        potential_evapotranspiration = potential_evapotranspiration[
            warmup_length:
        ]
    else:
        # 使用默认初始状态 (from Jinggang Chu)
        sa0 = 0.0
        ua0 = 0.0
        ya0 = 0.5

    time_steps = len(precipitation)

    # 模型常量
    time_interval = kwargs.get("time_interval_hours", 3.0)
    PAI = np.pi

    # 初始化状态变量
    sa = np.zeros(time_steps + 1)
    ua = np.zeros(time_steps + 1)
    ya = np.zeros(time_steps + 1)

    # 初始化产流变量
    RunoffSim = np.zeros(time_steps)
    QSim = np.zeros(time_steps)
    y0 = np.zeros(time_steps)
    PE = np.zeros(time_steps)
    yu = np.zeros(time_steps)
    yL = np.zeros(time_steps)
    y = np.zeros(time_steps)
    rL = np.zeros(time_steps)
    qs = np.zeros(time_steps)
    ql = np.zeros(time_steps)
    ET = np.zeros(time_steps)

    # 设置初始状态
    sa[0] = sa0
    ua[0] = ua0
    ya[0] = ya0

    # 限制初始状态
    if sa[0] > s0:
        sa[0] = s0
    if ua[0] > u0:
        ua[0] = u0

    # DHF产流计算
    for i in range(time_steps):
        # 限制当前状态
        if sa[i] > s0:
            sa[i] = s0
        if ua[i] > u0:
            ua[i] = u0

        # 初始化当前时段变量
        yu[i] = 0.0
        yL[i] = 0.0
        Eb = 0.0

        # 计算蒸散发
        if (
            potential_evapotranspiration is not None
            and len(potential_evapotranspiration) > i
        ):
            # 如果提供了蒸发数据，直接使用
            # print(f"使用提供的蒸发数据: {evapotranspiration[i]}")
            Ep = potential_evapotranspiration[i]
            EDt = kc * Ep

        else:
            raise ValueError("Potential evapotranspiration is required")

        PE[i] = precipitation[i] - EDt  # 净雨
        y0[i] = g * PE[i]  # 不透水面积产流
        Pc = PE[i] - y0[i]  # 净渗雨强
        ET[i] = EDt

        if Pc > 0.0:
            # 计算表层蓄水
            temp = (1 - sa[i] / s0) ** (1 / a)
            Sm = a * s0 * (1 - temp)

            if Sm + Pc < a * s0:
                temp = (1 - (Sm + Pc) / (a * s0)) ** a
                rr = Pc + sa[i] - s0 + s0 * temp
            else:
                rr = Pc - (s0 - sa[i])

            # 下层流计算
            temp = (1 - ua[i] / u0) ** (1 / b)
            un = b * u0 * (1 - temp)
            temp = (1 - ua[i] / u0) ** (u0 / (b * d0))
            dn = b * d0 * (1 - temp)

            Z1 = 1 - np.exp(-k2 * time_interval * u0 / d0)
            Z2 = 1 - np.exp(-k2 * time_interval)

            # 总流量计算
            if rr + Z2 * un < Z2 * b * u0:
                temp = (1 - (Z2 * un + rr) / (Z2 * b * u0)) ** b
                y[i] = rr + Z2 * (ua[i] - u0) + Z2 * u0 * temp
            else:
                y[i] = rr + Z2 * (ua[i] - u0)

            # 地面壤中流
            temp = (1 - ua[i] / u0) ** (u0 / d0)
            if Z1 * dn + rr < Z1 * b * d0:
                temp1 = 1 - (Z1 * dn + rr) / (Z1 * b * d0)
                temp2 = temp1**b
                yu[i] = rr - Z1 * d0 * temp + Z1 * d0 * temp2
            else:
                yu[i] = rr - Z1 * d0 * temp

            # 地下径流
            yL[i] = (y[i] - yu[i]) * kw

            # 更新状态变量
            if Sm + Pc < a * s0:
                temp1 = 1 - (Sm + Pc) / (a * s0)
                temp2 = temp1**a
                sa[i + 1] = s0 * (1 - temp2)
            else:
                sa[i + 1] = sa[i] + Pc - rr

            if sa[i + 1] > s0:
                sa[i + 1] = s0

            ua[i + 1] = ua[i] + rr - y[i]
            if ua[i + 1] > u0:
                ua[i + 1] = u0

            Eb = 0.0

        else:
            rr = 0.0
            Ec = EDt - precipitation[i]
            Eb = Eb + Ec

            # 表层可蒸发量Eu计算
            temp1 = (1 - (Eb - Ec) / (a * s0)) ** a
            temp2 = (1 - Eb / (a * s0)) ** a

            if (Eb / (a * s0) <= 0.999999) and (
                (Eb - Ec) / (a * s0) <= 0.999999
            ):
                Eu = s0 * (temp1 - temp2)
            elif (Eb / (a * s0) >= 1.00001) and (
                (Eb - Ec) / (a * s0) <= 0.999999
            ):
                Eu = s0 * temp1
            else:
                Eu = 0.00001

            if sa[i] - Eu < 0.0:
                EL = (Ec - sa[i]) * ua[i] / u0
                sa[i + 1] = 0.0
                ua[i + 1] = ua[i] - EL
                if ua[i + 1] < 0.0:
                    ua[i + 1] = 0.0
            else:
                EL = (Ec - Eu) * ua[i] / u0
                sa[i + 1] = sa[i] - Eu
                ua[i + 1] = ua[i] - EL
                if ua[i + 1] < 0.0:
                    ua[i + 1] = 0.0

            y[i] = 0.0
            y0[i] = 0.0
            yu[i] = 0.0
            yL[i] = 0.0

        # 确保边界条件
        if sa[i + 1] > s0:
            sa[i + 1] = s0
        if sa[i + 1] < 0:
            sa[i + 1] = 0
        if ua[i + 1] < 0:
            ua[i + 1] = 0
        if ua[i + 1] > u0:
            ua[i + 1] = u0

    # 计算总产流并更新ya
    for j in range(time_steps):
        RunoffSim[j] = y[j] + y0[j]
        if RunoffSim[j] < 0.0:
            RunoffSim[j] = 0.0
        rL[j] = yL[j]
        if rL[j] < 0.0:
            rL[j] = 0.0

        ya[j + 1] = (ya[j] + RunoffSim[j]) * ka
        if ya[j + 1] < 0.0:
            ya[j + 1] = 0.0

    # DHF汇流计算
    # 注意: 流域面积转换已在数据预处理中完成，这里使用标准化因子
    w0 = 1.0 / time_interval

    for i in range(time_steps):
        if ya[i] < 0.5:
            ya[i] = 0.5

        tempTm = (ya[i] + RunoffSim[i]) ** (-k0)
        LB = l / b0
        Tm = LB * tempTm  # 保持为浮点数

        TT = int(n * Tm)
        TS = int(coe * Tm)

        # 地下汇流参数计算
        if TT > 0:
            tempAAL = (PAI * coe / n) ** (ddl - 1)
            AAL = ccl / (ddl * tempAAL * np.tan(PAI * coe / n))
            K3L = 0.0
            for j in range(TT):
                tmp = (PAI * j / TT) ** ddl
                tmp1 = (np.sin(PAI * j / TT)) ** ccl
                K3L += np.exp(-AAL * tmp) * tmp1

            # 按照Java版本，没有除零检查，直接计算
            if K3L != 0:
                K3L = TT * w0 / K3L
            else:
                K3L = 0.0
        else:
            K3L = 0.0
            AAL = 0.0

        # 地表汇流参数计算
        K3 = 0.0  # 初始化K3
        if Tm > 0:
            tempAA = (PAI * coe) ** (dd - 1)
            AA = cc / (dd * tempAA * np.tan(PAI * coe))

            # Java版本：for (int j = 0; j < Tm; j++)
            j = 0
            while j < Tm:
                tmp = (PAI * j / Tm) ** dd
                tmp1 = (np.sin(PAI * j / Tm)) ** cc
                K3_old = K3  # 保存旧值用于跟踪
                K3 += np.exp(-AA * tmp) * tmp1
                j += 1  # 增加计数器

            # 按照Java版本，没有除零检查，直接计算
            if K3 != 0:
                K3_before = K3
                K3 = Tm * w0 / K3

            else:
                K3 = 0.0

        else:
            AA = 0.0

        TL = TT + TS - 1
        if TL <= 0:
            TL = 0

        # 汇流计算
        for j in range(TL):
            if i + j >= time_steps:  # 防止越界
                break

            # 地表汇流计算
            if Tm > 0:
                temp0 = PAI * j / Tm
                temp1 = temp0**dd
                temp2 = np.exp(-AA * temp1)
                temp3 = (np.sin(temp0)) ** cc
                Qs = (RunoffSim[i] - rL[i]) * K3 / Tm * temp2 * temp3
            else:
                Qs = 0.0

            if np.isnan(Qs):
                Qs = 0

            # 地下汇流计算
            if TT > 0:
                temp00 = PAI * (j - TS) / TT
                temp10 = temp00**ddl
                temp20 = np.exp(-AAL * temp10)
                temp30 = (np.sin(temp00)) ** ccl
                Ql = rL[i] * K3L / TT * temp20 * temp30
            else:
                Ql = 0.0

            # 按照Java版本的精确条件判断
            if j <= int(Tm):
                if j <= TS:
                    ql[i + j] += 0.0
                    qs[i + j] += Qs
                else:
                    qs[i + j] += Qs
                    ql[i + j] += Ql
            else:
                qs[i + j] += 0.0
                ql[i + j] += Ql

            QSim[i + j] = qs[i + j] + ql[i + j]
            if QSim[i + j] < 0.0:
                QSim[i + j] = 0.0

    # 状态变量处理（移除最后一个时步的状态，因为它对应时间步长+1）
    sa = sa[:-1]
    ua = ua[:-1]
    ya = ya[:-1]

    return {
        "QSim": QSim,
        "runoffSim": RunoffSim,
        "y0": y0,
        "yu": yu,
        "yl": yL,
        "y": y,
        "pe": PE,
        "sa": sa,
        "ua": ua,
        "ya": ya,
    }


def dhf_vectorized(
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
        np.ndarray,
    ],
]:
    """
    Vectorized DHF (Dahuofang) hydrological model - fully parallelized version

    This function implements the DHF model with full NumPy vectorization,
    processing all basins simultaneously using [seq, basin, feature] tensor operations.

    Parameters
    ----------
    p_and_e : np.ndarray
        precipitation and potential evapotranspiration, 3-dim: [time, basin, feature=2]
        where feature=0 is precipitation, feature=1 is potential evapotranspiration
    parameters : np.ndarray
        model parameters, 2-dim: [basin, parameter]
        Parameters: [S0, U0, D0, KC, KW, K2, KA, G, A, B, B0, K0, N, L, DD, CC, COE, DDL, CCL]
    warmup_length : int, optional
        the length of warmup period (default: 365)
    return_state : bool, optional
        if True, return internal state variables (default: False)
    normalized_params : Union[bool, str], optional
        parameter format specification:
        - "auto": automatically detect parameter format (default)
        - True: parameters are normalized (0-1 range), convert to original scale
        - False: parameters are already in original scale, use as-is
    **kwargs
        Additional keyword arguments, including time_interval_hours (default: 3.0)

    Returns
    -------
    result : np.ndarray or tuple
        if return_state is False: QSim array [time, basin, 1]
        if return_state is True: tuple of (QSim, runoffSim, y0, yu, yl, y, pe, sa, ua, ya)
    """

    # Get data dimensions
    time_steps, num_basins, _ = p_and_e.shape
    time_interval = kwargs.get("time_interval_hours", 3.0)
    pai = np.pi

    # Process parameters using unified parameter handling
    processed_parameters = parameters.copy()
    if normalized_params != False:
        model_param_dict = MODEL_PARAM_DICT.get("dhf")
        if model_param_dict is not None:
            param_ranges = model_param_dict["param_range"]
            processed_parameters = process_parameters(
                parameters, param_ranges, normalized=normalized_params
            )

    # Extract parameters - all are [basin] arrays
    s0 = processed_parameters[:, 0]  # Surface storage capacity
    u0 = processed_parameters[:, 1]  # Subsurface storage capacity
    d0 = processed_parameters[:, 2]  # Deep storage capacity
    kc = processed_parameters[:, 3]  # Evaporation coefficient
    kw = processed_parameters[:, 4]  # Subsurface flow coefficient
    k2 = processed_parameters[:, 5]  # Percolation coefficient
    ka = processed_parameters[:, 6]  # Total runoff adjustment coefficient
    g = processed_parameters[:, 7]  # Impervious area ratio
    a = processed_parameters[:, 8]  # Surface storage exponent
    b = processed_parameters[:, 9]  # Subsurface storage exponent
    b0 = processed_parameters[:, 10]  # Routing parameter
    k0 = processed_parameters[:, 11]  # Routing parameter
    n = processed_parameters[:, 12]  # Routing parameter
    l = processed_parameters[:, 13]  # Routing parameter
    dd = processed_parameters[:, 14]  # Surface routing parameter
    cc = processed_parameters[:, 15]  # Surface routing parameter
    coe = processed_parameters[:, 16]  # Routing parameter
    ddl = processed_parameters[:, 17]  # Subsurface routing parameter
    ccl = processed_parameters[:, 18]  # Subsurface routing parameter

    # Handle warmup period
    if warmup_length > 0:
        p_and_e_warmup = p_and_e[0:warmup_length, :, :]
        *_, sa0, ua0, ya0 = dhf_vectorized(
            p_and_e_warmup,
            parameters,
            warmup_length=0,
            return_state=True,
            normalized_params=False,  # Already processed
            **kwargs,
        )
    else:
        # Default initial states
        sa0 = np.zeros(s0.shape)
        ua0 = np.zeros(u0.shape)
        # just use d0's shape, ya0 is not d0, it is Pa, while d0 is the deep storage capacity
        ya0 = np.full(d0.shape, 0.5)

    inputs = p_and_e[warmup_length:, :, :]
    # Get actual time steps after warmup
    actual_time_steps = inputs.shape[0]

    # Initialize state and output arrays - [time, basin]
    sa = np.zeros((actual_time_steps, num_basins))
    ua = np.zeros((actual_time_steps, num_basins))
    ya = np.zeros((actual_time_steps, num_basins))
    # to store the accumulated deficit
    ebs = np.zeros((actual_time_steps, num_basins))

    # Initialize output arrays
    runoff_sim = np.zeros((actual_time_steps, num_basins))
    q_sim = np.zeros((actual_time_steps, num_basins))
    y0_out = np.zeros((actual_time_steps, num_basins))
    pe_out = np.zeros((actual_time_steps, num_basins))
    yu_out = np.zeros((actual_time_steps, num_basins))
    yl_out = np.zeros((actual_time_steps, num_basins))
    y_out = np.zeros((actual_time_steps, num_basins))
    et_out = np.zeros((actual_time_steps, num_basins))

    # Main time loop - DHF generation (runoff production)
    for i in range(actual_time_steps):
        if i == 5:
            print("i = 5")
        # Current precipitation and PET for all basins
        prcp = inputs[i, :, 0]
        pet = inputs[i, :, 1]
        if i == 0:
            eb = np.zeros(kc.shape)
        else:
            sa0 = sa[i - 1, :]
            ua0 = ua[i - 1, :]
            ya0 = ya[i - 1, :]
            # TODO: Because Chu version init eb as 0 at every time step, we keep the same now
            # if we want to make it same as the book, we need to value ebs after a time step's calculation
            eb = ebs[i - 1, :]

        # Limit current states
        sa0 = np.minimum(sa0, s0)
        ua0 = np.minimum(ua0, u0)

        # Calculate evapotranspiration and net precipitation (vectorized)
        edt = kc * pet
        pe = prcp - edt
        # Surface runoff calculation (vectorized)
        y0 = g * pe  # impervious area runoff
        pc = pe - y0  # net infiltration
        # Process based on whether we have net precipitation or evaporation
        # Actually, we should use pe > 0.0, but we use pc > 0.0 to make it same as Chu's version
        # as g<1, hence pe>0 means pc>0, so it is fine.
        net_precip_mask = pc > 0.0

        # For basins with net precipitation (pe > 0) - vectorized operations
        if np.any(net_precip_mask):
            # Apply mask to get relevant basin data
            sa_pos = sa0[net_precip_mask]
            ua_pos = ua0[net_precip_mask]
            s0_pos = s0[net_precip_mask]
            u0_pos = u0[net_precip_mask]
            d0_pos = d0[net_precip_mask]
            a_pos = a[net_precip_mask]
            b_pos = b[net_precip_mask]
            k2_pos = k2[net_precip_mask]
            kw_pos = kw[net_precip_mask]

            # Surface water storage calculation
            temp = (1 - sa_pos / s0_pos) ** (1 / a_pos)
            sm = a_pos * s0_pos * (1 - temp)

            # Calculate surface runoff
            rr = np.where(
                pc > 0.0,
                np.where(
                    sm + pc < a_pos * s0_pos,
                    pc
                    + sa_pos
                    - s0_pos
                    + s0_pos * (1 - (sm + pc) / (a_pos * s0_pos)) ** a_pos,
                    pc - (s0_pos - sa_pos),
                ),
                0.0,
            )

            # Subsurface flow calculation (vectorized)
            temp = (1 - ua_pos / u0_pos) ** (1 / b_pos)
            un = b_pos * u0_pos * (1 - temp)
            temp = (1 - ua_pos / u0_pos) ** (u0_pos / (b_pos * d0_pos))
            dn = b_pos * d0_pos * (1 - temp)

            z1 = 1 - np.exp(-k2_pos * time_interval * u0_pos / d0_pos)
            z2 = 1 - np.exp(-k2_pos * time_interval)

            # Calculate total flow
            y = np.where(
                rr + z2 * un < z2 * b_pos * u0_pos,
                rr
                + z2 * (ua_pos - u0_pos)
                + z2
                * u0_pos
                * (1 - (z2 * un + rr) / (z2 * b_pos * u0_pos)) ** b_pos,
                rr + z2 * (ua_pos - u0_pos),
            )

            # Calculate interflow
            temp = (1 - ua_pos / u0_pos) ** (u0_pos / d0_pos)
            yu = np.where(
                z1 * dn + rr < z1 * b_pos * d0_pos,
                rr
                - z1 * d0_pos * temp
                + z1
                * d0_pos
                * (1 - (z1 * dn + rr) / (z1 * b_pos * d0_pos)) ** b_pos,
                rr - z1 * d0_pos * temp,
            )

            # Calculate groundwater runoff
            yl = (y - yu) * kw_pos

            # Update storage states (vectorized)
            sa_new = np.where(
                pc > 0.0,
                np.where(
                    sm + pc < a_pos * s0_pos,
                    s0_pos * (1 - (1 - (sm + pc) / (a_pos * s0_pos)) ** a_pos),
                    sa_pos + pc - rr,
                ),
                sa_pos,
            )
            sa_new = np.clip(sa_new, 0.0, s0_pos)

            ua_new = ua_pos + rr - y
            ua_new = np.clip(ua_new, 0.0, u0_pos)
            # eb will be set to 0 when pc > 0
            eb = np.where(pc > 0.0, 0.0, eb)

            # Store results for basins with net precipitation
            y0_out[i, net_precip_mask] = y0
            yu_out[i, net_precip_mask] = yu
            yl_out[i, net_precip_mask] = yl
            y_out[i, net_precip_mask] = y
            sa[i, net_precip_mask] = sa_new
            ua[i, net_precip_mask] = ua_new

        # For basins with evaporation deficit (pe <= 0) - vectorized operations
        evap_mask = ~net_precip_mask
        if np.any(evap_mask):
            prcp_neg = prcp[evap_mask]
            edt_neg = edt[evap_mask]
            sa_neg = sa0[evap_mask]
            ua_neg = ua0[evap_mask]
            s0_neg = s0[evap_mask]
            u0_neg = u0[evap_mask]
            a_neg = a[evap_mask]

            ec = edt_neg - prcp_neg
            eb = eb + ec  # accumulated deficit

            # Calculate surface evaporation (vectorized)
            temp1 = (1 - (eb - ec) / (a_neg * s0_neg)) ** a_neg
            temp2 = (1 - eb / (a_neg * s0_neg)) ** a_neg

            eu = np.where(
                (eb / (a_neg * s0_neg) <= 0.999999)
                & ((eb - ec) / (a_neg * s0_neg) <= 0.999999),
                s0_neg * (temp1 - temp2),
                np.where(
                    (eb / (a_neg * s0_neg) >= 1.00001)
                    & ((eb - ec) / (a_neg * s0_neg) <= 0.999999),
                    s0_neg * temp1,
                    0.00001,
                ),
            )

            # Update storages after evaporation
            el = np.where(
                sa_neg - eu < 0.0,
                (ec - sa_neg) * ua_neg / u0_neg,
                (ec - eu) * ua_neg / u0_neg,
            )
            sa_new = np.where(sa_neg - eu < 0.0, 0.0, sa_neg - eu)
            ua_new = ua_neg - el
            ua_new = np.maximum(ua_new, 0.0)

            # Set runoff components to zero for evaporation basins
            y0_out[i, evap_mask] = 0.0
            yu_out[i, evap_mask] = 0.0
            yl_out[i, evap_mask] = 0.0
            y_out[i, evap_mask] = 0.0
            sa[i, evap_mask] = sa_new
            ua[i, evap_mask] = ua_new

        # Ensure states are within bounds
        sa[i, :] = np.clip(sa[i, :], 0.0, s0)
        ua[i, :] = np.clip(ua[i, :], 0.0, u0)

    # Calculate total runoff and update ya (vectorized)
    for i in range(actual_time_steps):
        if i == 0:
            ya0 = ya0
        else:
            ya0 = ya[i - 1, :]
        runoff_sim[i, :] = np.maximum(y_out[i, :] + y0_out[i, :], 0.0)
        ya[i, :] = np.maximum((ya0 + runoff_sim[i, :]) * ka, 0.0)

    # DHF routing calculation - still needs time loops due to convolution nature
    qs = np.zeros((actual_time_steps, num_basins))
    ql = np.zeros((actual_time_steps, num_basins))

    # Process routing for each basin (some vectorization possible)
    for basin_idx in range(num_basins):
        # Extract basin-specific data
        ya_basin = ya[:-1, basin_idx]  # Remove last time step
        runoff_basin = runoff_sim[:, basin_idx]
        yl_basin = yl_out[:, basin_idx]

        w0 = 1.0 / time_interval

        for i in range(actual_time_steps):
            ya_val = max(ya_basin[i], 0.5)  # Ensure stability

            temp_tm = (ya_val + runoff_basin[i]) ** (-k0[basin_idx])
            lb = l[basin_idx] / b0[basin_idx]
            tm = lb * temp_tm

            tt = int(n[basin_idx] * tm)
            ts = int(coe[basin_idx] * tm)

            # Calculate routing coefficients
            if tm > 0:
                temp_aa = (pai * coe[basin_idx]) ** (dd[basin_idx] - 1)
                aa_val = cc[basin_idx] / (
                    dd[basin_idx] * temp_aa * np.tan(pai * coe[basin_idx])
                )

                k3 = 0.0
                for j in range(int(tm) + 1):
                    temp = (pai * j / tm) ** dd[basin_idx]
                    temp1 = (np.sin(pai * j / tm)) ** cc[basin_idx]
                    k3 += np.exp(-aa_val * temp) * temp1

                k3 = tm * w0 / k3 if k3 != 0 else 0.0
            else:
                k3 = 0.0
                aa_val = 0.0

            if tt > 0:
                temp_aal = (pai * coe[basin_idx] / n[basin_idx]) ** (
                    ddl[basin_idx] - 1
                )
                aal_val = ccl[basin_idx] / (
                    ddl[basin_idx]
                    * temp_aal
                    * np.tan(pai * coe[basin_idx] / n[basin_idx])
                )

                k3l = 0.0
                for j in range(tt):
                    temp = (pai * j / tt) ** ddl[basin_idx]
                    temp1 = (np.sin(pai * j / tt)) ** ccl[basin_idx]
                    k3l += np.exp(-aal_val * temp) * temp1

                k3l = tt * w0 / k3l if k3l != 0 else 0.0
            else:
                k3l = 0.0
                aal_val = 0.0

            tl = tt + ts - 1
            if tl <= 0:
                tl = 0

            # Routing calculations
            for j in range(int(tl) + 1):
                if i + j >= actual_time_steps:
                    break

                # Surface routing
                if tm > 0:
                    temp0 = pai * j / tm
                    temp1 = temp0 ** dd[basin_idx]
                    temp2 = np.exp(-aa_val * temp1)
                    temp3 = (np.sin(temp0)) ** cc[basin_idx]
                    q_surface = (
                        (runoff_basin[i] - yl_basin[i])
                        * k3
                        / tm
                        * temp2
                        * temp3
                    )
                else:
                    q_surface = 0.0

                if np.isnan(q_surface):
                    q_surface = 0.0

                # Subsurface routing
                if tt > 0 and j >= ts:
                    temp00 = pai * (j - ts) / tt
                    temp10 = temp00 ** ddl[basin_idx]
                    temp20 = np.exp(-aal_val * temp10)
                    temp30 = (np.sin(temp00)) ** ccl[basin_idx]
                    q_subsurface = yl_basin[i] * k3l / tt * temp20 * temp30
                else:
                    q_subsurface = 0.0

                # Add contributions based on timing
                if j <= tm:
                    if j <= ts:
                        qs[i + j, basin_idx] += q_surface
                    else:
                        qs[i + j, basin_idx] += q_surface
                        ql[i + j, basin_idx] += q_subsurface
                else:
                    ql[i + j, basin_idx] += q_subsurface

    # Total discharge
    q_sim = qs + ql
    q_sim = np.maximum(q_sim, 0.0)

    # Remove final state step for output arrays
    sa_out = sa[:-1, :]
    ua_out = ua[:-1, :]
    ya_out = ya[:-1, :]

    # Format outputs to match original function signature
    q_sim_3d = np.expand_dims(q_sim, axis=2)  # [time, basin, 1]

    if return_state:
        runoff_sim_3d = np.expand_dims(runoff_sim, axis=2)
        y0_3d = np.expand_dims(y0_out, axis=2)
        yu_3d = np.expand_dims(yu_out, axis=2)
        yl_3d = np.expand_dims(yl_out, axis=2)
        y_3d = np.expand_dims(y_out, axis=2)
        pe_3d = np.expand_dims(pe_out, axis=2)
        sa_3d = np.expand_dims(sa_out, axis=2)
        ua_3d = np.expand_dims(ua_out, axis=2)
        ya_3d = np.expand_dims(ya_out, axis=2)

        return (
            q_sim_3d,
            runoff_sim_3d,
            y0_3d,
            yu_3d,
            yl_3d,
            y_3d,
            pe_3d,
            sa_3d,
            ua_3d,
            ya_3d,
        )
    else:
        return q_sim_3d


# Create new unified DHF function that replaces both dhf and run_dhf_single_basin
def dhf(
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
        np.ndarray,
    ],
]:
    """
    Unified DHF (Dahuofang) hydrological model - vectorized implementation

    This function merges the functionality of the original dhf() and run_dhf_single_basin()
    functions into a single, fully vectorized implementation that processes all basins
    simultaneously using NumPy tensor operations.

    Parameters
    ----------
    p_and_e : np.ndarray
        precipitation and potential evapotranspiration, 3-dim: [time, basin, feature=2]
        where feature=0 is precipitation, feature=1 is potential evapotranspiration
    parameters : np.ndarray
        model parameters, 2-dim: [basin, parameter]
        Parameters: [S0, U0, D0, KC, KW, K2, KA, G, A, B, B0, K0, N, L, DD, CC, COE, DDL, CCL]
    warmup_length : int, optional
        the length of warmup period (default: 365)
    return_state : bool, optional
        if True, return internal state variables (default: False)
    normalized_params : Union[bool, str], optional
        parameter format specification:
        - "auto": automatically detect parameter format (default)
        - True: parameters are normalized (0-1 range), convert to original scale
        - False: parameters are already in original scale, use as-is
    **kwargs
        Additional keyword arguments, including time_interval_hours (default: 3.0)

    Returns
    -------
    result : np.ndarray or tuple
        if return_state is False: QSim array [time, basin, 1]
        if return_state is True: tuple of (QSim, runoffSim, y0, yu, yl, y, pe, sa, ua, ya)

    Notes
    -----
    This function replaces both the original dhf() and run_dhf_single_basin() functions,
    implementing full vectorization to eliminate Java-style loops and enable
    [seq, basin, feature] tensor operations throughout the model.
    """
    return dhf_vectorized(
        p_and_e=p_and_e,
        parameters=parameters,
        warmup_length=warmup_length,
        return_state=return_state,
        normalized_params=normalized_params,
        **kwargs,
    )


def load_dhf_data_from_json(
    json_file_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从JSON文件加载DHF模型所需的时序数据和参数

    Parameters
    ----------
    json_file_path : str
        JSON文件路径，包含时间序列、降雨数据和模型参数

    Returns
    -------
    p_and_e : np.ndarray
        降雨和蒸发数据 [time, basin=1, feature=2]
        feature=1（蒸发）将在模型中根据月蒸发量计算
    parameters : np.ndarray
        模型参数 [basin=1, parameter=22]
    """
    # 读取JSON文件
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 解析时间序列和降雨数据
    dt = json.loads(data["dt"])  # 时间序列
    rain = json.loads(data["rain"])  # 降雨数据

    # 构建p_and_e数组 [time, basin=1, feature=2]
    # feature=1（蒸发）将在模型中根据月蒸发量计算
    time_steps = len(rain)
    p_and_e = np.zeros((time_steps, 1, 2))
    p_and_e[:, 0, 0] = rain  # 降雨数据
    p_and_e[:, 0, 1] = 0.0  # 蒸发数据占位（将在模型中计算）

    # 构建参数数组 [basin=1, parameter=22]
    # 参数顺序: [S0, U0, D0, K, KW, K2, KA, G, A, B, B0, K0, N, L, DD, CC, COE, DDL, CCL, SA0, UA0, YA0]
    parameters = np.array(
        [
            [
                float(data["S0"]),
                float(data["U0"]),
                float(data["D0"]),
                float(data["K"]),
                float(data["KW"]),
                float(data["K2"]),
                float(data["KA"]),
                float(data["G"]),
                float(data["A"]),
                float(data["B"]),
                float(data["B0"]),
                float(data["K0"]),
                float(data["N"]),
                float(data["L"]),
                float(data["DD"]),
                float(data["CC"]),
                float(data["COE"]),
                float(data["DDL"]),
                float(data["CCL"]),
                float(data["SA0"]),
                float(data["UA0"]),
                float(data["YA0"]),
            ]
        ]
    )

    return p_and_e, parameters


def load_dhf_data_from_csv_and_json(
    csv_file_path: str, json_params_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从CSV文件加载降雨和蒸发数据，从JSON文件加载参数并转换为标准格式

    Parameters
    ----------
    csv_file_path : str
        CSV文件路径，包含time, prcp(mm/h), pet(mm/h), flow(m^3/s)列
    json_params_path : str
        JSON参数文件路径，包含模型参数

    Returns
    -------
    p_and_e : np.ndarray
        precipitation and evapotranspiration data [time, basin=1, feature=2]
    parameters : np.ndarray
        model parameters [basin=1, parameter=22]
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)

    # 提取降雨和蒸发数据
    precipitation = df["prcp(mm/h)"].values
    evapotranspiration = df["pet(mm/h)"].values

    # 构建p_and_e数组 [time, basin=1, feature=2]
    time_steps = len(precipitation)
    p_and_e = np.zeros((time_steps, 1, 2))
    p_and_e[:, 0, 0] = precipitation  # precipitation
    p_and_e[:, 0, 1] = evapotranspiration  # evapotranspiration

    # 从JSON文件读取参数
    with open(json_params_path, "r", encoding="utf-8") as f:
        param_data = json.load(f)

    # 构建参数数组 [basin=1, parameter=22]
    # 参数顺序: [S0, U0, D0, K, KW, K2, KA, G, A, B, B0, K0, N, L, DD, CC, COE, DDL, CCL, SA0, UA0, YA0]
    parameters = np.array(
        [
            [
                float(param_data["S0"]),
                float(param_data["U0"]),
                float(param_data["D0"]),
                float(param_data["K"]),
                float(param_data["KW"]),
                float(param_data["K2"]),
                float(param_data["KA"]),
                float(param_data["G"]),
                float(param_data["A"]),
                float(param_data["B"]),
                float(param_data["B0"]),
                float(param_data["K0"]),
                float(param_data["N"]),
                float(param_data["L"]),
                float(param_data["DD"]),
                float(param_data["CC"]),
                float(param_data["COE"]),
                float(param_data["DDL"]),
                float(param_data["CCL"]),
                float(param_data["SA0"]),
                float(param_data["UA0"]),
                float(param_data["YA0"]),
            ]
        ]
    )

    return p_and_e, parameters


def main():
    """主函数示例"""
    try:
        # 从CSV和JSON文件加载数据
        p_and_e, parameters = load_dhf_data_from_csv_and_json(
            csv_file_path="hydromodel_dev/data/DHF.csv",
            json_params_path="hydromodel_dev/data/dhf_data.json",
        )

        # 读取原始CSV文件以获取时间列和观测流量
        df_original = pd.read_csv("hydromodel_dev/data/DHF.csv")

        # 设置预热期
        warmup_length = 0  # 使用0天作为预热期

        # 使用跟踪功能运行DHF模型
        print("=== 开始计算 ===")
        q_sim, all_results = dhf(
            p_and_e=p_and_e,
            parameters=parameters,
            warmup_length=warmup_length,
            return_state=True,
        )

        # 准备结果数据 - 考虑预热期
        result_dict = {
            "time": df_original["time"]
            .iloc[warmup_length:]
            .reset_index(drop=True),
            "precipitation": df_original["prcp(mm/h)"]
            .iloc[warmup_length:]
            .reset_index(drop=True),
            "pet": df_original["pet(mm/h)"]
            .iloc[warmup_length:]
            .reset_index(drop=True),
            "flow_obs": df_original["flow(m^3/s)"]
            .iloc[warmup_length:]
            .reset_index(drop=True),
            "QSim": q_sim.flatten(),
            "runoffSim": all_results["runoffSim"].flatten(),
        }

        # 创建结果DataFrame
        result_df = pd.DataFrame(result_dict)

        # 保存为CSV文件
        output_csv_path = "hydromodel_dev/data/dhf_result.csv"
        result_df.to_csv(output_csv_path, index=False)

        # 输出计算信息
        # print("\n=== DHF模型计算完成 ===")
        # print(f"预热期长度: {warmup_length} 天")
        # print(f"输入数据形状: {p_and_e.shape}")
        # print(f"参数形状: {parameters.shape}")
        # print(f"输出径流形状: {q_sim.shape}")
        # print(f"实际计算时段数: {len(q_sim)}")

        print(f"\n结果已保存到:")
        print(f"- CSV文件: {output_csv_path}")
        print(f"注意：输出结果已去除预热期 {warmup_length} 天的数据")

    except Exception as e:
        print(f"计算过程中发生错误: {e}")

        traceback.print_exc()


if __name__ == "__main__":
    main()
