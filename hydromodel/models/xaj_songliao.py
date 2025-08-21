"""
Author: zhuanglaihong
Date: 2025-08-20 23:01:02
LastEditTime: 2025-08-20 19:35:20
LastEditors: zhuanglaihong
Description: XinAnJiang Model - Python implementation based on Java version with vectorized operations
FilePath: /hydromodel/hydromodel/models/xaj_songliao.py
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


# @jit(nopython=True)
def dunne_mechanism_vectorized(
    precipitation: np.ndarray,
    edt: np.ndarray,
    pe: np.ndarray,
    wu: np.ndarray,
    wl: np.ndarray,
    wd: np.ndarray,
    wm: np.ndarray,
    um: np.ndarray,
    lm: np.ndarray,
    dm: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    im: np.ndarray,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    向量化蓄满产流模型 - 对应Java的dunneMechanism

    Parameters
    ----------
    precipitation : np.ndarray
        降水量
    edt : np.ndarray
        实际蒸散发
    pe : np.ndarray
        净雨
    wu, wl, wd : np.ndarray
        上层、下层、深层张力水含量
    wm, um, lm, dm : np.ndarray
        各层水容量参数
    b, c, im : np.ndarray
        模型参数

    Returns
    -------
    tuple
        更新后的状态变量和产流量
    """
    # 流域单点最大蓄水容量
    wmm = wm * (1.0 + b) / (1.0 - im)

    # 初始化输出变量
    r = np.zeros_like(pe)  # 总产流
    rim = np.zeros_like(pe)  # 不透水面积产流
    wu_new = wu.copy()
    wl_new = wl.copy()
    wd_new = wd.copy()

    # 处理有净雨的情况 (pe >= 0)
    pos_mask = pe >= 0.0
    if np.any(pos_mask):
        # 计算流域平均张力水蓄量
        wmt = wu[pos_mask] + wl[pos_mask] + wd[pos_mask]
        wmt = np.minimum(wmt, wm[pos_mask])

        # 计算总产流
        at = wmm[pos_mask] * (
            1.0
            - np.power((1.0 - wmt / wm[pos_mask]), 1.0 / (1.0 + b[pos_mask]))
        )

        # 两种情况计算产流
        case1_mask = (pe[pos_mask] + at) <= wmm[pos_mask]
        case2_mask = ~case1_mask

        # 情况1: pe + at <= wmm
        if np.any(case1_mask):
            r_case1 = (
                pe[pos_mask][case1_mask]
                + wmt[case1_mask]
                - wm[pos_mask][case1_mask]
                + wm[pos_mask][case1_mask]
                * np.power(
                    (
                        1
                        - (pe[pos_mask][case1_mask] + at[case1_mask])
                        / wmm[pos_mask][case1_mask]
                    ),
                    1 + b[pos_mask][case1_mask],
                )
            )

        # 情况2: pe + at > wmm
        if np.any(case2_mask):
            r_case2 = (
                pe[pos_mask][case2_mask]
                + wmt[case2_mask]
                - wm[pos_mask][case2_mask]
            )

        # 合并两种情况的结果
        r_temp = np.zeros_like(pe[pos_mask])
        if np.any(case1_mask):
            r_temp[case1_mask] = r_case1
        if np.any(case2_mask):
            r_temp[case2_mask] = r_case2
        r[pos_mask] = r_temp

        # 不透水面积产流
        rim[pos_mask] = pe[pos_mask] * im[pos_mask]

        # 更新张力水蓄量
        wpt = wu[pos_mask] + pe[pos_mask] - r[pos_mask]

        # 上层更新
        mask_u = wpt < um[pos_mask]
        wu_new[pos_mask] = np.where(mask_u, wpt, um[pos_mask])

        # 下层和深层更新
        wwpt = wu[pos_mask] + wl[pos_mask] + pe[pos_mask] - r[pos_mask]
        mask_ml = wwpt < (um[pos_mask] + lm[pos_mask])

        wl_new[pos_mask] = np.where(
            mask_u,
            wl[pos_mask],
            np.where(mask_ml, wwpt - um[pos_mask], lm[pos_mask]),
        )

        wd_new[pos_mask] = np.where(
            mask_u | mask_ml,
            wd[pos_mask],
            wmt
            + pe[pos_mask]
            - r[pos_mask]
            - wu_new[pos_mask]
            - wl_new[pos_mask],
        )

    # 处理无净雨的情况 (pe < 0) - 蒸散发
    neg_mask = pe < 0.0
    if np.any(neg_mask):
        r[neg_mask] = 0.0
        rim[neg_mask] = 0.0

        # 蒸散发计算
        # 上层蒸散发
        mask_u_sufficient = (wu[neg_mask] + pe[neg_mask]) >= 0.0
        wu_new[neg_mask] = np.where(
            mask_u_sufficient, wu[neg_mask] + pe[neg_mask], 0.0
        )

        # 下层和深层蒸散发
        mask_u_insufficient = ~mask_u_sufficient
        if np.any(mask_u_insufficient):
            eu = (
                wu[neg_mask][mask_u_insufficient]
                + precipitation[neg_mask][mask_u_insufficient]
            )

            # 下层蒸散发能力判断
            mask_l_high = (
                wl[neg_mask][mask_u_insufficient]
                >= c[neg_mask][mask_u_insufficient]
                * lm[neg_mask][mask_u_insufficient]
            )

            if np.any(mask_l_high):
                el = (
                    (
                        edt[neg_mask][mask_u_insufficient][mask_l_high]
                        - eu[mask_l_high]
                    )
                    * wl[neg_mask][mask_u_insufficient][mask_l_high]
                    / lm[neg_mask][mask_u_insufficient][mask_l_high]
                )
                wl_temp = wl[neg_mask].copy()
                wl_temp[mask_u_insufficient] = np.where(
                    mask_l_high,
                    wl[neg_mask][mask_u_insufficient] - el,
                    wl[neg_mask][mask_u_insufficient],
                )
                wl_new[neg_mask] = wl_temp

            # 深层蒸散发
            mask_l_low = ~mask_l_high
            if np.any(mask_l_low):
                deficit = c[neg_mask][mask_u_insufficient][mask_l_low] * (
                    edt[neg_mask][mask_u_insufficient][mask_l_low]
                    - eu[mask_l_low]
                )

                mask_deficit_small = (
                    wl[neg_mask][mask_u_insufficient][mask_l_low] >= deficit
                )

                wl_temp = wl[neg_mask].copy()
                wd_temp = wd[neg_mask].copy()

                if np.any(mask_deficit_small):
                    wl_temp[mask_u_insufficient] = np.where(
                        mask_l_low & mask_deficit_small,
                        wl[neg_mask][mask_u_insufficient] - deficit,
                        wl_temp[mask_u_insufficient],
                    )

                mask_deficit_large = ~mask_deficit_small
                if np.any(mask_deficit_large):
                    ed = (
                        deficit[mask_deficit_large]
                        - wl[neg_mask][mask_u_insufficient][mask_l_low][
                            mask_deficit_large
                        ]
                    )
                    wl_temp[mask_u_insufficient] = np.where(
                        mask_l_low & mask_deficit_large,
                        0.0,
                        wl_temp[mask_u_insufficient],
                    )
                    wd_temp[mask_u_insufficient] = np.where(
                        mask_l_low & mask_deficit_large,
                        wd[neg_mask][mask_u_insufficient] - ed,
                        wd_temp[mask_u_insufficient],
                    )

                wl_new[neg_mask] = wl_temp
                wd_new[neg_mask] = wd_temp

    # 确保非负值
    wu_new = np.maximum(wu_new, 0.0)
    wl_new = np.maximum(wl_new, 0.0)
    wd_new = np.maximum(wd_new, 0.0)

    return wu_new, wl_new, wd_new, r, pe, rim


# @jit(nopython=True)
def free_tank_vectorized(
    r: np.ndarray,
    pe: np.ndarray,
    s: np.ndarray,
    fr: np.ndarray,
    sm: np.ndarray,
    ex: np.ndarray,
    kg: np.ndarray,
    ki: np.ndarray,
    im: np.ndarray,
    rim: np.ndarray,
    time_interval: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    向量化分水源模型 - 对应Java的freeTank

    Parameters
    ----------
    r : np.ndarray
        总产流量
    pe : np.ndarray
        净雨
    s : np.ndarray
        自由水蓄量
    fr : np.ndarray
        产流面积系数
    sm, ex, kg, ki, im : np.ndarray
        模型参数
    rim : np.ndarray
        不透水面积产流
    time_interval : float
        时间间隔

    Returns
    -------
    tuple
        分水源结果和更新的状态变量
    """
    time_steps = len(r)

    # 初始化输出变量
    rs = np.zeros_like(r)  # 地表水
    ri = np.zeros_like(r)  # 壤中水
    rg = np.zeros_like(r)  # 地下水
    s_new = np.zeros(time_steps + 1)  # 自由水蓄量，多一个时间步
    fr_new = np.zeros_like(r)  # 产流面积系数

    # 初始化状态
    s_new[0] = s[0]
    fr_new[0] = fr[0]

    # 参数时段转换
    d = 24 / time_interval
    kid = (1 - np.power((1 - (ki + kg)), 1.0 / d)) / (1 + kg / ki)
    kgd = kid * kg / ki

    # 逐时间步处理
    for i in range(time_steps):
        # 范围控制
        s_new[i] = max(0.0, s_new[i])

        # 有产流量才需要分段计算
        if pe[i] > 0:
            # 计算产流面积
            fr_new[i] = r[i] / pe[i]
            fr_new[i] = max(0.0, min(1.0, fr_new[i]))

            if i != 0:
                # 换算自由水蓄量
                if fr_new[i] > 0:
                    s_new[i] = s_new[i] * fr_new[i - 1] / fr_new[i]
                else:
                    s_new[i] = 0.0

            # 分段计算
            if fr_new[i] > 0:
                q = r[i] / fr_new[i]  # 产流面积上净雨
            else:
                q = 0.0
            n = int(np.floor(q / 5 + 1))
            pe_seg = q / n  # 每段的净雨深

            smm = sm[i] * (1 + ex[i])  # 流域单点最大自由水蓄水容量

            # 产流面积上的单点最大自由水蓄水容量
            if ex[i] <= 0.0001:
                smmf = smm
            else:
                smmf = smm * (1.0 - np.power(1.0 - fr_new[i], 1.0 / ex[i]))

            smf = smmf / (1.0 + ex[i])  # 产流面积上平均自由水蓄水容量

            s_temp = s_new[i]  # 该时段初产流面积上的自由水蓄量
            rs_temp = 0.0  # 地表产流
            ri_temp = 0.0  # 土壤产流
            rg_temp = 0.0  # 地下产流

            # 精确分段计算
            for j in range(n):
                if s_temp >= smf:  # 自由水蓄水库已蓄满
                    rs_temp += pe_seg * fr_new[i]
                    ri_temp += smf * kid[i] * fr_new[i]
                    rg_temp += smf * kgd[i] * fr_new[i]
                    s_temp = smf - (ri_temp + rg_temp) / fr_new[i] / n
                else:
                    au = smmf * (
                        1 - np.power(1 - s_temp / smf, 1 / (1 + ex[i]))
                    )
                    if pe_seg + au < smmf:
                        rs_temp += fr_new[i] * (
                            pe_seg
                            - smf
                            + s_temp
                            + smf
                            * np.power(1 - (pe_seg + au) / smmf, 1 + ex[i])
                        )
                        smt = (
                            pe_seg + s_temp - rs_temp / fr_new[i] / n
                        )  # 扣除地表产流后的自由水蓄量
                        ri_temp += smt * kid[i] * fr_new[i]
                        rg_temp += smt * kgd[i] * fr_new[i]
                        s_temp += (
                            pe_seg
                            - (rs_temp + ri_temp + rg_temp) / fr_new[i] / n
                        )
                    else:
                        rs_temp += (pe_seg + s_temp - smf) * fr_new[i]
                        ri_temp += smf * kid[i] * fr_new[i]
                        rg_temp += smf * kgd[i] * fr_new[i]
                        s_temp = smf - (ri_temp + rg_temp) / fr_new[i] / n

            rs[i] = rs_temp
            ri[i] = ri_temp
            rg[i] = rg_temp
            s_new[i + 1] = s_temp

        else:
            # 无产流量
            if i != 0:
                fr_new[i] = fr_new[i - 1]
                fr_new[i] = max(0.0, min(1.0, fr_new[i]))

            rs[i] = 0.0
            ri[i] = kid[i] * s_new[i] * fr_new[i]
            rg[i] = kgd[i] * s_new[i] * fr_new[i]

            s_new[i + 1] = s_new[i] * (1 - kid[i] - kgd[i])
            s_new[i + 1] = max(0.0, min(sm[i], s_new[i + 1]))

        # 考虑不透水面积
        rs[i] = rs[i] * (1 - im[i]) + rim[i]  # 地表径流量
        ri[i] = ri[i] * (1 - im[i])  # 壤中径流量
        rg[i] = rg[i] * (1 - im[i])  # 地下径流量

    return rs, ri, rg, s_new[:-1], fr_new  # 返回与输入长度相同的s_new


# @jit(nopython=True)
def liner_reservoir_vectorized(
    rs: np.ndarray,
    ri: np.ndarray,
    rg: np.ndarray,
    time_interval: float,
    area: float,
    ci: np.ndarray,
    cg: np.ndarray,
    qsp: float = 0.0,
    qip: float = 0.0,
    qgp: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    向量化坡面汇流-线性水库 - 对应Java的LinerReservoir

    Parameters
    ----------
    rs, ri, rg : np.ndarray
        地表水、壤中水、地下水
    time_interval : float
        时间间隔
    area : float
        流域面积
    ci, cg : np.ndarray
        消退系数
    qsp, qip, qgp : float
        初始流量

    Returns
    -------
    tuple
        地表径流、壤中流、地下径流流量
    """
    time_steps = len(rs)

    # 单位转换系数
    u = area / (3.6 * time_interval)

    # 参数时段转换
    d = 24 / time_interval
    cid = np.power(ci, 1.0 / d)
    cgd = np.power(cg, 1.0 / d)

    # 初始化输出变量
    qs = np.zeros(time_steps)  # 地表径流流量
    qi = np.zeros(time_steps)  # 壤中流流量
    qg = np.zeros(time_steps)  # 地下径流流量

    # 第一个时间步
    qs[0] = rs[0] * u
    qi[0] = qip + ri[0] * (1 - cid[0]) * u
    qg[0] = qgp + rg[0] * (1 - cgd[0]) * u

    # 后续时间步
    for i in range(1, time_steps):
        qs[i] = rs[i] * u
        qi[i] = qi[i - 1] * cid[i] + ri[i] * (1 - cid[i]) * u
        qg[i] = qg[i - 1] * cgd[i] + rg[i] * (1 - cgd[i]) * u

    return qs, qi, qg


# @jit(nopython=True)
def time_lag_vectorized(
    q: np.ndarray,
    lag: float,
    cs: float,
) -> np.ndarray:
    """
    向量化河网汇流-滞时法 - 对应Java的TimeLag

    Parameters
    ----------
    q : np.ndarray
        输入流量
    lag : float
        滞时
    cs : float
        地面径流消退系数

    Returns
    -------
    np.ndarray
        滞后后的流量
    """
    time_steps = len(q)
    qf = np.zeros(time_steps)
    t = int(lag)

    if t <= 0:
        t = 0
        for i in range(time_steps):
            if i == 0:
                qf[i] = 0.0  # 简化初始条件
            else:
                qf[i] = cs * qf[i - 1] + (1 - cs) * q[i]
    else:
        for i in range(time_steps):
            if i < t:
                qf[i] = 0.0  # 简化初始条件
            else:
                qf[i] = cs * qf[i - 1] + (1 - cs) * q[i - t]

    return qf


# @jit(nopython=True)
def muskingum_vectorized(
    qr: np.ndarray,
    ke: float,
    xe: float,
    n: int,
    time_interval: float,
) -> np.ndarray:
    """
    向量化河道演进-分段马斯京根法 - 对应Java的muskingum

    Parameters
    ----------
    qr : np.ndarray
        入流
    ke : float
        子河段洪水波传播时间
    xe : float
        子河段流量比重因子
    n : int
        河段数
    time_interval : float
        时间间隔

    Returns
    -------
    np.ndarray
        出流
    """
    time_steps = len(qr)
    qc = np.zeros(n)  # 河段下断面时段初始流量
    q_routing = np.zeros(time_steps)  # 河道演进结果

    # 马斯京根系数
    cd = 0.5 * time_interval + ke - ke * xe
    c0 = (0.5 * time_interval - ke * xe) / cd
    c1 = (0.5 * time_interval + ke * xe) / cd
    c2 = 1 - c0 - c1

    if c0 >= 0.0 and c2 >= 0.0:  # 满足马斯京根法的适用条件
        for i in range(time_steps):
            if i == 0:
                qi1 = qr[i]
            else:
                qi1 = qr[i - 1]
            qi2 = qr[i]

            # 分段计算
            if n > 0:
                for j in range(n):
                    qo1 = qc[j]
                    qo2 = c0 * qi2 + c1 * qi1 + c2 * qo1
                    qi1 = qo1
                    qi2 = qo2
                    qc[j] = qo2
            else:
                qo2 = qr[i]

            if qo2 < 0.0001:
                qo2 = 0.0
            q_routing[i] = qo2
    else:
        # 不考虑河道演进
        q_routing = qr.copy()

    return q_routing


# @jit(nopython=True)
def channel_routing_combined(
    qtmp: np.ndarray,
    lag: float,
    cs: float,
    kk: float,
    x: float,
    mp: int,
    time_interval: float,
) -> np.ndarray:
    """
    组合河网汇流计算 - 结合TimeLag和muskingum

    Parameters
    ----------
    qtmp : np.ndarray
        总入流
    lag : float
        滞时
    cs : float
        消退系数
    kk : float
        马斯京根K参数
    x : float
        马斯京根X参数
    mp : int
        河段数
    time_interval : float
        时间间隔

    Returns
    -------
    np.ndarray
        最终出流
    """
    # 河网汇流-滞时法
    qf = time_lag_vectorized(qtmp, lag, cs)

    # 河道演进-马斯京根法
    q_sim = muskingum_vectorized(qf, kk, x, mp, time_interval)

    return q_sim



def xaj_songliao(
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
    向量化新安江松辽水文模型

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
                  SM, EX, KG, KI, CS, CI, CG, LAG, KK, X, MP]
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
        model_param_dict = MODEL_PARAM_DICT.get("xaj_songliao")
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

    # Calculate derived parameters
    um = wumx * wm  # Upper layer tension water capacity
    lm = wlmx * (wm - um)  # Lower layer tension water capacity
    dm = wm - um - lm  # Deep layer tension water capacity
    dm = np.maximum(dm, 0.0)  # Ensure non-negative

    # Handle warmup period
    if warmup_length > 0:
        p_and_e_warmup = p_and_e[0:warmup_length, :, :]
        *_, wu_init, wl_init, wd_init, s_init, fr_init = xaj_songliao(
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

    # Initialize state and output arrays - [time, basin]
    wu = np.zeros((actual_time_steps, num_basins))
    wl = np.zeros((actual_time_steps, num_basins))
    wd = np.zeros((actual_time_steps, num_basins))
    s = np.zeros((actual_time_steps, num_basins))
    fr = np.zeros((actual_time_steps, num_basins))

    # Initialize output arrays
    runoff_sim = np.zeros((actual_time_steps, num_basins))
    q_sim = np.zeros((actual_time_steps, num_basins))
    rs_out = np.zeros((actual_time_steps, num_basins))
    ri_out = np.zeros((actual_time_steps, num_basins))
    rg_out = np.zeros((actual_time_steps, num_basins))
    pe_out = np.zeros((actual_time_steps, num_basins))

    # Main time loop 
    for i in range(actual_time_steps):
        # Current precipitation and PET for all basins
        prcp = inputs[i, :, 0]
        pet = inputs[i, :, 1]

        # Get current states
        if i == 0:
            wu_curr = wu0
            wl_curr = wl0
            wd_curr = wd0
            s_curr = s0
            fr_curr = fr0
        else:
            wu_curr = wu[i - 1, :]
            wl_curr = wl[i - 1, :]
            wd_curr = wd[i - 1, :]
            s_curr = s[i - 1, :]
            fr_curr = fr[i - 1, :]

        # Step 1: Calculate evapotranspiration and net precipitation
        pe, edt = calculate_net_precipitation(
            prcp, pet, kc
        )

        # Step 2: Tension water storage and runoff generation (Dunne mechanism)
        wu_new, wl_new, wd_new, r, pe_calc, rim = dunne_mechanism_vectorized(
            prcp, edt, pe, wu_curr, wl_curr, wd_curr, wm, um, lm, dm, b, c, im
        )

        # Store states and outputs for this time step
        wu[i, :] = wu_new
        wl[i, :] = wl_new
        wd[i, :] = wd_new
        pe_out[i, :] = pe_calc

    # Step 3: Free water tank and flow separation - 批量处理
    for basin_idx in range(num_basins):
        # 获取该流域的时间序列数据
        r_basin = np.zeros(actual_time_steps)
        pe_basin = pe_out[:, basin_idx]
        rim_basin = np.zeros(actual_time_steps)
        s_basin = np.zeros(actual_time_steps + 1)
        fr_basin = np.zeros(actual_time_steps + 1)

        # 重新计算产流和不透水面积产流
        for i in range(actual_time_steps):
            wu_curr = wu0[basin_idx] if i == 0 else wu[i - 1, basin_idx]
            wl_curr = wl0[basin_idx] if i == 0 else wl[i - 1, basin_idx]
            wd_curr = wd0[basin_idx] if i == 0 else wd[i - 1, basin_idx]

            # 简化的产流计算
            wmt = wu_curr + wl_curr + wd_curr
            wmt = min(wmt, wm[basin_idx])

            if pe_basin[i] >= 0:
                wmm = (
                    wm[basin_idx]
                    * (1.0 + b[basin_idx])
                    / (1.0 - im[basin_idx])
                )
                at = wmm * (
                    1.0
                    - np.power(
                        (1.0 - wmt / wm[basin_idx]), 1.0 / (1.0 + b[basin_idx])
                    )
                )
                if pe_basin[i] + at <= wmm:
                    r_basin[i] = (
                        pe_basin[i]
                        + wmt
                        - wm[basin_idx]
                        + wm[basin_idx]
                        * np.power(
                            (1 - (pe_basin[i] + at) / wmm), 1 + b[basin_idx]
                        )
                    )
                else:
                    r_basin[i] = pe_basin[i] + wmt - wm[basin_idx]
                rim_basin[i] = pe_basin[i] * im[basin_idx]
            else:
                r_basin[i] = 0.0
                rim_basin[i] = 0.0

        # 初始化状态
        s_basin[0] = s0[basin_idx]
        fr_basin[0] = fr0[basin_idx]

        # 分水源计算
        rs_basin, ri_basin, rg_basin, s_new_basin, fr_new_basin = (
            free_tank_vectorized(
                r_basin,
                pe_basin,
                s_basin[:-1],
                fr_basin[:-1],
                np.full(actual_time_steps, sm[basin_idx]),
                np.full(actual_time_steps, ex[basin_idx]),
                np.full(actual_time_steps, kg[basin_idx]),
                np.full(actual_time_steps, ki[basin_idx]),
                np.full(actual_time_steps, im[basin_idx]),
                rim_basin,
                time_interval,
            )
        )

        # 存储结果
        rs_out[:, basin_idx] = rs_basin
        ri_out[:, basin_idx] = ri_basin
        rg_out[:, basin_idx] = rg_basin
        s[:, basin_idx] = s_new_basin
        fr[:, basin_idx] = fr_new_basin
        runoff_sim[:, basin_idx] = rs_basin + ri_basin + rg_basin

        # Step 4: Hillslope routing (Linear reservoir)
        qs, qi, qg = liner_reservoir_vectorized(
            rs_basin,
            ri_basin,
            rg_basin,
            time_interval,
            area,
            np.full(actual_time_steps, ci[basin_idx]),
            np.full(actual_time_steps, cg[basin_idx]),
        )

        # Total inflow to channel
        qtmp = qs + qi + qg
        qtmp = np.maximum(qtmp, 0.0)

        # Step 5: Channel routing (Time lag + Muskingum)
        q_sim[:, basin_idx] = channel_routing_combined(
            qtmp,
            lag[basin_idx],
            cs[basin_idx],
            kk[basin_idx],
            x[basin_idx],
            int(mp[basin_idx]),
            time_interval,
        )

    # Ensure non-negative discharge
    q_sim = np.maximum(q_sim, 0.0)

    # Format outputs to match DHF interface: [seq, batch, feature]
    q_sim = np.expand_dims(q_sim, axis=2)
    runoff_sim = np.expand_dims(runoff_sim, axis=2)
    rs_out = np.expand_dims(rs_out, axis=2)
    ri_out = np.expand_dims(ri_out, axis=2)
    rg_out = np.expand_dims(rg_out, axis=2)
    pe_out = np.expand_dims(pe_out, axis=2)
    wu = np.expand_dims(wu, axis=2)
    wl = np.expand_dims(wl, axis=2)
    wd = np.expand_dims(wd, axis=2)

    if return_state:
        return (
            q_sim,
            runoff_sim,
            rs_out,
            ri_out,
            rg_out,
            pe_out,
            wu,
            wl,
            wd,
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
        模型参数 [basin=1, parameter=23]
    """
    # 读取JSON文件
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 解析时间序列、降雨和蒸散发数据
    dt = json.loads(data["dt"])  # 时间序列
    rain = json.loads(data["rain"])  # 降雨数据
    evap = json.loads(data["evaporation"])  # 蒸散发数据

    # 构建p_and_e数组 [time, basin=1, feature=2]
    time_steps = len(rain)
    p_and_e = np.zeros((time_steps, 1, 2))
    p_and_e[:, 0, 0] = rain  # 降雨数据
    p_and_e[:, 0, 1] = evap  # 蒸散发数据

    # 构建参数数组 [basin=1, parameter=23]
    # 参数顺序: [WUP, WLP, WDP, SP, FRP, WM, WUMx, WLMx, K, B, C, IM,
    #          SM, EX, KG, KI, CS, CI, CG, LAG, KK, X, MP]
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
            ]
        ]
    )

    return p_and_e, parameters
