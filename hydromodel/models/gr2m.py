'''
Author: zhuanglaihong
Date: 2025-02-21 15:37:10
LastEditTime: 2025-03-10 17:32:13
LastEditors: zhuanglaihong
Description: 
FilePath: /zlh/hydromodel/hydromodel/models/gr2m.py
Copyright: Copyright (c) 2021-2024 zhuanglaihong. All rights reserved.
'''

import math
from typing import Optional, Tuple
import numpy as np
from numba import jit

from hydromodel.models.model_config import MODEL_PARAM_DICT
from hydromodel.models.xaj import uh_conv


def production(inputs, x1, s0):
    """
    GR2M模型的产流计算
    
    Parameters
    ----------
    inputs: ndarray
        2-dim input -- [basin, variable]: 降水和潜在蒸发
    x1: ndarray
        1-dim -- [basin]: 产流参数，表示产流库容量
    s0: ndarray
        1-dim -- [basin]: 初始产流库状态
        
    Returns
    -------
    tuple
        (pr, et, s): 产流量、蒸发量和产流库状态
    """
    p = inputs[:, 0]  # 降水
    e = inputs[:, 1]  # 潜在蒸发
    
    # 计算φ = tanh(P/X1)
    phi = np.tanh(p / x1)
    
    # 计算S1 = (S + X1*φ) / (1 + φ*S/X1)
    s1 = (s0 + x1 * phi) / (1 + phi * s0 / x1)
    
    # 计算P1 = P + S - S1
    p1 = p + s0 - s1
    
    # 计算ψ = tanh(E/X1)
    psi = np.tanh(e / x1)
    
    # 计算S2 = S1(1-ψ) / (1 + ψ(1-S1/X1))
    s2 = s1 * (1 - psi) / (1 + psi * (1 - s1 / x1))
    
    # 计算实际蒸发量
    et = s1 - s2
    
    # 计算S = S2 / [1 + (S2/X1)^3]^(1/3)
    s = s2 / np.power(1 + np.power(s2 / x1, 3), 1/3)
    
    # 计算P2 = S2 - S
    p2 = s2 - s
    
    # 计算P3 = P1 + P2
    p3 = p1 + p2
    
    return p3, et, s


def routing(p3, x5, r0):
    """
    GR2M模型的汇流计算
    
    Parameters
    ----------
    p3: ndarray
        1-dim -- [basin]: 产流量
    
    x5: ndarray
        1-dim -- [basin]: 汇流库出流系数
    r0: ndarray
        1-dim -- [basin]: 初始汇流库状态
        
    Returns
    -------
    tuple
        (q, r): 流量和汇流库状态
    """
    # 计算R1 = R + P3
    r1 = r0 + p3
    
    # 计算R2 = X5*R1
    r2 = x5 * r1
    
    # 计算Q 
    q = np.power(r2, 2) / (r2 + 60)
    
    # 计算R = R2 - Q
    r = r2 - q
    
    return q, r
def aggregate_to_monthly(p_and_e, warmup_length):
    """
    将小时尺度数据聚合为月尺度
    
    Parameters
    ----------
    p_and_e: ndarray
        3-dim input -- [time, basin, variable]: 小时尺度的降水和潜在蒸发
    warmup_length: int
        预热期长度（小时）
        
    Returns
    -------
    tuple
        (monthly_data, monthly_warmup_length): 月尺度数据和对应的预热期长度
    """
    # 获取时间维度
    total_hours = p_and_e.shape[0]
    
    # 将小时转换为天数
    total_days = total_hours // 24
    
    # 计算月份数（假设每月30天）
    n_months = (total_days + 30) // 30
    
    # 计算月尺度的预热期长度
    monthly_warmup_length = (warmup_length // 24 + 30) // 30
    
    # 创建月尺度数据数组
    monthly_data = np.zeros((n_months, p_and_e.shape[1], p_and_e.shape[2]))
    
    # 按月聚合数据
    for i in range(n_months):
        start_hour = i * 30 * 24  # 每月起始小时
        end_hour = min((i + 1) * 30 * 24, total_hours)  # 每月结束小时
        if end_hour > start_hour:
            # 降水累加
            monthly_data[i, :, 0] = np.sum(p_and_e[start_hour:end_hour, :, 0], axis=0)
            # 蒸发取平均后乘以月天数
            monthly_data[i, :, 1] = np.mean(p_and_e[start_hour:end_hour, :, 1], axis=0) * 30
    
    return monthly_data, monthly_warmup_length

def gr2m(p_and_e, parameters, warmup_length: int, return_state=False, **kwargs):
    """
    run GR2m model (simplified version)

    Parameters
    ----------
    p_and_e: ndarray
        3-dim input -- [time, basin, variable]: 日尺度的降水和潜在蒸发
    parameters
        2-dim variable -- [basin, parameter]:
        the parameters are x1, x5 (产流库容量和汇流库系数)
    warmup_length
        length of warmup period (days)
    return_state
        if True, return state values, mainly for warmup periods

    Returns
    -------
    Union[np.array, tuple]
        streamflow or (streamflow, states)
    """
    
    
    model_param_dict = kwargs.get("gr2m", None)
    if model_param_dict is None:
        model_param_dict = MODEL_PARAM_DICT["gr2m"]
    # params
    param_ranges = model_param_dict["param_range"]
    x1_scale = param_ranges["x1"]
    x5_scale = param_ranges["x5"]
    x1 = x1_scale[0] + parameters[:, 0] * (x1_scale[1] - x1_scale[0])
    x5 = x5_scale[0] + parameters[:, 1] * (x5_scale[1] - x5_scale[0])
    
    if warmup_length > 0:
        # 将数据转换为月尺度
        monthly_data, monthly_warmup_length = aggregate_to_monthly(p_and_e, warmup_length)
        
        
        s0 = 0.5 * x1
        r0 = np.zeros_like(x1)
        
        # 使用预热期数据进行预热
        inputs_warmup = monthly_data[0:monthly_warmup_length, :, :]
        for i in range(inputs_warmup.shape[0]):
            if i == 0:
                pr, et, s = production(inputs_warmup[i, :, :], x1, s0)
            else:
                pr, et, s = production(inputs_warmup[i, :, :], x1, s)
            
            if i == 0:
                q, r = routing(pr, x5, r0)
            else:
                q, r = routing(pr, x5, r)
            
            # 更新状态变量
            s0 = s
            r0 = r
    else:
        # 将小时尺度数据转换为月尺度
        monthly_data, monthly_warmup_length = aggregate_to_monthly(p_and_e, 0)
        s0 = 0.5 * x1
        r0 = np.zeros_like(x1)
    
    inputs = monthly_data[monthly_warmup_length:, :, :]
    streamflow_ = np.full(inputs.shape[:2], 0.0)
    prs = np.full(inputs.shape[:2], 0.0)
    ets = np.full(inputs.shape[:2], 0.0)
    
    # 逐月计算
    for i in range(inputs.shape[0]):
        if i == 0:
            pr, et, s = production(inputs[i, :, :], x1, s0)
        else:
            pr, et, s = production(inputs[i, :, :], x1, s)
        
        prs[i, :] = pr
        ets[i, :] = et
        
        if i == 0:
            q, r = routing(pr, x5, r0)
        else:
            q, r = routing(pr, x5, r)
        streamflow_[i, :] = q
    
    streamflow = np.expand_dims(streamflow_, axis=2)
    return (streamflow, ets, s, r) if return_state else (streamflow, ets)