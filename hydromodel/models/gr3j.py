'''
Author: zhuanglaihong
Date: 2025-03-13 09:35:22
LastEditTime: 2025-03-16 19:45:38
LastEditors: zhuanglaihong
Description: Core code for GR3J model
FilePath: /zlh/hydromodel/hydromodel/models/gr3j.py
Copyright: Copyright (c) 2021-2024 zhuanglaihong. All rights reserved.
'''

import math
from typing import Optional, Tuple
import numpy as np
from numba import jit

from hydromodel.models.model_config import MODEL_PARAM_DICT
from hydromodel.models.xaj import uh_conv

@jit(nopython=True)
def calculate_precip_store(s, precip_net, x1):
    """计算进入产水库的降水量"""
    n = precip_net * (1.0 - (s / x1) ** 2)
    d = 1.0 + (s / x1)
    return n / d

@jit(nopython=True)
def calculate_evap_store(s, evap_net, x1):
    """计算产水库的蒸发损失"""
    n = evap_net * (s / x1) * (2.0 - s / x1)
    d = 1.0 + (2.0 - s / x1)
    return n / d

def production(p_and_e: np.array, x1: np.array, s_level: Optional[np.array] = None) -> Tuple[np.array, np.array]:
    """GR3J的产流计算"""
    precip_difference = p_and_e[:, 0] - p_and_e[:, 1]
    precip_net = np.maximum(precip_difference, 0.0)
    evap_net = np.maximum(-precip_difference, 0.0)

    if s_level is None:
        s_level = 0.5 * x1

    s_level = np.clip(s_level, a_min=np.full(s_level.shape, 0.0), a_max=x1)
    precip_store = calculate_precip_store(s_level, precip_net, x1)
    evap_store = calculate_evap_store(s_level, evap_net, x1)
    
    s_update = s_level - evap_store + precip_store
    s_update = np.clip(s_update, a_min=np.full(s_update.shape, 0.0), a_max=x1)
    
    current_runoff = precip_net - precip_store
    return current_runoff, evap_store, s_update

@jit(nopython=True)
def s_curves1(t, x3):
    """SH1的S曲线"""
    if t <= 0:
        return 0
    elif t < x3:
        return (t / x3) ** 3
    else:
        return 1

@jit(nopython=True)
def s_curves2(t, x3):
    """SH2的S曲线"""
    if t <= 0:
        return 0
    elif t < x3:
        return 0.5 * (t / x3) ** 3
    elif t < 2 * x3:
        return 1 - 0.5 * (2 - t / x3) ** 3
    else:
        return 1

def uh_gr3j(x3):
    """生成GR3J的单位线核"""
    uh1_ordinates = []
    uh2_ordinates = []
    for i in range(len(x3)):
        n_uh1 = int(math.ceil(x3[i]))
        n_uh2 = int(math.ceil(2.0 * x3[i]))
        uh1_ordinate = np.zeros(n_uh1)
        uh2_ordinate = np.zeros(n_uh2)
        for t in range(1, n_uh1 + 1):
            uh1_ordinate[t - 1] = s_curves1(t, x3[i]) - s_curves1(t - 1, x3[i])
        for t in range(1, n_uh2 + 1):
            uh2_ordinate[t - 1] = s_curves2(t, x3[i]) - s_curves2(t - 1, x3[i])
        uh1_ordinates.append(uh1_ordinate)
        uh2_ordinates.append(uh2_ordinate)
    return uh1_ordinates, uh2_ordinates

def routing(q9: np.array, q1: np.array, x2, x3, r_level: Optional[np.array] = None):
    """GR3J的汇流计算"""
    if r_level is None:
        r_level = 0.5 * x3
    
    r_level = np.clip(r_level, a_min=np.full(r_level.shape, 0.0), a_max=x3)
    f = x2 * (r_level / x3) ** 4
    r_star = np.maximum(np.full(r_level.shape, 0.0), r_level + q9 + f)
    
    qr = r_star - (r_star ** -4 + x3 ** -4) ** (-1/4)
    r_updated = r_star - qr
    
    qd = np.maximum(np.full(f.shape, 0.0), q1 + f)
    q = qr + qd
    return q, r_updated

def gr3j(p_and_e, parameters, warmup_length: int, return_state=False, **kwargs):
    """
    run GR3j model

    Parameters
    ----------
    p_and_e: ndarray
        3-dim input -- [time, basin, variable]: precipitation and potential evaporation
    parameters
        2-dim variable -- [basin, parameter]:
        the parameters are x1, x2, x3 
    warmup_length
        length of warmup period
    return_state
        if True, return state values, mainly for warmup periods

    Returns
    -------
    Union[np.array, tuple]
        streamflow or (streamflow, states)
    """
    # 验证输入参数
    if len(p_and_e.shape) != 3 or p_and_e.shape[2] != 2:
        raise ValueError("p_and_e 应该是形状为 [time, basin, 2] 的三维数组")
    if len(parameters.shape) != 2 or parameters.shape[1] < 3:
        raise ValueError("parameters 应该是形状为 [basin, 3+] 的二维数组")
    
    model_param_dict = kwargs.get("gr3j", None)
    if model_param_dict is None:
        model_param_dict = MODEL_PARAM_DICT["gr3j"]
    # params
    param_ranges = model_param_dict["param_range"]
    x1_scale = param_ranges["x1"]
    x2_scale = param_ranges["x2"]  
    x3_scale = param_ranges["x3"]
    
    x1 = x1_scale[0] + parameters[:, 0] * (x1_scale[1] - x1_scale[0])
    x2 = x2_scale[0] + parameters[:, 1] * (x2_scale[1] - x2_scale[0]) 
    x3 = x3_scale[0] + parameters[:, 2] * (x3_scale[1] - x3_scale[0])
    
    # 生成单位线核
    conv_q9, conv_q1 = uh_gr3j(x3)
    
    # 改进预热处理，避免递归调用
    if warmup_length > 0:
        # set no_grad for warmup periods
        p_and_e_warmup = p_and_e[0:warmup_length, :, :]
        _, _, s0, r0 = gr3j(
            p_and_e_warmup, parameters, warmup_length=0, return_state=True, **kwargs
        )
    else:
        s0 = 0.5 * x1
        r0 = 0.5 * x3
        
    inputs = p_and_e[warmup_length:, :, :]
    streamflow_ = np.full(inputs.shape[:2], 0.0)
    prs = np.full(inputs.shape[:2], 0.0)
    ets = np.full(inputs.shape[:2], 0.0)
    
    # 产流计算
    for i in range(inputs.shape[0]):
        if i == 0:
            pr, et, s = production(inputs[i, :, :], x1, s0)
        else:
            pr, et, s = production(inputs[i, :, :], x1, s)
        prs[i, :] = pr
        ets[i, :] = et
        
    prs_x = np.expand_dims(prs, axis=2)
    q9 = np.full([inputs.shape[0], inputs.shape[1], 1], 0.0)
    q1 = np.full([inputs.shape[0], inputs.shape[1], 1], 0.0)
    
    # 分别计算90%和10%的产流部分
    for j in range(inputs.shape[1]):
        q9[:, j:j+1, :] = 0.9 * uh_conv(
            prs_x[:, j:j+1, :], conv_q9[j].reshape(-1, 1, 1)
        )
        q1[:, j:j+1, :] = 0.1 * uh_conv(
            prs_x[:, j:j+1, :], conv_q1[j].reshape(-1, 1, 1)
        )
    
    # 汇流计算
    for i in range(inputs.shape[0]):
        if i == 0:
            q, r = routing(q9[i, :, 0], q1[i, :, 0], x2, x3, r0)
        else:
            q, r = routing(q9[i, :, 0], q1[i, :, 0], x2, x3, r)
        streamflow_[i, :] = q
        
    streamflow = np.expand_dims(streamflow_, axis=2)
    return (streamflow, ets, s, r) if return_state else (streamflow, ets)