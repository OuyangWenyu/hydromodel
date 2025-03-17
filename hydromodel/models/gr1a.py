'''
Author: zhuanglaihong
Date: 2025-02-21 15:36:42
LastEditTime: 2025-03-17 10:33:45
LastEditors: zhuanglaihong
Description: Core code for GR1A model
FilePath: /zlh/hydromodel/hydromodel/models/gr1a.py
Copyright: Copyright (c) 2021-2024 zhuanglaihong. All rights reserved.
'''


import math
from typing import Optional, Tuple
import numpy as np
from numba import jit
from hydromodel.models.model_config import MODEL_PARAM_DICT


def calculate_qk(pk, pk_1, ek, x):
    """
    年径流的计算公式
    """
    # 计算分母项
    denominator = 1 + ((0.7 * pk + 0.3 *pk_1 ) / (x*ek)) ** 2
    
    # 计算整体公式
    qk = pk * (1 - 1 / (denominator ** 0.5))
    
    return qk

def gr1a(p_and_e, parameters, warmup_length: int, return_state=False, **kwargs):
    """
    run GR1a model

    Parameters
    ----------
    p_and_e: ndarray
        3-dim input -- [time, basin, variable]: yearly precipitation and potential evaporation
    parameters
        2-dim variable -- [basin, parameter]:
        the parameters is x
    warmup_length
        length of warmup period (years)
    return_state
        if True, return state values, mainly for warmup periods

    Returns
    -------
    Union[np.array, tuple]
        streamflow or (streamflow, states)
    """
    model_param_dict = kwargs.get("gr1a", None)
    if model_param_dict is None:
        model_param_dict = MODEL_PARAM_DICT["gr1a"]
    # params
    param_ranges = model_param_dict["param_range"]
    x1_scale = param_ranges["x1"]
    x1 = x1_scale[0] + parameters[:, 0] * (x1_scale[1] - x1_scale[0])

    if warmup_length > 0:
        # 使用预热期数据
        p_and_e_warmup = p_and_e[0:warmup_length, :, :]
        _, _, pk_1 ,r = gr1a(
            p_and_e_warmup, parameters, warmup_length=0, return_state=True, **kwargs
        )
    else:
        pk_1 = None

    # 获取输入数据
    inputs = p_and_e[warmup_length:, :, :]
    time_length, basin_num, _ = inputs.shape
    
    # 初始化年径流数组
    streamflow_ = np.zeros((time_length, basin_num))
    
    # 计算年径流
    for t in range(time_length):
        if t == 0:
            if pk_1 is None:
                pk_1 = np.mean(inputs[:, :, 0], axis=0) * 0.8 # 使用年均流量作为前一年流量
        else:
            pk_1 = inputs[t-1, :, 0]
        
        # 使用GR1A公式计算年径流
        streamflow_[t, :] = calculate_qk(
            inputs[t, :, 0],  # P
            pk_1,            # P_previous
            inputs[t, :, 1], # E
            x1
        )
    
    streamflow = np.expand_dims(streamflow_, axis=2)

    ets = inputs[:, :, 1]  # 使用潜在蒸发作为实际蒸发
    s = pk_1  # 使用前一年降水量作为状态变量
    r = streamflow_  # 使用径流作为汇流库状态
    return (streamflow, ets, s, r) if return_state else (streamflow, ets)
    
    
    