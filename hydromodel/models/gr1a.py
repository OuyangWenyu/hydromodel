'''
Author: zhuanglaihong
Date: 2025-02-21 15:36:42
LastEditTime: 2025-03-04 22:49:24
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

@jit(nopython=True)
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
        3-dim input -- [time, basin, variable]: precipitation and potential evaporation
    parameters
        2-dim variable -- [basin, parameter]:
        the parameters is x
    warmup_length
        length of warmup period
    return_state
        if True, return state values, mainly for warmup periods

    Returns
    -------
    Union[np.array, tuple]
        streamflow or (streamflow, states)
    """
    days_per_year = kwargs.get("days_per_year", 365)
    if warmup_length < days_per_year:
        raise ValueError(f"GR1A模型需要至少一年({days_per_year}天)的预热期，当前预热期为{warmup_length}天")
    
    model_param_dict = kwargs.get("gr1a", None)
    if model_param_dict is None:
        model_param_dict = MODEL_PARAM_DICT["gr1a"]
    # params
    param_ranges = model_param_dict["param_range"]
    x_scale = param_ranges["x"]
    x = x_scale[0] + parameters[:, 0] * (x_scale[1] - x_scale[0])

    # 处理预热期数据
    p_and_e_warmup = p_and_e[0:warmup_length, :, :]
    warmup_time_length, basin_num, _ = p_and_e_warmup.shape
    warmup_year_num = warmup_time_length // days_per_year
    
    # 计算预热期的年度数据
    warmup_annual_p = np.zeros((warmup_year_num, basin_num))
    warmup_annual_e = np.zeros((warmup_year_num, basin_num))
    
    # 聚合预热期的年数据
    for y in range(warmup_year_num):
        start_idx = y * days_per_year
        end_idx = (y + 1) * days_per_year
        warmup_annual_p[y, :] = np.sum(p_and_e_warmup[start_idx:end_idx, :, 0], axis=0)
        warmup_annual_e[y, :] = np.sum(p_and_e_warmup[start_idx:end_idx, :, 1], axis=0)
    
    # 获取最后一年的数据作为初始状态
    pk_1 = warmup_annual_p[-1, :]
    
    # 获取输入数据
    inputs = p_and_e[warmup_length:, :, :]
    time_length, basin_num, _ = inputs.shape
    
    # 假设日数据，计算每年的天数
    days_per_year = kwargs.get("days_per_year", 365)
    year_num = time_length // days_per_year
    
    # 初始化年尺度数据存储
    annual_p = np.zeros((year_num, basin_num))
    annual_e = np.zeros((year_num, basin_num))
    annual_q = np.zeros((year_num, basin_num))
    
    # 将日数据聚合为年数据
    for y in range(year_num):
        start_idx = y * days_per_year
        end_idx = (y + 1) * days_per_year
        
        # 计算年降水量和年蒸发量
        annual_p[y, :] = np.sum(inputs[start_idx:end_idx, :, 0], axis=0)
        annual_e[y, :] = np.sum(inputs[start_idx:end_idx, :, 1], axis=0)
    
    # 计算年径流
    for y in range(year_num):
        if y == 0:
            # 第一年，使用初始值或估计值作为上一年的降水量
            pk_1 = annual_p[0, :] * 0.8  # 假设上一年降水为当年的80%，可根据实际情况调整
        else:
            pk_1 = annual_p[y-1, :]
        
        # 使用GR1A公式计算年径流
        annual_q[y, :] = calculate_qk(annual_p[y, :], pk_1, annual_e[y, :], x)
    
    # 将年径流转换为日尺度输出（均匀分配到每一天）
    streamflow_ = np.zeros((time_length, basin_num))
    for y in range(year_num):
        start_idx = y * days_per_year
        end_idx = min((y + 1) * days_per_year, time_length)
        daily_q = annual_q[y, :] / days_per_year
        streamflow_[start_idx:end_idx, :] = np.tile(daily_q, (end_idx - start_idx, 1))
    
    # 处理可能的剩余天数
    if time_length > year_num * days_per_year:
        remaining_days = time_length - year_num * days_per_year
        # 使用最后一年的日均值填充
        streamflow_[year_num * days_per_year:, :] = np.tile(annual_q[-1, :] / days_per_year, (remaining_days, 1))
    
    streamflow = np.expand_dims(streamflow_, axis=2)
    
    if return_state:
        # 返回年径流数据和参数x
        return streamflow, annual_p, annual_e, annual_q, pk_1, x
    
    # 返回日尺度径流和参数x
    return streamflow, x
    
    