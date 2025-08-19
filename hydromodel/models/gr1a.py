"""
Author: zhuanglaihong
Date: 2025-02-21 15:36:42
LastEditTime: 2025-08-19 09:32:16
LastEditors: Wenyu Ouyang
Description: Core code for GR1A model
FilePath: \hydromodel\hydromodel\models\gr1a.py
Copyright: Copyright (c) 2021-2024 zhuanglaihong. All rights reserved.
"""

import math
from typing import Optional, Tuple
import numpy as np
from numba import jit
from hydromodel.models.model_config import MODEL_PARAM_DICT
from hydromodel.models.param_utils import process_parameters


def calculate_qk(pk, pk_1, ek, x):

    denominator = 1 + ((0.7 * pk + 0.3 * pk_1) / (x * ek)) ** 2

    return pk * (1 - 1 / (denominator**0.5))


def gr1a(
    p_and_e,
    parameters,
    warmup_length: int,
    return_state=False,
    normalized_params="auto",
    **kwargs,
):
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
    normalized_params
        parameter format specification:
        - "auto": automatically detect if parameters are normalized (0-1) or original scale (default)
        - True: parameters are normalized (0-1 range), will be converted to original scale
        - False: parameters are already in original scale, use as-is

    Returns
    -------
    Union[np.array, tuple]
        streamflow or (streamflow, states)
    """
    model_param_dict = kwargs.get("gr1a", None)
    if model_param_dict is None:
        model_param_dict = MODEL_PARAM_DICT["gr1a"]

    param_ranges = model_param_dict["param_range"]

    # Process parameters using unified parameter handling
    processed_params = process_parameters(
        parameters, param_ranges, normalized=normalized_params
    )

    # Extract individual parameters from processed array
    x1 = processed_params[:, 0]

    if warmup_length > 0:

        p_and_e_warmup = p_and_e[0:warmup_length, :, :]
        _, _, pk_1, r = gr1a(
            p_and_e_warmup,
            parameters,
            warmup_length=0,
            return_state=True,
            **kwargs,
        )
    else:
        pk_1 = None

    inputs = p_and_e[warmup_length:, :, :]
    time_length, basin_num, _ = inputs.shape

    streamflow_ = np.zeros((time_length, basin_num))

    for t in range(time_length):
        if t == 0:
            if pk_1 is None:
                pk_1 = (
                    inputs[0, :, 0] * 0.8
                )  # 使用当年降水量的80%作为前一年降水量 TODO
        else:
            pk_1 = inputs[t - 1, :, 0]

        streamflow_[t, :] = calculate_qk(
            inputs[t, :, 0], pk_1, inputs[t, :, 1], x1
        )

    streamflow = np.expand_dims(streamflow_, axis=2)

    ets = inputs[:, :, 1]  # 使用潜在蒸发作为实际蒸发
    s = pk_1  # 使用前一年降水量作为产流库状态
    r = streamflow_  # 使用径流作为汇流库状态
    return (streamflow, ets, s, r) if return_state else (streamflow, ets)
