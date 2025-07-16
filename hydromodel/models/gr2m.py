"""
Author: zhuanglaihong
Date: 2025-02-21 15:37:10
LastEditTime: 2025-07-08 19:03:43
LastEditors: Wenyu Ouyang
Description: Core code for GR2M model
FilePath: \hydromodel\hydromodel\models\gr2m.py
Copyright: Copyright (c) 2021-2024 zhuanglaihong. All rights reserved.
"""

import math
from typing import Optional, Tuple
import numpy as np
from numba import jit

from hydromodel.models.model_config import MODEL_PARAM_DICT
from hydromodel.models.unit_hydrograph import uh_conv


def production(inputs, x1, s0):
    """
    Calculates the production component of the GR2M model.

    Parameters
    ----------
    inputs : ndarray
        2D input array - [basin, variable]: Precipitation and potential evaporation.
    x1 : ndarray
        1D array - [basin]: Production store capacity.
    s0 : ndarray
        1D array - [basin]: Initial production store state.

    Returns
    -------
    tuple
        (pr, et, s): Production, actual evapotranspiration, and updated production store state.
    """
    p = inputs[:, 0]  # 降水
    e = inputs[:, 1]  # 潜在蒸发

    phi = np.tanh(p / x1)
    s1 = (s0 + x1 * phi) / (1 + phi * s0 / x1)
    p1 = p + s0 - s1
    psi = np.tanh(e / x1)
    s2 = s1 * (1 - psi) / (1 + psi * (1 - s1 / x1))
    et = s1 - s2
    s = s2 / np.power(1 + np.power(s2 / x1, 3), 1 / 3)
    p2 = s2 - s
    p3 = p1 + p2

    return p3, et, s


def routing(p3, x2, r0):
    """
    Calculates the routing component of the GR2M model.

    Parameters
    ----------
    p3 : ndarray
        1D array - [basin]: Production.
    x2 : ndarray
        1D array - [basin]: Routing store coefficient.
    r0 : ndarray
        1D array - [basin]: Initial routing store state.

    Returns
    -------
    tuple
        (q, r): Streamflow and updated routing store state.
    """
    r1 = r0 + p3
    r2 = x2 * r1
    q = np.power(r2, 2) / (r2 + 60)
    r = r2 - q

    return q, r


def gr2m(p_and_e, parameters, warmup_length: int, return_state=False, **kwargs):
    """
    run GR2m model

    Parameters
    ----------
    p_and_e : ndarray
        3D input array - [time, basin, variable]: Time series of precipitation and potential evaporation.
    parameters : ndarray
        2D parameter array - [basin, parameter]: Model parameters (x1, x2).
    warmup_length : int
        Length of the warmup period (in months).
    return_state : bool, optional
        Whether to return state variables, by default False.

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
    x2_scale = param_ranges["x2"]
    x1 = x1_scale[0] + parameters[:, 0] * (x1_scale[1] - x1_scale[0])
    x2 = x2_scale[0] + parameters[:, 1] * (x2_scale[1] - x2_scale[0])

    if warmup_length > 0:
        p_and_e_warmup = p_and_e[0:warmup_length, :, :]
        _, _, s0, r0 = gr2m(
            p_and_e_warmup, parameters, warmup_length=0, return_state=True, **kwargs
        )
    else:
        s0 = 0.5 * x1
        r0 = np.zeros_like(x1)

    inputs = p_and_e[warmup_length:, :, :]
    streamflow_ = np.full(inputs.shape[:2], 0.0)
    prs = np.full(inputs.shape[:2], 0.0)
    ets = np.full(inputs.shape[:2], 0.0)

    for i in range(inputs.shape[0]):
        if i == 0:
            pr, et, s = production(inputs[i, :, :], x1, s0)
        else:
            pr, et, s = production(inputs[i, :, :], x1, s)

        prs[i, :] = pr
        ets[i, :] = et

        q, r = routing(pr, x2, r0) if i == 0 else routing(pr, x2, r)
        streamflow_[i, :] = q

    streamflow = np.expand_dims(streamflow_, axis=2)
    return (streamflow, ets, s, r) if return_state else (streamflow, ets)
