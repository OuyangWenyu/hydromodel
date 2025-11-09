"""
Author: zhuanglaihong
Date: 2025-02-21 14:54:24
LastEditTime: 2025-08-19 09:34:44
LastEditors: Wenyu Ouyang
Description:
FilePath: /hydromodel/hydromodel/models/gr6j.py
Copyright: Copyright (c) 2021-2024 zhuanglaihong. All rights reserved.
"""

import math
from typing import Optional, Tuple
import numpy as np
from numba import jit

from hydromodel.models.model_config import MODEL_PARAM_DICT
from hydromodel.models.unit_hydrograph import uh_conv
from hydromodel.models.param_utils import process_parameters


# @jit
@jit(nopython=True)
def calculate_precip_store(s, precip_net, x1):
    """Calculates the amount of rainfall which enters the storage reservoir."""
    n = x1 * (1.0 - (s / x1) ** 2) * np.tanh(precip_net / x1)
    d = 1.0 + (s / x1) * np.tanh(precip_net / x1)
    return n / d


# @jit
@jit(nopython=True)
def calculate_evap_store(s, evap_net, x1):
    """Determines the evaporation loss from the production store"""
    n = s * (2.0 - s / x1) * np.tanh(evap_net / x1)
    d = 1.0 + (1.0 - s / x1) * np.tanh(evap_net / x1)
    return n / d


# @jit
@jit(nopython=True)
def calculate_perc(current_store, x1):
    """Determines how much water percolates out of the production store to streamflow"""
    return current_store * (
        1.0 - (1.0 + (4.0 / 9.0 * current_store / x1) ** 4) ** -0.25
    )


def production(
    p_and_e: np.array, x1: np.array, s_level: Optional[np.array] = None
) -> Tuple[np.array, np.array]:
    """
    an one-step calculation for production store in GR6j
    the dimension of the cell: [batch, feature]
    Parameters
    ----------
    p_and_e
        P is pe[:, 0] and E is pe[:, 1]; similar with the "input" in the RNNCell
    x1:
        Storage reservoir parameter;
    s_level
        s_level means S in the GR6j Model; similar with the "hx" in the RNNCell
        Initial value of storage in the storage reservoir.
    Returns
    -------
    tuple
        contains the Pr and updated S
    """
    # Calculate net precipitation and evapotranspiration
    precip_difference = p_and_e[:, 0] - p_and_e[:, 1]
    precip_net = np.maximum(precip_difference, 0.0)
    evap_net = np.maximum(-precip_difference, 0.0)

    if s_level is None:
        s_level = 0.6 * x1

    # s_level should not be larger than x1
    s_level = np.clip(s_level, a_min=np.full(s_level.shape, 0.0), a_max=x1)

    # Calculate the fraction of net precipitation that is stored
    precip_store = calculate_precip_store(s_level, precip_net, x1)

    # Calculate the amount of evaporation from storage
    evap_store = calculate_evap_store(s_level, evap_net, x1)

    # Update the storage by adding effective precipitation and
    # removing evaporation
    s_update = s_level - evap_store + precip_store
    # s_level should not be larger than self.x1
    s_update = np.clip(s_update, a_min=np.full(s_update.shape, 0.0), a_max=x1)

    # Update the storage again to reflect percolation out of the store
    perc = calculate_perc(s_update, x1)
    s_update = s_update - perc
    # perc is always lower than S because of the calculation itself, so we don't need clamp here anymore.

    # The precip. for routing is the sum of the rainfall which
    # did not make it to storage and the percolation from the store
    current_runoff = perc + (precip_net - precip_store)
    # TODO: check if evap_store is the real ET
    return current_runoff, evap_store, s_update


# @jit
@jit(nopython=True)
def s_curves1(t, x4):
    """
    Unit hydrograph ordinates for UH1 derived from S-curves.
    """

    if t <= 0:
        return 0
    elif t < x4:
        return (t / x4) ** 2.5
    else:  # t >= x4
        return 1


# @jit
@jit(nopython=True)
def s_curves2(t, x4):
    """
    Unit hydrograph ordinates for UH2 derived from S-curves.
    """

    if t <= 0:
        return 0
    elif t < x4:
        return 0.5 * (t / x4) ** 2.5
    elif t < 2 * x4:
        return 1 - 0.5 * (2 - t / x4) ** 2.5
    else:  # t >= x4
        return 1


def uh_gr6j(x4):
    """
    Generate the convolution kernel for the convolution operation in routing module of GR6j

    Parameters
    ----------
    x4
        the dim of x4 is [batch]
    Returns
    -------
    list
        UH1s and UH2s for all basins
    """
    uh1_ordinates = []
    uh2_ordinates = []
    for i in range(len(x4)):
        n_uh1 = int(math.ceil(x4[i]))
        n_uh2 = int(math.ceil(2.0 * x4[i]))
        uh1_ordinate = np.zeros(n_uh1)
        uh2_ordinate = np.zeros(n_uh2)
        for t in range(1, n_uh1 + 1):
            uh1_ordinate[t - 1] = s_curves1(t, x4[i]) - s_curves1(t - 1, x4[i])

        for t in range(1, n_uh2 + 1):
            uh2_ordinate[t - 1] = s_curves2(t, x4[i]) - s_curves2(t - 1, x4[i])
        uh1_ordinates.append(uh1_ordinate)
        uh2_ordinates.append(uh2_ordinate)

    return uh1_ordinates, uh2_ordinates


def routing_store(
    q9: np.array,
    q1: np.array,
    x2,
    x3,
    x5,
    SC=0.4,
    r1: Optional[np.array] = None,
):
    """
    the GR6j routing-module unit cell for time-sequence loop
    Parameters
    ----------
    q9

    x2
        Catchment water exchange parameter
    x3
        Routing reservoir parameters
    r_level
        Beginning value of storage in the routing reservoir.
    Returns
    -------
    """
    if r1 is None:
        r1 = 0.7 * x3
    # r_level should not be larger than self.x3
    r1 = np.clip(r1, a_min=np.full(r1.shape, 0.0), a_max=x3)
    groundwater_ex = x2 * r1 / x3 - x2 * x5
    r1_updated = np.maximum(
        np.full(r1.shape, 0.0), r1 + q9 * (1 - SC) + groundwater_ex
    )

    qr1 = r1_updated * (1.0 - (1.0 + (r1_updated / x3) ** 4) ** -0.25)
    r1_updated = r1_updated - qr1

    qd = np.maximum(np.full(groundwater_ex.shape, 0.0), q1 + groundwater_ex)
    q = qr1 + qd
    return q, r1_updated


def exponential_store(
    q9: np.array, x3, x6, SC=0.4, r2: Optional[np.array] = None
):
    """
    the GR6j exponential store module unit cell for time-sequence loop
    Parameters
    ----------
    q9

    x6
        exponential reservoir parameters
    r_2
        Beginning value of storage in the exponential reservoir.
    Returns
    -------
    """
    if r2 is None:
        r2 = 0.3 * x3
    # r_level should not be larger than self.x3
    r2 = np.clip(r2, a_min=np.full(r2.shape, 0.0), a_max=x3)
    r2 = q9 * SC + r2
    qr2 = x6 * np.log(1 + np.exp(r2 / x6))
    r2_updated = r2 - qr2

    r2_updated = np.maximum(r2_updated, np.zeros_like(r2_updated))

    return qr2, r2_updated


def gr6j(
    p_and_e,
    parameters,
    warmup_length: int,
    return_state=False,
    normalized_params="auto",
    **kwargs,
):
    """
    run GR6j model

    Parameters
    ----------
    p_and_e: ndarray
        3-dim input -- [time, basin, variable]: precipitation and potential evaporation
    parameters
        2-dim variable -- [basin, parameter]:
        the parameters are x1, x2, x3, x4, x5, x6
    warmup_length
        length of warmup period
    return_state
        if True, return state values, mainly for warmup periods
    normalized_params : Union[bool, str], optional
        Parameter format specification:
        - "auto": Automatically detect parameter format (default)
        - True: Parameters are normalized (0-1 range), convert to original scale
        - False: Parameters are already in original scale, use as-is

    Returns
    -------
    Union[np.array, tuple]
        streamflow or (streamflow, states)
    """
    model_param_dict = kwargs.get("gr6j", None)
    if model_param_dict is None:
        model_param_dict = MODEL_PARAM_DICT["gr6j"]
    # params
    param_ranges = model_param_dict["param_range"]

    # Process parameters using unified parameter handling
    processed_params = process_parameters(
        parameters, param_ranges, normalized=normalized_params
    )

    # Extract individual parameters from processed array
    x1 = processed_params[:, 0]
    x2 = processed_params[:, 1]
    x3 = processed_params[:, 2]
    x4 = processed_params[:, 3]
    x5 = processed_params[:, 4]
    x6 = processed_params[:, 5]
    if warmup_length > 0:
        # set no_grad for warmup periods
        p_and_e_warmup = p_and_e[0:warmup_length, :, :]
        _, _, s0, r1, r2 = gr6j(
            p_and_e_warmup,
            parameters,
            warmup_length=0,
            return_state=True,
            **kwargs,
        )
    else:
        s0 = 0.5 * x1
        r1 = 0.5 * x3
        r2 = 0.3 * x3
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
    prs_x = np.expand_dims(prs, axis=2)
    conv_q9, conv_q1 = uh_gr6j(x4)
    q9 = np.full([inputs.shape[0], inputs.shape[1], 1], 0.0)
    q1 = np.full([inputs.shape[0], inputs.shape[1], 1], 0.0)
    for j in range(inputs.shape[1]):
        q9[:, j : j + 1, :] = uh_conv(
            prs_x[:, j : j + 1, :], conv_q9[j].reshape(-1, 1, 1)
        )
        q1[:, j : j + 1, :] = uh_conv(
            prs_x[:, j : j + 1, :], conv_q1[j].reshape(-1, 1, 1)
        )

    SC = 0.4  # 分配系数
    for i in range(inputs.shape[0]):
        q, r1 = routing_store(q9[i, :, 0], q1[i, :, 0], x2, x3, x5, SC, r1)
        qr2, r2 = exponential_store(q9[i, :, 0], x3, x6, SC, r2)

        streamflow_[i, :] = qr2 + q
    streamflow = np.expand_dims(streamflow_, axis=2)
    return (streamflow, ets, s, r1, r2) if return_state else (streamflow, ets)
