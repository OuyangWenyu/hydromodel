import math
from typing import Optional, Tuple

import numpy as np
from numba import jit
from hydromodel.models.model_config import MODEL_PARAM_DICT

from hydromodel.models.xaj import uh_conv


@jit
def calculate_precip_store(s, precip_net, x1):
    """Calculates the amount of rainfall which enters the storage reservoir."""
    n = x1 * (1.0 - (s / x1) ** 2) * np.tanh(precip_net / x1)
    d = 1.0 + (s / x1) * np.tanh(precip_net / x1)
    return n / d


@jit
def calculate_evap_store(s, evap_net, x1):
    """Determines the evaporation loss from the production store"""
    n = s * (2.0 - s / x1) * np.tanh(evap_net / x1)
    d = 1.0 + (1.0 - s / x1) * np.tanh(evap_net / x1)
    return n / d


@jit
def calculate_perc(current_store, x1):
    """Determines how much water percolates out of the production store to streamflow"""
    return current_store * (
        1.0 - (1.0 + (4.0 / 9.0 * current_store / x1) ** 4) ** -0.25
    )


def production(
    p_and_e: np.array, x1: np.array, s_level: Optional[np.array] = None
) -> Tuple[np.array, np.array]:
    """
    an one-step calculation for production store in GR4J
    the dimension of the cell: [batch, feature]
    Parameters
    ----------
    p_and_e
        P is pe[:, 0] and E is pe[:, 1]; similar with the "input" in the RNNCell
    x1:
        Storage reservoir parameter;
    s_level
        s_level means S in the GR4J Model; similar with the "hx" in the RNNCell
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
    return current_runoff, s_update


@jit
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


@jit
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


def uh_gr4j(x4):
    """
    Generate the convolution kernel for the convolution operation in routing module of GR4J

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


def routing(q9: np.array, q1: np.array, x2, x3, r_level: Optional[np.array] = None):
    """
    the GR4J routing-module unit cell for time-sequence loop
    Parameters
    ----------
    q9
    q1
    x2
        Catchment water exchange parameter
    x3
        Routing reservoir parameters
    r_level
        Beginning value of storage in the routing reservoir.
    Returns
    -------
    """
    if r_level is None:
        r_level = 0.7 * x3
    # r_level should not be larger than self.x3
    r_level = np.clip(r_level, a_min=np.full(r_level.shape, 0.0), a_max=x3)
    groundwater_ex = x2 * (r_level / x3) ** 3.5
    r_updated = np.maximum(np.full(r_level.shape, 0.0), r_level + q9 + groundwater_ex)

    qr = r_updated * (1.0 - (1.0 + (r_updated / x3) ** 4) ** -0.25)
    r_updated = r_updated - qr

    qd = np.maximum(np.full(groundwater_ex.shape, 0.0), q1 + groundwater_ex)
    q = qr + qd
    return q, r_updated


def gr4j(p_and_e, parameters, warmup_length: int, return_state=False, **kwargs):
    """
    run GR4J model

    Parameters
    ----------
    p_and_e

    parameters
        2-dim variable -- [basin, parameter]:
        the parameters are x1, x2, x3 and x4
    warmup_length
        length of warmup period
    return_state
        if True, return state values, mainly for warmup periods

    Returns
    -------
    Union[np.array, tuple]
        streamflow or (streamflow, states)
    """
    x1_scale = MODEL_PARAM_DICT["gr4j"]["param_range"]["x1"]
    x2_sacle = MODEL_PARAM_DICT["gr4j"]["param_range"]["x2"]
    x3_scale = MODEL_PARAM_DICT["gr4j"]["param_range"]["x3"]
    x4_scale = MODEL_PARAM_DICT["gr4j"]["param_range"]["x4"]
    x1 = x1_scale[0] + parameters[:, 0] * (x1_scale[1] - x1_scale[0])
    x2 = x2_sacle[0] + parameters[:, 1] * (x2_sacle[1] - x2_sacle[0])
    x3 = x3_scale[0] + parameters[:, 2] * (x3_scale[1] - x3_scale[0])
    x4 = x4_scale[0] + parameters[:, 3] * (x4_scale[1] - x4_scale[0])

    if warmup_length > 0:
        # set no_grad for warmup periods
        p_and_e_warmup = p_and_e[0:warmup_length, :, :]
        _, s0, r0 = gr4j(
            p_and_e_warmup, parameters, warmup_length=0, return_state=True, **kwargs
        )
    else:
        s0 = 0.5 * x1
        r0 = 0.5 * x3
    inputs = p_and_e[warmup_length:, :, :]
    streamflow_ = np.full(inputs.shape[:2], 0.0)
    prs = np.full(inputs.shape[:2], 0.0)
    for i in range(inputs.shape[0]):
        if i == 0:
            pr, s = production(inputs[i, :, :], x1, s0)
        else:
            pr, s = production(inputs[i, :, :], x1, s)
        prs[i, :] = pr
    prs_x = np.expand_dims(prs, axis=2)
    conv_q9, conv_q1 = uh_gr4j(x4)
    q9 = np.full([inputs.shape[0], inputs.shape[1], 1], 0.0)
    q1 = np.full([inputs.shape[0], inputs.shape[1], 1], 0.0)
    for j in range(inputs.shape[1]):
        q9[:, j : j + 1, :] = uh_conv(
            prs_x[:, j : j + 1, :], conv_q9[j].reshape(-1, 1, 1)
        )
        q1[:, j : j + 1, :] = uh_conv(
            prs_x[:, j : j + 1, :], conv_q1[j].reshape(-1, 1, 1)
        )
    for i in range(inputs.shape[0]):
        if i == 0:
            q, r = routing(q9[i, :, 0], q1[i, :, 0], x2, x3, r0)
        else:
            q, r = routing(q9[i, :, 0], q1[i, :, 0], x2, x3, r)
        streamflow_[i, :] = q
    streamflow = np.expand_dims(streamflow_, axis=2)
    if return_state:
        return streamflow, s, r
    return streamflow
