"""
Author: Wenyu Ouyang
Date: 2025-02-18 10:20:58
LastEditTime: 2025-08-19 09:34:29
LastEditors: Wenyu Ouyang
Description: Core code for GR4J model
FilePath: /hydromodel/hydromodel/models/gr4j.py
Copyright: Copyright (c) 2021-2024 zhuanglaihong. All rights reserved.
"""

import math
from typing import Optional, Tuple
import numpy as np
from numba import jit

from hydromodel.models.model_config import MODEL_PARAM_DICT
from hydromodel.models.param_utils import process_parameters
from hydromodel.models.unit_hydrograph import uh_conv


# ---------------------------------------------------------------------------
# Fast path: a fully JIT-compiled core for single-basin GR4J.
#
# The historical implementation paid two large overheads in the hot loop:
#   1. A Python for-loop over time stepped through production() and routing(),
#      both of which were plain-Python functions that re-entered numba on every
#      step to call calculate_precip_store / calculate_evap_store / calculate_perc.
#      With SCE-UA running 10^4+ evaluations of 10^3+ time steps, the JIT
#      boundary crossings dominated runtime even though the leaf math was JITted.
#   2. The top-level gr4j() recursively re-invoked itself to warm up, doubling
#      the work per evaluation.
#
# _gr4j_jit_core fuses production, UH convolution and routing into a single
# nopython function operating on scalar parameters and 1D prcp/pet arrays for
# one basin. The wrapper below calls it once for warmup (to obtain seeded
# storage states) and once for the main window — no Python-level recursion,
# no per-step Python dispatch, no np.clip/np.full allocations per step.
# ---------------------------------------------------------------------------
@jit(nopython=True, cache=True)
def _gr4j_jit_core(prcp, pet, x1, x2, x3, x4, s0, r0):
    """Single-basin GR4J: production + UH convolution + routing in one pass.

    Returns
    -------
    qsim : 1D array [n_time]
        Routed streamflow.
    ets : 1D array [n_time]
        Evaporation from production store (same definition as legacy code).
    s_final, r_final : float
        Production / routing storage at end of the window.
    """
    n = prcp.shape[0]

    # --- Unit hydrograph ordinates (computed from differences of S-curves) ---
    n_uh1 = int(math.ceil(x4))
    n_uh2 = int(math.ceil(2.0 * x4))
    # Guard against pathological tiny x4 (shouldn't happen with sane ranges).
    if n_uh1 < 1:
        n_uh1 = 1
    if n_uh2 < 1:
        n_uh2 = 1
    uh1 = np.empty(n_uh1)
    uh2 = np.empty(n_uh2)

    for t in range(1, n_uh1 + 1):
        if t < x4:
            v1 = (t / x4) ** 2.5
        else:
            v1 = 1.0
        tp = t - 1
        if tp <= 0:
            v0 = 0.0
        elif tp < x4:
            v0 = (tp / x4) ** 2.5
        else:
            v0 = 1.0
        uh1[t - 1] = v1 - v0

    two_x4 = 2.0 * x4
    for t in range(1, n_uh2 + 1):
        if t < x4:
            v1 = 0.5 * (t / x4) ** 2.5
        elif t < two_x4:
            v1 = 1.0 - 0.5 * (2.0 - t / x4) ** 2.5
        else:
            v1 = 1.0
        tp = t - 1
        if tp <= 0:
            v0 = 0.0
        elif tp < x4:
            v0 = 0.5 * (tp / x4) ** 2.5
        elif tp < two_x4:
            v0 = 1.0 - 0.5 * (2.0 - tp / x4) ** 2.5
        else:
            v0 = 1.0
        uh2[t - 1] = v1 - v0

    # --- Production loop ---
    prs = np.empty(n)
    ets = np.empty(n)
    s = s0

    for i in range(n):
        p = prcp[i]
        e = pet[i]
        diff = p - e
        if diff > 0.0:
            pn = diff
            en = 0.0
        else:
            pn = 0.0
            en = -diff

        # Clip s into [0, x1] at the start of the step (matches np.clip semantics).
        if s > x1:
            s = x1
        elif s < 0.0:
            s = 0.0

        if pn > 0.0:
            th = math.tanh(pn / x1)
            s_ratio = s / x1
            ps = x1 * (1.0 - s_ratio * s_ratio) * th / (1.0 + s_ratio * th)
        else:
            ps = 0.0

        if en > 0.0:
            th = math.tanh(en / x1)
            s_ratio = s / x1
            es = s * (2.0 - s_ratio) * th / (1.0 + (1.0 - s_ratio) * th)
        else:
            es = 0.0

        s_new = s - es + ps
        if s_new > x1:
            s_new = x1
        elif s_new < 0.0:
            s_new = 0.0

        ratio = 4.0 / 9.0 * s_new / x1
        perc = s_new * (1.0 - (1.0 + ratio * ratio * ratio * ratio) ** -0.25)
        s = s_new - perc

        prs[i] = perc + (pn - ps)
        ets[i] = es

    # --- UH convolution (running sum, equivalent to np.convolve truncated) ---
    q9 = np.empty(n)
    q1 = np.empty(n)
    for i in range(n):
        kmax = n_uh1 if i + 1 > n_uh1 else i + 1
        a9 = 0.0
        for k in range(kmax):
            a9 += uh1[k] * prs[i - k]
        q9[i] = a9

        kmax = n_uh2 if i + 1 > n_uh2 else i + 1
        a1 = 0.0
        for k in range(kmax):
            a1 += uh2[k] * prs[i - k]
        q1[i] = a1

    # --- Routing loop ---
    qsim = np.empty(n)
    r = r0
    for i in range(n):
        if r > x3:
            r = x3
        elif r < 0.0:
            r = 0.0

        gw_ex = x2 * (r / x3) ** 3.5
        r_upd = r + q9[i] + gw_ex
        if r_upd < 0.0:
            r_upd = 0.0

        ratio = r_upd / x3
        qr = r_upd * (1.0 - (1.0 + ratio * ratio * ratio * ratio) ** -0.25)
        r = r_upd - qr

        qd_pre = q1[i] + gw_ex
        qd = qd_pre if qd_pre > 0.0 else 0.0
        qsim[i] = qr + qd

    return qsim, ets, s, r


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


def routing(
    q9: np.array, q1: np.array, x2, x3, r_level: Optional[np.array] = None
):
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
    r_updated = np.maximum(
        np.full(r_level.shape, 0.0), r_level + q9 + groundwater_ex
    )

    qr = r_updated * (1.0 - (1.0 + (r_updated / x3) ** 4) ** -0.25)
    r_updated = r_updated - qr

    qd = np.maximum(np.full(groundwater_ex.shape, 0.0), q1 + groundwater_ex)
    q = qr + qd
    return q, r_updated


def gr4j(
    p_and_e,
    parameters,
    warmup_length: int,
    return_state=False,
    normalized_params="auto",
    **kwargs,
):
    """
    run GR4J model

    Parameters
    ----------
    p_and_e: ndarray
        3-dim input -- [time, basin, variable]: precipitation and potential evaporation
    parameters
        2-dim variable -- [basin, parameter]:
        the parameters are x1, x2, x3 and x4
    warmup_length
        length of warmup period
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
    model_param_dict = kwargs.get("gr4j", None)
    if model_param_dict is None:
        model_param_dict = MODEL_PARAM_DICT["gr4j"]
    param_ranges = model_param_dict["param_range"]

    processed_params = process_parameters(
        parameters, param_ranges, normalized=normalized_params
    )

    n_basins = processed_params.shape[0]
    n_main = p_and_e.shape[0] - warmup_length

    streamflow_ = np.empty((n_main, n_basins))
    ets_out = np.empty((n_main, n_basins))
    s_final = np.empty(n_basins)
    r_final = np.empty(n_basins)

    # Loop over basins in Python; each basin's hot path is the single fused
    # JIT call below. For single-basin SCE-UA (the common case) this loop
    # executes exactly once per evaluation.
    for j in range(n_basins):
        x1 = float(processed_params[j, 0])
        x2 = float(processed_params[j, 1])
        x3 = float(processed_params[j, 2])
        x4 = float(processed_params[j, 3])

        # Ensure contiguous float64 1D slices for numba (cheap if already so).
        prcp = np.ascontiguousarray(p_and_e[:, j, 0], dtype=np.float64)
        pet = np.ascontiguousarray(p_and_e[:, j, 1], dtype=np.float64)

        if warmup_length > 0:
            # Warm up by running the same core over the warmup window; we keep
            # only the final storage states. No Python-level recursion into
            # gr4j() — this used to double the work per evaluation.
            _, _, s_warm, r_warm = _gr4j_jit_core(
                prcp[:warmup_length],
                pet[:warmup_length],
                x1,
                x2,
                x3,
                x4,
                0.5 * x1,
                0.5 * x3,
            )
            s_init, r_init = s_warm, r_warm
        else:
            s_init, r_init = 0.5 * x1, 0.5 * x3

        qsim_j, ets_j, s_end, r_end = _gr4j_jit_core(
            prcp[warmup_length:],
            pet[warmup_length:],
            x1,
            x2,
            x3,
            x4,
            s_init,
            r_init,
        )
        streamflow_[:, j] = qsim_j
        ets_out[:, j] = ets_j
        s_final[j] = s_end
        r_final[j] = r_end

    streamflow = np.expand_dims(streamflow_, axis=2)
    return (
        (streamflow, ets_out, s_final, r_final)
        if return_state
        else (streamflow, ets_out)
    )
