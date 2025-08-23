"""
Author: Wenyu Ouyang
Date: 2025-07-30 16:44:15
LastEditTime: 2025-08-23 10:40:21
LastEditors: Wenyu Ouyang
Description: Dahuofang Model - Python implementation based on Java version
FilePath: \flooddataaugmentationd:\Code\hydromodel\hydromodel\models\dhf.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union
import traceback
from numba import jit

from hydromodel.models.model_config import MODEL_PARAM_DICT
from hydromodel.models.param_utils import process_parameters


@jit(nopython=True)
def calculate_dhf_evapotranspiration(
    precipitation: np.ndarray,
    potential_evapotranspiration: np.ndarray,
    kc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate evapotranspiration and net precipitation for DHF model

    TODO: We writes some atomic functions for DHF model, but it is not used yet.

    """
    edt = kc * potential_evapotranspiration
    pe = precipitation - edt  # net precipitation
    return pe, edt


@jit(nopython=True)
def calculate_dhf_surface_runoff(
    pe: np.ndarray,
    sa: np.ndarray,
    s0: np.ndarray,
    a: np.ndarray,
    g: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate surface runoff and impervious area runoff"""
    y0 = g * pe  # impervious area runoff
    pc = pe - y0  # net infiltration

    # Calculate surface water storage contribution (vectorized)
    temp = np.where(sa > 0, (1 - sa / s0) ** (1 / a), 0.0)
    sm = a * s0 * (1 - temp)

    # Calculate runoff from surface storage
    rr = np.where(
        pc > 0.0,
        np.where(
            sm + pc < a * s0,
            pc + sa - s0 + s0 * (1 - (sm + pc) / (a * s0)) ** a,
            pc - (s0 - sa),
        ),
        0.0,
    )

    return y0, pc, rr


@jit(nopython=True)
def calculate_dhf_subsurface_flow(
    rr: np.ndarray,
    ua: np.ndarray,
    u0: np.ndarray,
    d0: np.ndarray,
    b: np.ndarray,
    k2: np.ndarray,
    kw: np.ndarray,
    time_interval: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate subsurface flow components"""
    # Calculate subsurface flow parameters
    temp = np.where(ua > 0, (1 - ua / u0) ** (1 / b), 0.0)
    un = b * u0 * (1 - temp)
    temp = np.where(ua > 0, (1 - ua / u0) ** (u0 / (b * d0)), 0.0)
    dn = b * d0 * (1 - temp)

    z1 = 1 - np.exp(-k2 * time_interval * u0 / d0)
    z2 = 1 - np.exp(-k2 * time_interval)

    # Calculate total flow
    y = np.where(
        rr + z2 * un < z2 * b * u0,
        rr
        + z2 * (ua - u0)
        + z2 * u0 * (1 - (z2 * un + rr) / (z2 * b * u0)) ** b,
        rr + z2 * (ua - u0),
    )

    # Calculate interflow
    temp = np.where(ua > 0, (1 - ua / u0) ** (u0 / d0), 0.0)
    yu = np.where(
        z1 * dn + rr < z1 * b * d0,
        rr
        - z1 * d0 * temp
        + z1 * d0 * (1 - (z1 * dn + rr) / (z1 * b * d0)) ** b,
        rr - z1 * d0 * temp,
    )

    # Calculate groundwater runoff
    yl = (y - yu) * kw

    return y, yu, yl


@jit(nopython=True)
def calculate_dhf_storage_update(
    sa: np.ndarray,
    ua: np.ndarray,
    pc: np.ndarray,
    rr: np.ndarray,
    y: np.ndarray,
    s0: np.ndarray,
    u0: np.ndarray,
    a: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Update water storage states"""
    # Calculate surface water storage parameters
    temp = np.where(sa > 0, (1 - sa / s0) ** (1 / a), 0.0)
    sm = a * s0 * (1 - temp)

    # Update surface storage
    sa_new = np.where(
        pc > 0.0,
        np.where(
            sm + pc < a * s0,
            s0 * (1 - (1 - (sm + pc) / (a * s0)) ** a),
            sa + pc - rr,
        ),
        sa,
    )
    sa_new = np.clip(sa_new, 0.0, s0)

    # Update subsurface storage
    ua_new = ua + rr - y
    ua_new = np.clip(ua_new, 0.0, u0)

    return sa_new, ua_new


@jit(nopython=True)
def calculate_dhf_evaporation_deficit(
    precipitation: np.ndarray,
    edt: np.ndarray,
    sa: np.ndarray,
    ua: np.ndarray,
    s0: np.ndarray,
    u0: np.ndarray,
    a: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate evaporation when precipitation is insufficient"""
    ec = edt - precipitation
    eb = ec  # accumulated deficit

    # Calculate surface evaporation
    temp1 = (1 - (eb - ec) / (a * s0)) ** a
    temp2 = (1 - eb / (a * s0)) ** a

    eu = np.where(
        (eb / (a * s0) <= 0.999999) & ((eb - ec) / (a * s0) <= 0.999999),
        s0 * (temp1 - temp2),
        np.where(
            (eb / (a * s0) >= 1.00001) & ((eb - ec) / (a * s0) <= 0.999999),
            s0 * temp1,
            0.00001,
        ),
    )

    # Update storages after evaporation
    el = np.where(sa - eu < 0.0, (ec - sa) * ua / u0, (ec - eu) * ua / u0)
    sa_new = np.where(sa - eu < 0.0, 0.0, sa - eu)
    ua_new = ua - el
    ua_new = np.maximum(ua_new, 0.0)

    return sa_new, ua_new


@jit(nopython=True)
def calculate_dhf_routing_params(
    ya: np.ndarray,
    runoff_sim: np.ndarray,
    l: np.ndarray,
    b0: np.ndarray,
    k0: np.ndarray,
    n: np.ndarray,
    coe: np.ndarray,
    dd: np.ndarray,
    cc: np.ndarray,
    ddl: np.ndarray,
    ccl: np.ndarray,
    time_interval: float,
    pai: float,
):
    """Calculate routing parameters for DHF model"""
    # Ensure ya >= 0.5 for stability
    ya = np.maximum(ya, 0.5)

    # Calculate timing parameters
    temp_tm = (ya + runoff_sim) ** (-k0)
    lb = l / b0
    tm = lb * temp_tm

    tt = (n * tm).astype(np.int32)
    ts = (coe * tm).astype(np.int32)

    # Surface flow routing coefficient
    w0 = 1.0 / time_interval

    # Calculate surface routing coefficient K3
    k3 = np.zeros_like(tm)
    aa = np.zeros_like(tm)

    mask = tm > 0
    if np.any(mask):
        temp_aa = (pai * coe[mask]) ** (dd[mask] - 1)
        aa[mask] = cc[mask] / (dd[mask] * temp_aa * np.tan(pai * coe[mask]))

        for j in range(int(np.max(tm[mask])) + 1):
            j_mask = mask & (j < tm)
            if np.any(j_mask):
                temp = (pai * j / tm[j_mask]) ** dd[j_mask]
                temp1 = (np.sin(pai * j / tm[j_mask])) ** cc[j_mask]
                k3[j_mask] += np.exp(-aa[j_mask] * temp) * temp1

        k3[mask] = tm[mask] * w0 / k3[mask]

    # Calculate subsurface routing coefficient K3L
    k3l = np.zeros_like(tm)
    aal = np.zeros_like(tm)

    tt_mask = tt > 0
    if np.any(tt_mask):
        temp_aal = (pai * coe[tt_mask] / n[tt_mask]) ** (ddl[tt_mask] - 1)
        aal[tt_mask] = ccl[tt_mask] / (
            ddl[tt_mask] * temp_aal * np.tan(pai * coe[tt_mask] / n[tt_mask])
        )

        for j in range(int(np.max(tt[tt_mask])) + 1):
            j_mask = tt_mask & (j < tt)
            if np.any(j_mask):
                temp = (pai * j / tt[j_mask]) ** ddl[j_mask]
                temp1 = (np.sin(pai * j / tt[j_mask])) ** ccl[j_mask]
                k3l[j_mask] += np.exp(-aal[j_mask] * temp) * temp1

        k3l[tt_mask] = tt[tt_mask] * w0 / k3l[tt_mask]

    return tm, k3, k3l, aa, aal, tt, ts


@jit(nopython=True)
def calculate_dhf_routing(
    runoff_sim: np.ndarray,
    rl: np.ndarray,
    tm: np.ndarray,
    k3: np.ndarray,
    k3l: np.ndarray,
    aa: np.ndarray,
    aal: np.ndarray,
    tt: np.ndarray,
    ts: np.ndarray,
    dd: np.ndarray,
    cc: np.ndarray,
    ddl: np.ndarray,
    ccl: np.ndarray,
    time_steps: int,
    pai: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate routing for DHF model"""
    qs = np.zeros_like(runoff_sim)
    ql = np.zeros_like(runoff_sim)

    for i in range(time_steps):
        tl = tt[i] + ts[i] - 1
        tl = max(tl, 0)

        for j in range(int(tl) + 1):
            if i + j >= time_steps:
                break

            # Surface routing
            if tm[i] > 0:
                temp0 = pai * j / tm[i]
                temp1 = temp0 ** dd[i]
                temp2 = np.exp(-aa[i] * temp1)
                temp3 = (np.sin(temp0)) ** cc[i]
                qs_contrib = (
                    (runoff_sim[i] - rl[i]) * k3[i] / tm[i] * temp2 * temp3
                )
            else:
                qs_contrib = 0.0

            # Subsurface routing
            if tt[i] > 0 and j >= ts[i]:
                temp00 = pai * (j - ts[i]) / tt[i]
                temp10 = temp00 ** ddl[i]
                temp20 = np.exp(-aal[i] * temp10)
                temp30 = (np.sin(temp00)) ** ccl[i]
                ql_contrib = rl[i] * k3l[i] / tt[i] * temp20 * temp30
            else:
                ql_contrib = 0.0

            # Add contributions based on timing conditions
            if j <= tm[i]:
                if j <= ts[i]:
                    qs[i + j] += qs_contrib
                else:
                    qs[i + j] += qs_contrib
                    ql[i + j] += ql_contrib
            else:
                ql[i + j] += ql_contrib

    return qs, ql


def dhf(
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
    Vectorized DHF (Dahuofang) hydrological model - fully parallelized version

    This function implements the DHF model with full NumPy vectorization,
    processing all basins simultaneously using [seq, basin, feature] tensor operations.

    Parameters
    ----------
    p_and_e : np.ndarray
        precipitation and potential evapotranspiration, 3-dim: [time, basin, feature=2]
        where feature=0 is precipitation, feature=1 is potential evapotranspiration
    parameters : np.ndarray
        model parameters, 2-dim: [basin, parameter]
        Parameters: [S0, U0, D0, KC, KW, K2, KA, G, A, B, B0, K0, N, DD, CC, COE, DDL, CCL]
    warmup_length : int, optional
        the length of warmup period (default: 365)
    return_state : bool, optional
        if True, return internal state variables (default: False)
    normalized_params : Union[bool, str], optional
        parameter format specification:
        - "auto": automatically detect parameter format (default)
        - True: parameters are normalized (0-1 range), convert to original scale
        - False: parameters are already in original scale, use as-is
    **kwargs
        Additional keyword arguments, including
        - time_interval_hours (default: 3.0)
        - main_river_length (default: None) means length of the main channel (km), for example, dahuofang's is 155.763
        - basin_area (default: None) means basin area (km^2), for example, dahuofang's is 5482.0

    Returns
    -------
    result : np.ndarray or tuple
        if return_state is False: QSim array [time, basin, 1]
        if return_state is True: tuple of (QSim, runoffSim, y0, yu, yl, y, sa, ua, ya)
    """

    # Get data dimensions
    time_steps, num_basins, _ = p_and_e.shape
    time_interval = kwargs.get("time_interval_hours", 3.0)
    pai = np.pi
    l = kwargs.get("main_river_length", None)  # km
    f = kwargs.get("basin_area", None)  # km^2

    if l is None or f is None:
        raise ValueError("l and f must be provided")

    # Process parameters using unified parameter handling
    processed_parameters = parameters.copy()
    if normalized_params != False:
        model_param_dict = MODEL_PARAM_DICT.get("dhf")
        if model_param_dict is not None:
            param_ranges = model_param_dict["param_range"]
            processed_parameters = process_parameters(
                parameters, param_ranges, normalized=normalized_params
            )

    # Extract parameters - all are [basin] arrays
    s0 = processed_parameters[:, 0]  # Surface storage capacity
    u0 = processed_parameters[:, 1]  # Subsurface storage capacity
    d0 = processed_parameters[:, 2]  # Deep storage capacity
    kc = processed_parameters[:, 3]  # Evaporation coefficient
    kw = processed_parameters[:, 4]  # Subsurface flow coefficient
    k2 = processed_parameters[:, 5]  # Percolation coefficient
    ka = processed_parameters[:, 6]  # Total runoff adjustment coefficient
    g = processed_parameters[:, 7]  # Impervious area ratio
    a = processed_parameters[:, 8]  # Surface storage exponent
    b = processed_parameters[:, 9]  # Subsurface storage exponent
    b0 = processed_parameters[:, 10]  # Routing parameter
    k0 = processed_parameters[:, 11]  # Routing parameter
    n = processed_parameters[:, 12]  # Routing parameter
    dd = processed_parameters[:, 13]  # Surface routing parameter
    cc = processed_parameters[:, 14]  # Surface routing parameter
    coe = processed_parameters[:, 15]  # Routing parameter
    ddl = processed_parameters[:, 16]  # Subsurface routing parameter
    ccl = processed_parameters[:, 17]  # Subsurface routing parameter

    # Handle warmup period
    if warmup_length > 0:
        p_and_e_warmup = p_and_e[0:warmup_length, :, :]
        *_, sa, ua, ya = dhf(
            p_and_e_warmup,
            parameters,
            warmup_length=0,
            return_state=True,
            normalized_params=False,  # Already processed
            **kwargs,
        )
        sa0 = sa[-1, :, 0].copy()
        ua0 = ua[-1, :, 0].copy()
        ya0 = ya[-1, :, 0].copy()
    else:
        # Default initial states
        sa0 = np.zeros(s0.shape)
        ua0 = np.zeros(u0.shape)
        # just use d0's shape, ya0 is not d0, it is Pa, while d0 is the deep storage capacity
        ya0 = np.full(d0.shape, 0.5)

    inputs = p_and_e[warmup_length:, :, :]
    # Get actual time steps after warmup
    actual_time_steps = inputs.shape[0]

    # Initialize state and output arrays - [time, basin]
    sa = np.zeros((actual_time_steps, num_basins))
    ua = np.zeros((actual_time_steps, num_basins))
    ya = np.zeros((actual_time_steps, num_basins))
    # to store the accumulated deficit
    ebs = np.zeros((actual_time_steps, num_basins))

    # Initialize output arrays
    runoff_sim = np.zeros((actual_time_steps, num_basins))
    q_sim = np.zeros((actual_time_steps, num_basins))
    y0_out = np.zeros((actual_time_steps, num_basins))
    yu_out = np.zeros((actual_time_steps, num_basins))
    yl_out = np.zeros((actual_time_steps, num_basins))
    y_out = np.zeros((actual_time_steps, num_basins))

    # Main time loop - DHF generation (runoff production)
    for i in range(actual_time_steps):
        # Current precipitation and PET for all basins
        prcp = inputs[i, :, 0]
        pet = inputs[i, :, 1]
        if i == 0:
            eb = np.zeros(kc.shape)
        else:
            sa0 = sa[i - 1, :]
            ua0 = ua[i - 1, :]
            # NOTE: Because Chu version init eb as 0 at every time step, we keep the same now
            # if we want to make it same as the book, we need to value ebs after a time step's calculation
            eb = ebs[i - 1, :]

        # Limit current states
        sa0 = np.minimum(sa0, s0)
        ua0 = np.minimum(ua0, u0)

        # Calculate evapotranspiration and net precipitation (vectorized)
        edt = kc * pet
        pe = prcp - edt
        # Surface runoff calculation (vectorized)
        y0 = g * pe  # impervious area runoff
        pc = pe - y0  # net infiltration
        # Process based on whether we have net precipitation or evaporation
        # Actually, we should use pe > 0.0, but we use pc > 0.0 to make it same as Chu's version
        # as g<1, hence pe>0 means pc>0, so it is fine.
        net_precip_mask = pc > 0.0

        # For basins with net precipitation (pe > 0) - vectorized operations
        if np.any(net_precip_mask):
            # Apply mask to get relevant basin data
            sa_pos = sa0[net_precip_mask]
            ua_pos = ua0[net_precip_mask]
            s0_pos = s0[net_precip_mask]
            u0_pos = u0[net_precip_mask]
            d0_pos = d0[net_precip_mask]
            a_pos = a[net_precip_mask]
            b_pos = b[net_precip_mask]
            k2_pos = k2[net_precip_mask]
            kw_pos = kw[net_precip_mask]

            # Surface water storage calculation
            temp = (1 - sa_pos / s0_pos) ** (1 / a_pos)
            sm = a_pos * s0_pos * (1 - temp)

            # Calculate surface runoff
            rr = np.where(
                pc > 0.0,
                np.where(
                    sm + pc < a_pos * s0_pos,
                    pc
                    + sa_pos
                    - s0_pos
                    + s0_pos * (1 - (sm + pc) / (a_pos * s0_pos)) ** a_pos,
                    pc - (s0_pos - sa_pos),
                ),
                0.0,
            )

            # Subsurface flow calculation (vectorized)
            temp = (1 - ua_pos / u0_pos) ** (1 / b_pos)
            un = b_pos * u0_pos * (1 - temp)
            temp = (1 - ua_pos / u0_pos) ** (u0_pos / (b_pos * d0_pos))
            dn = b_pos * d0_pos * (1 - temp)

            z1 = 1 - np.exp(-k2_pos * time_interval * u0_pos / d0_pos)
            z2 = 1 - np.exp(-k2_pos * time_interval)

            # Calculate total flow
            y = np.where(
                rr + z2 * un < z2 * b_pos * u0_pos,
                rr
                + z2 * (ua_pos - u0_pos)
                + z2
                * u0_pos
                * (1 - (z2 * un + rr) / (z2 * b_pos * u0_pos)) ** b_pos,
                rr + z2 * (ua_pos - u0_pos),
            )

            # Calculate interflow
            temp = (1 - ua_pos / u0_pos) ** (u0_pos / d0_pos)
            yu = np.where(
                z1 * dn + rr < z1 * b_pos * d0_pos,
                rr
                - z1 * d0_pos * temp
                + z1
                * d0_pos
                * (1 - (z1 * dn + rr) / (z1 * b_pos * d0_pos)) ** b_pos,
                rr - z1 * d0_pos * temp,
            )

            # Calculate groundwater runoff
            yl = (y - yu) * kw_pos

            # Update storage states (vectorized)
            sa_new = np.where(
                pc > 0.0,
                np.where(
                    sm + pc < a_pos * s0_pos,
                    s0_pos * (1 - (1 - (sm + pc) / (a_pos * s0_pos)) ** a_pos),
                    sa_pos + pc - rr,
                ),
                sa_pos,
            )
            sa_new = np.clip(sa_new, 0.0, s0_pos)

            ua_new = ua_pos + rr - y
            ua_new = np.clip(ua_new, 0.0, u0_pos)
            # eb will be set to 0 when pc > 0
            eb = np.where(pc > 0.0, 0.0, eb)

            # Store results for basins with net precipitation
            y0_out[i, net_precip_mask] = y0
            yu_out[i, net_precip_mask] = yu
            yl_out[i, net_precip_mask] = yl
            y_out[i, net_precip_mask] = y
            sa[i, net_precip_mask] = sa_new
            ua[i, net_precip_mask] = ua_new

        # For basins with evaporation deficit (pe <= 0) - vectorized operations
        evap_mask = ~net_precip_mask
        if np.any(evap_mask):
            prcp_neg = prcp[evap_mask]
            edt_neg = edt[evap_mask]
            sa_neg = sa0[evap_mask]
            ua_neg = ua0[evap_mask]
            s0_neg = s0[evap_mask]
            u0_neg = u0[evap_mask]
            a_neg = a[evap_mask]

            ec = edt_neg - prcp_neg
            eb = eb + ec  # accumulated deficit

            # Calculate surface evaporation (vectorized)
            temp1 = (1 - (eb - ec) / (a_neg * s0_neg)) ** a_neg
            temp2 = (1 - eb / (a_neg * s0_neg)) ** a_neg

            eu = np.where(
                (eb / (a_neg * s0_neg) <= 0.999999)
                & ((eb - ec) / (a_neg * s0_neg) <= 0.999999),
                s0_neg * (temp1 - temp2),
                np.where(
                    (eb / (a_neg * s0_neg) >= 1.00001)
                    & ((eb - ec) / (a_neg * s0_neg) <= 0.999999),
                    s0_neg * temp1,
                    0.00001,
                ),
            )

            # Update storages after evaporation
            el = np.where(
                sa_neg - eu < 0.0,
                (ec - sa_neg) * ua_neg / u0_neg,
                (ec - eu) * ua_neg / u0_neg,
            )
            sa_new = np.where(sa_neg - eu < 0.0, 0.0, sa_neg - eu)
            ua_new = ua_neg - el
            ua_new = np.maximum(ua_new, 0.0)

            # Set runoff components to zero for evaporation basins
            y0_out[i, evap_mask] = 0.0
            yu_out[i, evap_mask] = 0.0
            yl_out[i, evap_mask] = 0.0
            y_out[i, evap_mask] = 0.0
            sa[i, evap_mask] = sa_new
            ua[i, evap_mask] = ua_new

        # Ensure states are within bounds
        sa[i, :] = np.clip(sa[i, :], 0.0, s0)
        ua[i, :] = np.clip(ua[i, :], 0.0, u0)

    # get total runoff
    runoff_sim = np.maximum(y_out + y0_out, 0.0)

    # DHF routing calculation - vectorized version without basin loop
    qs = np.zeros((actual_time_steps, num_basins))
    ql = np.zeros((actual_time_steps, num_basins))

    w0 = f / (3.6 * time_interval)  # tmp value used for testing

    # Main time loop for routing (before convolution, getting ya)
    for i in range(actual_time_steps):
        # ya is precedent rain -- Pa
        if i > 0:
            ya0 = ya[i - 1, :]
        ya[i, :] = np.maximum((ya0 + runoff_sim[i, :]) * ka, 0.0)
        # Get current state for all basins (vectorized)
        # here we keep the same as Chu's version
        # You can see that when getting temp_tm, we use ya_val + runoff_sim[i, :]
        ya_val = np.maximum(ya0, 0.5)  # Ensure stability for all basins
        # yl is rL in Chu's version; it's subsurface reservoir's infiltration
        rl = yl_out[i, :]
        # keep same as Chu's version
        rl = np.maximum(rl, 0.0)

        # Calculate routing parameters for all basins (vectorized)
        temp_tm = (ya_val + runoff_sim[i, :]) ** (-k0)
        lb = l / b0
        tm = lb * temp_tm

        # Time indices for all basins (vectorized)
        tt = (n * tm).astype(int)
        ts = (coe * tm).astype(int)

        # Surface routing coefficient calculation (vectorized)
        k3 = np.zeros(num_basins)
        aa_val = np.zeros(num_basins)

        # Calculate routing coefficients for all basins
        temp_aa = (pai * coe) ** (dd - 1)
        aa_val = cc / (dd * temp_aa * np.tan(pai * coe))

        # Calculate k3 for all basins
        max_tm = int(np.ceil(np.max(tm)))
        # we use max_tm for all basins
        for j in range(max_tm):
            # Then, we only process basins for its periods, where j < tm
            j_mask = j < tm
            if np.any(j_mask):
                temp = (pai * j / tm[j_mask]) ** dd[j_mask]
                temp1 = (np.sin(pai * j / tm[j_mask])) ** cc[j_mask]
                k3[j_mask] += np.exp(-aa_val[j_mask] * temp) * temp1

        # Final k3 calculation with division check
        nonzero_k3 = k3 != 0
        if np.any(nonzero_k3):
            k3[nonzero_k3] = tm[nonzero_k3] * w0 / k3[nonzero_k3]

        # Subsurface routing coefficient calculation (vectorized)
        k3l = np.zeros(num_basins)
        aal_val = np.zeros(num_basins)

        # Calculate subsurface routing coefficients for all basins
        temp_aal = (pai * coe / n) ** (ddl - 1)
        aal_val = ccl / (ddl * temp_aal * np.tan(pai * coe / n))

        # Calculate k3l for all basins
        max_tt = int(np.ceil(np.max(tt)))
        for j in range(max_tt):
            # Only process basins where j < tt
            j_mask = j < tt
            if np.any(j_mask):
                temp = (pai * j / tt[j_mask]) ** ddl[j_mask]
                temp1 = (np.sin(pai * j / tt[j_mask])) ** ccl[j_mask]
                k3l[j_mask] += np.exp(-aal_val[j_mask] * temp) * temp1

        # Final k3l calculation with division check
        nonzero_k3l = k3l != 0
        if np.any(nonzero_k3l):
            k3l[nonzero_k3l] = tt[nonzero_k3l] * w0 / k3l[nonzero_k3l]

        # Calculate tl for all basins (vectorized)
        tl = np.maximum(tt + ts - 1, 0)

        # Process routing time steps (still need this loop for convolution)
        max_tl = int(np.ceil(np.max(tl)))
        for j in range(max_tl):
            # Check bounds to prevent overflow
            valid_idx = i + j < actual_time_steps
            if not valid_idx:
                break

            # Surface routing calculation (vectorized)
            q_surface = np.zeros(num_basins)
            # Subsurface routing calculation (vectorized)
            q_subsurface = np.zeros(num_basins)
            _j_mask = j <= tl
            if np.any(_j_mask):
                temp0 = pai * j / tm[_j_mask]
                temp1 = temp0 ** dd[_j_mask]
                temp2 = np.exp(-aa_val[_j_mask] * temp1)
                temp3 = (np.sin(temp0)) ** cc[_j_mask]
                q_surface[_j_mask] = (
                    (runoff_sim[i, _j_mask] - rl[_j_mask])
                    * k3[_j_mask]
                    / tm[_j_mask]
                    * temp2
                    * temp3
                )
                # Handle NaN values
                q_surface[np.isnan(q_surface)] = 0.0

                temp00 = pai * (j - ts[_j_mask]) / tt[_j_mask]
                temp10 = temp00 ** ddl[_j_mask]
                temp20 = np.exp(-aal_val[_j_mask] * temp10)
                temp30 = (np.sin(temp00)) ** ccl[_j_mask]
                q_subsurface[_j_mask] = (
                    rl[_j_mask] * k3l[_j_mask] / tt[_j_mask] * temp20 * temp30
                )

            # Add contributions based on timing conditions (vectorized)
            # Case 1: j <= tm and j <= ts
            mask1 = (j <= tm) & (j <= ts)
            qs[i + j, mask1] += q_surface[mask1]

            # Case 2: j <= tm and j > ts
            mask2 = (j <= tm) & (j > ts)
            qs[i + j, mask2] += q_surface[mask2]
            ql[i + j, mask2] += q_subsurface[mask2]

            # Case 3: j > tm
            mask3 = j > tm
            ql[i + j, mask3] += q_subsurface[mask3]

    # Total discharge
    q_sim = qs + ql
    q_sim = np.maximum(q_sim, 0.0)

    # seq, batch, feature
    q_sim = np.expand_dims(q_sim, axis=2)
    runoff_sim = np.expand_dims(runoff_sim, axis=2)
    y0_out = np.expand_dims(y0_out, axis=2)
    yu_out = np.expand_dims(yu_out, axis=2)
    yl_out = np.expand_dims(yl_out, axis=2)
    y_out = np.expand_dims(y_out, axis=2)
    sa = np.expand_dims(sa, axis=2)
    ua = np.expand_dims(ua, axis=2)
    ya = np.expand_dims(ya, axis=2)

    if return_state:
        return (
            q_sim,
            runoff_sim,
            y0_out,
            yu_out,
            yl_out,
            y_out,
            sa,
            ua,
            ya,
        )
    else:
        return q_sim
