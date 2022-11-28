# -*- coding: utf-8 -*-
import numpy as np
from numba import jit

from hydromodel.models.model_config import MODEL_PARAM_DICT


def hymod(p_and_e, parameters, warmup_length=30, return_state=False, **kwargs):
    """
    Run Hymod model

    See https://www.proc-iahs.net/368/180/2015/piahs-368-180-2015.pdf for a scientific paper:
    Quan, Z.; Teng, J.; Sun, W.; Cheng, T. & Zhang, J. (2015): Evaluation of the HYMOD model
    for rainfall–runoff simulation using the GLUE method. Remote Sensing and GIS for Hydrology
    and Water Resources, 180 - 185, IAHS Publ. 368. DOI: 10.5194/piahs-368-180-2015.

    Parameters
    ----------
    p_and_e
        precipitation and potential evapotranspiration, 3-dim variable: [time, basin, feature=1]
    parameters
         five parameters: cmax, bexp, alpha, ks, kq
    warmup_length
        the length of warmup period
    return_state
        if True, return x_slow, x_quick, x_loss, else only return streamflow

    Returns
    -------
    Union[list, np.array]
        streamflow, x_slow, x_quick, x_loss or streamflow
    """
    # parameter, 2-dim variable: [parameter=1, basin]
    cmax_scale = MODEL_PARAM_DICT["hymod"]["param_range"]["cmax"]
    bexp_sacle = MODEL_PARAM_DICT["hymod"]["param_range"]["bexp"]
    alpha_scale = MODEL_PARAM_DICT["hymod"]["param_range"]["alpha"]
    ks_scale = MODEL_PARAM_DICT["hymod"]["param_range"]["ks"]
    kq_scale = MODEL_PARAM_DICT["hymod"]["param_range"]["kq"]
    cmax = cmax_scale[0] + parameters[:, 0] * (cmax_scale[1] - cmax_scale[0])
    bexp = bexp_sacle[0] + parameters[:, 1] * (bexp_sacle[1] - bexp_sacle[0])
    alpha = alpha_scale[0] + parameters[:, 2] * (alpha_scale[1] - alpha_scale[0])
    ks = ks_scale[0] + parameters[:, 3] * (ks_scale[1] - ks_scale[0])
    kq = kq_scale[0] + parameters[:, 4] * (kq_scale[1] - kq_scale[0])
    if warmup_length > 0:
        # set no_grad for warmup periods
        p_and_e_warmup = p_and_e[0:warmup_length, :, :]
        _, x_slow, x_quick, x_loss = hymod(
            p_and_e_warmup,
            parameters,
            warmup_length=0,
            return_state=True,
            **kwargs,
        )
    else:
        # Initialize slow tank state
        # x_slow = 2.3503 / (ks * 22.5)
        x_slow = np.full(
            (p_and_e.shape[1], 1), 0.0
        )  # --> works ok if calibration data starts with low discharge
        # Initialize state(s) of quick tank(s)
        x_quick = np.full((p_and_e.shape[1], 3), 0.0)
        # HYMOD PROGRAM IS SIMPLE RAINFALL RUNOFF MODEL
        x_loss = np.full((p_and_e.shape[1], 1), 0.0)
    precip = p_and_e[warmup_length:, :, 0]
    pet = p_and_e[warmup_length:, :, 1]
    t = 0
    output = np.full(precip.shape, 0.0)
    # START PROGRAMMING LOOP WITH DETERMINING RAINFALL - RUNOFF AMOUNTS
    while t <= precip.shape[0] - 1:
        pval = precip[t, :]
        pet_val = pet[t, :]
        # Compute excess precipitation and evaporation
        er1, er2, x_loss = excess(x_loss, cmax, bexp, pval, pet_val)
        # Calculate total effective rainfall
        et = er1 + er2
        #  Now partition ER between quick and slow flow reservoirs
        uq = alpha * et
        us = (1 - alpha) * et
        # Route slow flow component with single linear reservoir
        x_slow, qs = linres(x_slow, us, ks)
        # Route quick flow component with linear reservoirs
        inflow = uq

        for i in range(3):
            # Linear reservoir
            x_quick[:, i], outflow = linres(x_quick[:, i], inflow, kq)
            inflow = outflow

        # Compute total flow for timestep
        output[t, :] = qs + outflow
        t = t + 1
    streamflow = np.expand_dims(output, axis=2)
    if return_state:
        return streamflow, x_slow, x_quick, x_loss
    return streamflow


@jit
def power(x, y):
    x = np.abs(x)  # Needed to capture invalid overflow with negative values
    return x**y


@jit
def linres(x_slow, inflow, rs):
    # Linear reservoir
    x_slow = (1 - rs) * x_slow + (1 - rs) * inflow
    outflow = (rs / (1 - rs)) * x_slow
    return x_slow, outflow


@jit
def excess(x_loss, cmax, bexp, pval, pet_val):
    # this function calculates excess precipitation and evaporation
    xn_prev = x_loss
    ct_prev = cmax * (
        1 - power((1 - ((bexp + 1) * (xn_prev) / cmax)), (1 / (bexp + 1)))
    )
    # Calculate Effective rainfall 1
    er1 = np.maximum((pval - cmax + ct_prev), 0.0)
    pval = pval - er1
    dummy = np.minimum(((ct_prev + pval) / cmax), 1)
    xn = (cmax / (bexp + 1)) * (1 - power((1 - dummy), (bexp + 1)))

    # Calculate Effective rainfall 2
    er2 = np.maximum(pval - (xn - xn_prev), 0)

    # Alternative approach
    evap = (
        1 - (((cmax / (bexp + 1)) - xn) / (cmax / (bexp + 1)))
    ) * pet_val  # actual ET is linearly related to the soil moisture state
    xn = np.maximum(xn - evap, 0)  # update state

    return er1, er2, xn
