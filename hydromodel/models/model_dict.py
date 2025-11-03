"""
Author: Wenyu Ouyang
Date: 2025-02-18 10:20:58
LastEditTime: 2025-08-30 08:53:43
LastEditors: Wenyu Ouyang
Description: LOSS_DICT and MODEL_DICT
FilePath: \hydromodel\hydromodel\models\model_dict.py
Copyright: Copyright (c) 2021-2024 zhuanglaihong. All rights reserved.
"""

import numpy as np
import spotpy.objectivefunctions as spotpy_obj

from hydromodel.models.semi_xaj import semi_xaj
from hydromodel.models.xaj import xaj
from hydromodel.models.gr1a import gr1a
from hydromodel.models.gr2m import gr2m
from hydromodel.models.gr3j import gr3j
from hydromodel.models.gr4j import gr4j
from hydromodel.models.gr5j import gr5j
from hydromodel.models.gr6j import gr6j
from hydromodel.models.hymod import hymod
from hydromodel.models.unit_hydrograph import (
    unit_hydrograph,
    categorized_unit_hydrograph,
)
from hydromodel.models.dhf import dhf
from hydromodel.models.xaj_slw import xaj_slw


def _auto_discover_spotpy_functions():
    """Auto-discover all callable functions in spotpy.objectivefunctions.

    Returns
    -------
    dict
        Dictionary with auto-discovered spotpy functions
    """
    auto_functions = {}

    # Get all functions from spotpy.objectivefunctions
    for attr_name in dir(spotpy_obj):
        attr = getattr(spotpy_obj, attr_name)
        # Check if it's a callable function and not private
        if (
            callable(attr)
            and not attr_name.startswith("_")
            and attr_name not in ["calculate_all_functions"]
        ):  # Exclude utility functions
            auto_functions[f"spotpy_{attr_name}"] = attr

    return auto_functions


def rmse43darr(obs, sim):
    """RMSE for 3D array

    Parameters
    ----------
    obs : np.ndarray
        observation data
    sim : np.ndarray
        simulation data

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    rmses = np.sqrt(np.nanmean((sim - obs) ** 2, axis=0))
    rmse = rmses.mean(axis=0)
    if np.isnan(rmse) or any(np.isnan(sim)):
        raise ValueError(
            "RMSE is nan or there are nan values in the simulation data, "
            "please check the input data."
        )
    # tolist is necessary for spotpy to get the value
    # otherwise the print will incur to an issue
    # https://github.com/thouska/spotpy/issues/319
    return rmse.tolist()


# Generate LOSS_DICT with custom functions and all spotpy functions
LOSS_DICT = {
    "RMSE": rmse43darr,
    **_auto_discover_spotpy_functions(),
}

MODEL_DICT = {
    "xaj_mz": xaj,
    "xaj": xaj,
    "gr4j": gr4j,
    "gr5j": gr5j,
    "gr6j": gr6j,
    "gr1a": gr1a,
    "gr2m": gr2m,
    "gr3j": gr3j,
    "hymod": hymod,
    "semi_xaj": semi_xaj,
    "unit_hydrograph": unit_hydrograph,
    "categorized_unit_hydrograph": categorized_unit_hydrograph,
    "dhf": dhf,
    "xaj_slw": xaj_slw,
}
