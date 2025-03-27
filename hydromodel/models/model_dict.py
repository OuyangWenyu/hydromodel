'''
Author: zhuanglaihong
Date: 2025-03-24 13:43:14
LastEditTime: 2025-03-27 11:16:28
LastEditors: zhuanglaihong
Description: 
FilePath: /zlh/hydromodel/hydromodel/models/model_dict.py
Copyright: Copyright (c) 2021-2024 zhuanglaihong. All rights reserved.
'''
"""
Author: Wenyu Ouyang
Date: 2025-02-18 10:20:58
LastEditTime: 2025-03-17 11:59:09
LastEditors: zhuanglaihong
Description: LOSS_DICT and MODEL_DICT
FilePath: /zlh/hydromodel/hydromodel/models/model_dict.py
Copyright: Copyright (c) 2021-2024 zhuanglaihong. All rights reserved.
"""

import numpy as np
from spotpy.objectivefunctions import rmse

from hydromodel.models.semi_xaj import semi_xaj
from hydromodel.models.xaj import xaj
from hydromodel.models.gr1a import gr1a
from hydromodel.models.gr2m import gr2m
from hydromodel.models.gr3j import gr3j
from hydromodel.models.gr4j import gr4j
from hydromodel.models.gr5j import gr5j
from hydromodel.models.gr6j import gr6j
from hydromodel.models.gr_model import GRModel
from hydromodel.models.hymod import hymod



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
            "RMSE is nan or there are nan values in the simulation data, please check the input data."
        )
    # tolist is necessary for spotpy to get the value
    # otherwise the print will incur to an issue https://github.com/thouska/spotpy/issues/319
    return rmse.tolist()



LOSS_DICT = {
    "RMSE": rmse43darr,
    "spotpy_rmse": rmse,
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
}
