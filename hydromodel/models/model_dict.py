"""
Author: Wenyu Ouyang
Date: 2024-03-23 08:25:49
LastEditTime: 2024-03-26 11:44:04
LastEditors: Wenyu Ouyang
Description: CRITERION_DICT and MODEL_DICT
FilePath: \hydro-model-xaj\hydromodel\models\model_dict.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import numpy as np
from spotpy.objectivefunctions import rmse
from hydromodel.models.xaj import xaj
from hydromodel.models.gr4j import gr4j
from hydromodel.models.hymod import hymod


def rmse43darr(obs, sim):
    rmses = np.sqrt(np.nanmean((sim - obs) ** 2, axis=0))
    rmse = rmses.mean(axis=0)
    if rmse is np.nan:
        raise ValueError("RMSE is nan, please check the input data.")
    return rmse


CRITERION_DICT = {
    "RMSE": rmse43darr,
    "spotpy_rmse": rmse,
}

MODEL_DICT = {
    "xaj_mz": xaj,
    "xaj": xaj,
    "gr4j": gr4j,
    "hymod": hymod,
}
