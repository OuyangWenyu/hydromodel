"""
Author: Wenyu Ouyang
Date: 2022-10-25 21:16:22
LastEditTime: 2022-12-25 16:06:05
LastEditors: Wenyu Ouyang
Description: some basic config for hydro-model-xaj models
FilePath: \hydro-model-xaj\hydromodel\models\model_config.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from collections import OrderedDict

# NOTE: Don't change the parameter settings

MODEL_PARAM_DICT = {
    "xaj": {
        "param_name": [
            # Allen, R.G., L. Pereira, D. Raes, and M. Smith, 1998.
            # Crop Evapotranspiration, Food and Agriculture Organization of the United Nations,
            # Rome, Italy. FAO publication 56. ISBN 92-5-104219-5. 290p.
            "K",  # ratio of potential evapotranspiration to reference crop evaporation generally from Allen, 1998
            "B",  # The exponent of the tension water capacity curve
            "IM",  # The ratio of the impervious to the total area of the basin
            "UM",  # Tension water capacity in the upper layer
            "LM",  # Tension water capacity in the lower layer
            "DM",  # Tension water capacity in the deepest layer
            "C",  # The coefficient of deep evapotranspiration
            "SM",  # The areal mean of the free water capacity of surface soil layer
            "EX",  # The exponent of the free water capacity curve
            "KI",  # Outflow coefficients of interflow
            "KG",  # Outflow coefficients of groundwater
            "CS",  # The recession constant of channel system
            "L",  # Lag time
            "CI",  # The recession constant of the lower interflow
            "CG",  # The recession constant of groundwater storage
        ],
        "param_range": OrderedDict(
            {
                # "K": [0.5, 1.0],
                # "B": [0.5, 0.4],
                # "IM": [0.07, 0.1],
                # "UM": [0.0, 20.0],
                # "LM": [60.0, 90.0],
                # "DM": [60.0, 120.0],
                # "C": [0.1, 0.2],
                # "SM": [50, 100.0],
                # "EX": [1.0, 1.5],
                # "KI": [0.3, 0.7],
                # "KG": [0.2, 0.7],
                # "CS": [0.5, 1.0],
                # "L": [1.0, 10.0],  # unit is day
                # "CI": [0.5, 0.9],
                # "CG": [0.98, 0.998],
                
                "K": [0.508600, 0.508600],
                "B": [0.283690, 0.283690],
                "IM": [0.065323, 0.065323],
                "UM": [1.247600, 1.247600],
                "LM": [88.770000, 88.770000],
                "DM": [74.856000, 74.856000],
                "C": [0.114600, 0.114600],
                "SM": [59.735000,  59.735000],
                "EX": [1.226550, 1.226550],
                "KI": [0.106610, 0.106610],
                "KG": [0.043708,0.043708],
                "CS": [0.865000, 0.865000],
                "L": [2.984500, 2.984500],  # unit is day
                "CI": [0.748800, 0.748800],
                "CG": [0.984275, 0.984275],
                
                # "K": [0.1, 1.0],
                # "B": [0.1, 0.4],
                # "IM": [0.01, 0.1],
                # "UM": [0.0, 20.0],
                # "LM": [60.0, 90.0],
                # "DM": [60.0, 120.0],
                # "C": [0.1, 0.2],
                # "SM": [20, 50.0],
                # "EX": [1.0, 1.5],
                # "KI": [0.0, 0.7],
                # "KG": [0.0, 0.7],
                # "CS": [0.0, 1.0],
                # "L": [1.0, 10.0],  # unit is day
                # "CI": [0.0, 0.9],
                # "CG": [0.98, 0.998],
                
            }
        ),
    },
    "xaj_mz": {
        "param_name": [
            # Allen, R.G., L. Pereira, D. Raes, and M. Smith, 1998.
            # Crop Evapotranspiration, Food and Agriculture Organization of the United Nations,
            # Rome, Italy. FAO publication 56. ISBN 92-5-104219-5. 290p.
            "K",  # ratio of potential evapotranspiration to reference crop evaporation generally from Allen, 1998
            "B",  # The exponent of the tension water capacity curve
            "IM",  # The ratio of the impervious to the total area of the basin
            "UM",  # Tension water capacity in the upper layer
            "LM",  # Tension water capacity in the lower layer
            "DM",  # Tension water capacity in the deepest layer
            "C",  # The coefficient of deep evapotranspiration
            "SM",  # The areal mean of the free water capacity of surface soil layer
            "EX",  # The exponent of the free water capacity curve
            "KI",  # Outflow coefficients of interflow
            "KG",  # Outflow coefficients of groundwater
            "A",  # parameter of mizuRoute
            "THETA",  # parameter of mizuRoute
            "CI",  # The recession constant of the lower interflow
            "CG",  # The recession constant of groundwater storage
            "KERNEL",  # kernel size of mizuRoute unit hydrograph when using convolution method
        ],
        "param_range": OrderedDict(
            {
                "K": [0.5, 1.0],
                "B": [0.2, 0.4],
                "IM": [0.07, 0.1],
                "UM": [0.0, 20.0],
                "LM": [60.0, 90.0],
                "DM": [60.0, 120.0],
                "C": [0.0, 0.2],
                "SM": [5, 10],
                "EX": [1.0, 1.5],
                "KI": [0.0, 0.7],
                "KG": [0.0, 0.7],
                "A": [0.0, 2.9],
                "THETA": [0.0, 6.5],
                "CI": [0.0, 0.9],
                "CG": [0.98, 0.998],
                "KERNEL": [1, 15],
                
                # "K": [0.1, 1.0],
                # "B": [0.1, 0.4],
                # "IM": [0.01, 0.1],
                # "UM": [0.0, 20.0],
                # "LM": [60.0, 90.0],
                # "DM": [60.0, 120.0],
                # "C": [0.0, 0.2],
                # "SM": [5, 10],
                # "EX": [1.0, 1.5],
                # "KI": [0.0, 0.7],
                # "KG": [0.0, 0.7],
                # "A": [0.0, 2.9],
                # "THETA": [0.0, 6.5],
                # "CI": [0.0, 0.9],
                # "CG": [0.98, 0.998],
                # "KERNEL": [1, 15],
            }
        ),
    },
    "gr4j": {
        "param_name": ["x1", "x2", "x3", "x4"],
        "param_range": OrderedDict(
            {
                "x1": [100.0, 1200.0],
                "x2": [-5.0, 3.0],
                "x3": [20.0, 300.0],
                "x4": [1.1, 2.9],
            }
        ),
    },
    "hymod": {
        "param_name": ["cmax", "bexp", "alpha", "ks", "kq"],
        "param_range": OrderedDict(
            {
                "cmax": [1.0, 500.0],
                "bexp": [0.1, 2.0],
                "alpha": [0.1, 0.99],
                "ks": [0.001, 0.10],
                "kq": [0.1, 0.99],
            }
        ),
    },
}
