"""
Author: Wenyu Ouyang
Date: 2022-10-25 21:16:22
LastEditTime: 2025-08-22 11:06:58
LastEditors: Wenyu Ouyang
Description: some basic config for hydro-model-xaj models
FilePath: /hydromodel/hydromodel/models/model_config.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from collections import OrderedDict

import yaml


def read_model_param_dict(file_path="param.yaml"):
    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)

        return {
            model: {
                "param_name": contents["param_name"],
                "param_range": OrderedDict(contents["param_range"]),
            }
            for model, contents in data.items()
        }
    except Exception as e:
        print(f"Error: {e}, we directly use the default MODEL_PARAM_DICT.")
        return MODEL_PARAM_DICT


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
                "K": [0.1, 1.0],
                "B": [0.1, 0.4],
                "IM": [0.01, 0.1],
                "UM": [0.0, 20.0],
                "LM": [60.0, 90.0],
                "DM": [60.0, 120.0],
                "C": [0.0, 0.2],
                "SM": [1, 100.0],
                # "SM": [50, 100.0],
                "EX": [1.0, 1.5],
                "KI": [0.0, 0.7],
                "KG": [0.0, 0.7],
                "CS": [0.0, 1.0],
                "L": [1.0, 10.0],  # unit is same as your time step
                "CI": [0.0, 0.9],
                "CG": [0.98, 0.998],
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
            # "KERNEL",  # kernel size of mizuRoute unit hydrograph when using convolution method
        ],
        "param_range": OrderedDict(
            {
                "K": [0.1, 1.0],
                # "K": [0.5, 1.0],
                "B": [0.1, 0.4],
                # "B": [0.2, 0.4],
                "IM": [0.01, 0.1],
                # "IM": [0.07, 0.1],
                "UM": [0.0, 20.0],
                "LM": [60.0, 90.0],
                "DM": [60.0, 120.0],
                "C": [0.0, 0.2],
                "SM": [1.0, 100.0],
                # "SM": [5, 10],
                "EX": [1.0, 1.5],
                "KI": [0.0, 0.7],
                "KG": [0.0, 0.7],
                "A": [0.0, 2.9],
                "THETA": [0.0, 6.5],
                "CI": [0.0, 0.9],
                "CG": [0.98, 0.998],
                # "KERNEL": [1, 15],
            }
        ),
    },
    "gr1a": {
        "param_name": ["x1"],
        "param_range": OrderedDict(
            {
                "x1": [0.01, 3.5],
            }
        ),
    },
    "gr2m": {
        "param_name": ["x1", "x2"],
        "param_range": OrderedDict(
            {
                "x1": [140, 2640],
                "x2": [0.21, 1.31],
            }
        ),
    },
    "gr3j": {
        "param_name": ["x1", "x2", "x3"],
        "param_range": OrderedDict(
            {
                "x1": [-5.0, 5.0],
                "x2": [0.5, 800],
                "x3": [0.3, 2.9],
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
    "gr5j": {
        "param_name": ["x1", "x2", "x3", "x4", "x5"],
        "param_range": OrderedDict(
            {
                "x1": [100.0, 1200.0],
                "x2": [-5.0, 3.0],
                "x3": [20.0, 300.0],
                "x4": [1.1, 2.9],
                "x5": [0, 1],
            }
        ),
    },
    "gr6j": {
        "param_name": ["x1", "x2", "x3", "x4", "x5", "x6"],
        "param_range": OrderedDict(
            {
                "x1": [100.0, 1200.0],
                "x2": [-5.0, 3.0],
                "x3": [20.0, 300.0],
                "x4": [1.1, 2.9],
                "x5": [0, 1],
                "x6": [1, 100],
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
    "dhf": {
        "param_name": [
            "S0",  # 表层蓄水容量
            "U0",  # 下层蓄水容量
            "D0",  # 深层蓄水容量
            "KC",  # 蒸发系数
            "KW",  # 下层流系数
            "K2",  # 渗透系数
            "KA",  # 总径流调节系数
            "G",  # 不透水面积比例
            "A",  # 表层蓄水指数
            "B",  # 下层蓄水指数
            "B0",  # 汇流参数
            "K0",  # 汇流参数
            "N",  # 汇流参数
            "DD",  # 汇流参数
            "CC",  # 汇流参数
            "COE",  # 汇流参数
            "DDL",  # 地下汇流参数
            "CCL",  # 地下汇流参数
        ],
        "param_range": OrderedDict(
            {
                "S0": [0.0, 50.0],  # 表层蓄水容量 (mm)
                "U0": [0.0, 90.0],  # 下层蓄水容量 (mm)
                "D0": [70.0, 160.0],  # 深层蓄水容量 (mm)
                "KC": [0.1, 0.9],  # 蒸发系数
                "KW": [0.0, 1.0],  # 下层流系数
                "K2": [0.2, 0.9],  # 渗透系数
                "KA": [0.7, 1.0],  # 总径流调节系数
                "G": [0.0, 1.0],  # 不透水面积比例
                "A": [0.0, 5.0],  # 表层蓄水指数
                "B": [1.0, 3.0],  # 下层蓄水指数
                "B0": [0.1, 2.0],  # 汇流参数
                "K0": [0.0, 0.8],  # 汇流参数
                "N": [2.0, 6.0],  # 汇流参数
                "DD": [0.5, 4.0],  # 汇流参数
                "CC": [0.5, 4.0],  # 汇流参数
                "COE": [0.0, 0.8],  # 汇流参数
                "DDL": [0.5, 4.0],  # 地下汇流参数
                "CCL": [0.5, 4.0],  # 地下汇流参数
            }
        ),
    },
    "xaj_slw": {
        "param_name": [
            "WUP",
            "WLP",
            "WDP",
            "SP",
            "FRP",
            "WM",
            "WUMx",
            "WLMx",
            "K",
            "B",
            "C",
            "IM",
            "SM",
            "EX",
            "KG",
            "KI",
            "CS",
            "CI",
            "CG",
            "LAG",
            "KK",
            "X",
            "MP",
            "QSP",
            "QIP",
            "QGP",
        ],
        "param_range": OrderedDict(
            {
                # Initial states and proportions
                "WUP": [0.0, 50.0],
                "WLP": [0.0, 60.0],
                "WDP": [0.0, 150.0],
                "SP": [0.0, 10.0],
                "FRP": [0.0, 1.0],
                # Generation parameters
                "WM": [80.0, 220.0],
                "WUMx": [0.05, 0.5],
                "WLMx": [0.5, 0.95],
                "K": [0.3, 1.2],
                "B": [0.05, 0.5],
                "C": [0.05, 0.35],
                "IM": [0.005, 0.1],
                "SM": [10.0, 120.0],
                "EX": [1.0, 2.0],
                "KG": [0.05, 0.7],
                "KI": [0.05, 0.7],
                # Routing/recession
                "CS": [0.1, 0.9],
                "CI": [0.3, 0.95],
                "CG": [0.95, 0.999],
                "LAG": [0.0, 10.0],
                "KK": [1.0, 15.0],
                "X": [0.0, 0.5],
                "MP": [1.0, 5.0],
                # Initial flows for routing
                "QSP": [0.0, 50.0],
                "QIP": [0.0, 50.0],
                "QGP": [0.0, 50.0],
            }
        ),
    },
}
