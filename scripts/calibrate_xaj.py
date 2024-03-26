"""
Author: Wenyu Ouyang
Date: 2022-11-19 17:27:05
LastEditTime: 2024-03-26 16:54:05
LastEditors: Wenyu Ouyang
Description: the script to calibrate a model for CAMELS basin
FilePath: \hydro-model-xaj\scripts\calibrate_xaj.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import json
import numpy as np
import argparse
import sys
import os
from pathlib import Path
import xarray as xr
import yaml

from hydrodataset import Camels
from hydrodata.utils.utils import streamflow_unit_conv


repo_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_path)
from hydromodel import SETTING
from hydromodel.datasets import *
from hydromodel.datasets.data_preprocess import cross_valid_data, split_train_test
from hydromodel.trainers.calibrate_sceua import calibrate_by_sceua


def calibrate(args):
    data_type = args.data_type
    data_dir = args.data_dir
    exp = args.exp
    cv_fold = args.cv_fold
    train_period = args.calibrate_period
    test_period = args.test_period
    periods = args.period
    warmup = args.warmup
    basin_ids = args.basin_id
    model_info = args.model
    algo_info = args.algorithm
    loss = args.loss
    if data_type == "camels":
        camels_data_dir = os.path.join(
            SETTING["local_data_path"]["datasets-origin"], "camels", data_dir
        )
        camels = Camels(camels_data_dir)
        ts_data = camels.read_ts_xrdataset(
            basin_ids, periods, ["prcp", "PET", "streamflow"]
        )
    elif data_type == "owndata":
        ts_data = xr.open_dataset(
            os.path.join(os.path.dirname(data_dir), "timeseries.nc")
        )
    else:
        raise NotImplementedError(
            "You should set the data type as 'camels' or 'owndata'"
        )

    where_save = Path(os.path.join(repo_path, "result", exp))
    if os.path.exists(where_save) is False:
        os.makedirs(where_save)

    if cv_fold <= 1:
        # no cross validation
        periods = np.sort(
            [train_period[0], train_period[1], test_period[0], test_period[1]]
        )
    if cv_fold > 1:
        train_and_test_data = cross_valid_data(ts_data, periods, warmup, cv_fold)
    else:
        # when using train_test_split, the warmup period is not used
        # so you should include the warmup period in the train and test period
        train_and_test_data = split_train_test(ts_data, train_period, test_period)
    print("Start to calibrate the model")

    if data_type == "camels":
        basin_area = camels.read_area(basin_ids)
        p_and_e = (
            train_and_test_data[0][["prcp", "PET"]]
            .to_array()
            .to_numpy()
            .transpose(2, 1, 0)
        )
        # trans unit to mm/day
        qobs_ = train_and_test_data[0][["streamflow"]]
        r_mmd = streamflow_unit_conv(qobs_, basin_area, target_unit="mm/d")
        qobs = np.expand_dims(r_mmd["streamflow"].to_numpy().transpose(1, 0), axis=2)
    elif data_type == "owndata":
        attr_data = xr.open_dataset(
            os.path.join(os.path.dirname(data_dir), "attributes.nc")
        )
        basin_area = attr_data["area"].values
        p_and_e = (
            train_and_test_data[0][[PRCP_NAME, PET_NAME]]
            .to_array()
            .to_numpy()
            .transpose(2, 1, 0)
        )
        qobs = np.expand_dims(
            train_and_test_data[0][[FLOW_NAME]].to_array().to_numpy().transpose(1, 0),
            axis=2,
        )
    else:
        raise NotImplementedError(
            "You should set the data type as 'camels' or 'owndata'"
        )
    calibrate_by_sceua(
        basin_ids,
        p_and_e,
        qobs,
        os.path.join(where_save, "sceua_xaj"),
        warmup,
        model=model_info,
        algorithm=algo_info,
        loss=loss,
    )
    # Convert the arguments to a dictionary
    args_dict = vars(args)
    # Save the arguments to a YAML file
    with open(os.path.join(where_save, "config.yaml"), "w") as f:
        yaml.dump(args_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hydro-model-xaj models with the CAMELS dataset"
    )
    parser.add_argument(
        "--data_type",
        dest="data_type",
        help="CAMELS dataset or your own data, such as 'camels' or 'owndata'",
        default="camels",
        type=str,
    )
    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        help="The directory of the CAMELS dataset or your own data, for CAMELS,"
        + " as we use SETTING to set the data path, you can directly choose camels_us;"
        + " for your own data, you should set the absolute path of your data directory",
        default="camels_us",
        type=str,
    )
    parser.add_argument(
        "--exp",
        dest="exp",
        help="An exp is corresponding to one data setting",
        default="expcamels001",
        type=str,
    )
    parser.add_argument(
        "--cv_fold",
        dest="cv_fold",
        help="the number of cross-validation fold",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--warmup",
        dest="warmup",
        help="the number of warmup days",
        default=365,
        type=int,
    )
    parser.add_argument(
        "--period",
        dest="period",
        help="The whole period",
        default=["2007-01-01", "2014-01-01"],
        nargs="+",
    )
    parser.add_argument(
        "--calibrate_period",
        dest="calibrate_period",
        help="The training period",
        default=["2007-01-01", "2014-01-01"],
        nargs="+",
    )
    parser.add_argument(
        "--test_period",
        dest="test_period",
        help="The testing period",
        default=["2007-01-01", "2014-01-01"],
        nargs="+",
    )
    parser.add_argument(
        "--basin_id",
        dest="basin_id",
        help="The basins' ids",
        default=["01439500", "06885500", "08104900", "09510200"],
        nargs="+",
    )
    parser.add_argument(
        "--model",
        dest="model",
        help="which hydro model you want to calibrate and the parameters setting for model function, note: not hydromodel parameters but function's parameters",
        default={
            "name": "xaj",
            "source_type": "sources5mm",
            "source_book": "HF",
        },
        type=json.loads,
    )
    parser.add_argument(
        "--algorithm",
        dest="algorithm",
        help="algorithm and its parameters used for calibrating algorithm. "
        "Here are some advices for the algorithmic parameter settings:"
        "rep is the maximum number of calling hydro-model, it is mainly impacted by ngs, if ngs is 30, one population need about 900 evaluations, at this time, 10000 maybe a good choice;"
        "ngs is the number of complex, better larger than your hydro-model-params number (nopt) but not too large, because the number of population's individuals is ngs * (2*nopt+1), larger ngs need more evaluations;"
        "kstop is the number of evolution (not evaluation) loops, some small numbers such as 2, 3, 5, ... are recommended, if too large it is hard to finish optimizing;"
        "peps and pcento are two loop-stop criterion, 0.1 (its unit is %, 0.1 means a relative change of 1/1000) is a good choice",
        default={
            "name": "SCE_UA",
            "random_seed": 1234,
            # these params are just for test
            "rep": 10,
            "ngs": 10,
            "kstop": 5,
            "peps": 0.1,
            "pcento": 0.1,
        },
        # default={
        #     "name": "GA",
        #     "random_seed": 1234,
        #     "run_counts": 2,
        #     "pop_num": 50,
        #     "cross_prob": 0.5,
        #     "mut_prob": 0.5,
        #     "save_freq": 1,
        # },
        type=json.loads,
    )
    parser.add_argument(
        "--loss",
        dest="loss",
        help="A tag for a plan, we will use it when postprocessing results",
        default={
            "type": "time_series",
            "obj_func": "RMSE",
            "events": None,
        },
        type=json.loads,
    )
    the_args = parser.parse_args()
    calibrate(the_args)
