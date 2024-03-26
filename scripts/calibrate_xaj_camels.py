"""
Author: Wenyu Ouyang
Date: 2022-11-19 17:27:05
LastEditTime: 2024-03-26 10:16:49
LastEditors: Wenyu Ouyang
Description: the script to calibrate a model for CAMELS basin
FilePath: \hydro-model-xaj\scripts\calibrate_xaj_camels.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import numpy as np
import argparse
import sys
import os
from pathlib import Path

from hydrodataset import Camels
from hydrodata.utils.utils import streamflow_unit_conv

repo_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_path)
from hydromodel import SETTING
from hydromodel.datasets.data_preprocess import cross_valid_data, split_train_test
from hydromodel.trainers.calibrate_sceua import calibrate_by_sceua


def main(args):
    camels_name = args.camels_name
    exp = args.exp
    cv_fold = args.cv_fold
    train_period = args.calibrate_period
    test_period = args.test_period
    periods = args.period
    warmup = args.warmup
    basin_ids = args.basin_id
    camels_data_dir = os.path.join(
        SETTING["local_data_path"]["datasets-origin"], "camels", camels_name
    )
    camels = Camels(camels_data_dir)
    ts_data = camels.read_ts_xrdataset(
        basin_ids, periods, ["prcp", "PET", "streamflow"]
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
    p_and_e = (
        train_and_test_data[0][["prcp", "PET"]].to_array().to_numpy().transpose(2, 1, 0)
    )
    # trans unit to mm/day
    qobs_ = train_and_test_data[0][["streamflow"]]
    basin_area = camels.read_area(basin_ids)
    r_mmd = streamflow_unit_conv(qobs_, basin_area, target_unit="mm/d")
    qobs = np.expand_dims(r_mmd["streamflow"].to_numpy().transpose(1, 0), axis=2)
    calibrate_by_sceua(
        basin_ids,
        p_and_e,
        qobs,
        os.path.join(where_save, "sceua_xaj"),
        warmup,
        model={
            "name": "xaj_mz",
            "source_type": "sources",
            "source_book": "HF",
        },
        algorithm={
            "name": "SCE_UA",
            "random_seed": 1234,
            "rep": 5,
            "ngs": 7,
            "kstop": 3,
            "peps": 0.1,
            "pcento": 0.1,
        },
        metric={
            "type": "time_series",
            "obj_func": "RMSE",
            "events": None,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hydro-model-xaj models with the CAMELS dataset"
    )
    parser.add_argument(
        "--camels_name",
        dest="camels_name",
        help="the name of CAMELS-formatted data directory",
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
    the_args = parser.parse_args()
    main(the_args)
