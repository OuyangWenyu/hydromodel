"""
Author: Wenyu Ouyang
Date: 2022-11-19 17:27:05
LastEditTime: 2024-08-15 16:56:41
LastEditors: Wenyu Ouyang
Description: the script to calibrate a model for CAMELS basin
FilePath: \hydromodel\scripts\calibrate_xaj.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import json
import argparse
import shutil
import sys
import os
from pathlib import Path
import yaml

repo_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_path)
from hydromodel.models.model_config import MODEL_PARAM_DICT
from hydromodel.datasets.data_preprocess import (
    _get_pe_q_from_ts,
    cross_val_split_tsdata,
)
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
    loss_info = args.loss
    param_range_file = args.param_range_file

    where_save = Path(os.path.join(repo_path, "result", exp))
    if os.path.exists(where_save) is False:
        os.makedirs(where_save)

    train_and_test_data = cross_val_split_tsdata(
        data_type,
        data_dir,
        cv_fold,
        train_period,
        test_period,
        periods,
        warmup,
        basin_ids,
    )

    print("Start to calibrate the model")

    if cv_fold <= 1:
        p_and_e, qobs = _get_pe_q_from_ts(train_and_test_data[0])
        calibrate_by_sceua(
            basin_ids,
            p_and_e,
            qobs,
            os.path.join(where_save, "sceua_xaj"),
            warmup,
            model=model_info,
            algorithm=algo_info,
            loss=loss_info,
            param_file=param_range_file,
        )
    else:
        for i in range(cv_fold):
            train_data, _ = train_and_test_data[i]
            p_and_e_cv, qobs_cv = _get_pe_q_from_ts(train_data)
            calibrate_by_sceua(
                basin_ids,
                p_and_e_cv,
                qobs_cv,
                os.path.join(where_save, f"sceua_xaj_cv{i+1}"),
                warmup,
                model=model_info,
                algorithm=algo_info,
                loss=loss_info,
                param_file=param_range_file,
            )
    # update the param_range_file path
    if param_range_file is None:
        param_range_file = os.path.join(where_save, "param_range.yaml")
        args.param_range_file = param_range_file
        yaml.dump(MODEL_PARAM_DICT, open(param_range_file, "w"))
    else:
        args.param_range_file = os.path.join(
            where_save, param_range_file.split(os.sep)[-1]
        )
        # Save the parameter range file to result directory
        shutil.copy(param_range_file, where_save)
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
        # default="camels",
        default="selfmadehydrodataset",
        # default="owndata",
        type=str,
    )
    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        help="The directory of the CAMELS dataset or your own data, for CAMELS,"
        + " as we use SETTING to set the data path, you can directly choose camels_us;"
        + " for your own data, you should set the absolute path of your data directory",
        # default="camels_us",
        # default="C:\\Users\\wenyu\\OneDrive\\data\\biliuhe",
        default="C:\\Users\\wenyu\\OneDrive\\data\\FD_sources",
        # default="/ftproot/basins-interim",
        type=str,
    )
    parser.add_argument(
        "--exp",
        dest="exp",
        help="An exp is corresponding to one data setting",
        # default="expcamels001",
        # default="exp21113800test001",
        default="expselfmadehydrodataset001",
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
        help="the number of warmup periods",
        # default=720,
        default=120,
        type=int,
    )
    parser.add_argument(
        "--period",
        dest="period",
        help="The whole period",
        # default=["2007-01-01", "2014-01-01"],
        # default=["2012-06-10 00:00", "2022-08-31 23:00"],
        default=["2010-01-01 08:00", "2015-11-02 14:00"],
        nargs="+",
    )
    parser.add_argument(
        "--calibrate_period",
        dest="calibrate_period",
        help="The training period",
        # default=["2007-01-01", "2014-01-01"],
        # default=["2012-06-10 00:00", "2017-08-31 23:00"],
        default=["2010-01-01 08:00", "2014-09-14 02:00"],
        nargs="+",
    )
    parser.add_argument(
        "--test_period",
        dest="test_period",
        help="The testing period",
        # default=["2007-01-01", "2014-01-01"],
        # default=["2017-09-01 00:00", "2022-08-31 23:00"],
        default=["2014-09-14 08:00", "2015-11-02 14:00"],
        nargs="+",
    )
    parser.add_argument(
        "--basin_id",
        dest="basin_id",
        help="The basins' ids",
        # default=["01439500", "06885500", "08104900", "09510200"],
        # default=["21401550"],
        default=["songliao_21401550"],
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
            "time_interval_hours": 6,
        },
        type=json.loads,
    )
    parser.add_argument(
        "--param_range_file",
        dest="param_range_file",
        help="The file of the parameter range",
        # default=None,
        default="C:\\Users\\wenyu\\OneDrive\\data\\biliuhe\\param_range.yaml",
        # default="C:\\Users\\wenyu\\Downloads\\21113800\\param_range.yaml",
        type=str,
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
