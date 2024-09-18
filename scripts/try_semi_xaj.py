"""
Author: Wenyu Ouyang
Date: 2022-11-19 17:27:05
LastEditTime: 2024-03-27 15:56:19
LastEditors: Wenyu Ouyang
Description: the script to calibrate a model for CAMELS basin
FilePath: \hydro-model-xaj\scripts\calibrate_xaj.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import json
import argparse
import shutil
import sys
import os
from pathlib import Path
import yaml
import numpy as np
import xarray as xr
import logging  # 去除debug信息
logging.basicConfig(level=logging.WARNING)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


repo_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_path)
from hydromodel.datasets import *
from hydromodel.datasets.data_preprocess import (
    _get_pe_q_from_ts,
    cross_val_split_tsdata,
)
from hydromodel.trainers.calibrate_semi_xaj_sceua import calibrate_semi_xaj_sceua
from hydromodel.models.semi_xaj import semi_xaj


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
    calibrate_id = args.calibrate_id
    model_info = args.model
    algo_info = args.algorithm
    loss_info = args.loss

    print(f'输入文件夹{data_dir}=====================')
    dt=3
    attributes = xr.open_dataset(f'{data_dir}/attributes.nc')  # 读取流域属性数据
    with open(f'{data_dir}/topo.txt', 'r') as f:
        topo = f.readlines()  # 加载拓扑
    with open(f'{data_dir}/ModelwithsameParas.json', 'r', encoding='utf-8') as file:
        modelwithsameParas = json.load(file)  # 加载率定参数
    with open(f'{data_dir}/params.json', 'r', encoding='utf-8') as file:
        params_range = json.load(file)  # 加载参数范围


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
    

    p_and_e, qobs = _get_pe_q_from_ts(train_and_test_data[0])
    para_seq=np.random.rand(sum(len(item['PARAMETER']) for item in modelwithsameParas))
    semi_xaj(p_and_e, attributes, modelwithsameParas, para_seq, params_range, topo, dt)


   #shutil.copy(param_range_file, where_save)
    #args.param_range_file = os.path.join(where_save, param_range_file.split(os.sep)[-1])
    args_dict = vars(args)
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
        default="owndata",
        type=str,
    )
    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        help="The directory of the CAMELS dataset or your own data, for CAMELS,"
        + " as we use SETTING to set the data path, you can directly choose camels_us;"
        + " for your own data, you should set the absolute path of your data directory",
        default=str(repo_path)+"\\input",
        type=str,
    )
    parser.add_argument(
        "--exp",
        dest="exp",
        help="An exp is corresponding to one data setting",
        default="yanwangbizi01",
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
        default=240,
        type=int,
    )
    parser.add_argument(
        "--period",
        dest="period",
        help="The whole period",
        default=["2010-01-01 00:00", "2015-10-29 21:00"],
        nargs="+",
    )
    parser.add_argument(
        "--calibrate_period",
        dest="calibrate_period",
        help="The training period",
        default=["2010-01-01 00:00", "2014-02-28 21:00"],
        nargs="+",
    )
    parser.add_argument(
        "--test_period",
        dest="test_period",
        help="The testing period",
        default=["2014-03-01 00:00", "2015-10-29 21:00"],
        nargs="+",
    )
    parser.add_argument(
        "--basin_id",
        dest="basin_id",
        help="The basins' ids",
        default=["1",'2','3'],
        nargs="+",
    )

    parser.add_argument(
        "--calibrate_id",
        dest="calibrate_id",
        help="The calibrate_id",
        default=2,
        nargs="+",
    )

    parser.add_argument(
        "--model",
        dest="model",
        help="which hydro model you want to calibrate and the parameters setting for model function, note: not hydromodel parameters but function's parameters",
        default={
            "name": "semi_xaj",
            "source_type": "sources5mm",
            "source_book": "HF",
        },
        type=json.loads,
    )
    parser.add_argument(
        "--param_range_file",
        dest="param_range_file",
        help="The file of the parameter range",
        # default=None,
        default=str(repo_path)+"\\result\\param_range.yaml",
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
            "rep": 5,
            "ngs": 5,
            "kstop": 5,
            "peps": 0.1,
            "pcento": 0.1,
        },
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
