"""
Author: Wenyu Ouyang
Date: 2024-03-26 12:00:12
LastEditTime: 2024-03-27 16:20:25
LastEditors: Wenyu Ouyang
Description: evaluate a calibrated hydrological model
FilePath: \hydro-model-xaj\scripts\evaluate_xaj.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import json
import argparse
import os
import sys
from pathlib import Path
import logging  # 去除debug信息
logging.basicConfig(level=logging.WARNING)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import xarray as xr
repo_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_path)
from hydromodel.datasets.data_preprocess_topo import cross_val_split_tsdata
from hydromodel.trainers.evaluate_topo import Evaluator, read_yaml_config


def evaluate(args):
    exp = args.exp
    cali_dir = Path(os.path.join(repo_path, "result", exp))
    cali_config = read_yaml_config(os.path.join(cali_dir, "config.yaml"))
    kfold = cali_config["cv_fold"]
    basins = cali_config["basin_id"]
    warmup = cali_config["warmup"]
    data_type = cali_config["data_type"]
    data_dir = cali_config["data_dir"]
    train_period = cali_config["calibrate_period"]
    test_period = cali_config["test_period"]
    periods = cali_config["period"]
    calibrate_id=cali_config["calibrate_id"]

    dt=3
    attributes = xr.open_dataset(f'{data_dir}/attributes.nc')  # 读取流域属性数据
    with open(f'{data_dir}/topo.txt', 'r') as f:
        topo = f.readlines()  # 加载拓扑
    with open(f'{data_dir}/ModelwithsameParas.json', 'r', encoding='utf-8') as file:
        modelwithsameParas = json.load(file)  # 加载率定参数
    with open(f'{data_dir}/params.json', 'r', encoding='utf-8') as file:
        params_range = json.load(file)  # 加载参数范围

    train_and_test_data = cross_val_split_tsdata(
        data_type,
        data_dir,
        kfold,
        train_period,
        test_period,
        periods,
        warmup,
        basins,
    )
    if kfold <= 1:
        print("Start to evaluate")
        # evaluate both train and test period for all basins
        train_data = train_and_test_data[0]
        test_data = train_and_test_data[1]
        param_dir = os.path.join(cali_dir, "sceua_xaj")
        _evaluate(cali_dir, param_dir, train_data, test_data,calibrate_id,attributes,modelwithsameParas,params_range,topo,dt)
        print("Finish evaluating")
    else:
        for fold in range(kfold):
            print(f"Start to evaluate the {fold+1}-th fold")
            fold_dir = os.path.join(cali_dir, f"sceua_xaj_cv{fold+1}")
            # evaluate both train and test period for all basins
            train_data = train_and_test_data[fold][0]
            test_data = train_and_test_data[fold][1]
            _evaluate(cali_dir, fold_dir, train_data, test_data,calibrate_id,attributes,modelwithsameParas,params_range,topo,dt)
            print(f"Finish evaluating the {fold}-th fold")


def _evaluate(cali_dir, param_dir, train_data, test_data,calibrate_id,attributes,modelwithsameParas,params_range,topo,dt):
    eval_train_dir = os.path.join(param_dir, "train")
    eval_test_dir = os.path.join(param_dir, "test")
    train_eval = Evaluator(cali_dir, param_dir, eval_train_dir)
    test_eval = Evaluator(cali_dir, param_dir, eval_test_dir)
    qsim_train, qobs_train = train_eval.predict(train_data,calibrate_id,attributes,modelwithsameParas,params_range,topo,dt)
    qsim_test, qobs_test = test_eval.predict(test_data,calibrate_id,attributes,modelwithsameParas,params_range,topo,dt)
    train_eval.save_results(
        train_data,
        qsim_train,
        qobs_train,
        calibrate_id,
    )
    test_eval.save_results(
        test_data,
        qsim_test,
        qobs_test,
        calibrate_id,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate a calibrated hydrological model."
    )
    parser.add_argument(
        "--exp",
        dest="exp",
        help="An exp is corresponding to a data plan from calibrate_xaj.py",
        default="yanwangbizi01",
        type=str,
    )
    the_args = parser.parse_args()
    evaluate(the_args)
