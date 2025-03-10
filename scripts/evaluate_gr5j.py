'''
Author: zhuanglaihong
Date: 2025-02-26 16:21:29
LastEditTime: 2025-02-26 16:21:42
LastEditors: zhuanglaihong
Description: 
FilePath: /zlh/hydromodel/scripts/evaluate_gr5j.py
Copyright: Copyright (c) 2021-2024 zhuanglaihong. All rights reserved.
'''


import argparse
import os
import sys
from pathlib import Path


repo_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_path)
from hydromodel.datasets.data_preprocess import cross_val_split_tsdata
from hydromodel.datasets import *
from hydromodel.trainers.evaluate import Evaluator, read_yaml_config


def evaluate(args):
    result_dir = args.result_dir
    exp = args.exp
    cali_dir = Path(os.path.join(result_dir, exp))
    cali_config = read_yaml_config(os.path.join(cali_dir, "config.yaml"))
    kfold = cali_config["cv_fold"]
    basins = cali_config["basin_id"]
    warmup = cali_config["warmup"]
    data_type = cali_config["data_type"]
    data_dir = cali_config["data_dir"]
    train_period = cali_config["calibrate_period"]
    test_period = cali_config["test_period"]
    periods = cali_config["period"]
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
        _evaluate_1fold(train_and_test_data, cali_dir)
    else:
        for fold in range(kfold):
            print(f"Start to evaluate the {fold+1}-th fold")
            fold_dir = os.path.join(cali_dir, f"sceua_xaj_cv{fold+1}")
            # evaluate both train and test period for all basins
            train_data = train_and_test_data[fold][0]
            test_data = train_and_test_data[fold][1]
            _evaluate(cali_dir, fold_dir, train_data, test_data)
            print(f"Finish evaluating the {fold}-th fold")


def _evaluate_1fold(train_and_test_data, cali_dir):
    print("Start to evaluate")
    # evaluate both train and test period for all basins
    train_data = train_and_test_data[0]
    test_data = train_and_test_data[1]
    param_dir = os.path.join(cali_dir, "sceua_gr5j")
    _evaluate(cali_dir, param_dir, train_data, test_data)
    print("Finish evaluating")


def _evaluate(cali_dir, param_dir, train_data, test_data):
    eval_train_dir = os.path.join(param_dir, "train")
    eval_test_dir = os.path.join(param_dir, "test")
    train_eval = Evaluator(cali_dir, param_dir, eval_train_dir)
    test_eval = Evaluator(cali_dir, param_dir, eval_test_dir)
    qsim_train, qobs_train, etsim_train = train_eval.predict(train_data)
    qsim_test, qobs_test, etsim_test = test_eval.predict(test_data)
    train_eval.save_results(
        train_data,
        qsim_train,
        qobs_train,
        etsim_train
    )
    test_eval.save_results(
        test_data,
        qsim_test,
        qobs_test,
        etsim_test
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate a calibrated hydrological model."
    )
    parser.add_argument(
        "--result_dir",
        dest="result_dir",
        help="The root directory of results",
        default=os.path.join(repo_path, "result"),
        type=str,
    )
    parser.add_argument(
        "--exp",
        dest="exp",
        help="An exp is corresponding to a data plan from calibrate_gr5j.py",
        default="expbiliuhe002",
        # default="exp21113800test001",
        # default="expselfmadehydrodataset001",
        type=str,
    )
    the_args = parser.parse_args()
    evaluate(the_args)
