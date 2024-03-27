"""
Author: Wenyu Ouyang
Date: 2024-03-26 12:00:12
LastEditTime: 2024-03-27 10:20:22
LastEditors: Wenyu Ouyang
Description: evaluate a calibrated hydrological model
FilePath: \hydro-model-xaj\scripts\evaluate_xaj.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

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
    exp = args.exp
    cali_dir = Path(os.path.join(repo_path, "result", exp))
    cali_config = read_yaml_config(os.path.join(cali_dir, "config.yaml"))
    kfold = cali_config["cv_fold"]
    algo_info = cali_config["algorithm"]
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
    for fold in range(kfold):
        print(f"Start to evaluate the {fold+1}-th fold")
        fold_dir = os.path.join(cali_dir, f"sceua_xaj_cv{fold+1}")
        if algo_info["name"] == "SCE_UA":
            # evaluate both train and test period for all basins
            train_data = train_and_test_data[fold][0]
            test_data = train_and_test_data[fold][1]
            eval_train_dir = os.path.join(fold_dir, "train")
            eval_test_dir = os.path.join(fold_dir, "test")
            train_eval = Evaluator(cali_dir, fold_dir, eval_train_dir)
            test_eval = Evaluator(cali_dir, fold_dir, eval_test_dir)
            qsim_train, qobs_train = train_eval.predict(train_data)
            qsim_test, qobs_test = test_eval.predict(test_data)
            train_eval.save_results(
                train_data,
                qsim_train,
                qobs_train,
            )
            test_eval.save_results(
                test_data,
                qsim_test,
                qobs_test,
            )
        else:
            raise NotImplementedError(
                "We don't provide this calibrate method! Choose from 'SCE_UA' or 'GA'!"
            )

        print(f"Finish evaluating the {fold}-th fold")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate a calibrated hydrological model."
    )
    parser.add_argument(
        "--exp",
        dest="exp",
        help="An exp is corresponding to a data plan from calibrate_xaj.py",
        default="expcamels001",
        type=str,
    )
    the_args = parser.parse_args()
    evaluate(the_args)
