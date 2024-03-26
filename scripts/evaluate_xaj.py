"""
Author: Wenyu Ouyang
Date: 2024-03-26 12:00:12
LastEditTime: 2024-03-26 21:50:14
LastEditors: Wenyu Ouyang
Description: 
FilePath: \hydro-model-xaj\scripts\evaluate_xaj.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import argparse
import yaml
import os
import sys
from pathlib import Path


repo_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_path)
from hydromodel.models.model_dict import MODEL_DICT
from hydromodel.datasets import *
from hydromodel.datasets.data_preprocess import cross_val_split_tsdata, get_pe_q_from_ts
from trainers.evaluate import (
    save_evaluate_results,
)
from hydromodel.trainers.evaluate import (
    read_all_basin_params,
)
from hydromodel.trainers.calibrate_ga import calibrate_by_ga, show_ga_result
from hydromodel.trainers.evaluate import (
    convert_streamflow_units,
    renormalize_params,
    summarize_metrics,
    summarize_parameters,
)


def read_yaml_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


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
    model_info = cali_config["model"]
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
        save_dir = os.path.join(cali_dir, f"sceua_xaj_cv{fold+1}")
        if algo_info["name"] == "SCE_UA":
            # evaluate both train and test period for all basins
            test_data = train_and_test_data[fold][1]
            test_p_and_e, _ = get_pe_q_from_ts(test_data)
            params = read_all_basin_params(basins, save_dir)
            # 计算模拟结果
            qsim, _ = MODEL_DICT[model_info["name"]](
                test_p_and_e,
                params,
                # we set the warmup_length=0 but later we get results from warmup_length to the end to evaluate
                warmup_length=0,
                **model_info,
            )
            # 创建 DataArray
            qsim, qobs = convert_streamflow_units(test_data, qsim, data_type, data_dir)
        elif algo_info["name"] == "GA":
            for i in range(len(data_info_train["basin"])):
                basin_id = data_info_train["basin"][i]
                basin_area = data_info_train["area"][i]
                # one directory for one model + one hyperparam setting and one basin
                deap_db_dir = os.path.join(
                    save_dir,
                    basin_id,
                )
                if not os.path.exists(deap_db_dir):
                    os.makedirs(deap_db_dir)
                calibrate_by_ga(
                    data_train[:, i : i + 1, 0:2],
                    data_train[:, i : i + 1, -1:],
                    deap_db_dir,
                    warmup_length=warmup,
                    model=model_info,
                    ga_param=algo_info,
                )
                show_ga_result(
                    deap_db_dir,
                    warmup_length=warmup,
                    basin_id=basin_id,
                    the_data=data_train[:, i : i + 1, :],
                    the_period=data_info_train["time"],
                    basin_area=basin_area,
                    model_info=model_info,
                    train_mode=True,
                )
                show_ga_result(
                    deap_db_dir,
                    warmup_length=warmup,
                    basin_id=basin_id,
                    the_data=test_p_and_e[:, i : i + 1, :],
                    the_period=data_info_test["time"],
                    basin_area=basin_area,
                    model_info=model_info,
                    train_mode=False,
                )
        else:
            raise NotImplementedError(
                "We don't provide this calibrate method! Choose from 'SCE_UA' or 'GA'!"
            )
        summarize_parameters(save_dir, model_info["name"], basins)
        renormalize_params(save_dir, model_info["name"], basins)
        # summarize_metrics(save_dir, model_info)
        save_evaluate_results(save_dir, model_info["name"], qsim, qobs, test_data)
        print(f"Finish calibrating the {fold}-th fold")


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
