"""
Author: Wenyu Ouyang
Date: 2022-11-19 17:27:05
LastEditTime: 2024-03-27 11:23:02
LastEditors: Wenyu Ouyang
Description: the script to postprocess results
FilePath: \hydro-model-xaj\scripts\post_process.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import argparse
import sys
import os
from pathlib import Path

repo_dir = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_dir)
from hydromodel.datasets.data_postprocess import plot_sim_and_obs
from hydromodel.trainers.evaluate import Evaluator, read_yaml_config


def visualize(args):
    exp = args.exp
    cali_dir = Path(os.path.join(repo_dir, "result", exp))
    cali_config = read_yaml_config(os.path.join(cali_dir, "config.yaml"))
    kfold = cali_config["cv_fold"]
    basins = cali_config["basin_id"]
    warmup = cali_config["warmup"]
    for fold in range(kfold):
        print(f"Start to evaluate the {fold+1}-th fold")
        fold_dir = os.path.join(cali_dir, f"sceua_xaj_cv{fold+1}")
        # evaluate both train and test period for all basins
        eval_train_dir = os.path.join(fold_dir, "train")
        eval_test_dir = os.path.join(fold_dir, "test")
        train_eval = Evaluator(cali_dir, fold_dir, eval_train_dir)
        test_eval = Evaluator(cali_dir, fold_dir, eval_test_dir)
        ds_train = train_eval.load_results()
        ds_test = test_eval.load_results()
        for basin in basins:
            save_fig_train = os.path.join(eval_train_dir, f"train_sim_obs_{basin}.png")
            plot_sim_and_obs(
                ds_train["time"].isel(time=slice(warmup, None)),
                ds_train["qsim"].sel(basin=basin).isel(time=slice(warmup, None)),
                ds_train["qobs"].sel(basin=basin).isel(time=slice(warmup, None)),
                save_fig_train,
                xlabel="Date",
                ylabel=None,
            )
            save_fig_test = os.path.join(eval_test_dir, f"test_sim_obs_{basin}.png")
            plot_sim_and_obs(
                ds_test["time"].isel(time=slice(warmup, None)),
                ds_test["qsim"].sel(basin=basin).isel(time=slice(warmup, None)),
                ds_test["qobs"].sel(basin=basin).isel(time=slice(warmup, None)),
                save_fig_test,
                xlabel="Date",
                ylabel=None,
            )
        print(f"Finish visualizing the {fold}-th fold")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="show results for calibrated hydro-model-xaj models."
    )
    parser.add_argument(
        "--exp",
        dest="exp",
        help="An exp is corresponding to one data setting",
        default="expcamels001",
        type=str,
    )
    the_args = parser.parse_args()
    visualize(the_args)
