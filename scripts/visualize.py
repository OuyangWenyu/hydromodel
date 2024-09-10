"""
Author: Wenyu Ouyang
Date: 2022-11-19 17:27:05
LastEditTime: 2024-08-15 17:02:21
LastEditors: Wenyu Ouyang
Description: the script to postprocess results
FilePath: \hydromodel\scripts\visualize.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import argparse
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt

repo_dir = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_dir)
from hydromodel.datasets.data_visualize import plot_sim_and_obs, plot_combined_figure
from hydromodel.trainers.evaluate import Evaluator, read_yaml_config

def visualize(args):
    exp = args.exp
    cali_dir = Path(os.path.join(repo_dir, "result", exp))
    cali_config = read_yaml_config(os.path.join(cali_dir, "config.yaml"))
    kfold = cali_config["cv_fold"]
    basins = cali_config["basin_id"]
    warmup = cali_config["warmup"]
    if kfold <= 1:
        print("Start to visualize the results")
        param_dir = os.path.join(cali_dir, "sceua_xaj")
        eval_train_dir = os.path.join(param_dir, "train")
        eval_test_dir = os.path.join(param_dir, "test")
        _visualize(cali_dir, basins, warmup, param_dir, eval_train_dir, eval_test_dir)
        print("Finish visualizing the results")
    else:
        for fold in range(kfold):
            print(f"Start to visualize the {fold+1}-th fold")
            param_dir = os.path.join(cali_dir, f"sceua_xaj_cv{fold+1}")
            eval_train_dir = os.path.join(param_dir, "train")
            eval_test_dir = os.path.join(param_dir, "test")
            _visualize(
                cali_dir, basins, warmup, param_dir, eval_train_dir, eval_test_dir
            )
            print(f"Finish visualizing the {fold+1}-th fold")

# def visualize(args):
#     exp = args.exp
#     cali_dir = Path(os.path.join(repo_dir, "result", exp))
#     cali_config = read_yaml_config(os.path.join(cali_dir, "config.yaml"))
#     kfold = cali_config["cv_fold"]
#     basins = cali_config["basin_id"]
#     warmup = cali_config["warmup"]
#     if kfold <= 1:
#         print("Start to visualize the results")
#         param_dir = os.path.join(cali_dir, "sceua_xaj")
#         eval_train_dir = os.path.join(param_dir, "train")
#         eval_test_dir = os.path.join(param_dir, "test")
#         _visualize(cali_dir, basins, warmup, param_dir, eval_train_dir, eval_test_dir)
#         print("Finish visualizing the results")
#     else:
#         for fold in range(kfold):
#             print(f"Start to visualize the {fold+1}-th fold")
#             param_dir = os.path.join(cali_dir, f"sceua_xaj_cv{fold+1}")
#             eval_train_dir = os.path.join(param_dir, "train")
#             eval_test_dir = os.path.join(param_dir, "test")
#             _visualize(
#                 cali_dir, basins, warmup, param_dir, eval_train_dir, eval_test_dir
#             )
#             print(f"Finish visualizing the {fold}-th fold")

def _visualize(cali_dir, basins, warmup, param_dir, eval_train_dir, eval_test_dir):
    train_eval = Evaluator(cali_dir, param_dir, eval_train_dir)
    test_eval = Evaluator(cali_dir, param_dir, eval_test_dir)
    ds_train = train_eval.load_results()
    ds_test = test_eval.load_results()
    for basin in basins:
        save_fig_train = os.path.join(eval_train_dir, f"train_combined_{basin}.png")
        start_time_train = ds_train["time"].isel(time=slice(warmup, None)).min().values
        end_time_train = ds_train["time"].isel(time=slice(warmup, None)).max().values
        # 调用绘图函数，包含降雨数据
        plot_combined_figure(
            ds_train["time"].isel(time=slice(warmup, None)),
            ds_train["qsim"].sel(basin=basin).isel(time=slice(warmup, None)),
            ds_train["qobs"].sel(basin=basin).isel(time=slice(warmup, None)),
            save_fig_train,
            basin,
            start_time_train,
            end_time_train
        )
        save_fig_test = os.path.join(eval_test_dir, f"test_combined_{basin}.png")
        start_time_test = ds_test["time"].isel(time=slice(warmup, None)).min().values
        end_time_test = ds_test["time"].isel(time=slice(warmup, None)).max().values
        plot_combined_figure(
            ds_test["time"].isel(time=slice(warmup, None)),
            ds_test["qsim"].sel(basin=basin).isel(time=slice(warmup, None)),
            ds_test["qobs"].sel(basin=basin).isel(time=slice(warmup, None)),
            save_fig_test,
            basin,
            start_time_test,
            end_time_test
        )



# def _visualize(cali_dir, basins, warmup, param_dir, eval_train_dir, eval_test_dir):
#     train_eval = Evaluator(cali_dir, param_dir, eval_train_dir)
#     test_eval = Evaluator(cali_dir, param_dir, eval_test_dir)
#     ds_train = train_eval.load_results()
#     ds_test = test_eval.load_results()
#     for basin in basins:
#         save_fig_train = os.path.join(eval_train_dir, f"train_sim_obs_{basin}.png")
#         plot_sim_and_obs(
#             ds_train["time"].isel(time=slice(warmup, None)),
#             ds_train["qsim"].sel(basin=basin).isel(time=slice(warmup, None)),
#             ds_train["qobs"].sel(basin=basin).isel(time=slice(warmup, None)),
#             save_fig_train,
#             xlabel="Date",
#             ylabel=None,
#         )
#         save_fig_test = os.path.join(eval_test_dir, f"test_sim_obs_{basin}.png")
#         plot_sim_and_obs(
#             ds_test["time"].isel(time=slice(warmup, None)),
#             ds_test["qsim"].sel(basin=basin).isel(time=slice(warmup, None)),
#             ds_test["qobs"].sel(basin=basin).isel(time=slice(warmup, None)),
#             save_fig_test,
#             xlabel="Date",
#             ylabel=None,
#         )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="show results for calibrated hydro-model-xaj models."
    )
    parser.add_argument(
        "--exp",
        dest="exp",
        help="An exp is corresponding to one data setting",
        default="expselfmadehydrodataset001",
        type=str,
    )
    the_args = parser.parse_args()
    visualize(the_args)
