"""
Author: Wenyu Ouyang
Date: 2022-11-19 17:27:05
LastEditTime: 2023-05-05 10:14:20
LastEditors: Wenyu Ouyang
Description: the script to postprocess calibrated models in hydro-model-xaj
FilePath: \hydro-model-xaj\hydromodel\app\datapostprocess4calibrate.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from tqdm import tqdm
import numpy as np
import pandas as pd
import fnmatch
import argparse
import sys
import os
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
import definitions
from hydromodel.data.data_postprocess import read_and_save_et_ouputs


def statistics(args):
    exp = args.exp
    cases = args.cases
    cv_fold = args.cv_fold
    where_save_cache = Path(
        os.path.join(definitions.ROOT_DIR, "hydromodel", "example", exp)
    )
    if os.path.exists(where_save_cache) is False:
        raise NotImplementedError(
            "You should run datapreprocess4calibrate.py and calibrate_xaj_camels_cc.py first."
        )
    if cases is None:
        cases = os.listdir(where_save_cache)

    print(
        "Compare evaluation results of different calibrated models in an exp directory"
    )
    mean_lst_test = []
    median_lst_test = []
    mean_lst_train = []
    median_lst_train = []
    comment_lst = []
    for case in cases:
        case_dir = where_save_cache.joinpath(case)
        if case_dir.is_dir():
            # test_metric_file is used only to confirm the test results exist
            test_metric_file = case_dir.joinpath("basins_metrics_test.csv")
            if not test_metric_file.exists():
                continue
            comment = case_dir.name.split("_")[-1]
            comment_lst.append(comment)
    comments = np.unique(comment_lst)

    for comment in tqdm(comments, desc="All settings in an exp directory"):
        comment_folds_test = []
        comment_folds_train = []
        for fold in range(cv_fold):
            comment_fold_dir = []
            for case in cases:
                case_dir = where_save_cache.joinpath(case)
                if case_dir.is_dir() and fnmatch.fnmatch(
                    case_dir.name, f"*_fold{str(fold)}_" + comment
                ):
                    comment_fold_dir.append(case_dir)
            comment_fold_dir_newest = np.sort(comment_fold_dir)[-1]
            read_and_save_et_ouputs(comment_fold_dir_newest, fold=fold)
            comment_fold_file_test = comment_fold_dir_newest.joinpath(
                "basins_metrics_test.csv"
            )
            comment_fold_file_train = comment_fold_dir_newest.joinpath(
                "basins_metrics_train.csv"
            )
            basins_test_metrics = pd.read_csv(comment_fold_file_test, index_col=0)
            basin_train_metrics = pd.read_csv(comment_fold_file_train, index_col=0)
            comment_folds_test.append(basins_test_metrics)
            comment_folds_train.append(basin_train_metrics)
        for i in range(len(comment_folds_test)):
            if i == 0:
                comment_folds_sum_test = comment_folds_test[i]
                comment_folds_sum_train = comment_folds_train[i]
            else:
                comment_folds_sum_test = comment_folds_sum_test + comment_folds_test[i]
                comment_folds_sum_train = (
                    comment_folds_sum_train + comment_folds_train[i]
                )
        comment_folds_mean_test = comment_folds_sum_test / len(comment_folds_test)
        comment_folds_mean_train = comment_folds_sum_train / len(comment_folds_train)
        comment_fold_save_file_test = where_save_cache.joinpath(
            "basins_metrics_test_" + comment + ".csv"
        )
        comment_fold_save_file_train = where_save_cache.joinpath(
            "basins_metrics_train_" + comment + ".csv"
        )
        comment_folds_mean_test.to_csv(comment_fold_save_file_test)
        comment_folds_mean_train.to_csv(comment_fold_save_file_train)
        ind_mean_test = comment_folds_mean_test.mean(axis=1)
        ind_median_test = comment_folds_mean_test.median(axis=1)
        ind_mean_train = comment_folds_mean_train.mean(axis=1)
        ind_median_train = comment_folds_mean_train.median(axis=1)
        ind_mean_test.name = comment
        ind_median_test.name = comment
        ind_mean_train.name = comment
        ind_median_train.name = comment
        mean_lst_test.append(ind_mean_test)
        median_lst_test.append(ind_median_test)
        mean_lst_train.append(ind_mean_train)
        median_lst_train.append(ind_median_train)
    mean_df_test = pd.concat(mean_lst_test, axis=1).T
    median_df_test = pd.concat(median_lst_test, axis=1).T
    mean_df_train = pd.concat(mean_lst_train, axis=1).T
    median_df_train = pd.concat(median_lst_train, axis=1).T
    mean_df_test.to_csv(
        where_save_cache.joinpath("basins_test_metrics_mean_all_cases.csv")
    )
    median_df_test.to_csv(
        where_save_cache.joinpath("basins_test_metrics_median_all_cases.csv")
    )
    mean_df_train.to_csv(
        where_save_cache.joinpath("basins_train_metrics_mean_all_cases.csv")
    )
    median_df_train.to_csv(
        where_save_cache.joinpath("basins_train_metrics_median_all_cases.csv")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="postprocess for calibrated hydro-model-xaj models."
    )
    parser.add_argument(
        "--exp",
        dest="exp",
        help="An exp is corresponding to one data setting",
        default="example",
        type=str,
    )
    parser.add_argument(
        "--cases",
        dest="cases",
        help="The cases directory of calibrating results",
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--cv_fold",
        dest="cv_fold",
        help="The number of cross-validation fold",
        default=1,
        type=int,
    )
    the_args = parser.parse_args()
    statistics(the_args)
