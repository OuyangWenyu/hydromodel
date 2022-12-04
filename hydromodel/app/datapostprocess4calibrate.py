"""
Author: Wenyu Ouyang
Date: 2022-11-19 17:27:05
LastEditTime: 2022-12-04 16:33:39
LastEditors: Wenyu Ouyang
Description: the script to postprocess calibrated models in hydro-model-xaj
FilePath: \hydro-model-xaj\hydromodel\app\datapostprocess4calibrate.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import numpy as np
import pandas as pd
import fnmatch
import argparse
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
import definitions


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
    mean_lst = []
    median_lst = []
    comment_lst = []
    for case in cases:
        case_dir = where_save_cache.joinpath(case)
        if case_dir.is_dir():
            test_metric_file = case_dir.joinpath("basins_metrics_test.csv")
            if not test_metric_file.exists():
                continue
            comment = case_dir.name.split("_")[-1]
            comment_lst.append(comment)
    comments = np.unique(comment_lst)

    for comment in comments:
        comment_folds = []
        for fold in range(cv_fold):
            comment_fold_dir = []
            for case in cases:
                case_dir = where_save_cache.joinpath(case)
                if case_dir.is_dir() and fnmatch.fnmatch(
                    case_dir.name, "*_fold" + str(fold) + "_" + comment
                ):
                    comment_fold_dir.append(case_dir)
            comment_fold_dir_newest = np.sort(comment_fold_dir)[-1]
            comment_fold_file = comment_fold_dir_newest.joinpath(
                "basins_metrics_test.csv"
            )
            basins_test_metrics = pd.read_csv(comment_fold_file, index_col=0)
            comment_folds.append(basins_test_metrics)
        for i in range(len(comment_folds)):
            if i == 0:
                comment_folds_sum = comment_folds[i]
            else:
                comment_folds_sum = comment_folds_sum + comment_folds[i]
        comment_folds_mean = comment_folds_sum / len(comment_folds)
        comment_fold_save_file = where_save_cache.joinpath(
            "basins_metrics_test_" + comment + ".csv"
        )
        comment_folds_mean.to_csv(comment_fold_save_file)
        ind_mean = comment_folds_mean.mean(axis=1)
        ind_median = comment_folds_mean.median(axis=1)
        ind_mean.name = comment
        ind_median.name = comment
        mean_lst.append(ind_mean)
        median_lst.append(ind_median)
    mean_df = pd.concat(mean_lst, axis=1).T
    median_df = pd.concat(median_lst, axis=1).T
    mean_df.to_csv(where_save_cache.joinpath("basins_test_metrics_mean_all_cases.csv"))
    median_df.to_csv(
        where_save_cache.joinpath("basins_test_metrics_median_all_cases.csv")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="postprocess for calibrated hydro-model-xaj models."
    )
    parser.add_argument(
        "--exp",
        dest="exp",
        help="An exp is corresponding to one data setting",
        default="exp201",
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
        default=2,
        type=int,
    )
    the_args = parser.parse_args()
    statistics(the_args)
