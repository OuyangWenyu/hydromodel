"""
Author: Wenyu Ouyang
Date: 2022-11-19 17:27:05
LastEditTime: 2022-12-01 13:29:30
LastEditors: Wenyu Ouyang
Description: the script to postprocess calibrated models in hydro-model-xaj
FilePath: \hydro-model-xaj\hydromodel\app\datapostprocess4calibrate.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import pandas as pd
import argparse
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
import definitions


def main(args):
    exp = args.exp
    cases = args.cases
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
    for case in cases:
        case_dir = where_save_cache.joinpath(case)
        if case_dir.is_dir():
            test_metric_file = case_dir.joinpath("basins_metrics_test.csv")
            if not test_metric_file.exists():
                continue
            basins_test_metrics = pd.read_csv(test_metric_file, index_col=0)
            dir_name = case_dir.stem
            ind_mean = basins_test_metrics.mean(axis=1)
            ind_median = basins_test_metrics.median(axis=1)
            ind_mean.name = dir_name
            ind_median.name = dir_name
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
        default="exp51210",
        type=str,
    )
    parser.add_argument(
        "--cases",
        dest="cases",
        help="The cases directory of calibrating results",
        default=None,
        nargs="+",
    )
    the_args = parser.parse_args()
    main(the_args)
