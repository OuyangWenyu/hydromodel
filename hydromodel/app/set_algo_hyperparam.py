"""
Author: Wenyu Ouyang
Date: 2022-11-19 17:45:09
LastEditTime: 2022-11-19 18:11:19
LastEditors: Wenyu Ouyang
Description: Set hyperparameters for hydro-model-xaj models
FilePath: \hydro-model-xaj\hydromodel\app\set_algo_hyperparam.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import argparse
import json

# TODO: It is not decided yet for how to set hyperparameters for each model
def main(args):
    exp = args.exp
    basin_ids = args.basin_id
    algo = args.algorithm
    hyperparam = args.hyperparam

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Set hyper-parameters for calibrating algorithms."
    )
    parser.add_argument(
        "--exp",
        dest="exp",
        help="An exp is corresponding to one data setting",
        default="exp001",
        type=str,
    )
    parser.add_argument(
        "--basin_id",
        dest="basin_id",
        help="The basins' ids",
        default=["60668", "61561", "63002", "63007", "63486", "92354", "94560"],
        nargs="+",
    )
    parser.add_argument(
        "--algorithm",
        dest="algorithm",
        help="the calibration algorithm",
        default="SCE-UA",
        type=str,
    )
    parser.add_argument(
        "--hyperparam",
        dest="hyperparam",
        help="hyper-parameters for calibrating algorithms",
        default={
            "random_seed": 1234,
            "rep": 1000,
            "ngs": 2000,
            "kstop": 500,
            "peps": 0.001,
            "pcento": 0.001,
        },
        type=json.loads,
    )
    the_args = parser.parse_args()
    main(the_args)
