"""
Author: Wenyu Ouyang
Date: 2022-11-19 17:27:05
LastEditTime: 2024-03-27 08:37:58
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
from hydromodel.trainers.evaluate import read_yaml_config


def statistics(args):
    exp = args.exp
    cali_dir = Path(os.path.join(repo_dir, "result", exp))
    cali_config = read_yaml_config(os.path.join(cali_dir, "config.yaml"))
    if os.path.exists(cali_dir) is False:
        raise NotImplementedError(
            "You should run prepare_data and calibrate scripts at first."
        )
    print(
        "Compare evaluation results of different calibrated models in an exp directory"
    )


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
    statistics(the_args)
