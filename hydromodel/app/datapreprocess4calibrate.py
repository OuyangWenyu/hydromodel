"""
Author: Wenyu Ouyang
Date: 2022-11-19 17:27:05
LastEditTime: 2022-11-19 17:48:35
LastEditors: Wenyu Ouyang
Description: the script to preprocess data for models in hydro-model-xaj
FilePath: \hydro-model-xaj\hydromodel\app\datapreprocess4calibrate.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import argparse
import sys
import os
from pathlib import Path
import hydrodataset

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
import definitions
from hydromodel.data.data_preprocess import (
    trans_camels_format_to_xaj_format,
    split_train_test,
)


def main(args):
    exp = args.exp
    train_period = args.calibrate_period
    test_period = args.test_period
    basin_ids = args.basin_id
    camels_data_dir = hydrodataset.ROOT_DIR
    # where_save_cache = hydrodataset.CACHE_DIR
    where_save_cache = Path(
        os.path.join(definitions.ROOT_DIR, "hydromodel", "example", exp)
    )
    if os.path.exists(where_save_cache) is False:
        os.makedirs(where_save_cache)
    json_file = where_save_cache.joinpath("data_info.json")
    npy_file = where_save_cache.joinpath("basins_lump_p_pe_q.npy")

    trans_camels_format_to_xaj_format(
        camels_data_dir.joinpath("camels", "camels_cc"),
        basin_ids,
        [train_period[0], test_period[1]],
        json_file,
        npy_file,
    )
    split_train_test(json_file, npy_file, train_period, test_period)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for hydro-model-xaj models.")
    parser.add_argument(
        "--exp",
        dest="exp",
        help="An exp is corresponding to one data setting",
        default="exp001",
        type=str,
    )
    parser.add_argument(
        "--calibrate_period",
        dest="calibrate_period",
        help="The period for calibrating",
        default=["2014-10-01", "2020-10-01"],
        nargs="+",
    )
    parser.add_argument(
        "--test_period",
        dest="test_period",
        help="The period for testing",
        default=["2019-10-01", "2021-10-01"],
        nargs="+",
    )
    parser.add_argument(
        "--basin_id",
        dest="basin_id",
        help="The basins' ids",
        default=["60668", "61561", "63002", "63007", "63486", "92354", "94560"],
        nargs="+",
    )
    the_args = parser.parse_args()
    main(the_args)
