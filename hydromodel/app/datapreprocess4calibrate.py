"""
Author: Wenyu Ouyang
Date: 2022-11-19 17:27:05
LastEditTime: 2022-12-02 15:14:36
LastEditors: Wenyu Ouyang
Description: the script to preprocess data for models in hydro-model-xaj
FilePath: \hydro-model-xaj\hydromodel\app\datapreprocess4calibrate.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import numpy as np
import hydrodataset
import argparse
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
import definitions
from hydromodel.data.data_preprocess import (
    trans_camels_format_to_xaj_format,
    split_train_test,
)


def main(args):
    camels_name = args.camels_name
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

    # training period can be behind test period
    periods = np.sort(
        [train_period[0], train_period[1], test_period[0], test_period[1]]
    )
    trans_camels_format_to_xaj_format(
        camels_data_dir.joinpath("camels", camels_name),
        basin_ids,
        [periods[0], periods[-1]],
        json_file,
        npy_file,
    )
    split_train_test(json_file, npy_file, train_period, test_period)


# python datapreprocess4calibrate.py --camels_name camels_cc --exp exp003 --calibrate_period 2014-10-01 2019-10-01 --test_period 2018-10-01 2021-10-01 --basin_id 60668 61561 63002 63007 63486 92354 94560
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for hydro-model-xaj models."
    )
    parser.add_argument(
        "--camels_name",
        dest="camels_name",
        help="the name of CAMELS-formatted data directory",
        # default="camels_cc",
        default="camels_us",
        type=str,
    )
    parser.add_argument(
        "--exp",
        dest="exp",
        help="An exp is corresponding to one data setting",
        default="camels4basins",
        type=str,
    )
    parser.add_argument(
        "--calibrate_period",
        dest="calibrate_period",
        help="The period for calibrating",
        # default=["2014-10-01", "2020-10-01"],
        default=["2007-01-01", "2014-01-01"],
        nargs="+",
    )
    parser.add_argument(
        "--test_period",
        dest="test_period",
        help="The period for testing",
        # default=["2019-10-01", "2021-10-01"],
        default=["2001-01-01", "2008-01-01"],
        nargs="+",
    )
    parser.add_argument(
        "--basin_id",
        dest="basin_id",
        help="The basins' ids",
        # default=["60668", "61561", "63002", "63007", "63486", "92354", "94560"],
        default=["01439500", "06885500", "08104900", "09510200"],
        nargs="+",
    )
    the_args = parser.parse_args()
    main(the_args)
