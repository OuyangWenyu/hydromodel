'''
Author: zhuanglaihong
Date: 2025-02-18 10:20:58
LastEditTime: 2025-03-20 00:58:57
LastEditors: zhuanglaihong
Description: Script for preparing data
FilePath: /zlh/hydromodel/scripts/prepare_data.py
Copyright: Copyright (c) 2021-2024 zhuanglaihong. All rights reserved.
'''

from pathlib import Path
import sys
import os
import argparse

current_script_path = Path(os.path.realpath(__file__))
repo_root_dir = current_script_path.parent.parent
sys.path.append(str(repo_root_dir))
from hydromodel.datasets.data_preprocess import process_and_save_data_as_nc


def main(args):
    data_path = args.origin_data_dir
    origin_data_scale = args.origin_data_scale
    target_data_scale = args.target_data_scale

    if process_and_save_data_as_nc(data_path, origin_data_scale,target_data_scale, save_folder=data_path):
        print("Data is ready!")
    else:
        print("Data format is incorrect! Please check the data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data.")
    parser.add_argument(
        "--origin_data_dir",
        type=str,
        help="Path to your hydrological data foler",
        default="/home/zlh/hydromodel/data/camels_11532500",
        # default="C:\\Users\\wenyu\\Downloads\\biliuhe",
    )
    parser.add_argument(
        "--origin_data_scale",
        type=str,
        help="your origin data time scale",
        default="D",
        # default="D"or"h"
    )
    parser.add_argument(
        "--target_data_scale",
        type=str,
        help="your input data time scale",
        default="Y",
        # default="D"or"M"or"Y"
    )
    args = parser.parse_args()
    main(args)
