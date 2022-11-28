"""
Author: Wenyu Ouyang
Date: 2021-07-26 08:51:23
LastEditTime: 2022-11-16 18:47:10
LastEditors: Wenyu Ouyang
Description: some configs for hydro-model-xaj
FilePath: \hydro-model-xaj\definitions.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import os
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
path = Path(ROOT_DIR)
DATASET_DIR = os.path.join(path.parent.parent.absolute(), "data")
print("Please Check your directory:")
print("ROOT_DIR of the repo: ", ROOT_DIR)
print("DATASET_DIR of the repo: ", DATASET_DIR)
