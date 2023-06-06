"""
Author: Wenyu Ouyang
Date: 2023-06-03 11:05:28
LastEditTime: 2023-06-05 20:55:46
LastEditors: Wenyu Ouyang
Description: Script to run dpl4xaj
FilePath: \hydro-model-xaj\scripts\run_dpl4xaj.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os
import sys
from omegaconf import DictConfig
import hydra

project_dir = os.path.abspath("")
# import the module using a relative path
sys.path.append(project_dir)
from hydromodel.calibrate.train_dpl import train_model


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train_dpl(cfg: DictConfig):
    train_model(cfg)


if __name__ == "__main__":
    train_dpl()
