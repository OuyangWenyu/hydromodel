"""
Author: Wenyu Ouyang
Date: 2023-06-03 11:05:28
LastEditTime: 2023-06-04 19:38:46
LastEditors: Wenyu Ouyang
Description: Script to run dpl4xaj
FilePath: \hydro-model-xaj\scripts\run_dpl4xaj.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os
import sys
import pandas as pd
from omegaconf import DictConfig
import hydra
from hydrodataset import Camels, CACHE_DIR
import xarray as xr

project_dir = os.path.abspath("")
# import the module using a relative path
sys.path.append(project_dir)
from hydromodel.calibrate.train_dpl import train_model


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def my_app(cfg: DictConfig):
    basins_num = 2
    camels_us = Camels()
    camels_us.cache_attributes_feather()
    try:
        streamflow_ds = xr.open_dataset(CACHE_DIR.joinpath("camels_streamflow.nc"))
        forcing_ds = xr.open_dataset(CACHE_DIR.joinpath("camels_daymet_forcing.nc"))
        attrs = pd.read_feather(CACHE_DIR.joinpath("camels_attributes_v2.0.feather"))
    except FileNotFoundError:
        print("cache downloaded data to nc and feather files firstly.")
        camels_us.cache_attributes_feather()
        camels_us.cache_forcing_xrdataset()
        camels_us.cache_streamflow_xrdataset()
        streamflow_ds = xr.open_dataset(CACHE_DIR.joinpath("camels_streamflow.nc"))
        forcing_ds = xr.open_dataset(CACHE_DIR.joinpath("camels_daymet_forcing.nc"))
        attrs = pd.read_feather(CACHE_DIR.joinpath("camels_attributes_v2.0.feather"))
    chosen_basins = camels_us.camels_sites["gauge_id"][:basins_num].values

    train_model(cfg, chosen_basins, streamflow_ds, forcing_ds, attrs)


if __name__ == "__main__":
    my_app()
