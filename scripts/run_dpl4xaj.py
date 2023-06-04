"""
Author: Wenyu Ouyang
Date: 2023-06-03 11:05:28
LastEditTime: 2023-06-04 09:36:46
LastEditors: Wenyu Ouyang
Description: Script to run dpl4xaj
FilePath: /hydro-model-xaj/scripts/run_dpl4xaj.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import os
import sys
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra
from hydrodataset import Camels
import xarray as xr
from torch.utils.data import DataLoader


project_dir = os.path.abspath("")
# import the module using a relative path
sys.path.append(project_dir)
from hydromodel.data.data_sets import CamelsDataset, load_streamflow
from hydromodel.calibrate.train_dpl import train_model
from hydromodel.models.dpl4xaj import DplLstmXaj


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def my_app(cfg: DictConfig):
    basins_num = 2
    camels_us = Camels()
    data_dir = camels_us.data_source_dir
    try:
        streamflow_ds = xr.open_dataset(data_dir.joinpath("camels_streamflow.nc"))
        forcing_ds = xr.open_dataset(data_dir.joinpath("camels_daymet_forcing.nc"))
        attrs = pd.read_feather(data_dir.joinpath("camels_attributes_v2.0.feather"))
    except FileNotFoundError:
        print("cache downloaded data to nc and feather files firstly.")
        camels_us.cache_attributes_feather()
        camels_us.cache_forcing_xrdataset()
        camels_us.cache_streamflow_xrdataset()
        streamflow_ds = xr.open_dataset(data_dir.joinpath("camels_streamflow.nc"))
        forcing_ds = xr.open_dataset(data_dir.joinpath("camels_daymet_forcing.nc"))
        attrs = pd.read_feather(data_dir.joinpath("camels_attributes_v2.0.feather"))
    chosen_basins = camels_us.camels_sites["gauge_id"][:basins_num].values
    train_times = cfg.train_times
    valid_times = cfg.valid_times
    chosen_forcing_vars = cfg.chosen_forcing_vars
    chosen_attrs_vars = cfg.chosen_attrs_vars
    # 需要的属性
    chosen_attrs = attrs[attrs["gauge_id"].isin(chosen_basins)][
        ["gauge_id"] + chosen_attrs_vars
    ]
    chosen_attrs = chosen_attrs.set_index("gauge_id")
    # 需要的气象时序数据
    train_forcings = forcing_ds[chosen_forcing_vars].sel(
        basin=chosen_basins, time=slice(train_times[0], train_times[1])
    )
    valid_forcings = forcing_ds[chosen_forcing_vars].sel(
        basin=chosen_basins, time=slice(valid_times[0], valid_times[1])
    )
    # 需要的径流数据
    # NOTE： 这里把径流单位转换为 mm/day
    train_flow = load_streamflow(
        streamflow_ds, chosen_attrs, chosen_basins, train_times
    )
    valid_flow = load_streamflow(
        streamflow_ds, chosen_attrs, chosen_basins, valid_times
    )
    # settings
    input_size = len(chosen_attrs_vars) + len(chosen_forcing_vars)
    hidden_size = 10  # Number of LSTM cells
    dropout_rate = 0.0  # Dropout rate of the final fully connected Layer [0.0, 1.0]
    learning_rate = 1e-3  # Learning rate used to update the weights
    sequence_length = 100  # Length of the meteorological record provided to the network
    batch_size = 32

    # Training data
    ds_train = CamelsDataset(
        basins=chosen_basins,
        dates=train_times,
        data_attr=chosen_attrs,
        data_forcing=train_forcings,
        data_flow=train_flow,
        loader_type="train",
        seq_length=sequence_length,
        means=None,
        stds=None,
    )
    tr_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    # Validation data
    means = ds_train.get_means()
    stds = ds_train.get_stds()
    ds_val = CamelsDataset(
        basins=chosen_basins,
        dates=valid_times,
        data_attr=chosen_attrs,
        data_forcing=valid_forcings,
        data_flow=valid_flow,
        loader_type="valid",
        seq_length=sequence_length,
        means=means,
        stds=stds,
    )
    valid_batch_size = 1000
    val_loader = DataLoader(ds_val, batch_size=valid_batch_size, shuffle=False)

    # Here we create our model, feel free
    model = DplLstmXaj(
        input_size=input_size, hidden_size=hidden_size, dropout_rate=dropout_rate
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()
    n_epochs = 2  # Number of training epochs

    train_model(
        basins_num, tr_loader, ds_val, val_loader, model, optimizer, loss_func, n_epochs
    )


if __name__ == "__main__":
    my_app()
