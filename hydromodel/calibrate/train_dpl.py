"""
Author: Wenyu Ouyang
Date: 2023-06-03 17:21:19
LastEditTime: 2023-06-05 21:39:03
LastEditors: Wenyu Ouyang
Description: Function to train and test a Model
FilePath: \hydro-model-xaj\hydromodel\calibrate\train_dpl.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import random
import HydroErr as he
import numpy as np
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from hydromodel.data.data_cache import cache_data_source
from hydromodel.data.data_dict import data_sources_dict, pytorch_dataset_dict
from hydromodel.data.data_sets import CamelsDataset, load_streamflow
from hydromodel.models.model_func_dict import (
    pytorch_model_dict,
    pytorch_opt_dict,
    pytorch_loss_dict,
)


def set_random_seed(seed):
    """
    Set a random seed to guarantee the reproducibility

    Parameters
    ----------
    seed
        a number

    Returns
    -------
    None
    """
    print("Random seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, optimizer, loader, loss_func, epoch, accelerator):
    """Train model for a single epoch"""
    # set model to train mode (important for dropout)
    model.train()
    pbar = tqdm(loader)
    pbar.set_description(f"Epoch {epoch}")
    # request mini-batch of data from the loader
    for xs, ys in pbar:
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # get model predictions
        y_hat = model(xs)
        # calculate loss
        loss = loss_func(y_hat, ys)
        # calculate gradients
        # loss.backward()
        # we use huggingface accelerate
        accelerator.backward(loss)
        # update the weights
        optimizer.step()
        # write current loss in the progress bar
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")


def eval_model(model, loader):
    """Evaluate the model"""
    # set model to eval mode (important for dropout)
    model.eval()
    obs = []
    preds = []
    # in inference mode, we don't need to store intermediate steps for
    # backprob
    with torch.no_grad():
        # request mini-batch of data from the loader
        for xs, ys in loader:
            # get model predictions
            y_hat = model(xs)
            obs.append(ys)
            preds.append(y_hat)
    # tensor is sequence-first, so the default cat dimension is 1
    cat_dim = 1
    return torch.cat(obs, dim=cat_dim), torch.cat(preds, dim=cat_dim)


def get_dataloaders(cfg, chosen_basins, streamflow_ds, forcing_ds, attrs):
    train_times = OmegaConf.to_container(cfg.data_params.t_range_train)
    valid_times = OmegaConf.to_container(cfg.data_params.t_range_valid)
    chosen_forcing_vars = OmegaConf.to_container(cfg.data_params.relevant_cols)
    chosen_attrs_vars = OmegaConf.to_container(cfg.data_params.constant_cols)
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

    def collate_fn_seq_first(a_batch_lst):
        the_batch = default_collate(a_batch_lst)
        # Transpose the data to have sequence dimension as the first dimension
        return [the_batch[0].transpose(0, 1), the_batch[1].transpose(0, 1)]

    # Length of the meteorological record provided to the network
    sequence_length = cfg.training_params.seq_length
    batch_size = cfg.training_params.batch_size
    # Training data
    ds_train = pytorch_dataset_dict[cfg.training_params.dataset](
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

    batch_first = cfg.training_params.batch_first
    if batch_first:
        tr_loader = DataLoader(
            ds_train, batch_size=batch_size, shuffle=True, collate_fn=default_collate
        )
    else:
        tr_loader = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_seq_first,
        )
    # Validation data
    means = ds_train.get_means()
    stds = ds_train.get_stds()
    ds_val = pytorch_dataset_dict[cfg.training_params.dataset](
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
    valid_batch_size = cfg.training_params.val_batch_size
    if batch_first:
        val_loader = DataLoader(
            ds_val,
            batch_size=valid_batch_size,
            shuffle=False,
            collate_fn=default_collate,
        )
    else:
        val_loader = DataLoader(
            ds_val,
            batch_size=valid_batch_size,
            shuffle=False,
            collate_fn=collate_fn_seq_first,
        )
    return tr_loader, val_loader


def train_model(config):
    seed = int(config.training_params.random_seed)
    set_random_seed(seed)
    data_source_name = config.data_params.data_source_name
    data_source = data_sources_dict[data_source_name]()
    streamflow_ds, forcing_ds, attrs = cache_data_source(data_source)
    basins = config.data_params.object_ids
    # Initialize accelerator
    accelerator = Accelerator()
    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    num_epochs = int(config.training_params.train_epoch)
    train_dataloader, eval_dataloader = get_dataloaders(
        config, basins, streamflow_ds, forcing_ds, attrs
    )
    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    model = pytorch_model_dict[config.model_params["model_name"]](
        **config.model_params["model_param"]
    )
    optimizer = pytorch_opt_dict[config.training_params["optimizer"]](
        model.parameters(), **config.training_params["optim_params"]
    )
    loss_func = pytorch_loss_dict[config.training_params["loss_func"]]()
    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Now we train the model
    for epoch in range(num_epochs):
        train_epoch(model, optimizer, train_dataloader, loss_func, epoch, accelerator)
        obs, preds = eval_model(model, eval_dataloader)
        preds = eval_dataloader.dataset.local_denormalization(
            preds.cpu().numpy(), variable="streamflow"
        )
        obs = obs.cpu().numpy().reshape(2, -1)
        preds = preds.reshape(2, -1)
        nse = np.array([he.nse(preds[i], obs[i]) for i in range(obs.shape[0])])
        tqdm.write(f"epoch {epoch} -- Validation NSE mean: {nse.mean():.2f}")
