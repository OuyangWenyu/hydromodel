"""
Author: Wenyu Ouyang
Date: 2023-06-03 17:21:19
LastEditTime: 2023-06-03 20:00:29
LastEditors: Wenyu Ouyang
Description: Function to train and test a Model
FilePath: /hydro-model-xaj/hydromodel/calibrate/train_dpl.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import fnmatch
import os
import random
import HydroErr as he
import numpy as np
from typing import Dict, Tuple, Union
import pandas as pd
import torch
from tqdm import tqdm

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


def train_epoch(model, optimizer, loader, loss_func, epoch):
    """Train model for a single epoch"""
    # set model to train mode (important for dropout)
    model.train()
    pbar = tqdm.notebook.tqdm(loader)
    pbar.set_description(f"Epoch {epoch}")
    # request mini-batch of data from the loader
    for xs, ys in pbar:
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # push data to GPU (if available)
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        # get model predictions
        y_hat = model(xs)
        # calculate loss
        loss = loss_func(y_hat, ys)
        # calculate gradients
        loss.backward()
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
            # push data to GPU (if available)
            xs = xs.to(DEVICE)
            # get model predictions
            y_hat = model(xs)
            obs.append(ys)
            preds.append(y_hat)

    return torch.cat(obs), torch.cat(preds)


def train_model(
    basins_num, tr_loader, ds_val, val_loader, model, optimizer, loss_func, n_epochs
):
    for i in range(n_epochs):
        train_epoch(model, optimizer, tr_loader, loss_func, i + 1)
        obs, preds = eval_model(model, val_loader)
        preds = ds_val.local_denormalization(preds.cpu().numpy(), variable="streamflow")
        obs = obs.numpy().reshape(basins_num, -1)
        preds = preds.reshape(basins_num, -1)
        nse = np.array([he.nse(preds[i], obs[i]) for i in range(obs.shape[0])])
        tqdm.notebook.tqdm.write(f"Validation NSE mean: {nse.mean():.2f}")
