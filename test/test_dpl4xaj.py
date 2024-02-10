"""
Author: Wenyu Ouyang
Date: 2023-06-02 17:17:51
LastEditTime: 2023-06-02 17:21:47
LastEditors: Wenyu Ouyang
Description: Test for dpl4xaj
FilePath: /hydro-model-xaj/test/test_dpl4xaj.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import numpy as np
import pytest

import torch

from hydromodel.models.dpl4xaj import DplLstmXaj
from hydromodel.models.dpl_basic import uh_conv, uh_gamma


@pytest.fixture()
def device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture()
def dpl(device):
    dpl_ = DplLstmXaj(5, 15, 64, kernel_size=15, warmup_length=5)
    return dpl_.to(device)


def test_dpl_lstm_xaj(device, dpl):
    # sequence-first tensor: time_sequence, batch, feature_size (assume that they are p, pet, srad, tmax, tmin)
    x = torch.rand(20, 10, 5).to(device)
    z = torch.rand(20, 10, 5).to(device)
    q = dpl(x, z)
    assert len(q.shape) == 3
    assert q.shape == (15, 10, 1)
    assert type(q) == torch.Tensor


def test_uh_gamma():
    # batch = 10
    tempa = torch.Tensor(np.full(10, [2.5]))
    tempb = torch.Tensor(np.full(10, [3.5]))
    # repeat for 20 periods and add one dim as feature: time_seq, batch, feature
    routa = tempa.repeat(20, 1).unsqueeze(-1)
    routb = tempb.repeat(20, 1).unsqueeze(-1)
    uh = uh_gamma(routa, routb, len_uh=15)
    np.testing.assert_almost_equal(
        uh.numpy()[:, 0, :],
        np.array(
            [
                [0.0069],
                [0.0314],
                [0.0553],
                [0.0738],
                [0.0860],
                [0.0923],
                [0.0939],
                [0.0919],
                [0.0875],
                [0.0814],
                [0.0744],
                [0.0670],
                [0.0597],
                [0.0525],
                [0.0459],
            ]
        ),
        decimal=3,
    )


def test_uh():
    uh_from_gamma = torch.full((5, 3, 1), 1.0)
    rf = torch.Tensor(np.arange(30).reshape(10, 3, 1) / 100)
    qs = uh_conv(rf, uh_from_gamma)
    np.testing.assert_almost_equal(
        np.array(
            [
                [0.0000, 0.0100, 0.0200],
                [0.0300, 0.0500, 0.0700],
                [0.0900, 0.1200, 0.1500],
                [0.1800, 0.2200, 0.2600],
                [0.3000, 0.3500, 0.4000],
                [0.4500, 0.5000, 0.5500],
                [0.6000, 0.6500, 0.7000],
                [0.7500, 0.8000, 0.8500],
                [0.9000, 0.9500, 1.0000],
                [1.0500, 1.1000, 1.1500],
            ]
        ),
        qs.numpy()[:, :, 0],
        decimal=3,
    )
