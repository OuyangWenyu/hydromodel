"""
Author: Wenyu Ouyang
Date: 2025-08-06 22:52:45
LastEditTime: 2025-08-19 10:12:54
LastEditors: Wenyu Ouyang
Description: Test XAJ model
FilePath: \hydromodel\test\test_xaj.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""
import numpy as np
import pytest

from hydromodel.models.xaj import xaj, uh_gamma


@pytest.fixture()
def params():
    # all parameters are in range [0,1]
    return np.tile([0.5], (1, 15))


def test_uh_gamma():
    # repeat for 20 periods and add one dim as feature: time_seq=20, batch=10, feature=1
    routa = np.tile(2.5, (20, 10, 1))
    routb = np.tile(3.5, (20, 10, 1))
    uh = uh_gamma(routa, routb, len_uh=15)
    np.testing.assert_almost_equal(
        uh[:, 0, :],
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


def test_xaj(p_and_e, params, warmup_length):
    qsim, e = xaj(
        p_and_e,
        params,
        warmup_length=warmup_length,
        name="xaj",
        source_book="HF",
        source_type="sources",
    )
    np.testing.assert_array_equal(qsim.shape[0], p_and_e.shape[0] - warmup_length)


def test_xaj_mz(p_and_e, params, warmup_length):
    qsim, e = xaj(
        p_and_e,
        params,
        warmup_length=warmup_length,
        name="xaj_mz",
        source_book="HF",
        source_type="sources",
    )
    np.testing.assert_array_equal(qsim.shape[0], p_and_e.shape[0] - warmup_length)
