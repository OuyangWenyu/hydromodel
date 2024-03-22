import numpy as np
import pytest

from hydromodel.models.xaj import xaj, uh_gamma, uh_conv


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


def test_uh():
    uh_from_gamma = np.tile(1, (5, 3, 1))
    # uh_from_gamma = np.arange(15).reshape(5, 3, 1)
    rf = np.arange(30).reshape(10, 3, 1) / 100
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
        qs[:, :, 0],
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
        np.tile([0.5], (1, 16)),
        warmup_length=warmup_length,
        name="xaj_mz",
        source_book="HF",
        source_type="sources",
    )
    np.testing.assert_array_equal(qsim.shape[0], p_and_e.shape[0] - warmup_length)
