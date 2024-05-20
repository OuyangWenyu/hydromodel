import os
import numpy as np
import pytest

from hydrodataset import Camels
from hydromodel import SETTING


@pytest.fixture()
def warmup_length():
    return 30


@pytest.fixture()
def camels():
    # for methods testing, we simply use the CAMELS dataset
    root_dir = SETTING["local_data_path"]["datasets-origin"]
    return Camels(os.path.join(root_dir, "camels", "camels_us"))


@pytest.fixture()
def basins():
    return ["01013500"]


@pytest.fixture()
def basin_area(camels, basins):
    return camels.read_area(basins)


@pytest.fixture()
def p_and_e(camels, basins):
    p_and_e = camels.read_ts_xrdataset(
        basins, ["2010-01-01", "2014-01-01"], ["prcp", "PET"]
    )
    # three dims: sequence (time), batch (basin), feature (variable)
    return p_and_e.to_array().to_numpy().transpose(2, 1, 0)


@pytest.fixture()
def qobs(basin_area, camels, basins):
    import pint_xarray  # noqa

    qobs_ = camels.read_ts_xrdataset(
        basins, ["2010-01-01", "2014-01-01"], ["streamflow"]
    )
    # we use pint package to handle the unit conversion
    # trans unit to mm/time_interval
    basin_area = basin_area.pint.quantify()
    qobs = qobs_.pint.quantify()
    target_unit = "mm/d"
    r = qobs["streamflow"] / basin_area["area_gages2"]
    r_mmd = r.pint.to(target_unit)
    return np.expand_dims(r_mmd.to_numpy().transpose(1, 0), axis=2)
