"""
Author: Wenyu Ouyang
Date: 2024-08-14 16:34:32
LastEditTime: 2024-09-17 14:32:47
LastEditors: Wenyu Ouyang
Description: Some common fixtures for testing
FilePath: \hydromodel\test\conftest.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

# ═══════════════════════════════════════════════════════════════════════════
# Test isolation: redirect CACHE_DIR to a temp directory BEFORE any
# hydrodatasource / hydrodataset lazy init fires.  This prevents tests from
# reading or writing production cache files in E:\data\.cache.
# ═══════════════════════════════════════════════════════════════════════════
import atexit
import os
import shutil
import tempfile

_TEST_CACHE_DIR = tempfile.mkdtemp(prefix="hydromodel_test_cache_")


def _cleanup_test_cache() -> None:
    shutil.rmtree(_TEST_CACHE_DIR, ignore_errors=True)


atexit.register(_cleanup_test_cache)

import hydrodatasource.configs.config as _ds_config  # noqa: E402

# Run the full lazy init (LOCAL_ROOT, FS, MINIO_PARAM, SETTING), then
# override only CACHE_DIR so tests never touch the production cache.
_ds_config._init_settings()
_ds_config._lazy["CACHE_DIR"] = _TEST_CACHE_DIR

# ── normal imports (after cache isolation) ────────────────────────────────
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import spotpy  # noqa: E402
from spotpy.examples.spot_setup_hymod_python import spot_setup  # noqa: E402

from hydrodataset.camels_us import CamelsUs as _CamelsUs  # noqa: E402
from hydrodataset.hydro_dataset import StandardVariable  # noqa: E402
from hydromodel.configs.data_resolver import resolve_data_cfgs  # noqa: E402


@pytest.fixture()
def warmup_length():
    return 30


@pytest.fixture()
def camels():
    """Provide a CamelsUs datasource using the unified data resolution."""
    resolved = resolve_data_cfgs({"dataset": "camels_us"})
    return _CamelsUs(resolved["uri"])


@pytest.fixture()
def CamelsUs(camels):
    return camels


@pytest.fixture()
def basins():
    return ["01013500"]


@pytest.fixture()
def basin_area(camels, basins):
    return camels.read_area(basins)


@pytest.fixture()
def p_and_e(camels, basins):
    p_and_e = camels.read_ts_xrdataset(
        basins, ["2010-01-01", "2014-01-01"],
        [StandardVariable.PRECIPITATION, StandardVariable.POTENTIAL_EVAPOTRANSPIRATION],
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
    r = qobs["streamflow"] / basin_area["area"]
    r_mmd = r.pint.to(target_unit)
    return np.expand_dims(r_mmd.to_numpy().transpose(1, 0), axis=2)


@pytest.fixture(scope="session")
def hymod_setup():
    """
    A pytest fixture that runs the hymod calibration and returns the results.
    This will run before any test that requires it.
    """
    setup = spot_setup(spotpy.objectivefunctions.rmse)
    if not os.path.exists("test/SCEUA_hymod.csv"):
        # Set up the hymod model and sampler

        # Create SCE-UA sampler
        sampler = spotpy.algorithms.sceua(
            setup, dbname="test/SCEUA_hymod", dbformat="csv"
        )

        # Calibration parameters
        repetitions = 5000  # Maximum iterations

        # Run the sampler
        sampler.sample(repetitions, ngs=7, kstop=3, peps=0.1, pcento=0.1)

    # Return the results for further use
    return setup
