"""Regression tests for streamflow unit conversion with hydroutils 0.2.0.

These tests lock in the behavior hydromodel relies on after adapting to
hydroutils 0.2.0 / hydrodataset 0.3.0:

- ``streamflow_unit_conv`` dropped the ``inverse`` parameter; direction is
  auto-detected from source/target units.
- pint-xarray (imported below, as ``test/conftest.py`` does for CAMELS data)
  cannot parse custom interval units such as ``mm/3h`` or ``mm/1d`` — so
  hydromodel emits pint-safe units (``mm/d``, ``mm/h``, ``mm/day``) and scales
  the data for other cadences.
- scalar basin area breaks the pint-xarray conversion path; area must be
  broadcast to the time dimension.
"""

import numpy as np
import pytest
import xarray as xr

import pint_xarray  # noqa: F401  (enable the .pint accessor, as conftest does)
from hydroutils.hydro_units import streamflow_unit_conv

N_TIME = 6
AREA = 1000.0  # km^2


def _make_data(unit: str, values: np.ndarray = None) -> xr.Dataset:
    data = xr.Dataset({"qsim": ("time", np.ones(N_TIME) if values is None else values)})
    data["qsim"].attrs["units"] = unit
    return data


def _broadcast_area() -> np.ndarray:
    return np.full(N_TIME, AREA)


class TestPintSafeUnits:
    """Pint-safe depth units convert correctly under pint-xarray."""

    def test_mm_per_day(self):
        r = streamflow_unit_conv(
            _make_data("mm/day"), _broadcast_area(), target_unit="m^3/s"
        )
        # 1 mm/day over 1000 km^2 = 11.574 m^3/s
        assert r["qsim"].values[0] == pytest.approx(11.574, rel=1e-3)

    def test_mm_per_d(self):
        r = streamflow_unit_conv(
            _make_data("mm/d"), _broadcast_area(), target_unit="m^3/s"
        )
        assert r["qsim"].values[0] == pytest.approx(11.574, rel=1e-3)

    def test_mm_per_h(self):
        r = streamflow_unit_conv(
            _make_data("mm/h"), _broadcast_area(), target_unit="m^3/s"
        )
        # 1 mm/h over 1000 km^2 = 277.778 m^3/s
        assert r["qsim"].values[0] == pytest.approx(277.778, rel=1e-3)


class TestCustomIntervalUnits:
    """Custom mm/Nh units are NOT parseable by pint — hydromodel must scale."""

    @pytest.mark.parametrize("unit", ["mm/1d", "mm/1D", "mm/3h", "mm/7d"])
    def test_custom_interval_units_raise(self, unit):
        with pytest.raises(ValueError):
            streamflow_unit_conv(
                _make_data(unit), _broadcast_area(), target_unit="m^3/s"
            )

    def test_scaled_3h_equivalent_to_mm_h(self):
        """mm/3h data scaled to mm/h (divide by 3) converts correctly."""
        r = streamflow_unit_conv(
            _make_data("mm/h", np.full(N_TIME, 1.0 / 3.0)),
            _broadcast_area(),
            target_unit="m^3/s",
        )
        # 1 mm per 3 hours over 1000 km^2 = 92.593 m^3/s
        assert r["qsim"].values[0] == pytest.approx(92.593, rel=1e-3)


class TestAreaHandling:
    """Scalar area breaks the pint-xarray path; broadcast area works."""

    def test_scalar_area_raises(self):
        with pytest.raises(ValueError):
            streamflow_unit_conv(_make_data("mm/day"), AREA, target_unit="m^3/s")

    def test_broadcast_area_works(self):
        r = streamflow_unit_conv(
            _make_data("mm/day"), _broadcast_area(), target_unit="m^3/s"
        )
        assert r["qsim"].values[0] == pytest.approx(11.574, rel=1e-3)
