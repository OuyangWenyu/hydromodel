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


def _fake_basin_area() -> xr.Dataset:
    area = xr.Dataset({"area": (("basin",), [AREA])}, coords={"basin": ["basin_001"]})
    area["area"].attrs["units"] = "km^2"
    return area

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


class TestSimulateUnitConversion:
    """Exercises UnifiedSimulator.trans_sim_results_unit."""

    def _simulator(self, basin):
        from hydromodel.trainers.unified_simulate import UnifiedSimulator

        u = UnifiedSimulator.__new__(UnifiedSimulator)
        u.basin = basin
        return u

    def test_no_conversion_when_basin_none(self):
        u = self._simulator(None)
        conv, meta = u.trans_sim_results_unit(
            np.ones((3, 1, 1)), output_unit="mm", time_interval_hours=3
        )
        assert conv is None
        assert meta["applied"] is False

    def test_int_time_interval_hours_accepted(self):
        class FakeBasin:
            unit_areas = np.array([AREA])

        u = self._simulator(FakeBasin())
        conv, meta = u.trans_sim_results_unit(
            np.ones((3, 1, 1)), output_unit="m^3/s", time_interval_hours=3
        )
        # 1 mm/3h over 1000 km^2 = 92.59 m^3/s
        assert conv[0, 0, 0] == pytest.approx(92.59, rel=1e-3)
        assert meta["applied"] is True

    def test_sub_hourly_scales_to_mm_h(self):
        class FakeBasin:
            unit_areas = np.array([AREA])

        u = self._simulator(FakeBasin())
        conv, _ = u.trans_sim_results_unit(
            np.ones((3, 1, 1)), output_unit="m^3/s", time_interval_hours=0.5
        )
        # 1 mm/30min scaled to mm/h (x2) over 1000 km^2 = 555.56 m^3/s
        assert conv[0, 0, 0] == pytest.approx(555.56, rel=1e-3)


class TestEvaluatorConversion:
    """Exercises Evaluator._convert_streamflow_units end-to-end."""

    def _convert(self, times, obs_units, monkeypatch):
        from hydromodel.trainers.evaluate import Evaluator

        import hydromodel.trainers.evaluate as ev_module

        ev = Evaluator.__new__(Evaluator)
        ev.data_type = "test"
        ev.data_dir = "test"
        monkeypatch.setattr(
            ev_module, "get_basin_area", lambda *a, **k: _fake_basin_area()
        )

        test_data = xr.Dataset(
            {"flow": (("time", "basin"), np.ones((len(times), 1)))},
            coords={"time": times, "basin": ["basin_001"]},
        )
        test_data["flow"].attrs["units"] = obs_units
        qsim = np.ones((len(times), 1))
        etsim = np.ones((len(times), 1))
        ds_simflow, ds_obsflow, ds_et = ev._convert_streamflow_units(
            test_data, qsim, etsim
        )
        return ds_simflow["flow"].values[0, 0]

    def test_3h_data_scales_to_mm_h(self, monkeypatch):
        times = np.array(
            ["2010-01-01 00:00", "2010-01-01 03:00", "2010-01-01 06:00"],
            dtype="datetime64[ns]",
        )
        # 1 mm/3h scaled to mm/h (1/3) over 1000 km^2 = 92.59 m^3/s
        assert self._convert(times, "m^3/s", monkeypatch) == pytest.approx(
            92.59, rel=1e-3
        )

    def test_daily_obs_unit_preserved(self, monkeypatch):
        times = np.array(
            ["2010-01-01", "2010-01-02", "2010-01-03"], dtype="datetime64[ns]"
        )
        # obs mm/d (CAMELS US) -> sim stays mm/d -> 11.57 m^3/s
        assert self._convert(times, "mm/d", monkeypatch) == pytest.approx(
            11.574, rel=1e-3
        )
