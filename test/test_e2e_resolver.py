"""End-to-end integration test for the unified data resolution pipeline.

Requires real CAMELS-US data on disk (configured via .hydro_setting.yml).
Skipped automatically when the data is not available on the current machine.
"""

import pytest
from pathlib import Path

from hydromodel.configs.data_resolver import (
    resolve_config,
    DatasetResolutionError,
)
from hydromodel.configs.config_manager import validate_config
from hydromodel.datasets.unified_data_loader import UnifiedDataLoader
from hydromodel.trainers.unified_calibrate import calibrate


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def e2e_raw_config(tmp_path_factory):
    """Minimal calibration config targeting CAMELS-US basin 12025000."""
    out_dir = tmp_path_factory.mktemp("e2e_output")
    return {
        "data_cfgs": {
            "dataset": "camels_us",
            "source": "local",
            "basin_ids": ["12025000"],
            "warmup_length": 365,
            "variables": [
                "precipitation",
                "potential_evapotranspiration",
                "daylight_duration",
                "solar_radiation",
                "temperature_max",
                "temperature_min",
                "vapor_pressure",
                "streamflow",
            ],
            "train_period": ["1981-01-01", "2004-12-31"],
            "test_period": ["2010-01-01", "2014-12-31"],
        },
        "model_cfgs": {
            "name": "xaj_mz",
            "params": {
                "source_type": "sources",
                "source_book": "HF",
                "kernel_size": 15,
            },
        },
        "training_cfgs": {
            "algorithm": "SCE_UA",
            "SCE_UA": {"rep": 10, "ngs": 5},
            "loss": "RMSE",
            "output_dir": str(out_dir),
            "experiment_name": "e2e_test",
            "save_config": False,
        },
    }


@pytest.fixture(scope="module")
def e2e_config(e2e_raw_config):
    """Resolved config; skips the whole module if CAMELS-US data is absent."""
    try:
        return resolve_config(e2e_raw_config)
    except DatasetResolutionError as exc:
        pytest.skip(f"CAMELS-US data not available on this machine: {exc}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestResolverE2E:
    """End-to-end tests for unified data resolution + calibration pipeline."""

    def test_resolve_config_produces_existing_uri(self, e2e_config):
        data_cfgs = e2e_config["data_cfgs"]
        assert data_cfgs["source"] == "local"
        assert data_cfgs["reader"] == "camels_us"
        uri = data_cfgs["uri"]
        assert Path(
            uri
        ).exists(), f"Resolved URI does not exist on disk: {uri}"

    def test_validate_config_passes(self, e2e_config):
        result = validate_config(e2e_config)
        assert result["valid"], f"Config validation failed: {result['errors']}"

    def test_data_loader_returns_correct_shapes(self, e2e_config):
        loader = UnifiedDataLoader(
            e2e_config["data_cfgs"], is_train_val_test="train"
        )
        p_and_e, qobs = loader.load_data()
        # Expected shapes: [time, basin, features]
        assert p_and_e.ndim == 3, "p_and_e must be 3-D"
        assert p_and_e.shape[1] == 1, "expected exactly one basin"
        assert p_and_e.shape[2] == 2, "p_and_e must have 2 features (P, E)"
        assert qobs.ndim == 3, "qobs must be 3-D"
        assert qobs.shape[1] == 1, "expected exactly one basin"
        assert qobs.shape[2] == 1, "qobs must have 1 feature"

    @pytest.mark.slow
    def test_calibrate_returns_basin_results(self, e2e_config):
        results = calibrate(e2e_config)
        assert results, "calibrate() returned empty results"
        basin_id = list(results.keys())[0]
        r = results[basin_id]
        assert "best_params" in r, "result must contain best_params"
        assert "objective_value" in r, "result must contain objective_value"
