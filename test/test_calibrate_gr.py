"""Tests for GR model calibration scripts (calibrate_gr.py, evaluate_gr.py).

These tests verify that the GR calibration/evaluation scripts work with
the unified config-driven API.
"""
import os
import subprocess
import sys
import pytest
from pathlib import Path


@pytest.fixture
def example_gr_config(tmp_path):
    """Create a minimal GR calibration config for testing."""
    import yaml
    config = {
        "data_cfgs": {
            "dataset": "camels_us",
            "source": "local",
            "basin_ids": ["12025000"],
            "warmup_length": 30,
            "variables": ["precipitation", "potential_evapotranspiration", "streamflow"],
            "train_period": ["1990-01-01", "1995-12-31"],
            "test_period": ["1996-01-01", "2000-12-31"],
        },
        "model_cfgs": {
            "name": "gr4j",
        },
        "training_cfgs": {
            "algorithm": "SCE_UA",
            "SCE_UA": {"rep": 10, "ngs": 5},
            "loss": "RMSE",
            "output_dir": str(tmp_path / "results"),
            "experiment_name": "test_gr_cal",
            "save_config": False,
        },
    }
    config_path = tmp_path / "example_gr_config.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


class TestCalibrateGR:
    """Tests for calibrate_gr.py."""

    def test_help(self):
        """Script should show help without error."""
        result = subprocess.run(
            [sys.executable, "scripts/calibrate_gr.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower()

    def test_dry_run(self, example_gr_config, tmp_path):
        """Script should pass validation with a valid config (dry-run mode)."""
        result = subprocess.run(
            [sys.executable, "scripts/calibrate_gr.py",
             "--config", str(example_gr_config),
             "--dry-run"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "validated" in result.stdout.lower()
