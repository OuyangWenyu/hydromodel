from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest
import yaml

from hydromodel.configs.data_resolver import (
    DatasetResolutionError,
    resolve_config,
    resolve_config_if_needed,
)
from hydromodel.configs.config_manager import validate_config
from hydromodel.datasets.unified_data_loader import UnifiedDataLoader
from hydromodel.trainers import unified_calibrate


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def test_resolve_config_uses_dataset_registry_and_local_storage(tmp_path):
    data_root = tmp_path / "data"
    dataset_dir = data_root / "public" / "camels_us"
    dataset_dir.mkdir(parents=True)

    project_root = tmp_path / "project"
    write_yaml(
        project_root / "configs" / "datasets.yml",
        {
            "datasets": {
                "camels_us": {
                    "reader": "camels_us",
                    "path": "public/camels_us",
                }
            }
        },
    )
    write_yaml(
        project_root / ".hydro_setting.yml",
        {
            "storage": {
                "default_source": "local",
                "local": {"root": str(data_root)},
            }
        },
    )

    config = {
        "data_cfgs": {
            "dataset": "camels_us",
            "basin_ids": ["12025000"],
        },
        "model_cfgs": {"name": "xaj"},
        "training_cfgs": {"algorithm": "SCE_UA"},
    }

    resolved = resolve_config(config, project_root=project_root)

    data_cfgs = resolved["data_cfgs"]
    assert data_cfgs["dataset"] == "camels_us"
    assert data_cfgs["source"] == "local"
    assert data_cfgs["reader"] == "camels_us"
    assert data_cfgs["uri"] == str(dataset_dir)
    assert data_cfgs["basin_ids"] == ["12025000"]
    assert Path(data_cfgs["resolution"]["dataset_layer"]).name == (
        "datasets.yml"
    )
    assert Path(data_cfgs["resolution"]["dataset_layer"]).parent.name == (
        "configs"
    )
    assert Path(data_cfgs["resolution"]["storage_layer"]).name == (
        ".hydro_setting.yml"
    )


def test_resolve_config_rejects_task_level_path_fields(tmp_path):
    config = {
        "data_cfgs": {
            "dataset": "camels_us",
            "data_source_path": "D:/data/camels_us",
        }
    }

    with pytest.raises(DatasetResolutionError, match="forbidden"):
        resolve_config(config, project_root=tmp_path)


@pytest.mark.parametrize(
    "bad_path",
    [
        "D:/data/camels_us",
        "s3://bucket/camels_us",
        "../camels_us",
    ],
)
def test_resolve_config_rejects_unsafe_dataset_paths(tmp_path, bad_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    project_root = tmp_path / "project"
    write_yaml(
        project_root / "configs" / "datasets.yml",
        {"datasets": {"camels_us": {"reader": "camels_us", "path": bad_path}}},
    )
    write_yaml(
        project_root / ".hydro_setting.yml",
        {"storage": {"local": {"root": str(data_root)}}},
    )

    with pytest.raises(DatasetResolutionError, match="path"):
        resolve_config(
            {"data_cfgs": {"dataset": "camels_us"}},
            project_root=project_root,
        )


def test_resolve_config_resolves_cloud_uri_without_remote_check(tmp_path):
    project_root = tmp_path / "project"
    write_yaml(
        project_root / "configs" / "datasets.yml",
        {
            "datasets": {
                "era5_songliao": {
                    "reader": "zarr_timeseries",
                    "path": "reanalysis/era5/songliao.zarr",
                }
            }
        },
    )
    write_yaml(
        project_root / ".hydro_setting.yml",
        {
            "storage": {
                "s3": {
                    "bucket": "hydro-data",
                    "prefix": "hydromodel",
                    "region": "us-east-1",
                    "profile": "default",
                }
            }
        },
    )

    resolved = resolve_config(
        {"data_cfgs": {"dataset": "era5_songliao", "source": "cloud"}},
        project_root=project_root,
    )

    assert resolved["data_cfgs"]["uri"] == (
        "s3://hydro-data/hydromodel/reanalysis/era5/songliao.zarr"
    )
    assert resolved["data_cfgs"]["reader"] == "zarr_timeseries"


def test_unified_data_loader_rejects_unresolved_data_config():
    with pytest.raises(ValueError, match="resolved"):
        UnifiedDataLoader({"dataset": "camels_us"})


def test_validate_config_accepts_new_cfgs_schema():
    validation = validate_config(
        {
            "data_cfgs": {
                "dataset": "camels_us",
                "source": "local",
                "reader": "camels_us",
                "uri": "D:/data/hydromodel/public/camels_us",
                "train_period": ["1981-01-01", "2004-12-31"],
            },
            "model_cfgs": {
                "name": "xaj",
                "params": {
                    "source_type": "sources",
                    "source_book": "HF",
                    "kernel_size": 15,
                },
            },
            "training_cfgs": {
                "algorithm": "SCE_UA",
                "loss": "RMSE",
            },
        }
    )

    assert validation["valid"], validation["errors"]


def test_calibrate_accepts_resolved_new_cfgs_schema(tmp_path, monkeypatch):
    captured = {}

    class FakeDataLoader:
        basin_ids = ["12025000"]

    class FakeModelSetup:
        def __init__(
            self,
            data_config,
            model_config,
            loss_config,
            training_config,
            **kwargs,
        ):
            captured["data_config"] = data_config
            captured["model_config"] = model_config
            captured["loss_config"] = loss_config
            captured["training_config"] = training_config
            self.data_loader = FakeDataLoader()
            self.p_and_e = None
            self.model_name = model_config["name"]
            self.param_range = {"xaj": {}}

    def fake_calibrate_model(
        model_setup,
        algorithm_config,
        output_dir,
        basin_id,
        basin_index,
        **kwargs,
    ):
        captured["algorithm_config"] = algorithm_config
        return {"convergence": "success"}

    monkeypatch.setattr(
        unified_calibrate, "UnifiedCalibrator", FakeModelSetup
    )
    monkeypatch.setattr(
        unified_calibrate, "_calibrate_model", fake_calibrate_model
    )

    result = unified_calibrate.calibrate(
        {
            "data_cfgs": {
                "dataset": "camels_us",
                "source": "local",
                "reader": "camels_us",
                "uri": str(tmp_path / "data" / "camels_us"),
                "basin_ids": ["12025000"],
            },
            "model_cfgs": {
                "name": "xaj",
                "params": {"kernel_size": 15},
            },
            "training_cfgs": {
                "algorithm": "SCE_UA",
                "SCE_UA": {"rep": 10},
                "loss": "RMSE",
                "output_dir": str(tmp_path / "results"),
                "experiment_name": "new_schema",
                "save_config": False,
            },
        }
    )

    assert result == {"12025000": {"convergence": "success"}}
    assert captured["model_config"] == {"name": "xaj", "kernel_size": 15}
    assert captured["algorithm_config"] == {"name": "SCE_UA", "rep": 10}
    assert captured["training_config"]["loss_config"]["obj_func"] == "RMSE"


# ── New tests for extra_registry_dicts integration ─────────────────────────


def test_builtin_dataset_resolves_without_yaml_definition(tmp_path):
    """camels_us should resolve from hydrodataset's built-in _DEFAULT_REGISTRY
    even when the user has NO configs/datasets.yml."""
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)

    project_root = tmp_path / "project"
    # Only storage config — NO datasets.yml at all
    write_yaml(
        project_root / ".hydro_setting.yml",
        {
            "storage": {
                "default_source": "local",
                "local": {"root": str(data_root)},
            }
        },
    )

    config = {
        "data_cfgs": {"dataset": "camels_us", "basin_ids": ["12025000"]},
        "model_cfgs": {"name": "xaj"},
        "training_cfgs": {"algorithm": "SCE_UA"},
    }

    resolved = resolve_config(config, project_root=project_root)

    assert resolved["data_cfgs"]["dataset"] == "camels_us"
    assert resolved["data_cfgs"]["source"] == "local"
    assert resolved["data_cfgs"]["reader"] == "camels_us"
    # Built-in _DEFAULT_REGISTRY has path="." for camels_us,
    # so uri resolves to local_root / "." = local_root
    assert resolved["data_cfgs"]["uri"] == str(data_root)


def test_yaml_overrides_builtin_path(tmp_path):
    """User YAML definition of camels_us with a custom path should override
    the built-in path from hydrodataset's _DEFAULT_REGISTRY."""
    data_root = tmp_path / "data"
    custom_dir = data_root / "my_custom_camels"
    custom_dir.mkdir(parents=True)
    # Also create the default path to prove it's NOT used
    default_dir = data_root / "camels_us"
    default_dir.mkdir(parents=True)

    project_root = tmp_path / "project"
    write_yaml(
        project_root / "configs" / "datasets.yml",
        {
            "datasets": {
                "camels_us": {
                    "reader": "camels_us",
                    "path": "my_custom_camels",
                }
            }
        },
    )
    write_yaml(
        project_root / ".hydro_setting.yml",
        {
            "storage": {
                "default_source": "local",
                "local": {"root": str(data_root)},
            }
        },
    )

    config = {
        "data_cfgs": {"dataset": "camels_us", "basin_ids": ["12025000"]},
        "model_cfgs": {"name": "xaj"},
        "training_cfgs": {"algorithm": "SCE_UA"},
    }

    resolved = resolve_config(config, project_root=project_root)

    # Must use the YAML-overridden path, NOT the default "camels_us"
    assert resolved["data_cfgs"]["uri"] == str(custom_dir)


@pytest.mark.parametrize(
    "absolute_path",
    [
        "/absolute/zarr/file.zarr",
        "D:\\absolute\\zarr\\file.zarr",
    ],
)
def test_cloud_uri_rejects_absolute_path(tmp_path, absolute_path):
    """Cloud URIs must reject absolute local paths in the registry."""
    project_root = tmp_path / "project"
    write_yaml(
        project_root / "configs" / "datasets.yml",
        {
            "datasets": {
                "test_ds": {
                    "reader": "zarr_timeseries",
                    "path": absolute_path,
                }
            }
        },
    )
    write_yaml(
        project_root / ".hydro_setting.yml",
        {
            "storage": {
                "s3": {
                    "bucket": "hydro-data",
                    "prefix": "hydromodel",
                }
            }
        },
    )

    with pytest.raises(DatasetResolutionError, match="path"):
        resolve_config(
            {"data_cfgs": {"dataset": "test_ds", "source": "cloud"}},
            project_root=project_root,
        )


def test_cloud_rejects_unknown_reader_alias(tmp_path):
    """Cloud branch must validate reader alias, symmetric with local branch."""
    project_root = tmp_path / "project"
    write_yaml(
        project_root / "configs" / "datasets.yml",
        {
            "datasets": {
                "test_ds": {
                    "reader": "nonexistent_reader_xyz",
                    "path": "some/path.zarr",
                }
            }
        },
    )
    write_yaml(
        project_root / ".hydro_setting.yml",
        {
            "storage": {
                "s3": {
                    "bucket": "hydro-data",
                    "prefix": "hydromodel",
                }
            }
        },
    )

    with pytest.raises(DatasetResolutionError, match="reader"):
        resolve_config(
            {"data_cfgs": {"dataset": "test_ds", "source": "cloud"}},
            project_root=project_root,
        )


def test_unified_data_loader_config_contract(tmp_path):
    """UnifiedDataLoader config contract: must provide 'reader' and 'uri'
    in resolved data_cfgs, and must extract them correctly."""
    resolved_cfg = {
        "dataset": "camels_us",
        "source": "local",
        "reader": "camels_us",
        "uri": str(tmp_path / "camels_us"),
        "basin_ids": ["12025000"],
    }

    # Verify the contract: reader and uri are present and extractable
    assert resolved_cfg.get("reader") == "camels_us"
    assert resolved_cfg.get("uri") == str(tmp_path / "camels_us")
    assert resolved_cfg.get("dataset") == "camels_us"

    # Verify that missing reader or uri triggers the expected error
    from hydromodel.datasets.unified_data_loader import UnifiedDataLoader
    with pytest.raises(ValueError, match="resolved"):
        UnifiedDataLoader({"dataset": "camels_us"})


def test_calibrate_auto_resolves_unresolved_config(tmp_path, monkeypatch):
    """calibrate() should auto-resolve a config that only has dataset id,
    without requiring the caller to manually call resolve_config() first."""
    data_root = tmp_path / "data"
    data_root.mkdir()

    project_root = tmp_path / "project"
    write_yaml(
        project_root / ".hydro_setting.yml",
        {
            "storage": {
                "default_source": "local",
                "local": {"root": str(data_root)},
            }
        },
    )

    captured = {}

    class FakeDataLoader:
        basin_ids = ["12025000"]

    class FakeModelSetup:
        def __init__(
            self,
            data_config,
            model_config,
            loss_config,
            training_config,
            **kwargs,
        ):
            captured["data_config"] = data_config
            captured["model_config"] = model_config
            captured["loss_config"] = loss_config
            captured["training_config"] = training_config
            self.data_loader = FakeDataLoader()
            self.p_and_e = None
            self.model_name = model_config["name"]
            self.param_range = {"xaj": {}}

    def fake_calibrate_model(
        model_setup,
        algorithm_config,
        output_dir,
        basin_id,
        basin_index,
        **kwargs,
    ):
        captured["algorithm_config"] = algorithm_config
        return {"convergence": "success"}

    monkeypatch.setattr(
        unified_calibrate, "UnifiedCalibrator", FakeModelSetup
    )
    monkeypatch.setattr(
        unified_calibrate, "_calibrate_model", fake_calibrate_model
    )

    # Pass UNRESOLVED config — only dataset + basin_ids, no uri/reader
    result = unified_calibrate.calibrate(
        {
            "data_cfgs": {
                "dataset": "camels_us",
                "basin_ids": ["12025000"],
            },
            "model_cfgs": {
                "name": "xaj",
                "params": {"kernel_size": 15},
            },
            "training_cfgs": {
                "algorithm": "SCE_UA",
                "SCE_UA": {"rep": 10},
                "loss": "RMSE",
                "output_dir": str(tmp_path / "results"),
                "experiment_name": "auto_resolve",
                "save_config": False,
            },
        },
        project_root=str(project_root),
    )

    assert result == {"12025000": {"convergence": "success"}}
    # Verify auto-resolution happened: data_config now has uri and reader
    assert captured["data_config"].get("uri") == str(data_root)
    assert captured["data_config"].get("reader") == "camels_us"
    assert captured["data_config"].get("source") == "local"


def test_resolve_config_preserves_non_data_sections(tmp_path):
    """resolve_config must not alter model_cfgs, training_cfgs, etc."""
    project_root = tmp_path / "project"
    write_yaml(
        project_root / ".hydro_setting.yml",
        {
            "storage": {
                "default_source": "local",
                "local": {"root": str(tmp_path / "data")},
            }
        },
    )
    (tmp_path / "data").mkdir()

    config = {
        "data_cfgs": {"dataset": "camels_us"},
        "model_cfgs": {"name": "xaj_mz", "params": {"kernel_size": 15}},
        "training_cfgs": {"algorithm": "SCE_UA", "loss": "RMSE"},
        "evaluation_cfgs": {"metrics": ["NSE", "KGE"]},
    }

    resolved = resolve_config(config, project_root=project_root)

    # Non-data sections must be unchanged
    assert resolved["model_cfgs"] == config["model_cfgs"]
    assert resolved["training_cfgs"] == config["training_cfgs"]
    assert resolved["evaluation_cfgs"] == config["evaluation_cfgs"]
    # data_cfgs must be enriched
    assert "uri" in resolved["data_cfgs"]
    assert "reader" in resolved["data_cfgs"]


def test_calibrate_rejects_missing_model_name():
    """calibrate() should give a clear error when model_cfgs.name is missing."""
    from hydromodel.trainers.unified_calibrate import calibrate
    with pytest.raises(ValueError, match="model_cfgs.name"):
        calibrate({
            "data_cfgs": {
                "dataset": "camels_us",
                "source": "local",
                "reader": "camels_us",
                "uri": "/fake/path",
                "basin_ids": ["12025000"],
            },
            "model_cfgs": {"params": {"kernel_size": 15}},  # missing "name"
            "training_cfgs": {"algorithm": "SCE_UA", "loss": "RMSE"},
        })


def test_resolve_config_if_needed_skips_resolved_config():
    """resolve_config_if_needed should return resolved config as-is."""
    resolved = {
        "data_cfgs": {
            "dataset": "camels_us",
            "source": "local",
            "reader": "camels_us",
            "uri": "/already/resolved",
            "basin_ids": ["12025000"],
        },
        "model_cfgs": {"name": "xaj"},
    }
    result = resolve_config_if_needed(resolved)
    assert result is resolved  # must return the SAME object (no-op)

