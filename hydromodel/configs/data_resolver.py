"""
Unified data path resolution for hydromodel runtime configs.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Optional, Tuple

import yaml


class DatasetResolutionError(ValueError):
    """Raised when data configuration cannot be resolved deterministically."""


from hydrodatasource.configs.data_resolver import (  # noqa: E402
    READER_ALIASES as _MERGED_READER_ALIASES,
)

READER_ALIASES: Dict[str, Dict[str, str]] = dict(_MERGED_READER_ALIASES)

# Add hydromodel-specific aliases on top
READER_ALIASES["zarr_timeseries"] = {
    "module": "hydromodel.datasets.zarr_loader",
    "class": "ZarrTimeSeriesDatasource",
    "category": "zarr",
}


FORBIDDEN_DATA_CFG_FIELDS = {
    "path",
    "data_source_path",
    "data_source_type",
}


def resolve_config(
    config: Dict[str, Any],
    *,
    project_root: Optional[os.PathLike[str] | str] = None,
    user_setting_path: Optional[os.PathLike[str] | str] = None,
    check_remote: bool = False,
) -> Dict[str, Any]:
    """Resolve data configuration for a full hydromodel config."""
    resolved = copy.deepcopy(config)
    resolved["data_cfgs"] = resolve_data_cfgs(
        resolved.get("data_cfgs"),
        project_root=project_root,
        user_setting_path=user_setting_path,
        check_remote=check_remote,
    )
    return resolved


def resolve_data_cfgs(
    data_cfgs: Dict[str, Any],
    *,
    project_root: Optional[os.PathLike[str] | str] = None,
    user_setting_path: Optional[os.PathLike[str] | str] = None,
    check_remote: bool = False,
) -> Dict[str, Any]:
    """Resolve a task data_cfgs block into a canonical runtime data_cfgs."""
    del check_remote  # Reserved for future explicit remote existence checks.

    if not isinstance(data_cfgs, dict):
        raise DatasetResolutionError("data_cfgs must be a mapping")

    forbidden = sorted(FORBIDDEN_DATA_CFG_FIELDS.intersection(data_cfgs))
    if forbidden:
        raise DatasetResolutionError(
            "data_cfgs contains forbidden path/source fields: "
            + ", ".join(forbidden)
        )

    dataset_id = data_cfgs.get("dataset")
    if not dataset_id:
        raise DatasetResolutionError("data_cfgs.dataset is required")

    root = Path(project_root or Path.cwd())
    datasets, dataset_layers = _load_dataset_registry(
        root, user_setting_path=user_setting_path
    )
    if dataset_id not in datasets:
        raise DatasetResolutionError(
            f"Unknown dataset id '{dataset_id}'. "
            "Declare it in configs/datasets.yml."
        )

    dataset_spec = datasets[dataset_id]
    reader = dataset_spec.get("reader")
    if not reader:
        raise DatasetResolutionError(
            f"Dataset '{dataset_id}' must define reader"
        )
    if reader not in READER_ALIASES:
        raise DatasetResolutionError(
            f"Unknown reader alias '{reader}' for dataset '{dataset_id}'"
        )

    relative_path = dataset_spec.get("path")
    if not relative_path:
        raise DatasetResolutionError(
            f"Dataset '{dataset_id}' must define path"
        )
    _validate_relative_path(relative_path, dataset_id)

    storage, storage_layers = _load_storage(
        root, user_setting_path=user_setting_path
    )
    source = (
        data_cfgs.get("source")
        or storage.get("default_source")
        or "local"
    )
    if source not in {"local", "cloud"}:
        raise DatasetResolutionError(
            "data_cfgs.source must be either 'local' or 'cloud'"
        )

    uri, storage_layer = _resolve_uri(
        source, relative_path, storage, storage_layers
    )

    resolved = copy.deepcopy(data_cfgs)
    resolved["source"] = source
    resolved["reader"] = reader
    resolved["uri"] = uri
    resolved["resolution"] = {
        "dataset_layer": str(dataset_layers.get(dataset_id, "")),
        "storage_layer": str(storage_layer),
    }
    return resolved


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded or {}


def _load_dataset_registry(
    project_root: Path,
    *,
    user_setting_path: Optional[os.PathLike[str] | str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Path]]:
    layers = [
        project_root / "configs" / "datasets.yml",
        (
            Path(user_setting_path)
            if user_setting_path
            else Path.home() / "hydro_setting.yml"
        ),
        project_root / ".hydro_setting.yml",
    ]

    datasets: Dict[str, Dict[str, Any]] = {}
    dataset_layers: Dict[str, Path] = {}
    saw_registry = False
    for layer in layers:
        data = _load_yaml(layer)
        layer_datasets = data.get("datasets")
        if layer_datasets is None:
            continue
        if not isinstance(layer_datasets, dict):
            raise DatasetResolutionError(
                f"Dataset registry in {layer} must be a mapping"
            )
        saw_registry = True
        for dataset_id, spec in layer_datasets.items():
            if not isinstance(spec, dict):
                raise DatasetResolutionError(
                    f"Dataset '{dataset_id}' in {layer} must be a mapping"
                )
            datasets[dataset_id] = copy.deepcopy(spec)
            dataset_layers[dataset_id] = layer

    if not saw_registry:
        raise DatasetResolutionError(
            "dataset registry not found; create configs/datasets.yml"
        )

    return datasets, dataset_layers


def _load_storage(
    project_root: Path,
    *,
    user_setting_path: Optional[os.PathLike[str] | str],
) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    layers = [
        (
            Path(user_setting_path)
            if user_setting_path
            else Path.home() / "hydro_setting.yml"
        ),
        project_root / ".hydro_setting.yml",
    ]

    storage: Dict[str, Any] = {}
    storage_layers: Dict[str, Path] = {}
    for layer in layers:
        data = _load_yaml(layer)
        layer_storage = data.get("storage")
        if layer_storage is None:
            continue
        if not isinstance(layer_storage, dict):
            raise DatasetResolutionError(
                f"Storage config in {layer} must be a mapping"
            )
        storage.update(copy.deepcopy(layer_storage))
        for key in layer_storage:
            storage_layers[key] = layer

    return storage, storage_layers


def _validate_relative_path(path_value: str, dataset_id: str) -> None:
    if not isinstance(path_value, str):
        raise DatasetResolutionError(
            f"Dataset '{dataset_id}' path must be a string"
        )
    if "://" in path_value:
        raise DatasetResolutionError(
            f"Dataset '{dataset_id}' path must be relative, not a URI"
        )

    windows_path = PureWindowsPath(path_value)
    posix_path = PurePosixPath(path_value)
    if windows_path.is_absolute() or posix_path.is_absolute():
        raise DatasetResolutionError(
            f"Dataset '{dataset_id}' path must be relative"
        )
    if ".." in windows_path.parts or ".." in posix_path.parts:
        raise DatasetResolutionError(
            f"Dataset '{dataset_id}' path cannot contain '..'"
        )


def _resolve_uri(
    source: str,
    relative_path: str,
    storage: Dict[str, Any],
    storage_layers: Dict[str, Path],
) -> Tuple[str, Path | str]:
    if source == "local":
        local = storage.get("local")
        if not isinstance(local, dict):
            raise DatasetResolutionError("storage.local is required")
        root = local.get("root")
        if not root:
            raise DatasetResolutionError("storage.local.root is required")
        root_path = Path(root)
        if not root_path.exists():
            raise DatasetResolutionError(
                f"storage.local.root does not exist: {root_path}"
            )
        resolved_path = root_path / Path(relative_path)
        if not resolved_path.exists():
            raise DatasetResolutionError(
                f"resolved local dataset path does not exist: {resolved_path}"
            )
        return str(resolved_path), storage_layers.get("local", "")

    s3 = storage.get("s3")
    if not isinstance(s3, dict):
        raise DatasetResolutionError("storage.s3 is required for cloud source")
    bucket = s3.get("bucket")
    if not bucket:
        raise DatasetResolutionError("storage.s3.bucket is required")
    prefix = str(s3.get("prefix") or "").strip("/")
    rel = relative_path.replace("\\", "/").strip("/")
    path = f"{prefix}/{rel}" if prefix else rel
    return f"s3://{bucket}/{path}", storage_layers.get("s3", "")
