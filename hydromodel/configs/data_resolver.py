"""
Unified data path resolution for hydromodel runtime configs.

Delegates path resolution to hydrodataset's canonical resolve_data_path()
while adding hydromodel-specific concerns:

- Task config validation (rejects deprecated raw-path fields like
  ``data_source_path``, ``data_source_type``)
- Layered config loading: project ``.hydro_setting.yml`` overrides
  user ``~/hydro_setting.yml``
- Hydromodel-specific reader alias (``zarr_timeseries``)

The heavy lifting -- path validation, local-root joining, cloud S3 URI
construction -- lives in hydrodataset and is not duplicated here.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Optional, Tuple

import yaml

# Canonical resolution engine -- the single source of truth for path math.
from hydrodataset.configs.data_resolver import (
    resolve_data_path as _hd_resolve_data_path,
    DatasetResolutionError,  # noqa: F401 -- re-exported for callers
)

# Fully-merged reader aliases (hydrodataset 33 + hydrodatasource 11).
from hydrodatasource.configs.data_resolver import (
    HDS_DATASETS,
    READER_ALIASES as _MERGED_READER_ALIASES,
)

# ── hydromodel-specific aliases ───────────────────────────────────────────

READER_ALIASES: Dict[str, Dict[str, str]] = dict(_MERGED_READER_ALIASES)
READER_ALIASES["zarr_timeseries"] = {
    "module": "hydromodel.datasets.zarr_loader",
    "class": "ZarrTimeSeriesDatasource",
    "category": "zarr",
}

# ── task-config hygiene ───────────────────────────────────────────────────

FORBIDDEN_DATA_CFG_FIELDS = {
    "path",
    "data_source_path",
    "data_source_type",
}

__all__ = [
    "DatasetResolutionError",
    "FORBIDDEN_DATA_CFG_FIELDS",
    "READER_ALIASES",
    "resolve_config",
    "resolve_config_if_needed",
    "resolve_data_cfgs",
]


# ── public API ────────────────────────────────────────────────────────────


def resolve_config_if_needed(
    config: Dict[str, Any],
    **kwargs,
) -> Dict[str, Any]:
    """Auto-resolve *config* if *data_cfgs* lacks ``uri``, else return as-is.

    Consumes ``project_root`` and ``user_setting_path`` from *kwargs*
    (popped, not forwarded).  All other kwargs are left untouched for the
    caller to forward downstream.
    """
    data_cfgs_raw = config.get("data_cfgs", {})
    if (
        isinstance(data_cfgs_raw, dict)
        and "uri" not in data_cfgs_raw
        and "dataset" in data_cfgs_raw
    ):
        project_root = kwargs.pop("project_root", None)
        user_setting_path = kwargs.pop("user_setting_path", None)
        return resolve_config(
            config,
            project_root=project_root,
            user_setting_path=user_setting_path,
        )
    return config


def resolve_config(
    config: Dict[str, Any],
    *,
    project_root: Optional[os.PathLike[str] | str] = None,
    user_setting_path: Optional[os.PathLike[str] | str] = None,
) -> Dict[str, Any]:
    """Resolve *data_cfgs* inside a full hydromodel task config.

    Every other top-level section (model_cfgs, training_cfgs, …) is
    returned unchanged.  Only ``data_cfgs`` is rewritten.
    """
    resolved = copy.deepcopy(config)
    resolved["data_cfgs"] = resolve_data_cfgs(
        resolved.get("data_cfgs"),
        project_root=project_root,
        user_setting_path=user_setting_path,
    )
    return resolved


def resolve_data_cfgs(
    data_cfgs: Dict[str, Any],
    *,
    project_root: Optional[os.PathLike[str] | str] = None,
    user_setting_path: Optional[os.PathLike[str] | str] = None,
) -> Dict[str, Any]:
    """Resolve a task ``data_cfgs`` block into its canonical runtime form.

    Returns a shallow copy of *data_cfgs* enriched with the resolved
    ``reader``, ``uri``, ``source``, and ``resolution`` metadata.

    Dataset registry is built by hydrodataset's 3-layer cascade:
    DEFAULT(33) → HDS_DATASETS(1) → YAML overrides(N) → project YAML.
    hydromodel injects its YAML-based datasets via *extra_registry_dicts*.
    """
    if not isinstance(data_cfgs, dict):
        raise DatasetResolutionError("data_cfgs must be a mapping")

    # ── 1. reject deprecated raw-path fields in task config ────────────
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

    # ── 2. collect YAML-based dataset overrides ────────────────────────
    yaml_datasets, dataset_layers = _collect_yaml_datasets(
        root, user_setting_path=user_setting_path
    )

    # ── 3. load layered storage config ────────────────────────────────
    storage, storage_layers = _load_storage(
        root, user_setting_path=user_setting_path
    )

    # ── 4. determine source ───────────────────────────────────────────
    source = (
        data_cfgs.get("source")
        or storage.get("default_source")
        or "local"
    )
    if source not in {"local", "cloud"}:
        raise DatasetResolutionError(
            "data_cfgs.source must be either 'local' or 'cloud'"
        )

    # ── 5. extract reader name from registry ──────────────────────────
    # Merge hydrodatasource datasets with YAML overrides for lookup
    merged_for_lookup = dict(HDS_DATASETS)
    merged_for_lookup.update(yaml_datasets)
    reader = merged_for_lookup.get(dataset_id, {}).get("reader", dataset_id)

    # ── 6. validate reader alias ──────────────────────────────────────
    if reader not in READER_ALIASES:
        raise DatasetResolutionError(
            f"Unknown reader alias '{reader}' for dataset '{dataset_id}'. "
            f"Known readers: {', '.join(sorted(READER_ALIASES))}"
        )

    # ── 7. resolve URI ────────────────────────────────────────────────
    if source == "local":
        local_root = (
            Path(storage["local"]["root"])
            if isinstance(storage.get("local"), dict)
            and storage["local"].get("root")
            else None
        )
        resolved_path = _hd_resolve_data_path(
            dataset_id,
            source="local",
            project_root=str(root),
            local_root=local_root,
            extra_registry_dicts=[HDS_DATASETS, yaml_datasets],
            extra_reader_aliases=dict(READER_ALIASES),
        )
        uri = str(resolved_path)
        storage_layer = storage_layers.get("local", "")
    else:
        # Cloud: hydrodataset's resolve_data_path reads storage from
        # ~/hydro_setting.yml only, but hydromodel uses layered config.
        # Build the S3 URI ourselves from the already-merged storage.
        #
        # NOTE: merged_for_lookup only contains HDS_DATASETS + YAML
        # overrides, NOT hydrodataset's _DEFAULT_REGISTRY (33 entries).
        # Cloud source therefore requires the dataset to be explicitly
        # defined in a YAML file. This is intentional — cloud storage
        # is primarily used for zarr_timeseries datasets, not for
        # traditional hydrodataset-served datasets like CAMELS.
        spec = merged_for_lookup.get(dataset_id)
        if not spec:
            raise DatasetResolutionError(
                f"Unknown dataset id '{dataset_id}'. "
                f"Define it in configs/datasets.yml or hydro_setting.yml."
            )
        uri = _build_cloud_uri(spec.get("path", ""), storage)
        storage_layer = storage_layers.get("s3", "")

    # ── 8. assemble resolved data_cfgs ────────────────────────────────
    resolved = copy.deepcopy(data_cfgs)
    resolved["source"] = source
    resolved["reader"] = reader
    resolved["uri"] = uri
    resolved["resolution"] = {
        "dataset_layer": str(dataset_layers.get(dataset_id, "")),
        "storage_layer": str(storage_layer),
    }
    return resolved


# ── internal helpers ──────────────────────────────────────────────────────


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Return parsed YAML, or ``{}`` when the file is missing or empty."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        loaded = yaml.safe_load(f)
    return loaded or {}


def _collect_yaml_datasets(
    project_root: Path,
    *,
    user_setting_path: Optional[os.PathLike[str] | str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Path]]:
    """Collect dataset definitions from layered YAML sources.

    These are injected as *extra_registry_dicts* into hydrodataset's
    ``resolve_data_path()``, where they override the built-in
    ``_DEFAULT_REGISTRY`` entries.

    Priority (last wins):
    1. ``{project_root}/configs/datasets.yml``
    2. ``~/hydro_setting.yml``  (or *user_setting_path*)
    3. ``{project_root}/.hydro_setting.yml``
    """
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

    for layer in layers:
        data = _load_yaml(layer)
        layer_datasets = data.get("datasets")
        if layer_datasets is None:
            continue
        if not isinstance(layer_datasets, dict):
            raise DatasetResolutionError(
                f"Dataset registry in {layer} must be a mapping"
            )
        for dataset_id, spec in layer_datasets.items():
            if not isinstance(spec, dict):
                raise DatasetResolutionError(
                    f"Dataset '{dataset_id}' in {layer} must be a mapping"
                )
            datasets[dataset_id] = copy.deepcopy(spec)
            dataset_layers[dataset_id] = layer

    return datasets, dataset_layers


def _build_cloud_uri(relative_path: str, storage: Dict[str, Any]) -> str:
    """Build an S3 URI from a relative path and merged storage config."""
    # Reject absolute and dangerous paths
    if (
        "://" in relative_path
        or ".." in relative_path
        or PurePosixPath(relative_path).is_absolute()
        or PureWindowsPath(relative_path).is_absolute()
    ):
        raise DatasetResolutionError(
            f"Cloud dataset path must be relative: '{relative_path}'"
        )

    s3 = storage.get("s3")
    if not isinstance(s3, dict):
        raise DatasetResolutionError("storage.s3 is required for cloud source")
    bucket = s3.get("bucket")
    if not bucket:
        raise DatasetResolutionError("storage.s3.bucket is required")
    prefix = str(s3.get("prefix") or "").strip("/")
    rel = relative_path.replace("\\", "/").strip("/")
    path = f"{prefix}/{rel}" if prefix else rel
    return f"s3://{bucket}/{path}"


def _load_storage(
    project_root: Path,
    *,
    user_setting_path: Optional[os.PathLike[str] | str],
) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    """Merge storage config from layered sources.

    Priority (last wins):
    1. ``~/hydro_setting.yml``  (or *user_setting_path*)
    2. ``{project_root}/.hydro_setting.yml``
    """
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
