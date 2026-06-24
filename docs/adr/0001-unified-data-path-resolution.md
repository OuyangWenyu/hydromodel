# ADR 0001: Unified Data Path Resolution

## Status

Accepted.

## Context

Hydromodel currently resolves input data paths through multiple independent
routes. Some code reads `~/hydro_setting.yml` during package import, some code
loads it in the configuration manager, and `UnifiedDataLoader` also tries to
resolve missing paths by itself. This creates non-idempotent behavior: the same
dataset can resolve to different paths depending on whether calibration,
simulation, evaluation, or a saved configuration is used.

The old configuration also mixes separate concepts:

- A dataset id, such as `camels_us`.
- A reader implementation, such as a hydrodataset or hydrodatasource class.
- A physical storage location, such as a local directory or S3 prefix.
- A task-specific configuration, such as basin ids and time periods.

The project also needs a clean path to cloud-native datasets, especially S3
stores and Zarr layouts, without making every task configuration verbose.

## Decision

Hydromodel will use a single deterministic dataset resolver. All data tasks
must resolve their data configuration before calibration, simulation,
evaluation, or data loading starts.

The configuration model is split into three layers:

1. Environment storage configuration.
2. Dataset registry.
3. Task configuration.

The resolver combines these layers into a resolved runtime `data_cfgs` block.
Business code and loaders consume only the resolved block.

## Storage Configuration

Storage is environment-specific and belongs in `~/hydro_setting.yml` or
workspace-local `.hydro_setting.yml`.

```yaml
storage:
  default_source: local
  local:
    root: D:/data/hydromodel
  s3:
    bucket: hydro-data
    prefix: hydromodel
    region: us-east-1
    profile: default
```

Rules:

- `source` is restricted to `local` or `cloud`.
- `local` always maps to `storage.local`.
- `cloud` always maps to `storage.s3`.
- `data_cfgs.source` is optional.
- If `data_cfgs.source` is omitted, use `storage.default_source`.
- If `storage.default_source` is omitted, use `local`.
- Only validate the storage backend selected by the final source.
- Local input roots must exist.
- Local resolved dataset paths must exist.
- Cloud paths are validated for configuration shape by default, but remote
  existence checks require an explicit future option such as `--check-remote`.
- Secrets must not be stored in YAML. Use profiles, environment credentials, or
  cloud-native identity. Experiment snapshots may store credential references,
  never access keys or secret keys.

## Dataset Registry

Dataset identity and relative physical layout are declared in a dataset
registry. The project registry is `configs/datasets.yml`. User and workspace
settings may add or override registry entries, but overrides must be recorded in
the experiment provenance.

Dataset registry layers are merged in this order:

1. `configs/datasets.yml`
2. `~/hydro_setting.yml`
3. `.hydro_setting.yml`

Later layers override earlier layers. Overrides should produce a warning during
resolution and be recorded in the resolved provenance.

```yaml
datasets:
  camels_us:
    reader: camels_us
    path: public/camels_us

  songliao_event:
    reader: floodevent
    path: projects/songliao/event

  era5_songliao:
    reader: zarr_timeseries
    path: reanalysis/era5/songliao.zarr
```

Rules:

- Every dataset id used by a data task must be declared in the merged registry.
- Every dataset spec must explicitly contain `reader` and `path`.
- `path` must be a safe relative path.
- Absolute local paths are invalid.
- Complete URIs such as `s3://...` are invalid in dataset specs.
- Parent traversal with `..` is invalid.
- The same dataset path is used for local and cloud sources.
- `format` is not part of the required schema. The reader is responsible for
  interpreting the resolved URI.
- Dataset specs do not support inheritance. Entries must be explicit.
- Dataset registry YAML cannot define Python module paths. Reader aliases are
  code capabilities and must be registered in code.

## Reader Aliases

The project keeps a small code-defined reader alias registry. It replaces the
old use of `DATASET_MAPPING` as a mixed dataset/path/reader registry.

Examples:

```python
READER_ALIASES = {
    "camels_us": ...,
    "floodevent": ...,
    "selfmade": ...,
    "zarr_timeseries": ...,
}
```

Rules:

- `data_cfgs.dataset` names a data asset.
- `datasets.<id>.reader` names a reader alias.
- Reader aliases are registered in code, not user YAML.
- If a user writes a reader alias as a dataset id, resolution fails and the
  error should explain how to declare a dataset.

## Task Configuration

Task configuration uses the existing `*_cfgs` naming style.

```yaml
data_cfgs:
  dataset: camels_us
  source: local
  basin_ids: ["12025000"]
  variables: ["prcp", "PET", "streamflow"]
  warmup_length: 365
  train_period: ["1981-01-01", "2004-12-31"]
  valid_period: ["2005-01-01", "2009-12-31"]
  test_period: ["2010-01-01", "2014-12-31"]

model_cfgs:
  name: xaj
  params:
    source_type: sources
    source_book: HF
    kernel_size: 15

training_cfgs:
  algorithm: SCE_UA
  loss: RMSE

evaluation_cfgs:
  metrics: ["NSE", "RMSE", "KGE", "PBIAS"]
```

Rules:

- Task configuration must not contain paths.
- `data_cfgs.path`, `data_cfgs.data_source_path`, and
  `data_cfgs.data_source_type` are invalid in the new contract.
- `data_cfgs.dataset` is required.
- `data_cfgs.source` is optional.

## Resolved Runtime Configuration

Before any business operation starts, the resolver produces a canonical runtime
configuration.

```yaml
data_cfgs:
  dataset: camels_us
  source: local
  reader: camels_us
  uri: D:/data/hydromodel/public/camels_us
  basin_ids: ["12025000"]
  variables: ["prcp", "PET", "streamflow"]
  train_period: ["1981-01-01", "2004-12-31"]
  valid_period: ["2005-01-01", "2009-12-31"]
  test_period: ["2010-01-01", "2014-12-31"]
  resolution:
    dataset_layer: configs/datasets.yml
    storage_layer: ~/.hydro_setting.yml
```

`UnifiedDataLoader` and downstream trainers must consume this resolved
configuration. They must not read `hydro_setting.yml`, guess paths, or perform
fallback path resolution.

## Experiment Snapshots

Calibration outputs should save a resolved data snapshot for reproducibility.
The snapshot records the resolved URI and lightweight provenance.

```yaml
data_cfgs:
  dataset: camels_us
  source: local
  reader: camels_us
  uri: D:/data/hydromodel/public/camels_us
  resolution:
    dataset_layer: configs/datasets.yml
    storage_layer: ~/.hydro_setting.yml
```

Evaluation and simulation from a calibration directory should use the saved
resolved data configuration by default. A future explicit refresh option may
re-resolve the current registry and storage configuration.

## Fail-Fast Rules

Resolution must fail immediately for:

- Missing dataset registry on data tasks.
- Unknown `data_cfgs.dataset`.
- Missing dataset `reader`.
- Missing dataset `path`.
- Unknown reader alias.
- Invalid `source` values outside `local` and `cloud`.
- Missing selected storage backend.
- Missing local storage root.
- Missing local resolved dataset path.
- Absolute dataset paths.
- Complete URIs in dataset paths.
- Parent traversal in dataset paths.
- Path fields in task configuration.

No silent fallback to home directories, `basins-origin`, `datasets-origin`, or
loader-level guessing is allowed.

## Consequences

This decision intentionally removes backward compatibility with the old
`data_source_type` and `data_source_path` contract in the new execution path.
Migration should be direct rather than layered through legacy fields.

Benefits:

- One deterministic path resolver for calibration, simulation, and evaluation.
- Clear separation between data identity, reader capability, storage backend,
  and task parameters.
- Cleaner cloud-native support for S3 and Zarr.
- Better experiment reproducibility through resolved snapshots.
- Earlier and clearer configuration errors.

Costs:

- Existing example configs and scripts must be updated.
- `UnifiedDataLoader` must be refactored to consume resolved specs only.
- Users must declare every dataset in the dataset registry.
- Tests must cover resolver behavior rather than relying on implicit fallback.

## Non-Goals

- No task-level path overrides.
- No absolute dataset paths in registries.
- No custom `source` names beyond `local` and `cloud`.
- No YAML-defined dynamic Python imports for readers.
- No dataset inheritance in the registry.
- No implicit path defaults for dataset ids.
- No automatic creation of input data directories.
