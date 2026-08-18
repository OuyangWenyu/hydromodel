# Unified Data Path Resolution Implementation Plan

## Goal

Implement the configuration contract from
`docs/adr/0001-unified-data-path-resolution.md` and remove hidden path
resolution from calibration, simulation, evaluation, and data loading paths.

## Phase 1: Resolver Foundation

1. Add a resolver module under `hydromodel/configs/`.
2. Define a code-owned `READER_ALIASES` registry.
3. Load dataset registry layers in this order:
   - `configs/datasets.yml`
   - `~/hydro_setting.yml`
   - `.hydro_setting.yml`
4. Load storage layers in this order:
   - `~/hydro_setting.yml`
   - `.hydro_setting.yml`
5. Merge layers deterministically.
6. Record which layer supplied or overrode each dataset and storage value.
7. Validate task `data_cfgs` before resolution.
8. Reject task path fields.
9. Validate dataset specs require `reader` and `path`.
10. Validate reader aliases.
11. Validate source is `local` or `cloud`.
12. Resolve `uri` from selected source and dataset relative path.
13. Validate local root and resolved local path exist.
14. Return canonical resolved `data_cfgs` with `dataset`, `source`, `reader`,
    `uri`, and `resolution`.

## Phase 2: Configuration Loading

1. Update config loading to accept the new `*_cfgs` schema directly.
2. Remove simplified-config conversion paths that generate
   `data_source_type` and `data_source_path`.
3. Add `configs/datasets.yml` with explicit entries used by examples.
4. Update example configs to remove task-level paths.
5. Ensure `--dry-run` runs the resolver and validation without starting
   calibration.

## Phase 3: Data Loader Refactor

1. Refactor `UnifiedDataLoader` to require resolved `data_cfgs.uri`.
2. Remove loader reads of `~/hydro_setting.yml`.
3. Remove loader default path guessing.
4. Replace `data_source_type` usage with `dataset` and `reader`.
5. Instantiate the reader from the resolved reader alias.
6. Pass only the resolved `uri` and task options to the reader.
7. Keep data-reading errors in the loader or reader layer, not in the resolver.

## Phase 4: Runtime Entrypoints

1. Update `scripts/run_xaj_calibration.py` to load, resolve, validate, and then
   call calibration.
2. Update simulation and evaluation entrypoints to use the same resolver.
3. Update `calibrate()` to expect resolved config.
4. Save resolved `data_cfgs` and provenance into calibration snapshots.
5. Make evaluation from a calibration directory use the saved resolved
   `data_cfgs`.

## Phase 5: Tests

Add focused tests for:

1. Local resolution with valid registry and storage.
2. `source` omitted and `storage.default_source` used.
3. `source` omitted and default to `local`.
4. Unknown dataset fails.
5. Missing dataset registry fails.
6. Missing dataset `reader` fails.
7. Missing dataset `path` fails.
8. Unknown reader alias fails.
9. Absolute local path fails.
10. Complete S3 URI in dataset path fails.
11. Parent traversal path fails.
12. Task `data_cfgs.path` fails.
13. Task `data_source_path` fails.
14. Missing local root fails.
15. Missing local resolved dataset path fails.
16. Cloud source resolves to expected S3 URI without remote checks.
17. Workspace `.hydro_setting.yml` overrides user storage.
18. Dataset override emits provenance.
19. `UnifiedDataLoader` rejects unresolved data configs.
20. Calibration snapshot preserves resolved `uri` and `resolution`.

## Phase 6: Documentation

1. Update installation docs to show the new `storage` schema.
2. Update data guide to explain dataset registry, storage, and task config.
3. Update quickstart examples to use `data_cfgs.dataset`.
4. Remove documentation that recommends `data_source_path`.
5. Add migration notes for old configs.

## Acceptance Criteria

- All data tasks call one resolver before business execution.
- `UnifiedDataLoader` never reads `hydro_setting.yml`.
- `UnifiedDataLoader` never guesses a path.
- New task configs contain no path fields.
- Dataset registry entries contain explicit `reader` and safe relative `path`.
- Local path errors fail before calibration starts.
- Cloud URI resolution is deterministic and does not require network by
  default.
- Calibration snapshots contain resolved data provenance.
- Tests cover fail-fast path and registry errors.
