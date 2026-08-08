# FAQ

## How do I use cloud (OSS/S3) data?

Set `data_cfgs.source: cloud` and configure `storage.s3` in `~/hydro_setting.yml`
(bucket, prefix, access keys, endpoint URL). Cloud reads use zarr caches under
`s3://<bucket>/zarr/`; missing caches are generated from raw files (write access
required). `source` is a hard selection - it does not fall back to local automatically.

## Why do I get an EndpointConnectionError when using cloud?

Check that this machine can reach the OSS/S3 endpoint, and that the `storage.s3` block
in `~/hydro_setting.yml` is correct. Remember that a project-level
`.hydro_setting.yml` overrides the user-level file key-by-key.

## Why does my streamflow contain negative values like -28.28?

Raw USGS files mark missing flow as -999 (cfs). If the -999 values are converted to
m^3/s (multiplied by 0.0283168) instead of being replaced with NaN, you get negative
garbage values that corrupt metrics. Regenerate the data cache/zarr with the -999 ->
NaN handling so missing days are excluded from objective functions.

## What does "param_range_file not provided ... Using default MODEL_PARAM_DICT ranges" mean?

It is informational: no custom parameter range file was given, so the built-in
parameter ranges are used. To customize ranges, set `training_cfgs.param_range_file`
to a YAML file (strict validation: missing files, unknown/missing parameters, or
invalid `[min, max]` ranges fail fast).

## What is the difference between calibration, evaluation, and simulation?

- **Calibration** searches for the parameter set that minimizes the objective (e.g. RMSE).
- **Evaluation** runs the calibrated parameters on a period (e.g. test) and reports
  metrics (NSE, KGE, RMSE, PBIAS, ...).
- **Simulation** runs the model with any parameters and returns the simulated series.

Evaluation internally runs a simulation, so you normally do not need to run simulation
separately after calibration unless you want the raw simulated series.

## Why is `source: cloud` in my YAML ignored?

`source` is read from `data_cfgs.source`. If it is missing, `storage.default_source`
in `hydro_setting.yml` is used. Also verify you are running from the project root - a
project-level `.hydro_setting.yml` overrides the user-level one.

## Which Python versions are supported?

Python >= 3.11 (see `pyproject.toml`).
