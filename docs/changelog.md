# Changelog

## v0.3.0 - 2025-11-05

**Multi-Basin Support**:

- **Fixed Multi-Basin Unit Conversion**:
  - Fixed broadcasting error in `streamflow_unit_conv` for multi-basin data
  - Modified `UnifiedDataLoader._check_and_convert_units()` to process basins individually
  - Modified `_save_evaluation_results()` to convert units basin-by-basin
  - Now properly handles dimension order (`["basin", "time"]` or `["time", "basin"]`)
  - Successfully tested calibration and evaluation on CAMELS-US multi-basin datasets

**Calibration Improvements**:

- **Parameter Display Enhancement**:
  - All three algorithms (SCE-UA, GA, scipy) now display denormalized parameters (actual physical ranges)
  - Changed console output from normalized values (0-1) to actual parameter ranges
  - Uses the same denormalization formula as `process_parameters()` in `param_utils.py`
  - Format: "Best parameters (actual ranges)" with physical values

**Files Changed**:

- `hydromodel/trainers/unified_calibrate.py`: Parameter display for all three algorithms
- `hydromodel/datasets/unified_data_loader.py`: Multi-basin unit conversion
- `hydromodel/trainers/unified_evaluate.py`: Multi-basin evaluation results saving

**Testing**:

- Verified multi-basin calibration on CAMELS-US dataset
- Verified multi-basin evaluation on CAMELS-US dataset
- All three calibration algorithms tested with multi-basin data

## v0.3.0 - 2025-11-04

**Major Improvements**:

- **Configurable File Saving**:
  - Added `save_config` option in `training_cfgs` (default: `True`)
  - Controls saving of `calibration_config.yaml` and `param_range.yaml`
  - `param_range.yaml` now only contains the current model's parameters (not all models)
  - `calibration_config.yaml` records actual `param_range_file` path instead of `null`
  - Command-line flag `--no-save-config` to disable saving

**Bug Fixes**:

- Fixed `param_range.yaml` containing all models instead of only the current model
- Fixed `calibration_config.yaml` not recording the actual `param_range_file` path

**Documentation**:

- Updated README.md and README_zh.md with `save_config` usage
- Documented `param_range.yaml` behavior (only saves current model)
- Updated example_config.yaml with save_config option

**Testing**:

- Verified all three calibration algorithms (SCE-UA, GA, scipy) work correctly
- Tested configuration file saving functionality

## v0.3.0 - 2025-11-04

**Major Improvements**:

- **Unified Calibration Interface**: All three algorithms (SCE-UA, GA, scipy) now use a consistent API
- **Standardized Results Format**:
  - All algorithms save best parameters to `calibration_results.json` (unified format)
  - All algorithms save detailed iteration history to CSV with parameter values
  - CSV format unified across GA and scipy (`objective_value` + `param_{name}` columns)
- **Enhanced Progress Tracking**:
  - Real-time progress display for all algorithms
  - Progress bars for GA using tqdm
  - Iteration-by-iteration output for scipy
  - Generation-by-generation statistics for GA
- **Improved Parameter Loading**:
  - Automatic fallback: JSON → GA CSV → scipy CSV → SCE-UA CSV → legacy TXT
  - Clear error messages showing all attempted file paths
  - Support for all algorithm result formats in evaluation

**New Features**:

- **Genetic Algorithm (GA) Enhancements**:
  - Saves detailed generation history with all parameter values
  - Custom bounded mutation operator
  - Generation statistics (min, mean, max, best fitness)
  - Compatible CSV format with scipy results
- **scipy Optimizer Enhancements**:
  - Iteration history tracking with parameter values
  - Progress display every N iterations
  - Detailed convergence information
  - Support for multiple scipy methods (SLSQP, L-BFGS-B, TNC, etc.)
- **Unified Configuration**:
  - All algorithm parameters configurable via YAML
  - Example config includes all three algorithms with defaults
  - Algorithm aliases: `SCE_UA`/`sceua`, `GA`/`genetic_algorithm`, `scipy`/`scipy_minimize`

**Bug Fixes**:

- Fixed GA evaluation function returning tuple of list instead of tuple of float
- Fixed scipy/GA algorithm name matching using `in` operator instead of `or`
- Fixed parameter loading failure when `calibration_results.json` not found
- Added error handling for all parameter loading methods

**Documentation**:

- Updated README.md with complete algorithm parameter documentation
- Updated README_zh.md with Chinese documentation
- Updated quickstart.md with algorithm comparison guide
- Added convergence analysis examples
- Added results format explanation
- Updated configuration examples with all algorithm parameters

## v0.2.11 - 2025-11-02

**Improvement**:

- Refactored visualization and cleanup codebase
- Updated GitHub workflows

**New Features**:

- Added unified simulate interface
- Added parameter file support for simulation

## v0.0.1 - 2024-XX-XX

**Initial Release**:

- Basic XAJ model implementation
- SCE-UA calibration support
- CAMELS dataset integration
