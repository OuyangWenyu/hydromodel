# Changelog

## v0.3.0 - 2025-11-07

**Critical Bug Fixes for Multi-Basin Flood Event Data**:

- **Fixed Time Array Misalignment in Evaluation**:
  - Fixed critical issue where multi-basin flood events used incorrect time arrays in NetCDF output
  - Previous behavior: Used longest basin's time array for all basins, causing time misalignment for shorter basins
  - Example issue: Basin 21100150's Event 26 showed "2020-08-31 to 2022-08-02" (spanning 2 years) when actual data ended at 2020-09-05
  - New behavior: Creates unified time array from all basins' unique timestamps, then remaps each basin's data to correct time positions
  - Modified `_save_all_results()` in `unified_evaluate.py` (lines 403-470)
  - Each basin now has accurate event timestamps in evaluation results

- **Fixed Data Access Issues in Calibration**:
  - Fixed `AttributeError` when calibrating multi-basin flood event data
  - Previous behavior: `UnifiedModelSetup` tried to access `p_and_e.shape[1]` when `p_and_e=None` for separate basin data
  - Added proper detection of `basin_data_separate` mode in `UnifiedModelSetup.__init__()`
  - Modified `set_basin_index()` to extract basin-specific data from `basin_data_separate` when available
  - Each basin now uses its own time series without padding during calibration
  - Modified `unified_calibrate.py` (lines 116-253, 377-386)

- **Code Quality Improvements**:
  - Removed debug print statements from data loading pipeline
  - Cleaned up verbose event statistics output in `unified_data_loader.py`
  - Removed redundant basin information prints in `floodevent.py`
  - Fixed `SyntaxWarning` in `run_xaj_calibration.py` by using raw string for file path

- **Visualization Enhancements**:
  - Fixed metrics text overlap with legend in flood event plots
  - Moved metrics display position from (0.98, 0.97) to (0.98, 0.70) to avoid overlap
  - Metrics (NSE/RMSE/PBIAS) now display cleanly below legend on right side
  - Modified `data_visualize.py` (line 255)

**Impact**:

- **Evaluation**: Multi-basin flood event evaluations now produce correct NetCDF files with accurate timestamps for each basin
- **Calibration**: Multi-basin flood event calibrations now work correctly without data access errors
- **Visualization**: Event plots are cleaner with better layout and no overlapping text

**Files Changed**:

- `hydromodel/trainers/unified_evaluate.py`: Time array unification and remapping logic
- `hydromodel/trainers/unified_calibrate.py`: Basin data separation support and basin_ids detection
- `hydromodel/datasets/unified_data_loader.py`: Removed debug print statements
- `hydromodel/datasets/data_visualize.py`: Adjusted metrics text position
- `scripts/run_xaj_calibration.py`: Fixed path string syntax warning

**Testing**:

- Verified multi-basin flood event evaluation produces correct timestamps
- Verified multi-basin flood event calibration accesses correct basin data
- Confirmed clean console output without excessive debug information

## v0.3.0 - 2025-11-06

**Flood Event Data Support**:

- **Event ID System**:
  - Added `event_id` tracking to group multi-peak flood events together
  - Modified `floodevent.py` to extract and pass `event_id` and `event_name` from original data
  - Modified `_convert_events_to_arrays()` to include `event_id` as 4th feature in p_and_e array
  - p_and_e shape changed from `[time, features=3]` to `[time, features=4]` where features = `[prcp, pet, marker, event_id]`
  - Prevents original flood events with multiple peaks from being split into separate visualizations

- **Simulation Code Fixes**:
  - Fixed `_simulate_event_data()` in `unified_simulate.py` to handle 4-feature input
  - Changed flood_event_marker extraction from index `-1` to index `2` (to account for event_id as 4th feature)
  - Only passes first 3 features `[:3]` to model function, excluding event_id
  - Updated validation error message to reflect 3+ features format

- **Visualization Improvements**:
  - Modified `_identify_flood_events()` to support grouping by `event_id` when available
  - Only plots flood period data (`marker==1`), excluding warmup (`marker==NaN`) and gaps (`marker==0`)
  - Filters time and data arrays to only include flood period timesteps
  - Fixes issue where plots would span entire year or show overlapping x-axis ticks
  - Event count now correctly reflects original flood events (e.g., 26 events instead of 39 peaks)

- **Time Handling Fixes**:
  - Fixed time array padding for multi-basin flood event data
  - Extends time arrays with regular intervals using `pd.date_range()` when padding data
  - Fixed pandas FutureWarning by replacing `'H'` with `'h'` and `'D'` with `'d'` in frequency strings
  - Extracts real event timestamps from original flood event data instead of generating dummy times

- **Evaluation Results**:
  - Modified `_save_evaluation_results()` to save `event_id` as additional variable in NetCDF output
  - Enables visualization to group multi-peak events by original event_id

**Files Changed**:

- `hydrodatasource/reader/floodevent.py`: Event ID and time extraction
- `hydromodel/datasets/unified_data_loader.py`: 4-feature p_and_e array, time padding
- `hydromodel/trainers/unified_simulate.py`: 4-feature input handling
- `hydromodel/trainers/unified_evaluate.py`: Event ID saving to NetCDF
- `hydromodel/datasets/data_visualize.py`: Event grouping and flood period filtering

**Known Issues**:

- Visualization still shows some x-axis overlap issues for certain events (pending further investigation)

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
