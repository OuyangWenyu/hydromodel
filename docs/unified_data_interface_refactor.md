# Unified Data Interface Refactor

## Overview

This document describes the major refactoring of the hydromodel data interface to support both continuous time series and flood event data through a unified `data_config` approach.

## Problem Statement

Previously, the hydromodel package had different interfaces for different data types:
- Continuous data: Used `p_and_e` and `qobs` arrays directly
- Event data: Used `List[Dict]` with a problematic `_convert_event_data_to_standard_format()` function

This led to several issues:
1. Inconsistent interfaces between data types
2. Difficult data handling for mixed scenarios
3. The conversion function was fundamentally flawed for proper event handling
4. No unified approach to data loading across different sources

## Solution Architecture

### 1. Unified Data Configuration

Replaced the direct `p_and_e`, `qobs` parameters with a unified `data_config` dictionary that follows the same pattern as `model_config` and `loss_config`:

```python
# Old approach
UnifiedModelSetup(p_and_e, qobs, model_config, loss_config)

# New approach  
UnifiedModelSetup(data_config, model_config, loss_config)
```

### 2. UnifiedDataLoader Class

Created a new `UnifiedDataLoader` class that:
- Uses `hydrodatasource.read_ts_xrdataset()` as the unified interface
- Supports multiple data source types (FloodEvent, Camels, SelfMade)
- Automatically converts all data to standard `(p_and_e, qobs)` format
- Handles data type detection and appropriate processing

### 3. Event vs Continuous Data Handling

#### For Unit Hydrograph Models
- Direct processing of event data (no change needed)
- Uses existing event-based interfaces

#### For Traditional Models (XAJ, GR series, etc.)
- **Continuous data**: Standard processing (no change)
- **Event data**: Special segmented processing:
  1. Identify event segments (continuous non-zero periods)
  2. Run model on each segment separately
  3. Combine results while maintaining timeline
  4. Handle gaps between events properly

### 4. Removed Problematic Functions

Eliminated `_convert_event_data_to_standard_format()` function which was fundamentally flawed for proper event data handling.

## Key Components

### UnifiedDataLoader

Located in `hydromodel/datasets/unified_data_loader.py`

**Features:**
- Supports `floodevent`, `camels`, and `selfmade` data types
- Graceful handling of missing hydrodatasource dependencies
- Automatic data format conversion
- Event metadata extraction (when available)

**Usage:**
```python
data_config = {
    "data_type": "floodevent",
    "data_path": "path/to/data",
    "basin_ids": ["basin1"],
    "time_periods": {"calibration": ["2020-01-01", "2022-12-31"]},
    "variables": ["P_eff", "Q_obs_eff"]
}

loader = UnifiedDataLoader(data_config)
p_and_e, qobs = loader.load_data()
```

### Enhanced UnifiedModelSetup

Updated `hydromodel/trainers/unified_calibrate.py`

**Key Changes:**
- Constructor now accepts `data_config` instead of `p_and_e`, `qobs`
- Automatic data loading through `UnifiedDataLoader`
- Event data detection and special processing
- Segmented simulation for traditional models with event data

**Event Processing Algorithm:**
```python
def _simulate_event_data(self, params):
    # 1. Find event segments in precipitation data
    event_segments = self._find_event_segments(rain_series)
    
    # 2. Process each segment separately
    for start_idx, end_idx in event_segments:
        event_data = self.p_and_e[start_idx:end_idx+1, ...]
        event_result = model_function(event_data, params, warmup_length=0)
        output[start_idx:end_idx+1, ...] = event_result
    
    return output
```

### Updated Configuration System

Enhanced `hydromodel/configs/unified_config.py` with comprehensive configuration structure:

```yaml
data_cfgs:
  data_type: "selfmade"
  data_path: null
  dataset_name: "experiment"
  basin_ids: []
  variables: ["prcp", "PET", "streamflow"]
  time_unit: ["1D"]
  warmup_length: 365
  # Event-specific configuration
  net_rain_key: "P_eff"
  obs_flow_key: "Q_obs_eff"
  # Additional parameters
  datasource_kwargs: {}
  read_kwargs: {}

training_cfgs:
  algorithm_name: "SCE_UA"
  algorithm_params:
    # Algorithm-specific parameters
    rep: 1000
    ngs: 1000
  # Training objective (loss function)
  loss_config:
    type: "time_series"
    obj_func: "RMSE"
  # Parameter range file for traditional models
  param_range_file: "param_range.yaml"
  output_dir: "results"
  experiment_name: "experiment"

evaluation_cfgs:
  # Final evaluation metrics (after training)
  metrics: ["RMSE", "NSE", "KGE", "Bias"]
  evaluation_period: "testing"
  save_results: true
  plot_results: true
```

**Key Changes in Configuration Structure:**
- **`loss_config` moved to `training_cfgs`**: The loss function is the training objective, not an evaluation metric
- **`param_range_file` moved to `training_cfgs`**: Parameter ranges are training configurations
- **`evaluation_cfgs` simplified**: Only contains final evaluation metrics and output settings
- **Clear separation**: Training vs evaluation concerns are now properly separated

## Usage Examples

### Flood Event Data with XAJ Model

```yaml
# floodevent_xaj_example.yaml
data_cfgs:
  data_type: "floodevent"
  data_path: "path/to/flood/events"
  basin_ids: ["21401550"]
  time_unit: ["3h"]
  variables: ["P_eff", "Q_obs_eff"]
  net_rain_key: "P_eff"
  obs_flow_key: "Q_obs_eff"
  warmup_length: 0

model_cfgs:
  model_name: "xaj"
  model_params:
    time_interval_hours: 3

training_cfgs:
  algorithm_name: "SCE_UA"
  loss_config:
    type: "time_series"
    obj_func: "RMSE"
  param_range_file: "param_range.yaml"
  output_dir: "results/floodevent_xaj"

evaluation_cfgs:
  metrics: ["RMSE", "NSE", "KGE", "Bias"]
  evaluation_period: "testing"
```

### Continuous Data with XAJ Model

```yaml
# continuous_xaj_example.yaml
data_cfgs:
  data_type: "camels"
  data_path: "path/to/camels"
  basin_ids: ["01013500"]
  variables: ["prcp", "PET", "streamflow"]
  warmup_length: 365

model_cfgs:
  model_name: "xaj"
  model_params:
    time_interval_hours: 24

training_cfgs:
  algorithm_name: "SCE_UA"
  loss_config:
    type: "time_series"
    obj_func: "RMSE"
  param_range_file: "param_range.yaml"
  output_dir: "results/continuous_xaj"

evaluation_cfgs:
  metrics: ["RMSE", "NSE", "KGE", "Bias"]
  evaluation_period: "testing"
```

### Programmatic Usage

```python
from hydromodel.configs.unified_config import UnifiedConfig
from hydromodel.trainers.unified_calibrate import calibrate

# Method 1: Using UnifiedConfig (Recommended)
config = UnifiedConfig(config_file="config.yaml")
results = calibrate(config)

# Method 2: Using config dictionary
config_dict = {
    "data_cfgs": {...},     # Data configuration
    "model_cfgs": {...},    # Model configuration  
    "training_cfgs": {...}, # Training configuration (includes algorithm, loss, params)
    "evaluation_cfgs": {...} # Evaluation configuration
}
results = calibrate(config_dict)
```

## Benefits

### 1. Consistency
- Ultra-simple interface: `calibrate(config)`
- Same calibration function for all data types
- Single configuration parameter eliminates confusion

### 2. Flexibility
- Easy switching between data sources
- Support for both event and continuous data
- Extensible to new data source types

### 3. Maintainability
- Eliminated problematic conversion functions
- Clear separation of concerns
- Proper handling of different data characteristics

### 4. Robustness
- Graceful handling of missing dependencies
- Proper event segmentation for traditional models
- Error handling and validation

## Migration Guide

### For Existing Code

**Old:**
```python
# Manual data preparation
p_and_e, qobs = prepare_data_somehow()

# Create model setup
setup = UnifiedModelSetup(p_and_e, qobs, model_config, loss_config)
```

**New:**
```python
# Create data configuration
data_config = {
    "data_type": "your_data_type",
    "data_path": "path/to/data",
    "basin_ids": ["basin1"],
    # ... other config
}

# Create model setup (data loading is automatic)
setup = UnifiedModelSetup(data_config, model_config, loss_config)
```

### For Function Calls

**Old (multiple separate configs):**
```python
calibrate(
    data_config=data_config,
    model_config=model_config,
    algorithm_config=algorithm_config,
    loss_config=loss_config,
    output_dir=output_dir,
    warmup_length=365,
    param_file="param.yaml",
    basin_ids=["basin1"]
)
```

**New (clean three-config interface):**
```python
# All training-related configs unified
training_config = {
    "algorithm_name": "SCE_UA",
    "algorithm_params": {...},
    "loss_config": {"type": "time_series", "obj_func": "RMSE"},
    "param_range_file": "param.yaml",
    "output_dir": "results"
}

calibrate(
    data_config=data_config,
    model_config=model_config,
    training_config=training_config
)
```

## Future Enhancements

1. **Multi-event handling**: Enhanced support for multiple events in calibration
2. **Advanced event detection**: More sophisticated event segmentation algorithms
3. **Data validation**: Built-in data quality checks and validation
4. **Caching**: Intelligent data caching for faster repeated access
5. **Parallel processing**: Multi-basin parallel calibration support

## Files Modified

- `hydromodel/trainers/unified_calibrate.py` - Major refactor of UnifiedModelSetup
- `hydromodel/configs/unified_config.py` - Enhanced data configuration
- `hydromodel/datasets/unified_data_loader.py` - New unified data loader
- `hydromodel/datasets/__init__.py` - Updated imports
- `configs/examples/` - New example configurations
- `examples/unified_calibration_example.py` - Usage examples

## Testing

The refactored code maintains backward compatibility through configuration-based interfaces. Existing model functions (XAJ, GR series, unit hydrograph) remain unchanged - only the data loading and setup interfaces have been modified.

Testing should focus on:
1. Data loading with different source types
2. Event vs continuous data handling
3. Model simulation with segmented event data
4. Configuration validation and error handling
