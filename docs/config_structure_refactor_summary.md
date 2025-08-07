# Configuration Structure Refactor Summary

## Overview

This document summarizes the refactoring of the hydromodel configuration structure to address issues with parameter placement and interface consistency.

## Problem Analysis

The original configuration structure had several conceptual issues:

1. **`param_range_file` misplaced in `data_cfgs`**: Parameter ranges are training configurations, not data configurations
2. **`loss_config` misplaced in `evaluation_cfgs`**: Loss functions are training objectives, not evaluation metrics
3. **Redundant function parameters**: Parameters were duplicated in both config and function interfaces
4. **Unclear separation**: Training vs evaluation concerns were mixed

## Solution: Reorganized Configuration Structure

### Before (Problematic)

```yaml
data_cfgs:
  # ... data configs ...
  param_range_file: "param_range.yaml"  # WRONG: Not data-related

evaluation_cfgs:
  loss_type: "time_series"              # WRONG: Not evaluation metric
  objective_function: "RMSE"            # WRONG: This is training objective
  metrics: ["RMSE", "NSE", "KGE"]       # Correct: Final evaluation metrics
```

Function interface had redundant parameters:
```python
calibrate(
    data_config=data_config,
    model_config=model_config, 
    algorithm_config=algorithm_config,
    loss_config=loss_config,
    warmup_length=365,          # Redundant: Already in data_config
    param_file="param.yaml",    # Redundant: Should be in training_config
    basin_ids=["basin1"],       # Redundant: Already in data_config
)
```

### After (Corrected)

```yaml
data_cfgs:
  # Pure data configurations
  data_type: "floodevent"
  data_path: "path/to/data"
  basin_ids: ["basin1"]
  warmup_length: 365
  variables: ["P_eff", "Q_obs_eff"]
  # ... other data configs ...

training_cfgs:
  # Algorithm configuration
  algorithm_name: "SCE_UA"
  algorithm_params:
    rep: 1000
    ngs: 1000
  
  # Training objective (loss function)
  loss_config:
    type: "time_series"
    obj_func: "RMSE"
  
  # Parameter ranges for traditional models
  param_range_file: "param_range.yaml"
  
  # Output configuration
  output_dir: "results"
  experiment_name: "experiment"

evaluation_cfgs:
  # Pure evaluation configurations
  metrics: ["RMSE", "NSE", "KGE", "Bias"]  # Final evaluation metrics
  evaluation_period: "testing"
  save_results: true
  plot_results: true
  export_format: ["json", "csv"]
```

Simplified function interface:
```python
calibrate(
    data_config=data_config,      # All data configs
    model_config=model_config,    # All model configs  
    algorithm_config=algorithm_config,  # All training configs
    loss_config=loss_config,      # Training objective
    output_dir=output_dir,        # Output directory
)
```

## Key Changes Made

### 1. Configuration Structure Changes

**`data_cfgs` (Pure Data Configuration):**
- ✅ Kept: `data_type`, `data_path`, `basin_ids`, `warmup_length`, `variables`
- ❌ Removed: `param_range_file` (moved to `training_cfgs`)

**`training_cfgs` (Training & Optimization Configuration):**
- ✅ Added: `loss_config` (moved from `evaluation_cfgs`)
- ✅ Added: `param_range_file` (moved from `data_cfgs`)
- ✅ Kept: `algorithm_name`, `algorithm_params`, `output_dir`

**`evaluation_cfgs` (Post-Training Evaluation Configuration):**
- ❌ Removed: `loss_type`, `objective_function` (moved to `training_cfgs`)
- ✅ Kept: `metrics`, `evaluation_period`
- ✅ Added: `save_results`, `plot_results`, `export_format`

### 2. Function Interface Simplification

**Removed redundant parameters from `calibrate()`:**
- `warmup_length` → Available in `data_config`
- `param_file` → Available in `training_config`
- `basin_ids` → Available in `data_config`

**Updated `UnifiedModelSetup` constructor:**
```python
# Before
UnifiedModelSetup(p_and_e, qobs, model_config, loss_config, warmup_length, param_file)

# After  
UnifiedModelSetup(data_config, model_config, loss_config, training_config)
```

### 3. Configuration Access Methods

Updated `UnifiedConfig` methods:

```python
# Loss config now comes from training_cfgs
def get_loss_config(self) -> Dict:
    training_cfg = self.training_cfgs
    loss_cfg = training_cfg.get("loss_config", {})
    return {
        "type": loss_cfg.get("type", "time_series"),
        "obj_func": loss_cfg.get("obj_func", "RMSE"),
        "events": loss_cfg.get("events_config")
    }

# Algorithm config includes param_range_file
def get_algorithm_config(self) -> Dict:
    # ... existing algorithm params ...
    config["param_range_file"] = self.training_cfgs.get("param_range_file")
    return config
```

## Conceptual Clarity

### Training vs Evaluation

**Training Configuration (`training_cfgs`):**
- **Purpose**: Controls the optimization/calibration process
- **Contains**: Algorithm settings, loss function, parameter ranges, output settings
- **Used during**: Model calibration/training phase

**Evaluation Configuration (`evaluation_cfgs`):**  
- **Purpose**: Controls post-training model assessment
- **Contains**: Evaluation metrics, test periods, result formatting
- **Used during**: Model evaluation phase (after training is complete)

### Data vs Model vs Training

**Data Configuration (`data_cfgs`):**
- **Purpose**: Specifies what data to load and how
- **Contains**: Data sources, time periods, variables, preprocessing settings

**Model Configuration (`model_cfgs`):**
- **Purpose**: Specifies model type and model-specific parameters
- **Contains**: Model name, model hyperparameters

**Training Configuration (`training_cfgs`):**
- **Purpose**: Specifies how to train/calibrate the model
- **Contains**: Optimization algorithm, loss function, parameter ranges

## Benefits

1. **Conceptual Clarity**: Clear separation of concerns
2. **Consistency**: Configuration structure matches its purpose
3. **Simplicity**: Fewer redundant parameters in function interfaces
4. **Maintainability**: Easier to understand and modify
5. **Extensibility**: New configurations can be added to appropriate sections

## Migration Guide

### Configuration Files

**Old structure:**
```yaml
data_cfgs:
  param_range_file: "param.yaml"  # Move to training_cfgs

evaluation_cfgs:
  loss_type: "time_series"        # Move to training_cfgs.loss_config.type
  objective_function: "RMSE"      # Move to training_cfgs.loss_config.obj_func
```

**New structure:**
```yaml
training_cfgs:
  loss_config:
    type: "time_series"
    obj_func: "RMSE"
  param_range_file: "param.yaml"

evaluation_cfgs:
  metrics: ["RMSE", "NSE", "KGE"]
```

### Code Changes

**Function calls:**
```python
# Old (with redundant parameters)
calibrate(data_config, model_config, algorithm_config, loss_config, 
          output_dir, warmup_length=365, param_file="param.yaml", basin_ids=["basin1"])

# New (single unified function, two usage patterns)
# Pattern 1: Using config object
calibrate(config=config)

# Pattern 2: Using individual configs  
calibrate(data_config, model_config, training_config)
```

**Model setup:**
```python
# Old
setup = UnifiedModelSetup(p_and_e, qobs, model_config, loss_config, warmup_length, param_file)

# New
setup = UnifiedModelSetup(data_config, model_config, loss_config, training_config)
```

## Files Modified

- `hydromodel/configs/unified_config.py` - Updated defaults and access methods
- `hydromodel/trainers/unified_calibrate.py` - Simplified function interfaces
- `configs/examples/*.yaml` - Updated example configurations
- `docs/unified_data_interface_refactor.md` - Updated documentation

This refactoring provides a much cleaner and more logical configuration structure that properly separates data, model, training, and evaluation concerns.

## Final Simplification: Single Function Design

The final design uses only one `calibrate()` function that supports two usage patterns:

```python
from hydromodel.trainers.unified_calibrate import calibrate

# Pattern 1: UnifiedConfig object (recommended for most users)
config = UnifiedConfig(config_file="config.yaml")
results = calibrate(config=config)

# Pattern 2: Individual config dictionaries (for programmatic use)
results = calibrate(
    data_config=data_config,
    model_config=model_config, 
    training_config=training_config
)
```

This eliminates the need for separate `calibrate_with_config()` function while maintaining full flexibility and backward compatibility.
