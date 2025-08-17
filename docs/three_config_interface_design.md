# Three-Config Interface Design

## Overview

This document describes the final design of the hydromodel calibration interface, which uses a clean three-configuration approach: `data_config`, `model_config`, and `training_config`.

## Design Philosophy

### Problem with Previous Approach

The previous interface had multiple separate configuration objects being passed to the calibration function:

```python
# Previous approach (confusing and verbose)
calibrate(
    data_config=data_config,
    model_config=model_config,
    algorithm_config=algorithm_config,  # Training-related
    loss_config=loss_config,            # Training-related
    output_dir=output_dir,              # Training-related
    warmup_length=365,                  # Duplicated from data_config
    param_file="param.yaml",            # Training-related
    basin_ids=["basin1"]                # Duplicated from data_config
)
```

**Issues:**
- Multiple training-related configs scattered across parameters
- Duplicate information (warmup_length, basin_ids)
- Unclear grouping of related configurations
- Verbose function signature

### Solution: Three-Config Approach

```python
# New approach (clean and logical)
calibrate(
    data_config=data_config,        # All data-related settings
    model_config=model_config,      # All model-related settings
    training_config=training_config # All training-related settings
)
```

## Configuration Structure

### 1. Data Configuration (`data_config`)

**Purpose**: Specifies what data to load and how to process it.

```yaml
data_cfgs:
  # Data source specification
  data_type: "floodevent"           # or "camels", "selfmade"
  data_path: "path/to/data"
  dataset_name: "experiment"
  
  # Basin and time configuration
  basin_ids: ["basin1", "basin2"]
  time_periods:
    calibration: ["2020-01-01", "2022-12-31"]
    testing: ["2023-01-01", "2023-12-31"]
  
  # Data processing settings
  variables: ["net_rain", "inflow"]
  warmup_length: 365
  time_unit: ["3h"]
  
  # Event-specific settings (for flood events)
  net_rain_key: "net_rain"
  obs_flow_key: "inflow"
  
  # Additional datasource parameters
  datasource_kwargs: {}
  read_kwargs: {}
```

### 2. Model Configuration (`model_config`)

**Purpose**: Specifies the model type and model-specific hyperparameters.

```yaml
model_cfgs:
  model_name: "xaj"                 # or "unit_hydrograph", "gr4j", etc.
  model_params:
    # Model-specific parameters
    source_type: "sources"
    source_book: "HF"
    kernel_size: 15
    time_interval_hours: 3
    
    # Unit hydrograph specific (if applicable)
    n_uh: 24
    smoothing_factor: 0.1
```

### 3. Training Configuration (`training_config`)

**Purpose**: Specifies how to train/calibrate the model, including algorithm, objective function, and output settings.

```yaml
training_cfgs:
  # Optimization algorithm
  algorithm_name: "SCE_UA"
  algorithm_params:
    random_seed: 1234
    rep: 1000
    ngs: 1000
    kstop: 50
    peps: 0.1
    pcento: 0.1
  
  # Training objective (loss function)
  loss_config:
    type: "time_series"
    obj_func: "RMSE"
  
  # Parameter ranges (for traditional models)
  param_range_file: "param_range.yaml"
  
  # Output configuration
  output_dir: "results"
  experiment_name: "experiment"
```

### 4. Evaluation Configuration (`evaluation_config`)

**Purpose**: Specifies post-training evaluation metrics and output format (separate from training).

```yaml
evaluation_cfgs:
  # Final evaluation metrics (calculated after training)
  metrics: ["RMSE", "NSE", "KGE", "Bias"]
  evaluation_period: "testing"
  save_results: true
  plot_results: true
  export_format: ["json", "csv"]
```

## Usage Examples

### 1. Using UnifiedConfig (Recommended)

```python
from hydromodel.configs.unified_config import UnifiedConfig
from hydromodel.trainers.unified_calibrate import calibrate

# Load configuration from file
config = UnifiedConfig(config_file="config.yaml")

# Run calibration with unified interface
results = calibrate(config)
```

### 2. Using Config Dictionary

```python
from hydromodel.trainers.unified_calibrate import calibrate

# Create complete config dictionary
config_dict = {
    "data_cfgs": {
        "data_type": "floodevent",
        "data_path": "path/to/data",
        "basin_ids": ["basin1"],
        "variables": ["P_eff", "Q_obs_eff"],
        "warmup_length": 0
    },
    "model_cfgs": {
        "model_name": "xaj",
        "model_params": {
            "source_type": "sources",
            "time_interval_hours": 3
        }
    },
    "training_cfgs": {
        "algorithm_name": "SCE_UA",
        "algorithm_params": {"rep": 1000, "ngs": 1000},
        "loss_config": {"type": "time_series", "obj_func": "RMSE"},
        "param_range_file": "param_range.yaml",
        "output_dir": "results"
    },
    "evaluation_cfgs": {
        "metrics": ["RMSE", "NSE", "KGE"],
        "evaluation_period": "testing"
    }
}

# Run calibration
results = calibrate(config_dict)
```

### 3. Programmatic Configuration Building

```python
from hydromodel.configs.unified_config import UnifiedConfig

# Create configuration programmatically
config_dict = {
    "data_cfgs": data_config,
    "model_cfgs": {
        "model_name": "xaj",
        "model_params": model_config
    },
    "training_cfgs": training_config,
    "evaluation_cfgs": {
        "metrics": ["RMSE", "NSE", "KGE"],
        "evaluation_period": "testing"
    }
}

config = UnifiedConfig(config_dict=config_dict)
results = calibrate(config)
```

## Benefits

### 1. Conceptual Clarity

**Clear separation of concerns:**
- **Data**: What data to use and how to load it
- **Model**: What model to use and its hyperparameters  
- **Training**: How to train the model (algorithm, objective, output)
- **Evaluation**: How to assess the trained model

### 2. Interface Simplicity

**Before (7+ parameters):**
```python
calibrate(data_config, model_config, algorithm_config, loss_config, 
          output_dir, warmup_length, param_file, basin_ids)
```

**After (1 parameter):**
```python
calibrate(config)
```

### 3. Logical Grouping

**Training-related settings are now unified:**
- Algorithm settings (optimizer, parameters)
- Objective function (loss configuration)
- Parameter ranges (for traditional models)
- Output settings (directories, experiment names)

### 4. No Information Duplication

- `warmup_length` only in `data_config`
- `basin_ids` only in `data_config`
- `param_range_file` only in `training_config`
- Output settings only in `training_config`

### 5. Extensibility

Easy to add new settings to appropriate configuration sections:
- New data sources → Add to `data_config`
- New model types → Add to `model_config`
- New algorithms → Add to `training_config`
- New evaluation metrics → Add to `evaluation_config`

## Internal Implementation

### Configuration Processing

```python
def calibrate(data_config, model_config, training_config):
    # Extract algorithm configuration
    algorithm_config = {
        "name": training_config.get("algorithm_name", "SCE_UA"),
        **training_config.get("algorithm_params", {})
    }
    
    # Extract loss configuration
    loss_config = training_config.get("loss_config", {
        "type": "time_series", 
        "obj_func": "RMSE"
    })
    
    # Extract output directory
    output_dir = os.path.join(
        training_config.get("output_dir", "results"),
        training_config.get("experiment_name", "experiment")
    )
    
    # Create model setup
    model_setup = UnifiedModelSetup(
        data_config=data_config,
        model_config=model_config,
        loss_config=loss_config,
        training_config=training_config
    )
    
    # Run calibration
    return _calibrate_model(model_setup, algorithm_config, output_dir, ...)
```

### Configuration Validation

```python
class UnifiedConfig:
    def _validate_and_set_defaults(self):
        # Validate required sections exist
        required_sections = ["data_cfgs", "model_cfgs", "training_cfgs", "evaluation_cfgs"]
        
        # Set appropriate defaults for each section
        self._set_data_defaults()
        self._set_model_defaults()
        self._set_training_defaults()
        self._set_evaluation_defaults()
    
    def get_training_config(self) -> Dict:
        """Get complete training configuration."""
        return self.training_cfgs
```

## Migration Guide

### From Previous Interface

**Old code:**
```python
# Multiple scattered configurations
results = calibrate(
    data_config=data_config,
    model_config=model_config,
    algorithm_config={"name": "SCE_UA", "rep": 1000},
    loss_config={"type": "time_series", "obj_func": "RMSE"},
    output_dir="results",
    warmup_length=365,
    param_file="param.yaml"
)
```

**New code:**
```python
# Unified training configuration
training_config = {
    "algorithm_name": "SCE_UA",
    "algorithm_params": {"rep": 1000},
    "loss_config": {"type": "time_series", "obj_func": "RMSE"},
    "param_range_file": "param.yaml",
    "output_dir": "results"
}

results = calibrate(
    data_config=data_config,
    model_config=model_config,
    training_config=training_config
)
```

### Configuration File Updates

**Move settings to appropriate sections:**
- `warmup_length` → `data_cfgs`
- `basin_ids` → `data_cfgs`
- `param_range_file` → `training_cfgs`
- `loss_config` → `training_cfgs`
- `output_dir` → `training_cfgs`

## Summary

The three-config interface provides:

1. **Clarity**: Each configuration has a clear purpose
2. **Simplicity**: Only three parameters to the calibration function
3. **Consistency**: Logical grouping of related settings
4. **Maintainability**: Easy to understand and modify
5. **Extensibility**: Natural place to add new features

This design makes the hydromodel calibration interface much more intuitive and easier to use, while maintaining full functionality and flexibility.
