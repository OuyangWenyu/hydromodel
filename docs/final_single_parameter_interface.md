# Final Single Parameter Interface

## Overview

After iterative refinement, the hydromodel calibration interface has been simplified to the absolute minimum: **a single `calibrate(config)` function**.

## The Problem with Previous Designs

### Original Approach (Confusing)
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
**Issues:** Too many parameters, redundant information, unclear relationships.

### Intermediate Approach (Still Confusing)
```python
calibrate(
    data_config=data_config,
    model_config=model_config,
    training_config=training_config,
    config=config  # Redundant with above!
)
```
**Issues:** Mixing individual configs with unified config was confusing and redundant.

## Final Solution: Single Parameter

```python
def calibrate(config) -> Dict[str, Any]:
    """
    Unified calibration interface for all hydrological models.
    
    Parameters
    ----------
    config : UnifiedConfig or Dict
        Complete configuration containing all settings
    """
```

## Usage Patterns

### 1. UnifiedConfig Object (Recommended)

```python
from hydromodel.configs.unified_config import UnifiedConfig
from hydromodel.trainers.unified_calibrate import calibrate

# From file
config = UnifiedConfig(config_file="config.yaml")
results = calibrate(config)

# Programmatic
config_dict = {
    "data_cfgs": {...},
    "model_cfgs": {...}, 
    "training_cfgs": {...},
    "evaluation_cfgs": {...}
}
config = UnifiedConfig(config_dict=config_dict)
results = calibrate(config)
```

### 2. Direct Dictionary (For Advanced Users)

```python
from hydromodel.trainers.unified_calibrate import calibrate

config_dict = {
    "data_cfgs": {
        "data_type": "floodevent",
        "data_path": "path/to/data",
        "basin_ids": ["basin1"],
        "variables": ["P_eff", "Q_obs_eff"]
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
        "algorithm_params": {"rep": 1000},
        "loss_config": {"type": "time_series", "obj_func": "RMSE"},
        "output_dir": "results"
    },
    "evaluation_cfgs": {
        "metrics": ["RMSE", "NSE", "KGE"]
    }
}

results = calibrate(config_dict)
```

## Benefits

### 1. Ultimate Simplicity
- **Single function**: `calibrate(config)`
- **Single parameter**: No confusion about which parameters to use
- **Single responsibility**: Configuration contains everything

### 2. Flexibility
- **Supports UnifiedConfig objects**: Full validation and defaults
- **Supports dictionaries**: Direct programmatic control
- **Backwards compatible**: Existing configs work without change

### 3. Eliminates Confusion
- **No redundant parameters**: All information in one place
- **No multiple ways**: Only one function signature to remember
- **No parameter order**: Only one parameter to pass

### 4. Logical Structure
```
config
├── data_cfgs      # What data to use
├── model_cfgs     # What model to use  
├── training_cfgs  # How to train
└── evaluation_cfgs # How to evaluate
```

## Error Handling

```python
# Missing required sections
config_dict = {"data_cfgs": {"data_type": "camels"}}
calibrate(config_dict)  # ValueError: Missing model_cfgs, training_cfgs

# Invalid config type  
calibrate("invalid")  # ValueError: Config must be UnifiedConfig or dict

# Valid but incomplete
config_dict = {
    "data_cfgs": {...},
    "model_cfgs": {...},
    "training_cfgs": {...}
    # evaluation_cfgs is optional
}
calibrate(config_dict)  # Works fine
```

## Implementation Details

The function internally handles two config types:

```python
def calibrate(config):
    if hasattr(config, 'data_cfgs'):
        # UnifiedConfig object - use its methods
        data_config = config.data_cfgs
        model_config = config.get_model_config()
        training_config = config.get_training_config()
    elif isinstance(config, dict):
        # Dictionary - extract sections directly
        data_config = config['data_cfgs']
        model_config = {
            "name": config['model_cfgs'].get("model_name"),
            **config['model_cfgs'].get("model_params", {})
        }
        training_config = config['training_cfgs']
    else:
        raise ValueError("Invalid config type")
    
    # Continue with calibration...
```

## Migration Guide

### From Previous Versions

**Old (Multiple Parameters):**
```python
calibrate(
    data_config=data_config,
    model_config=model_config,
    training_config=training_config
)
```

**New (Single Parameter):**
```python
# Option 1: Create config dictionary
config_dict = {
    "data_cfgs": data_config,
    "model_cfgs": {
        "model_name": model_config["name"],
        "model_params": {k: v for k, v in model_config.items() if k != "name"}
    },
    "training_cfgs": training_config
}
calibrate(config_dict)

# Option 2: Use UnifiedConfig (recommended)
config = UnifiedConfig(config_dict=config_dict) 
calibrate(config)
```

## Comparison with Other Libraries

Many scientific libraries follow similar patterns:

```python
# scikit-learn
model.fit(X, y)  # Single data parameter

# TensorFlow  
model.fit(dataset)  # Single dataset parameter

# Our design
calibrate(config)  # Single config parameter
```

This aligns with the principle of **"configuration objects over parameter lists"**.

## Summary

The final `calibrate(config)` interface represents the ultimate simplification:

- ✅ **One function**: `calibrate()`
- ✅ **One parameter**: `config`
- ✅ **Two input types**: `UnifiedConfig` or `dict`
- ✅ **Zero confusion**: No redundant or optional parameters

This design is:
- **Intuitive**: Everything goes in config
- **Flexible**: Supports different config types
- **Maintainable**: Single entry point
- **Extensible**: Easy to add new config options

The interface is now as simple as it can possibly be while maintaining full functionality.
