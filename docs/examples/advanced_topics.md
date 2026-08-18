## Advanced Topics

### MODEL_DICT Registry

All models are registered in `models/model_dict.py`:

```python
from hydromodel.models.model_dict import MODEL_DICT

# Available models
print(MODEL_DICT.keys())  # ['xaj', 'xaj_mz', 'gr4j', ...]

# Model signature
model_func = MODEL_DICT["xaj_mz"]
qsim, intermediates = model_func(
    p_and_e,           # [time, basin, 2]
    params,            # [basin, n_params]
    warmup_length=365,
    **model_params
)
```

**Adding a new model**:

1. Implement model in `models/my_model.py`
2. Register in `MODEL_DICT`
3. Add parameter ranges to `model_config.py`
4. Model immediately works with all APIs

### Custom Loss Functions

```python
def my_custom_loss(obs, sim):
    """
    Custom objective function.

    Parameters
    ----------
    obs, sim : np.ndarray
        Shape [time, basin, 1]

    Returns
    -------
    float
        Loss value (to minimize)
    """
    # Example: Combine NSE and PBIAS
    nse = calculate_nse(obs, sim)
    pbias = calculate_pbias(obs, sim)
    return -nse + abs(pbias) / 100

# Use in configuration
config["training_cfgs"]["loss_config"] = {
    "type": "custom",
    "obj_func": my_custom_loss
}
```

### Batch Processing

Process multiple experiments programmatically:

```python
experiments = [
    {"name": "exp1", "basins": ["01013500"], "algorithm": "SCE_UA"},
    {"name": "exp2", "basins": ["01022500"], "algorithm": "GA"},
    {"name": "exp3", "basins": ["01030500"], "algorithm": "scipy"},
]

for exp in experiments:
    config["data_cfgs"]["basin_ids"] = exp["basins"]
    config["training_cfgs"]["algorithm"] = exp["algorithm"]
    config["training_cfgs"]["experiment_name"] = exp["name"]

    results = calibrate(config)
    print(f"Completed {exp['name']}")
```

### Parallel Basin Calibration

Calibrate multiple basins in parallel:

```python
from multiprocessing import Pool

def calibrate_basin(basin_id):
    """Calibrate single basin."""
    config_copy = config.copy()
    config_copy["data_cfgs"]["basin_ids"] = [basin_id]
    config_copy["training_cfgs"]["experiment_name"] = f"exp_{basin_id}"
    return calibrate(config_copy)

# Parallel execution
basin_ids = ["01013500", "01022500", "01030500"]
with Pool(processes=3) as pool:
    results = pool.map(calibrate_basin, basin_ids)
```

### Custom Parameter Ranges

Override default parameter ranges:

```yaml
# param_range.yaml
xaj_mz:
  param_name:
    - K
    - B
    - IM
  param_range:
    K: [0.5, 1.5]
    B: [0.1, 0.5]
    IM: [0.01, 0.1]
```

```python
config["training_cfgs"]["param_range_file"] = "param_range.yaml"
```

When `param_range_file` is set explicitly, the file must exist and must define
exactly the model parameters in `param_name`. Hydromodel reorders
`param_range` by `param_name`, validates each `[min, max]` pair, and fails on
missing, extra, or invalid ranges. When `param_range_file` is omitted,
hydromodel uses the built-in defaults and emits a warning.

### Intermediate States

Return intermediate model states:

```python
results = simulator.simulate(
    inputs=p_and_e,
    qobs=qobs,
    warmup_length=365,
    return_intermediate=True  # ← Enable intermediate outputs
)

# Access intermediate states (model-specific)
if "EU" in results:
    eu = results["EU"]  # Upper layer soil moisture
if "EL" in results:
    el = results["EL"]  # Lower layer soil moisture
```

---

## Best Practices

### 1. Configuration Management

- ✅ Use YAML files for all experiments
- ✅ Save configs with results (`output_dir/calibration_config.yaml`)
- ✅ Version control configurations in git
- ✅ Document parameter choices in comments

### 2. Data Quality

```python
# Always verify data before calibration
data = data_loader.load_data()
print(f"Data shape: {data[0].shape}")
print(f"Missing values: {np.isnan(data[0]).sum()}")
print(f"Data range: [{data[0].min():.2f}, {data[0].max():.2f}]")
```

### 3. Warmup Period

- ✅ Always use adequate warmup (typically 365 days)
- ✅ Exclude warmup from evaluation metrics
- ✅ Longer warmup for longer memory models

### 4. Reproducibility

```python
# Set random seeds
import numpy as np
import random

np.random.seed(1234)
random.seed(1234)

# Save exact package versions
# requirements.txt or environment.yml
```

### 5. Result Validation

```python
# After calibration, always evaluate on independent test period
results_train = evaluate(config, eval_period="train")
results_test = evaluate(config, eval_period="test")

# Compare performance
print(f"Train NSE: {results_train['metrics']['01013500']['NSE']:.3f}")
print(f"Test NSE: {results_test['metrics']['01013500']['NSE']:.3f}")
```

---

## Troubleshooting

### Common Issues

**1. Shape mismatch errors:**

```python
# Check data shapes
print(f"p_and_e: {p_and_e.shape}")  # Should be [time, basin, 2]
print(f"params: {params.shape}")     # Should be [basin, n_params]
```

**2. Parameter out of bounds:**

```python
# Check parameter ranges
from hydromodel.models.model_config import read_model_param_dict
param_dict = read_model_param_dict(None)
print(param_dict["xaj_mz"]["param_range"])
```

**3. Memory issues:**

```python
# Process basins in batches
batch_size = 10
for i in range(0, len(all_basins), batch_size):
    batch = all_basins[i:i+batch_size]
    config["data_cfgs"]["basin_ids"] = batch
    calibrate(config)
```

**4. Slow calibration:**

- Use `xaj_mz` instead of full `xaj` (fewer parameters)
- Reduce `rep` and `ngs` for testing
- Consider faster algorithms (GA, scipy)
- Profile code to find bottlenecks

---

