# Usage Guide

This guide demonstrates how to use hydromodel for hydrological modeling, calibration, and evaluation using the unified API architecture.

## Overview

hydromodel provides a unified interface for:
- Model calibration with multiple algorithms (SCE-UA, GA, scipy optimizers)
- Model evaluation with comprehensive metrics
- Direct model simulation
- Configuration-based workflows for reproducibility

## Quick Start

### 1. Basic Calibration

The simplest way to calibrate a model is using the unified calibration API:

```python
from hydromodel.trainers.unified_calibrate import calibrate

config = {
    "data_cfgs": {
        "data_source_type": "camels_us",
        "data_source_path": "D:/data/camels",
        "basin_ids": ["01013500"],
        "train_period": ["1990-10-01", "2000-09-30"],
        "test_period": ["2000-10-01", "2010-09-30"],
        "warmup_length": 365,
    },
    "model_cfgs": {
        "model_name": "xaj_mz",
        "source_type": "sources",
        "source_book": "HF",
    },
    "training_cfgs": {
        "algorithm": "SCE_UA",
        "loss_func": "RMSE",
        "output_dir": "results",
        "experiment_name": "my_experiment",
        "rep": 10000,
        "ngs": 100,
    },
    "evaluation_cfgs": {
        "metrics": ["NSE", "KGE", "RMSE"],
    },
}

# Run calibration
results = calibrate(config)
print("Calibration complete!")
```

### 2. Basic Evaluation

After calibration, evaluate the model:

```python
from hydromodel.trainers.unified_evaluate import evaluate

# Evaluate on test period
results = evaluate(
    config,
    param_dir="results/my_experiment",
    eval_period="test"
)

# Print metrics
print("\nEvaluation Metrics:")
for basin_id, metrics in results['metrics'].items():
    print(f"{basin_id}:")
    print(f"  NSE: {metrics['NSE']:.3f}")
    print(f"  KGE: {metrics['KGE']:.3f}")
    print(f"  RMSE: {metrics['RMSE']:.3f}")
```

## Configuration-Based Workflow

For production use, we recommend using YAML configuration files:

### Step 1: Create Configuration File

Create `calibration_config.yaml`:

```yaml
data:
  dataset: "camels_us"
  path: "D:/data/camels"
  basin_ids: ["01013500", "01022500"]
  train_period: ["1990-10-01", "2000-09-30"]
  test_period: ["2000-10-01", "2010-09-30"]
  warmup_length: 365
  output_dir: "results"
  experiment_name: "xaj_camels_multi"

model:
  name: "xaj_mz"
  params:
    source_type: "sources"
    source_book: "HF"
    kernel_size: 15
    time_interval_hours: 24

training:
  algorithm: "SCE_UA"
  loss: "RMSE"
  SCE_UA:
    random_seed: 1234
    rep: 10000
    ngs: 100
    kstop: 50
    peps: 0.1
    pcento: 0.1

evaluation:
  metrics: ["NSE", "KGE", "RMSE", "PBIAS"]
```

### Step 2: Run from Command Line

```bash
# Calibration
python scripts/run_xaj_calibration.py --config calibration_config.yaml

# Evaluation
python scripts/run_xaj_evaluate.py --exp xaj_camels_multi --eval-period test
```

### Step 3: Access Results

Results are organized in a clear directory structure:

```
results/xaj_camels_multi/
├── calibration_config.yaml           # Configuration used
├── param_range.yaml                  # Parameter ranges
├── 01013500_sceua.csv               # Calibration history for basin 1
├── 01022500_sceua.csv               # Calibration history for basin 2
└── evaluation_test/                  # Test period evaluation
    ├── basins_metrics.csv            # Performance metrics
    ├── basins_denorm_params.csv      # Calibrated parameters
    └── xaj_mz_evaluation_results.nc  # Full simulation results
```

## Data Sources

### Using CAMELS Datasets

hydromodel integrates seamlessly with [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) for CAMELS data:

```python
config = {
    "data_cfgs": {
        "data_source_type": "camels_us",  # or camels_gb, camels_aus, etc.
        "data_source_path": "D:/data/camels",
        "basin_ids": ["01013500", "01022500"],
        # ... other settings
    }
}
```

The data will be automatically downloaded and cached on first access.

### Using Custom Data

For your own data, use the `selfmadehydrodataset` format:

```python
config = {
    "data_cfgs": {
        "data_source_type": "selfmadehydrodataset",
        "data_source_path": "D:/my_data/basins",
        "basin_ids": ["basin_001", "basin_002"],
        # ... other settings
    }
}
```

Data should be organized as:
```
D:/my_data/basins/
├── attributes.nc              # Static basin attributes
└── timeseries/
    ├── basin_001_lump.nc      # Time series for basin 001
    └── basin_002_lump.nc      # Time series for basin 002
```

Required variables:
- Time series: `prcp` (precipitation), `PET` (potential ET), `streamflow`
- Attributes: `area` (basin area in km²)

## Model Configuration

### Available Models

- **xaj**: Standard XAJ model with traditional routing
- **xaj_mz**: XAJ with Muskingum routing (recommended)

```python
model_cfgs = {
    "model_name": "xaj_mz",
    "source_type": "sources",  # Runoff generation method
    "source_book": "HF",       # Parameter source book
    "kernel_size": 15,         # Unit hydrograph kernel size
    "time_interval_hours": 24, # Time step (hours)
}
```

### Parameter Ranges

You can customize parameter ranges using a YAML file:

```yaml
# param_range.yaml
xaj_mz:
  K:
    min: 0.1
    max: 2.0
  B:
    min: 0.1
    max: 0.5
  IM:
    min: 0.01
    max: 0.05
  # ... more parameters
```

Then specify in training config:

```python
training_cfgs = {
    "param_range_file": "path/to/param_range.yaml",
    # ... other settings
}
```

## Calibration Algorithms

### SCE-UA (Recommended)

Shuffled Complex Evolution - University of Arizona:

```python
training_cfgs = {
    "algorithm": "SCE_UA",
    "loss_func": "RMSE",  # or "NSE", "KGE"
    "rep": 10000,         # Maximum iterations
    "ngs": 100,           # Number of complexes
    "kstop": 50,          # Stopping criteria
    "peps": 0.1,          # Convergence threshold
    "pcento": 0.1,        # Convergence percentage
    "random_seed": 1234,  # For reproducibility
}
```

### Genetic Algorithm

```python
training_cfgs = {
    "algorithm": "GA",
    "loss_func": "RMSE",
    "run_counts": 2,      # Number of runs
    "pop_num": 50,        # Population size
    "cross_prob": 0.5,    # Crossover probability
    "mut_prob": 0.5,      # Mutation probability
    "random_seed": 1234,
}
```

### Scipy Optimizers

```python
training_cfgs = {
    "algorithm": "scipy",
    "loss_func": "RMSE",
    "method": "Nelder-Mead",  # or "Powell", "COBYLA", etc.
    "options": {
        "maxiter": 1000,
        "disp": True,
    },
}
```

## Evaluation Options

### Evaluation Periods

```python
# Evaluate on training period
results = evaluate(config, param_dir="...", eval_period="train")

# Evaluate on test period (default)
results = evaluate(config, param_dir="...", eval_period="test")

# Evaluate on custom period
results = evaluate(
    config,
    param_dir="...",
    eval_period="custom",
    custom_period=["2010-10-01", "2015-09-30"]
)
```

### Available Metrics

The evaluation calculates multiple performance metrics:

- **NSE**: Nash-Sutcliffe Efficiency (higher is better, max=1)
- **KGE**: Kling-Gupta Efficiency (higher is better, max=1)
- **RMSE**: Root Mean Square Error (lower is better, min=0)
- **PBIAS**: Percent Bias (closer to 0 is better)
- **FHV**: High flow volume error
- **FLV**: Low flow volume error
- **FMS**: Mid-segment slope of flow duration curve

```python
evaluation_cfgs = {
    "metrics": ["NSE", "KGE", "RMSE", "PBIAS", "FHV", "FLV", "FMS"]
}
```

## Direct Model Simulation

For research or custom workflows, you can use models directly:

```python
from hydromodel.models.model_factory import model_factory
import numpy as np

# Create model
model = model_factory(
    model_name="xaj_mz",
    source_type="sources",
    source_book="HF"
)

# Prepare input data
n_basins = 2
n_time = 1000
p = np.random.rand(n_basins, n_time) * 10      # Precipitation (mm/day)
pet = np.random.rand(n_basins, n_time) * 5     # PET (mm/day)

# Model parameters (normalized [0,1])
n_params = model.param_limits.shape[0]
params = np.random.rand(n_basins, n_params)

# Run simulation
q_sim, intermediates = model.run(p, pet, params)

print(f"Simulated flow shape: {q_sim.shape}")  # (n_basins, n_time)
print(f"Intermediate states: {intermediates.keys()}")
```

### Parameter Normalization

Models use normalized parameters [0,1]. To convert:

```python
from hydromodel.datasets.data_preprocess import denormalize_params

# Denormalize parameters
param_ranges = model.param_limits  # (n_params, 2) array
physical_params = denormalize_params(params, param_ranges)

print("Normalized:", params[0, :5])
print("Physical:", physical_params[0, :5])
```

## Cross-Validation

Enable k-fold cross-validation:

```yaml
data:
  cv_fold: 5  # 5-fold cross-validation
  # ... other settings
```

The calibration will:
1. Split training period into 5 folds
2. Calibrate on 4 folds, validate on 1 fold
3. Repeat for all combinations
4. Report performance on each fold

## Multi-Basin Calibration

Calibrate multiple basins independently:

```python
config = {
    "data_cfgs": {
        "basin_ids": ["01013500", "01022500", "01030500"],
        # ... other settings
    }
}

results = calibrate(config)
# Results will contain calibrated parameters for each basin
```

Each basin is calibrated separately with its own parameter set.

## Advanced Usage

### Resuming Calibration

If calibration is interrupted, you can resume:

```python
training_cfgs = {
    "resume_from": "results/my_experiment/checkpoint.pkl",
    # ... other settings
}
```

### Custom Loss Functions

Define custom objective functions:

```python
def my_custom_loss(obs, sim):
    """Custom loss combining NSE and PBIAS."""
    nse = 1 - np.sum((obs - sim)**2) / np.sum((obs - np.mean(obs))**2)
    pbias = np.sum(sim - obs) / np.sum(obs) * 100
    return -nse + abs(pbias) / 100  # Minimize this

# Use in config
training_cfgs = {
    "loss_func": my_custom_loss,
    # ... other settings
}
```

### Batch Processing

Process multiple experiments:

```python
experiments = [
    {"name": "exp1", "basin_ids": ["01013500"]},
    {"name": "exp2", "basin_ids": ["01022500"]},
    {"name": "exp3", "basin_ids": ["01030500"]},
]

for exp in experiments:
    config["data_cfgs"]["basin_ids"] = exp["basin_ids"]
    config["training_cfgs"]["experiment_name"] = exp["name"]

    results = calibrate(config)
    print(f"Completed {exp['name']}")
```

## Best Practices

### 1. Start Small
Begin with a single basin and short period to verify setup:
```python
"basin_ids": ["01013500"],
"train_period": ["1990-10-01", "1991-09-30"],  # 1 year
```

### 2. Warmup Period
Always use adequate warmup (typically 365 days):
```python
"warmup_length": 365,  # Days
```

### 3. Save Configurations
Always save configs for reproducibility:
```bash
python run_xaj_calibration.py --config config.yaml --save-config
```

### 4. Version Control
Track your configuration files in git:
```bash
git add calibration_config.yaml param_range.yaml
git commit -m "Add calibration config for experiment X"
```

### 5. Monitor Progress
Check calibration progress regularly:
```python
# In your script
if iteration % 100 == 0:
    print(f"Iteration {iteration}, Best loss: {best_loss:.4f}")
```

### 6. Validate Results
Always evaluate on independent test period:
```python
# Training period: 1990-2000
# Test period: 2000-2010 (completely independent)
```

## Troubleshooting

### Common Issues

**1. "Data not found" error:**
```python
# Verify data path
from hydrodataset.camels_us import CamelsUs
ds = CamelsUs("D:/data/camels")
basin_ids = ds.read_object_ids()
print(f"Found {len(basin_ids)} basins")
```

**2. Calibration converges too fast:**
```python
# Increase complexity limits
training_cfgs = {
    "rep": 20000,  # More iterations
    "kstop": 100,  # Stricter stopping
}
```

**3. Memory issues with large datasets:**
```python
# Process basins in batches
batch_size = 10
for i in range(0, len(all_basins), batch_size):
    config["basin_ids"] = all_basins[i:i+batch_size]
    calibrate(config)
```

**4. Slow calibration:**
- Use fewer basins initially
- Reduce `rep` and `ngs` for testing
- Consider using faster algorithm (scipy)
- Use xaj_mz instead of xaj (fewer parameters)

## Next Steps

- See [Examples](examples.md) for complete workflows
- Read [XAJ Model](models/xaj.md) for model details
- Check [API Reference](api.md) for detailed documentation
- Browse [FAQ](faq.md) for common questions
