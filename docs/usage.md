# Usage Guide

> **Document Purpose**: This guide is designed for **developers** who need detailed understanding of hydromodel's code architecture, API design, and internal workflows. For end users who want to quickly start using hydromodel, please refer to [Quick Start Guide](quickstart.md).

This guide demonstrates how to use hydromodel's unified API architecture for hydrological modeling, calibration, evaluation, and simulation. It provides comprehensive documentation on the codebase structure and design principles.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Data Loading](#data-loading)
- [Model Calibration](#model-calibration)
- [Model Evaluation](#model-evaluation)
- [Model Simulation](#model-simulation)
- [Configuration System](#configuration-system)
- [Advanced Topics](#advanced-topics)

---

## Overview

### Unified API Design

hydromodel provides a **completely unified interface** for all hydrological models:

```
Data → UnifiedDataLoader → UnifiedCalibrator → UnifiedEvaluator
                                ↓
                           UnifiedSimulator
```

**Core Design Principles:**

1. **Unified Interfaces**: All models (XAJ, GR series, etc.) use the same API
2. **Configuration-Based**: YAML configs for reproducibility
3. **Decoupled Components**: Calibration, evaluation, and simulation are independent
4. **Flexible Integration**: Works with CAMELS datasets and custom data

### Key Components

| Component | Purpose | Module |
|-----------|---------|--------|
| `UnifiedDataLoader` | Load and preprocess data | `datasets.unified_data_loader` |
| `calibrate()` | Model calibration | `trainers.unified_calibrate` |
| `evaluate()` | Model evaluation | `trainers.unified_evaluate` |
| `UnifiedSimulator` | Direct model simulation | `trainers.unified_simulate` |
| `MODEL_DICT` | Model registry | `models.model_dict` |

---

## Architecture

### Code Structure

```
hydromodel/
├── models/                         # Model implementations
│   ├── xaj.py                     # XAJ model core
│   ├── xaj_mz.py                  # XAJ with Muskingum routing
│   ├── gr4j.py                    # GR4J model
│   ├── model_dict.py              # Model registry (MODEL_DICT)
│   └── model_config.py            # Parameter ranges and configs
│
├── trainers/                       # Calibration, evaluation, simulation
│   ├── unified_calibrate.py       # Unified calibration interface
│   ├── unified_evaluate.py        # Unified evaluation interface
│   ├── unified_simulate.py        # Unified simulation interface
│   ├── calibrate_sceua.py         # SCE-UA algorithm
│   └── calibrate_ga.py            # Genetic algorithm
│
├── datasets/                       # Data loading and preprocessing
│   ├── unified_data_loader.py     # Unified data loading interface
│   └── data_preprocess.py         # Data preprocessing utilities
│
└── scripts/                        # Command-line interfaces
    ├── run_xaj_calibration.py     # Calibration script
    ├── run_xaj_evaluate.py        # Evaluation script
    └── run_xaj_simulate.py        # Simulation script
```

### Data Flow

```
1. Configuration (YAML)
   ↓
2. UnifiedDataLoader
   ↓ (load time series, attributes)
3. Data Tensors [time, basin, features]
   ↓
4. UnifiedCalibrator/UnifiedSimulator
   ↓ (run model with parameters)
5. Results [time, basin, outputs]
   ↓
6. UnifiedEvaluator
   ↓ (calculate metrics)
7. Performance Metrics + NetCDF outputs
```

---

## Data Loading

### UnifiedDataLoader

`UnifiedDataLoader` provides a **consistent interface** for loading data from multiple sources:

```python
from hydromodel.datasets.unified_data_loader import UnifiedDataLoader

# Configuration
data_config = {
    "data_source_type": "camels_us",  # or "selfmadehydrodataset"
    "basin_ids": ["01013500"],
    "test_period": ["2000-10-01", "2010-09-30"],
    "warmup_length": 365,
    "variables": ["precipitation", "potential_evapotranspiration", "streamflow"]
}

# Load data
data_loader = UnifiedDataLoader(data_config, is_train_val_test="test")
p_and_e, qobs = data_loader.load_data()

# Data shapes
print(f"Inputs: {p_and_e.shape}")   # [time, basins, 2] (prcp + pet)
print(f"Qobs: {qobs.shape}")         # [time, basins, 1]

# Get basin attributes
basin_configs = data_loader.get_basin_configs()
basin_area = basin_configs["01013500"]["area"]  # km²
```

### Data Format

All data is returned in **standardized tensors**:

- **Shape**: `[time_steps, num_basins, num_features]`
- **Type**: `numpy.ndarray` (float32 or float64)
- **Order**: Time-major for efficiency

**Example**:
```python
# p_and_e shape: [3653, 2, 2]
# - 3653 time steps
# - 2 basins
# - 2 features (precipitation, PET)

# Access basin 0, all times, precipitation:
prcp_basin0 = p_and_e[:, 0, 0]

# Access all basins, time 100, PET:
pet_t100 = p_and_e[100, :, 1]
```

### Supported Data Sources

| Data Source | `data_source_type` | Package |
|-------------|-------------------|---------|
| CAMELS-US | `camels_us` | hydrodataset |
| CAMELS-GB | `camels_gb` | hydrodataset |
| CAMELS-AUS | `camels_aus` | hydrodataset |
| ... | ... | hydrodataset |
| Custom Data | `selfmadehydrodataset` | hydrodatasource |

See [Data Guide](data_guide.md) for detailed data preparation instructions.

---

## Model Calibration

### Unified Calibration API

The `calibrate()` function provides a **completely unified** calibration interface:

```python
from hydromodel.trainers.unified_calibrate import calibrate

config = {
    "data_cfgs": {
        "data_source_type": "camels_us",
        "basin_ids": ["01013500"],
        "train_period": ["1990-10-01", "2000-09-30"],
        "test_period": ["2000-10-01", "2010-09-30"],
        "warmup_length": 365,
    },
    "model_cfgs": {
        "model_name": "xaj_mz",
        "model_params": {
            "source_type": "sources",
            "source_book": "HF",
        },
    },
    "training_cfgs": {
        "algorithm_name": "SCE_UA",
        "algorithm_params": {
            "rep": 10000,
            "ngs": 100,
            "random_seed": 1234,
        },
        "loss_config": {
            "type": "time_series",
            "obj_func": "RMSE",
        },
        "output_dir": "results",
        "experiment_name": "my_experiment",
    },
    "evaluation_cfgs": {
        "metrics": ["NSE", "KGE", "RMSE"],
    },
}

# Run calibration
results = calibrate(config)
```

### Internal Workflow

```
calibrate()
  ↓
1. Parse configuration
  ↓
2. UnifiedDataLoader.load_data()
  ↓
3. Create UnifiedModelSetup (wraps MODEL_DICT)
  ↓
4. Select algorithm (SCE_UA, GA, scipy)
  ↓
5. For each basin:
     a. Initialize parameters (normalized [0,1])
     b. Run optimization loop
     c. For each iteration:
        - Denormalize parameters
        - Call MODEL_DICT[model_name](inputs, params, ...)
        - Calculate objective function
        - Update parameters
     d. Save best parameters
  ↓
6. Save results to output_dir
```

### Algorithm Implementations

#### SCE-UA (Recommended)

Uses `spotpy` library:

```python
training_cfgs = {
    "algorithm_name": "SCE_UA",
    "algorithm_params": {
        "rep": 10000,         # Maximum iterations
        "ngs": 100,           # Number of complexes
        "kstop": 50,          # Stopping criteria
        "peps": 0.1,          # Convergence threshold
        "pcento": 0.1,        # Convergence percentage
        "random_seed": 1234,
    },
    "loss_config": {
        "type": "time_series",
        "obj_func": "RMSE",   # or "NSE", "KGE"
    },
}
```

**Output**: `{basin_id}_sceua.csv` with columns:
- `like1`: Objective function value
- `parK`, `parB`, ...: Parameter values (with `par` prefix)
- `simulation1_1`, ...: Simulation results for each iteration

#### Genetic Algorithm

Uses `DEAP` library:

```python
training_cfgs = {
    "algorithm_name": "GA",
    "algorithm_params": {
        "run_counts": 2,      # Number of evolutionary runs
        "pop_num": 50,        # Population size
        "cross_prob": 0.5,    # Crossover probability
        "mut_prob": 0.5,      # Mutation probability
        "save_freq": 1,       # Save frequency
        "random_seed": 1234,
    },
}
```

**Output**: Pickled checkpoints (`epoch{N}.pkl`) containing:
- `population`: Current population
- `halloffame`: Best individuals
- `logbook`: Optimization history

#### Scipy Optimizers

```python
training_cfgs = {
    "algorithm_name": "scipy",
    "algorithm_params": {
        "method": "Nelder-Mead",  # or "Powell", "COBYLA"
        "options": {
            "maxiter": 1000,
            "disp": True,
        },
    },
}
```

### Parameter Management

Parameters are **always normalized** to [0, 1] during optimization:

```python
from hydromodel.models.model_config import read_model_param_dict

# Get parameter ranges
param_dict = read_model_param_dict(None)  # Uses default
param_ranges = param_dict["xaj_mz"]

print(param_ranges["param_name"])   # ['K', 'B', 'IM', ...]
print(param_ranges["param_range"])  # [[min, max], [min, max], ...]

# During optimization:
# 1. Optimizer works with normalized params [0, 1]
# 2. Before model call: denormalize to physical range
# 3. Run model with physical parameters
# 4. Calculate objective function
```

### Output Files

```
results/{experiment_name}/
├── {basin_id}_sceua.csv            # SCE-UA calibration history
├── calibration_config.yaml          # Config used (for reproducibility)
└── param_range.yaml                 # Parameter ranges used
```

---

## Model Evaluation

### Unified Evaluation API

```python
from hydromodel.trainers.unified_evaluate import evaluate

# Evaluate on test period
results = evaluate(
    config,
    param_dir="results/my_experiment",
    eval_period="test"  # "train", "test", or "custom"
)

# Evaluate on custom period
results = evaluate(
    config,
    param_dir="results/my_experiment",
    eval_period="custom",
    custom_period=["2010-10-01", "2015-09-30"]
)
```

### Internal Workflow

```
evaluate()
  ↓
1. Load configuration and calibrated parameters
  ↓
2. Load data for evaluation period
  ↓
3. For each basin:
     a. Load best parameters from calibration
     b. Create UnifiedSimulator
     c. Run simulation
     d. Calculate metrics
  ↓
4. Save results:
     - basins_metrics.csv
     - basins_denorm_params.csv
     - {model_name}_evaluation_results.nc
```

### Metrics

All metrics are calculated using `hydroutils.hydro_stat`:

| Metric | Description | Range | Optimal |
|--------|-------------|-------|---------|
| NSE | Nash-Sutcliffe Efficiency | (-∞, 1] | 1.0 |
| KGE | Kling-Gupta Efficiency | (-∞, 1] | 1.0 |
| RMSE | Root Mean Square Error | [0, ∞) | 0.0 |
| PBIAS | Percent Bias | (-∞, ∞) | 0.0 |
| FHV | High Flow Volume Error | [0, ∞) | 0.0 |
| FLV | Low Flow Volume Error | [0, ∞) | 0.0 |
| FMS | Mid-segment Slope of FDC | (-∞, ∞) | 0.0 |

**Implementation**:
```python
from hydroutils import hydro_stat

# qobs, qsim shape: [n_basins, n_time]
metrics = hydro_stat.stat_error(qobs, qsim)

print(metrics["NSE"])    # [n_basins]
print(metrics["KGE"])    # [n_basins]
```

### Output Files

```
results/{experiment_name}/evaluation_{period}/
├── basins_metrics.csv                    # Performance metrics for all basins
├── basins_norm_params.csv                # Normalized parameters [0,1]
├── basins_denorm_params.csv              # Physical parameters
└── {model_name}_evaluation_results.nc    # Full simulation results (NetCDF)
```

**NetCDF structure**:
```python
import xarray as xr

ds = xr.open_dataset("xaj_mz_evaluation_results.nc")

print(ds.dims)      # {'basin': N, 'time': T}
print(ds.data_vars) # qsim, qobs, prcp, pet, ...

# Access data
qsim = ds['qsim'].values  # [basin, time]
qobs = ds['qobs'].values  # [basin, time]
```

---

## Model Simulation

### Important Design Principle

⚠️ **Simulation does NOT require prior calibration!**

`UnifiedSimulator` is a **completely independent** simulation interface. You can:
- Run simulations with any parameter values
- Use parameters from literature
- Use calibrated parameters
- Perform sensitivity analysis

**Simulation and calibration are fully decoupled** - this is a core design principle.

### UnifiedSimulator API

```python
from hydromodel.trainers.unified_simulate import UnifiedSimulator
from hydromodel.datasets.unified_data_loader import UnifiedDataLoader

# Step 1: Load data
data_loader = UnifiedDataLoader(data_config, is_train_val_test="test")
p_and_e, qobs = data_loader.load_data()
basin_configs = data_loader.get_basin_configs()

# Step 2: Define parameters (from anywhere!)
parameters = {
    "K": 0.75, "B": 0.25, "IM": 0.06,
    "UM": 18.0, "LM": 80.0, "DM": 95.0,
    "C": 0.18, "SM": 120.0, "EX": 1.5,
    "KI": 0.35, "KG": 0.45, "A": 0.85,
    "THETA": 0.012, "CI": 0.85, "CG": 0.95
}

# Step 3: Create simulator
model_config = {
    "model_name": "xaj_mz",
    "model_params": {
        "source_type": "sources",
        "source_book": "HF",
    },
    "parameters": parameters
}

basin_id = data_config["basin_ids"][0]
simulator = UnifiedSimulator(model_config, basin_configs[basin_id])

# Step 4: Run simulation
results = simulator.simulate(
    inputs=p_and_e,
    qobs=qobs,
    warmup_length=365,
    return_intermediate=False
)

# Step 5: Extract results
qsim = results["qsim"]  # [time, basin, 1] simulated streamflow

# Calculate metrics
from hydroutils import hydro_stat
qsim_2d = qsim[365:, 0, 0].reshape(1, -1)
qobs_2d = qobs[365:, 0, 0].reshape(1, -1)
metrics = hydro_stat.stat_error(qobs_2d, qsim_2d)
print(f"NSE: {metrics['NSE'][0]:.3f}")
```

### UnifiedSimulator Design

**Core Philosophy**: All models use the **same interface** regardless of internal complexity.

```python
class UnifiedSimulator:
    def __init__(self, model_config, basin_config):
        """
        Parameters
        ----------
        model_config : dict
            - model_name: str (e.g., "xaj_mz", "gr4j")
            - model_params: dict (model-specific configs)
            - parameters: OrderedDict (calibratable parameters)

        basin_config : dict (optional)
            - basin_area: float (km²)
            - other basin attributes
        """
        self.model_name = model_config["model_name"]
        self.parameters = model_config["parameters"]
        # Initialize model from MODEL_DICT

    def simulate(self, inputs, qobs=None, warmup_length=0, return_intermediate=False):
        """
        Run model simulation.

        Parameters
        ----------
        inputs : np.ndarray
            Shape [time, basin, features] (e.g., [T, N, 2] for prcp+pet)
        qobs : np.ndarray, optional
            Shape [time, basin, 1], observed streamflow
        warmup_length : int
            Number of warmup time steps
        return_intermediate : bool
            Return intermediate states?

        Returns
        -------
        dict
            Model-specific outputs (e.g., {"qsim": [...], "es": [...]})
        """
        # Normalize parameters to [0,1] if needed
        # Call MODEL_DICT[model_name](inputs, params, ...)
        # Return results in unified format
```

### Parameter Loading

UnifiedSimulator accepts parameters from **any source**:

#### 1. From Calibration (CSV)

```python
import pandas as pd
from collections import OrderedDict

# Load SCE-UA results
df = pd.read_csv("results/exp/01013500_sceua.csv")
best_idx = df["like1"].idxmin()
best_row = df.iloc[best_idx]

# Extract parameters
param_names = ["K", "B", "IM", "UM", "LM", "DM", "C", "SM", "EX", "KI", "KG", "A", "THETA", "CI", "CG"]
parameters = OrderedDict()
for name in param_names:
    parameters[name] = float(best_row[f"par{name}"])
```

#### 2. From YAML

```yaml
# configs/my_params.yaml
K: 0.75
B: 0.25
IM: 0.06
# ...
```

```python
import yaml
from collections import OrderedDict

with open("configs/my_params.yaml", "r") as f:
    parameters = OrderedDict(yaml.safe_load(f))
```

#### 3. From Literature or Expert Knowledge

```python
from collections import OrderedDict

# Parameters from published study
parameters = OrderedDict({
    "K": 0.8,
    "B": 0.3,
    # ... other parameters
})
```

### Command-Line Simulation Script

The `scripts/run_xaj_simulate.py` is a **minimal template** for users to customize:

```bash
# Using custom parameters (recommended)
python scripts/run_xaj_simulate.py \
    --config configs/example_simulate_config.yaml \
    --param-file configs/example_xaj_params.yaml \
    --output results.csv \
    --plot

# Using calibrated parameters (CSV format, SCE-UA only)
python scripts/run_xaj_simulate.py \
    --param-file results/exp/01013500_sceua.csv \
    --plot

# Specify basin and warmup
python scripts/run_xaj_simulate.py \
    --param-file configs/params.yaml \
    --basin-id 01013500 \
    --warmup 730
```

**Script design**:
- Simple, readable code for users to understand
- Easy to modify for custom workflows
- Demonstrates UnifiedSimulator usage

### Common Use Cases

#### 1. Parameter Sensitivity Analysis

```python
# Vary one parameter, observe impact
base_params = load_parameters(...)

results_dict = {}
for k_value in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    params = base_params.copy()
    params["K"] = k_value

    # Update simulator
    simulator.parameters = params
    results = simulator.simulate(inputs, qobs, warmup_length=365)

    # Store results
    results_dict[k_value] = results["qsim"]

# Analyze sensitivity
import matplotlib.pyplot as plt
for k, qsim in results_dict.items():
    plt.plot(qsim[:, 0, 0], label=f"K={k}")
plt.legend()
plt.show()
```

#### 2. Model Comparison

```python
models = ["xaj", "xaj_mz", "gr4j"]
results_comparison = {}

for model_name in models:
    model_config["model_name"] = model_name
    # Adjust parameters for each model as needed

    simulator = UnifiedSimulator(model_config, basin_config)
    results = simulator.simulate(inputs, qobs, warmup_length=365)
    results_comparison[model_name] = results

# Compare performance
for model_name, results in results_comparison.items():
    qsim_2d = results["qsim"][365:, 0, 0].reshape(1, -1)
    metrics = hydro_stat.stat_error(qobs_2d, qsim_2d)
    print(f"{model_name}: NSE={metrics['NSE'][0]:.3f}")
```

#### 3. Ensemble Simulations

```python
# Run multiple parameter sets (e.g., from different calibration runs)
parameter_sets = [params1, params2, params3, ...]
ensemble_results = []

for params in parameter_sets:
    simulator.parameters = params
    results = simulator.simulate(inputs, qobs, warmup_length=365)
    ensemble_results.append(results["qsim"])

# Calculate ensemble mean and spread
ensemble_array = np.array(ensemble_results)  # [n_members, time, basin, 1]
ensemble_mean = ensemble_array.mean(axis=0)
ensemble_std = ensemble_array.std(axis=0)
```

### Relationship with Evaluation

```
run_xaj_simulate.py       # Simple, flexible user template
    ↑
    | (demonstrates API usage)
    ↓
UnifiedSimulator          # Core simulation interface
    ↑
    | (used by)
    ↓
run_xaj_evaluate.py       # Complete evaluation workflow
                           # (with NetCDF saving, batch processing, etc.)
```

- **`run_xaj_simulate.py`**: Simple script for custom workflows
- **`run_xaj_evaluate.py`**: Standardized evaluation pipeline

---

## Configuration System

### Configuration Structure

All APIs use a **consistent configuration format**:

```python
config = {
    "data_cfgs": {
        # Data source and loading
        "data_source_type": str,
        "data_source_path": str,  # Optional for CAMELS
        "basin_ids": list[str],
        "train_period": [str, str],
        "test_period": [str, str],
        "warmup_length": int,
        "variables": list[str],
    },
    "model_cfgs": {
        # Model configuration
        "model_name": str,
        "model_params": dict,
    },
    "training_cfgs": {
        # Calibration settings
        "algorithm_name": str,
        "algorithm_params": dict,
        "loss_config": dict,
        "output_dir": str,
        "experiment_name": str,
    },
    "evaluation_cfgs": {
        # Evaluation metrics
        "metrics": list[str],
    },
}
```

### YAML Configuration

For reproducibility, use YAML files:

```yaml
# config.yaml
data_cfgs:
  data_source_type: "camels_us"
  basin_ids: ["01013500"]
  train_period: ["1990-10-01", "2000-09-30"]
  test_period: ["2000-10-01", "2010-09-30"]
  warmup_length: 365
  variables: ["precipitation", "potential_evapotranspiration", "streamflow"]

model_cfgs:
  model_name: "xaj_mz"
  model_params:
    source_type: "sources"
    source_book: "HF"

training_cfgs:
  algorithm_name: "SCE_UA"
  algorithm_params:
    rep: 10000
    ngs: 100
    random_seed: 1234
  loss_config:
    type: "time_series"
    obj_func: "RMSE"
  output_dir: "results"
  experiment_name: "my_exp"

evaluation_cfgs:
  metrics: ["NSE", "KGE", "RMSE", "PBIAS"]
```

Load and use:

```python
import yaml
from hydromodel.trainers.unified_calibrate import calibrate

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

results = calibrate(config)
```

---

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
    config["training_cfgs"]["algorithm_name"] = exp["algorithm"]
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
    - [0.5, 1.5]    # K range
    - [0.1, 0.5]    # B range
    - [0.01, 0.1]   # IM range
```

```python
config["training_cfgs"]["param_range_file"] = "param_range.yaml"
```

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

## Summary

### Key Takeaways

1. **Unified Design**: All models, algorithms, and data sources use the same API
2. **Configuration-Based**: YAML configs ensure reproducibility
3. **Decoupled Components**: Calibration, evaluation, and simulation are independent
4. **Flexible Integration**: Works with CAMELS and custom data
5. **Extensible**: Easy to add new models, algorithms, and metrics

### Core APIs

```python
# Data loading
from hydromodel.datasets.unified_data_loader import UnifiedDataLoader
data_loader = UnifiedDataLoader(config)
p_and_e, qobs = data_loader.load_data()

# Calibration
from hydromodel.trainers.unified_calibrate import calibrate
results = calibrate(config)

# Evaluation
from hydromodel.trainers.unified_evaluate import evaluate
metrics = evaluate(config, param_dir="results/exp", eval_period="test")

# Simulation
from hydromodel.trainers.unified_simulate import UnifiedSimulator
simulator = UnifiedSimulator(model_config, basin_config)
results = simulator.simulate(inputs, qobs, warmup_length=365)
```

---

## Additional Resources

- **Quick Start**: [quickstart.md](quickstart.md) - End user guide for quick setup
- **Data Guide**: [data_guide.md](data_guide.md) - Data preparation and management
- **FAQ**: [faq.md](faq.md) - Common questions and solutions
- **API Reference**: Full API documentation (auto-generated)
- **GitHub**: https://github.com/OuyangWenyu/hydromodel
- **Issues**: https://github.com/OuyangWenyu/hydromodel/issues

---

## Contributing

For developers interested in contributing:

1. Fork the repository
2. Create a feature branch
3. Follow the unified API design principles
4. Add tests for new features
5. Update documentation
6. Submit a pull request

See [contributing.md](contributing.md) for detailed guidelines.
