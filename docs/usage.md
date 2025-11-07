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
- [Results Visualization](#results-visualization)
- [Flood Event Data](#flood-event-data)
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

## Results Visualization

### Overview

The `visualize.py` script provides a simple command-line interface for visualizing evaluation results. It generates time series plots with precipitation and streamflow comparisons.

**Key Features:**
- Time series plots with observed vs simulated streamflow
- Precipitation displayed as inverted bars (top-down)
- Automatic loading from NetCDF evaluation results
- Basin-level or multi-basin visualization

### Command-Line Usage

**Basic usage** (visualize all basins):

```bash
python scripts/visualize.py --eval-dir results/xaj_mz_SCE_UA/evaluation_test
```

**Visualize specific basins:**

```bash
python scripts/visualize.py \
    --eval-dir results/xaj_mz_SCE_UA/evaluation_test \
    --basins 01013500 01022500
```

**Custom output directory:**

```bash
python scripts/visualize.py \
    --eval-dir results/xaj_mz_SCE_UA/evaluation_test \
    --output-dir my_figures
```

### Python API Usage

For programmatic visualization:

```python
from hydromodel.datasets.data_visualize import visualize_evaluation

# Visualize all basins
visualize_evaluation(
    eval_dir="results/xaj_mz_SCE_UA/evaluation_test",
    output_dir="figures",  # Optional, defaults to eval_dir/figures
    basins=None  # Optional, defaults to all basins
)

# Visualize specific basins
visualize_evaluation(
    eval_dir="results/xaj_mz_SCE_UA/evaluation_test",
    basins=["01013500", "01022500"]
)
```

### Input Requirements

The visualization script expects:

1. **NetCDF file**: `*_evaluation_results.nc` containing:
   - `qobs`: Observed streamflow `[time, basin]`
   - `qsim`: Simulated streamflow `[time, basin]`
   - `prcp`: Precipitation `[time, basin]` (optional)
   - `basin`: Basin IDs
   - `time`: Time coordinates

2. **Directory structure**:
   ```
   results/xaj_mz_SCE_UA/
   └── evaluation_test/
       ├── test_evaluation_results.nc
       └── figures/  # Output directory (auto-created)
           ├── 01013500_timeseries.png
           ├── 01022500_timeseries.png
           └── ...
   ```

### Output

For each basin, generates:
- **Time series plot**: `{basin_id}_timeseries.png`
  - Upper panel: Precipitation (inverted bars)
  - Lower panel: Observed vs simulated streamflow

**Plot features:**
- Date formatting (YYYY-MM)
- Dual-axis precipitation/streamflow
- Legend with simulation vs observation
- PNG format with 300 DPI

### Advanced Visualization

For custom plots beyond the CLI tool, use the core plotting functions directly:

```python
from hydromodel.datasets.data_visualize import (
    plot_sim_and_obs,
    plot_sim_and_obs_streamflow,
    plot_precipitation
)
import xarray as xr

# Load evaluation results
ds = xr.open_dataset("results/xaj_mz_SCE_UA/evaluation_test/test_evaluation_results.nc")

# Extract data for a specific basin
basin_idx = 0
time = ds['time'].values
qobs = ds['qobs'].values[:, basin_idx]
qsim = ds['qsim'].values[:, basin_idx]
prcp = ds['prcp'].sel(basin=ds['basin'].values[basin_idx])

# Create custom plot
plot_sim_and_obs(
    date=time,
    prcp=prcp,
    sim=qsim,
    obs=qobs,
    save_fig="my_custom_plot.png",
    ylabel="Streamflow (m³/s)"
)
```

---

## Flood Event Data

### Overview

hydromodel provides specialized support for **flood event datasets**, where data consists of discrete flood events rather than continuous time series. This is particularly useful for:

- Event-based model calibration and validation
- Flood forecasting applications
- Multi-peak flood event analysis
- Studies focusing on extreme hydrological conditions

**Key Features:**

- Multi-basin support with correct time alignment (no padding issues)
- Automatic event grouping for multi-peak floods
- Event-specific visualizations showing only flood periods
- Proper handling of gaps and warmup periods
- Backward compatible with all existing hydromodel APIs

### Data Format

Flood event data uses the `floodevent` data source type from `hydrodatasource`:

```python
data_config = {
    "data_source_type": "floodevent",  # or "selfmadehydrodataset"
    "dataset_name": "my_flood_data",   # Dataset folder name
    "basin_ids": ["basin_001"],
    # ... other parameters
}
```

**Required Data Structure:**

```
my_flood_data/
├── attributes/
│   └── attributes.csv              # Basin metadata
├── timeseries/
│   ├── 1h/                         # Hourly flood event data
│   │   ├── basin_001.csv          # Time series with marker column
│   │   ├── basin_002.csv
│   │   └── ...
│   └── 1h_units_info.json          # Variable units
```

**Time series CSV format:**

```csv
time,prcp,PET,streamflow,marker,event_id
2020-08-01 00:00,0.5,0.1,10.2,1,25
2020-08-01 01:00,1.2,0.1,12.5,1,25
...
2020-08-05 23:00,0.2,0.05,8.1,1,25
2020-08-06 00:00,0.0,0.0,0.0,0,0
...
2020-09-01 00:00,0.8,0.12,15.3,1,26
```

**Special columns:**

- `marker`:
  - `1` = flood event data (valid)
  - `0` = gap between events (invalid, not used in calibration/evaluation)
  - `NaN` = warmup period (used for model spinup, excluded from metrics)
- `event_id`: Integer identifier grouping related flood peaks together

### Configuration

**YAML Configuration Example:**

```yaml
# configs/flood_event_config.yaml
data_cfgs:
  data_source_type: "floodevent"
  dataset_name: "songliao_flood_events"
  basin_ids: ["songliao_21401550", "songliao_21100150"]
  train_period: ["2019-01-01", "2020-12-31"]  # Filter events by date range
  test_period: ["2021-01-01", "2022-12-31"]
  warmup_length: 15  # Hours for event-based data
  time_unit: ["1h"]  # Hourly resolution
  variables: ["prcp", "PET", "streamflow"]

  # Optional: Filter by event_id
  event_ids: [25, 26, 27]  # Only use these events

model_cfgs:
  model_name: "xaj_mz"
  model_params:
    source_type: "sources"
    source_book: "HF"
    kernel_size: 15

training_cfgs:
  algorithm_name: "SCE_UA"
  algorithm_params:
    rep: 5000
    ngs: 1000
    random_seed: 1234
  loss_config:
    type: "time_series"
    obj_func: "RMSE"
  output_dir: "results"
  experiment_name: "flood_event_calibration"

evaluation_cfgs:
  metrics: ["NSE", "KGE", "RMSE", "PBIAS"]
```

### Command-Line Scripts

#### Quick Start with Default Configuration

The `run_event_calibration.py` script provides sensible defaults for flood event calibration:

```bash
# Use default configuration
python scripts/run_event_calibration.py --default

# Verify configuration without running
python scripts/run_event_calibration.py --default --dry-run

# Customize basin IDs and algorithm
python scripts/run_event_calibration.py --default \
    --basin-ids songliao_21401550 songliao_21100150 \
    --algorithm GA \
    --output-dir results/flood_ga
```

**Default configuration:**
- Data source: `floodevent`
- Dataset: `songliao_flood_events`
- Basin: `songliao_21401550`
- Algorithm: `SCE_UA`
- Warmup: 15 hours
- Model: `xaj_mz`

#### Using Configuration Files

```bash
# Calibration
python scripts/run_event_calibration.py --config configs/flood_event_config.yaml

# Or use standard calibration script (works identically)
python scripts/run_xaj_calibration.py --config configs/flood_event_config.yaml

# Evaluation (same as continuous data)
python scripts/run_xaj_evaluate.py \
    --calibration-dir results/flood_event_calibration \
    --eval-period test

# Visualization (event-specific plots)
python scripts/visualize.py \
    --eval-dir results/flood_event_calibration/evaluation_test
```

### Python API Usage

Flood event data works seamlessly with the unified API:

```python
from hydromodel.trainers.unified_calibrate import calibrate
from hydromodel.trainers.unified_evaluate import evaluate

# Configuration
config = {
    "data_cfgs": {
        "data_source_type": "floodevent",
        "dataset_name": "songliao_flood_events",
        "basin_ids": ["songliao_21401550"],
        "train_period": ["2019-01-01", "2020-12-31"],
        "test_period": ["2021-01-01", "2022-12-31"],
        "warmup_length": 15,
        "time_unit": ["1h"],
        "variables": ["prcp", "PET", "streamflow"],
    },
    "model_cfgs": {
        "model_name": "xaj_mz",
    },
    "training_cfgs": {
        "algorithm_name": "SCE_UA",
        "algorithm_params": {"rep": 5000, "ngs": 1000},
        "loss_config": {"type": "time_series", "obj_func": "RMSE"},
        "output_dir": "results",
        "experiment_name": "flood_event_exp",
    },
    "evaluation_cfgs": {
        "metrics": ["NSE", "KGE", "RMSE"],
    },
}

# Calibration
results = calibrate(config)

# Evaluation
metrics = evaluate(config, param_dir="results/flood_event_exp", eval_period="test")
```

### Multi-Basin Time Alignment

**Problem**: Different basins may have flood events at different times, creating challenges for multi-basin calibration.

**Solution**: hydromodel automatically handles time alignment:

```python
# Example: Two basins with different event times
# Basin A: Events on 2020-08-01 to 2020-08-05 (Event 25)
# Basin B: Events on 2020-09-01 to 2020-09-05 (Event 26)

# hydromodel creates a unified time array:
# - Merges all unique timestamps from both basins
# - Maps each basin's data to correct time positions
# - Fills gaps with marker=0 (invalid data, excluded from loss)
# - Preserves event_id for visualization

# No manual intervention required!
```

**Internal Workflow:**

1. Load each basin's event data separately
2. Extract unique timestamps across all basins
3. Create unified sorted time array
4. Remap each basin's data to unified time indices
5. Mark missing periods with `marker=0`, `event_id=0`
6. Calibration/evaluation uses only `marker=1` data

**NetCDF Output Structure:**

```python
import xarray as xr

ds = xr.open_dataset("results/flood_event_exp/evaluation_test/test_evaluation_results.nc")

print(ds.dims)
# {'basin': 2, 'time': 1500}  # Unified time array

print(ds['event_id'])
# Shows event_id for each time step and basin
# event_id=0 indicates gaps (not used in metrics)

print(ds['qsim'])
# Simulated streamflow [time, basin]
# Zero values where marker=0 (gaps)
```

### Event-Specific Visualization

Flood event plots automatically highlight event periods:

```python
from hydromodel.datasets.data_visualize import visualize_evaluation

# Visualize flood events
visualize_evaluation(
    eval_dir="results/flood_event_calibration/evaluation_test",
    basins=["songliao_21401550"]
)
```

**Plot Features:**

- Only shows periods where `marker=1` (flood events)
- Gaps between events are excluded from visualization
- Event IDs displayed in plot titles
- Precipitation and streamflow on separate panels
- Performance metrics (NSE, RMSE, PBIAS) shown in text box

### Multi-Peak Event Grouping

Use `event_id` to group related flood peaks:

```python
# Example: Typhoon with multiple peaks
# Event 25:
#   - Peak 1: 2020-08-01 to 2020-08-03
#   - Gap:    2020-08-04 to 2020-08-05 (marker=0)
#   - Peak 2: 2020-08-06 to 2020-08-08
#   - All marked as event_id=25

# During calibration:
# - Model sees both peaks with correct warmup
# - Gap period (marker=0) excluded from loss calculation
# - event_id preserved for analysis

# During visualization:
# - Both peaks plotted together as "Event 25"
# - Gap period shown but marked differently
```

### Filtering Events

**By Time Range:**

```python
config["data_cfgs"]["train_period"] = ["2019-07-01", "2020-09-30"]
# Only loads events within this date range
```

**By Event ID:**

```python
config["data_cfgs"]["event_ids"] = [25, 26, 27]
# Only loads these specific events
```

**By Basin:**

```python
config["data_cfgs"]["basin_ids"] = ["songliao_21401550"]
# Single basin calibration
```

### Comparison with Continuous Data

| Feature | Continuous Data | Flood Event Data |
|---------|----------------|------------------|
| Data source | CAMELS, selfmadehydrodataset | floodevent |
| Time structure | Continuous time series | Discrete events with gaps |
| Warmup | Days (typically 365) | Hours (typically 15) |
| marker column | Not used | Required (1=valid, 0=gap) |
| event_id column | Not used | Required for grouping |
| API usage | Identical | Identical |
| Output format | NetCDF with continuous time | NetCDF with unified time array |

### Best Practices

**1. Adequate Warmup:**

```python
# For hourly data, use at least 15 hours warmup
config["data_cfgs"]["warmup_length"] = 15

# For event data with long recessions, increase warmup
config["data_cfgs"]["warmup_length"] = 30
```

**2. Event Selection:**

```python
# Start with well-observed, significant events
config["data_cfgs"]["event_ids"] = [25, 26, 27]  # Major floods only

# Avoid events with missing data or measurement errors
```

**3. Multi-Basin Calibration:**

```python
# Ensure basins have overlapping events for better parameter transfer
basin_ids = ["basin_A", "basin_B", "basin_C"]

# Check event coverage before calibration
from hydrodatasource import FloodEventDataSource
ds = FloodEventDataSource(...)
for basin in basin_ids:
    events = ds.get_events(basin)
    print(f"{basin}: {len(events)} events")
```

**4. Validation:**

```python
# Always validate on independent events
config["data_cfgs"]["train_period"] = ["2018-01-01", "2020-12-31"]
config["data_cfgs"]["test_period"] = ["2021-01-01", "2022-12-31"]

# Or use event-based split
train_events = [20, 21, 22, 23, 24]
test_events = [25, 26, 27]
```

### Troubleshooting

**Issue 1: Time Misalignment in Multi-Basin Results**

**Symptoms:** NetCDF shows incorrect time ranges for events (e.g., Event 26 spanning years instead of days)

**Solution:** This has been fixed in v0.3.0. Ensure you're using the latest version.

**Issue 2: AttributeError in Calibration**

**Symptoms:** `'NoneType' object has no attribute 'shape'`

**Solution:** This has been fixed in v0.3.0. The issue occurred when accessing basin data in separate mode.

**Issue 3: Missing Event Data**

**Symptoms:** Fewer events loaded than expected

**Check:**
```python
# Verify event filtering
print(f"Time range: {config['data_cfgs']['train_period']}")
print(f"Event IDs: {config['data_cfgs'].get('event_ids', 'All')}")

# Check raw data
from hydrodatasource import FloodEventDataSource
ds = FloodEventDataSource(data_path=..., time_unit=["1h"])
events = ds.get_events(basin_id)
print(f"Available events: {len(events)}")
```

**Issue 4: Poor Calibration Performance**

**Possible causes:**
- Insufficient warmup period
- Events too short for model spinup
- Mixed event types (e.g., snowmelt and rainfall floods)

**Solutions:**
- Increase warmup: `warmup_length = 30`
- Filter events by type or magnitude
- Check data quality (missing values, outliers)

### Example Workflow

Complete workflow for flood event calibration:

```python
from hydromodel.trainers.unified_calibrate import calibrate
from hydromodel.trainers.unified_evaluate import evaluate
from hydromodel.datasets.data_visualize import visualize_evaluation

# 1. Configuration
config = {
    "data_cfgs": {
        "data_source_type": "floodevent",
        "dataset_name": "songliao_flood_events",
        "basin_ids": ["songliao_21401550", "songliao_21100150"],
        "train_period": ["2019-01-01", "2020-12-31"],
        "test_period": ["2021-01-01", "2022-12-31"],
        "warmup_length": 15,
        "time_unit": ["1h"],
        "variables": ["prcp", "PET", "streamflow"],
    },
    "model_cfgs": {"model_name": "xaj_mz"},
    "training_cfgs": {
        "algorithm_name": "SCE_UA",
        "algorithm_params": {"rep": 5000, "ngs": 1000},
        "loss_config": {"type": "time_series", "obj_func": "RMSE"},
        "output_dir": "results",
        "experiment_name": "flood_2basin",
    },
    "evaluation_cfgs": {"metrics": ["NSE", "KGE", "RMSE", "PBIAS"]},
}

# 2. Calibration
print("Starting calibration...")
results = calibrate(config)
print(f"Calibration completed: {results}")

# 3. Evaluation on test period
print("Evaluating on test period...")
test_metrics = evaluate(
    config,
    param_dir="results/flood_2basin",
    eval_period="test"
)
print(f"Test NSE: {test_metrics}")

# 4. Visualization
print("Creating visualizations...")
visualize_evaluation(
    eval_dir="results/flood_2basin/evaluation_test",
    output_dir="figures/flood_events"
)
print("Done!")
```

### Related Documentation

- **Data Preparation**: [Data Guide](data_guide.md#custom-data) - How to prepare flood event data
- **API Reference**: [Unified API](#unified-calibration-api) - Core API documentation
- **Configuration**: [Configuration System](#configuration-system) - Full configuration options
- **hydrodatasource**: [GitHub](https://github.com/OuyangWenyu/hydrodatasource) - Data source package

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
