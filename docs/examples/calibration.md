## Model Calibration

### Unified Calibration API

The `calibrate()` function provides a **completely unified** calibration interface:

```python
from hydromodel.trainers.unified_calibrate import calibrate

config = {
    "data_cfgs": {
        "dataset": "camels_us",
        "basin_ids": ["01013500"],
        "train_period": ["1990-10-01", "2000-09-30"],
        "test_period": ["2000-10-01", "2010-09-30"],
        "warmup_length": 365,
    },
    "model_cfgs": {
        "name": "xaj_mz",
        "params": {
            "source_type": "sources",
            "source_book": "HF",
        },
    },
    "training_cfgs": {
        "algorithm": "SCE_UA",
        "SCE_UA": {
            "rep": 10000,
            "ngs": 100,
            "random_seed": 1234,
        },
        "loss_config": {
            "type": "time_series",
            "obj_func": "RMSE",  # User objective: RMSE, NSE, KGE, LOGNSE
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
3. Create UnifiedCalibrator (wraps MODEL_DICT)
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
    "algorithm": "SCE_UA",
    "SCE_UA": {
        "rep": 10000,         # Maximum iterations
        "ngs": 100,           # Number of complexes
        "kstop": 50,          # Stopping criteria
        "peps": 0.1,          # Convergence threshold
        "pcento": 0.1,        # Convergence percentage
        "random_seed": 1234,
    },
    "loss_config": {
        "type": "time_series",
        "obj_func": "RMSE",   # user objective: RMSE, NSE, KGE, LOGNSE
    },
}
```

Hydromodel optimizers always minimize. User-facing objectives `NSE`,
`KGE`, and `LOGNSE` are resolved internally to negated objectives such as
`neg_nashsutcliffe`, `neg_kge`, and `neg_lognashsutcliffe`. Evaluation
metrics remain positive hydrological metrics such as `NSE`, `KGE`, `RMSE`,
and `PBIAS`.

**Output**: `{basin_id}_sceua.csv` with columns:
- `like1`: Objective function value
- `parK`, `parB`, ...: Parameter values (with `par` prefix)
- `simulation1_1`, ...: Simulation results for each iteration

#### Genetic Algorithm

Uses `DEAP` library:

```python
training_cfgs = {
    "algorithm": "GA",
    "GA": {
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
    "algorithm": "scipy",
    "scipy": {
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

`calibration_results.json` is always written in the same directory. It keeps
the legacy `best_params` field as normalized `[0,1]` values for evaluation
compatibility. New fields make the contract explicit:

- `parameter_format`: currently `"normalized"`.
- `best_params_normalized`: same values as legacy `best_params`.
- `best_params_denormalized`: physical values from the resolved range.
- `param_range_source`: `default`, `explicit`, or `artifact`.
- `loss_config.requested_obj_func`: user objective such as `KGE`.
- `loss_config.resolved_obj_func`: minimized internal objective such as
  `neg_kge`.

---

