## Model Simulation

### Important Design Principle

⚠️ **Simulation does NOT require prior calibration!**

`simulate(config)` is the top-level simulation interface, mirroring `calibrate(config)`.
It loads data, builds the simulator, runs it, and returns results — all from one config dict.

### Top-Level Simulation API (Recommended)

```python
from hydromodel import simulate

config = {
    "data_cfgs": {
        "dataset": "camels_us",
        "source": "local",
        "basin_ids": ["01013500"],
        "warmup_length": 365,
        "variables": ["precipitation", "potential_evapotranspiration", "streamflow"],
        "test_period": ["2005-10-01", "2014-09-30"],
    },
    "model_cfgs": {
        "name": "xaj",
        "params": {"source_type": "sources", "source_book": "HF"},
        "parameters": {
            "K": 0.75, "B": 0.25, "IM": 0.06,
            "UM": 18.0, "LM": 80.0, "DM": 95.0,
            "C": 0.18, "SM": 120.0, "EX": 1.5,
            "KI": 0.35, "KG": 0.45,
            "CS": 0.5, "L": 5.5, "CI": 0.85, "CG": 0.95,
        },
    },
}

results = simulate(config)

# Access results
qsim = results["simulation"]["qsim"]  # simulated streamflow
qobs = results["qobs"]                 # observed streamflow (if available)
print(f"Parameters used: {results['parameters']['K']}")
print(f"Basins: {results['basin_ids']}")
```

**Return format:**

| Key | Type | Description |
|-----|------|-------------|
| `simulation` | `dict` | Model output arrays (keys vary by model, usually `{"qsim": array}`) |
| `qobs` | `ndarray` or `None` | Observed streamflow (if loaded) |
| `parameters` | `dict` | The parameter values used |
| `model_name` | `str` | Model name |
| `basin_ids` | `list` | Basin IDs simulated |

### UnifiedSimulator API (Advanced)

For advanced use cases (custom basin configs, step-by-step control, multi-step simulation),
use `UnifiedSimulator` directly:

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
    "KI": 0.35, "KG": 0.45,
    "CS": 0.5, "L": 5.5, "CI": 0.85, "CG": 0.95,
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
# configs/example_xaj_params.yaml
K: 0.75
B: 0.25
IM: 0.06
# ...
```

```python
import yaml
from collections import OrderedDict

with open("configs/example_xaj_params.yaml", "r") as f:
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
    --param-file configs/example_xaj_params.yaml \
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

