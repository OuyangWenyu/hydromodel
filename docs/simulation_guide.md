# Model Simulation Guide

## Important Note

⚠️ **Simulation does NOT require prior calibration!**

`UnifiedSimulator` is an **independent simulation interface**. You can:
- ✅ Run simulations with any parameter values you choose
- ✅ Use parameters from literature
- ✅ Use calibrated parameters
- ✅ Perform parameter sensitivity analysis

**Simulation and calibration are completely decoupled** - this is the core design principle of `UnifiedSimulator`.

## Core Concept

`UnifiedSimulator` is a **unified simulator interface** providing consistent access to all hydrological models:

```python
# 1. Prepare configuration
model_config = {
    "model_name": "xaj_mz",
    "parameters": {"K": 0.8, "B": 0.3, ...}
}

# 2. Create simulator
simulator = UnifiedSimulator(model_config, basin_config)

# 3. Run simulation
results = simulator.simulate(inputs=p_and_e, qobs=qobs)
```

**Core Value**:
- ✅ All models (XAJ, GR series, HyMod, etc.) use the same interface
- ✅ Unified input format: `[time, basin, features]`
- ✅ Unified output format: `Dict[str, np.ndarray]`
- ✅ Reusable instance without reinitialization

## Quick Start

### Simplest Usage

```bash
# Run simulation with calibrated parameters
python scripts/run_xaj_simulate.py \
    --param-file results/xaj_mz_SCE_UA/01013500_sceua.csv
```

### Common Options

```bash
# Specify config and parameter files
python scripts/run_xaj_simulate.py \
    --config configs/example_simulate_config.yaml \
    --param-file configs/example_xaj_params.yaml

# Save results to CSV
python scripts/run_xaj_simulate.py \
    --param-file results/xaj_mz_SCE_UA/01013500_sceua.csv \
    --output simulation_results.csv

# Show visualization
python scripts/run_xaj_simulate.py \
    --param-file results/xaj_mz_SCE_UA/01013500_sceua.csv \
    --plot

# Specify basin and warmup period
python scripts/run_xaj_simulate.py \
    --param-file my_params.yaml \
    --basin-id 01013500 \
    --warmup 730
```

## Script Structure

`run_xaj_simulate.py` is a **minimal example** demonstrating how to use `UnifiedSimulator`:

```python
# Step 1: Load configuration
config = yaml.safe_load(open("config.yaml"))

# Step 2: Load parameters
parameters = {"K": 0.8, "B": 0.3, ...}

# Step 3: Load data
data_loader = UnifiedDataLoader(config)
p_and_e, qobs = data_loader.load_data()

# Step 4: Create and run simulator
model_config = {"model_name": "xaj_mz", "parameters": parameters}
simulator = UnifiedSimulator(model_config, basin_config)
results = simulator.simulate(inputs=p_and_e, qobs=qobs)

# Step 5: Use results
qsim = results["qsim"]  # Main output for XAJ models
```

## Use as Template

This script is designed as a **starting point for customization**. You can:

1. **Modify parameter loading**:
```python
# Custom parameter source
parameters = my_custom_parameter_loader()
```

2. **Add custom analysis**:
```python
# Run simulation
results = simulator.simulate(...)

# Custom analysis
qsim = results["qsim"]  # Extract simulated flow
if "es" in results:
    es = results["es"]  # Extract evaporation (if available)
my_analysis(qsim, es)
```

3. **Batch simulations**:
```python
# Multiple parameter sets
for param_set in parameter_sets:
    simulator.update_parameters(param_set)
    results = simulator.simulate(inputs)
    # Process results...
```

4. **Different models**:
```python
# Just change model_name
for model in ["xaj", "xaj_mz", "gr4j"]:
    model_config["model_name"] = model
    simulator = UnifiedSimulator(model_config)
    # Rest of code unchanged
```

## UnifiedSimulator API

### Initialization

```python
simulator = UnifiedSimulator(
    model_config={
        "model_name": "xaj_mz",           # Model name
        "model_params": {...},            # Model-specific parameters
        "parameters": OrderedDict(...)    # Calibratable parameters
    },
    basin_config={                        # Optional: Basin configuration
        "basin_area": 1000.0,
        ...
    }
)
```

### Run Simulation

```python
results = simulator.simulate(
    inputs=p_and_e,                # [time, basin, features] input data
    qobs=qobs,                     # [time, basin, 1] observed flow (optional)
    warmup_length=365,             # Warmup period length
    return_intermediate=False      # Return intermediate states?
)
```

### Return Format

```python
# UnifiedSimulator returns model-specific output names
results = {
    "qsim": np.ndarray,            # [time, basin, 1] simulated flow
    "es": np.ndarray,              # [time, basin, 1] evaporation (if model has it)
    # ... other model-specific outputs
    # If return_intermediate=True, also includes intermediate states
}

# Note: Different models return different keys
# XAJ/XAJ_MZ: "qsim", "es"
# GR4J: "qsim", "ets"
# etc.
```

## Relationship with Other Scripts

```
run_xaj_calibration.py    # Calibration: Find optimal parameters
    ↓ Generate parameter files
run_xaj_simulate.py       # Simulation: Use parameters to run (simple example)
    ↓
run_xaj_evaluate.py       # Evaluation: Complete evaluation workflow (save NetCDF, etc.)
```

- `run_xaj_simulate.py`: **Simple, flexible**, as user template
- `run_xaj_evaluate.py`: **Complete, standardized**, for standard evaluation workflow

## Common Use Cases

### 1. Parameter Sensitivity Analysis

```python
# Modify one parameter, observe impact
base_params = load_parameters(...)

for k_value in [0.5, 0.6, 0.7, 0.8, 0.9]:
    params = base_params.copy()
    params["K"] = k_value
    simulator.update_parameters(params)
    results = simulator.simulate(inputs)
    analyze_results(results)
```

### 2. Model Comparison

```python
models = ["xaj", "xaj_mz", "gr4j"]
results_comparison = {}

for model_name in models:
    config["model_name"] = model_name
    simulator = UnifiedSimulator(config)
    results = simulator.simulate(inputs)
    results_comparison[model_name] = results
```

### 3. Different Time Periods

```python
# Same parameters, different periods
periods = [
    ("2000-2005", dry_period_data),
    ("2010-2015", wet_period_data)
]

for period_name, data in periods:
    results = simulator.simulate(inputs=data)
    save_results(period_name, results)
```

## Output Description

The script prints:
- Configuration info (model, basin, period)
- Parameter values
- Simulation progress
- Performance metrics (NSE, RMSE, etc.)

Optional outputs:
- CSV file (`--output`): Time series results
- Visualization plot (`--plot`): Simulated vs observed comparison

## Example Parameter Files

### YAML Format (Recommended)
```yaml
# configs/example_xaj_params.yaml
K: 0.75
B: 0.25
IM: 0.06
UM: 18.0
LM: 80.0
# ... other parameters
```

### CSV Format (Calibration Results)
Automatically extracts optimal parameters from SCE-UA calibration results.

## Further Customization

If you need:
- More complex output formats (NetCDF, HDF5, etc.)
- Unit conversion and post-processing
- Complete evaluation pipeline

Please refer to `run_xaj_evaluate.py` or extend based on `run_xaj_simulate.py`.

## Help

```bash
python scripts/run_xaj_simulate.py --help
```
