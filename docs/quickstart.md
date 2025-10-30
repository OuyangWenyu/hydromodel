# Quick Start Guide

Get started with hydromodel in 5 minutes! This guide will walk you through a complete workflow from installation to model evaluation.

## Prerequisites

- Python 3.9 or higher
- Basic knowledge of Python
- Understanding of hydrological modeling concepts

## Step 1: Installation (2 minutes)

Install hydromodel and hydrodataset:

```bash
pip install hydromodel hydrodataset
```

Or using uv (faster):

```bash
uv pip install hydromodel hydrodataset
```

## Step 2: Configure Data Path (1 minute)

Create `hydro_setting.yml` in your home directory:

**Windows:** `C:\Users\YourUsername\hydro_setting.yml`

```yaml
local_data_path:
  datasets-origin: 'D:\data'
  cache: 'D:\data\.cache'
```

**Linux/Mac:** `~/hydro_setting.yml`

```yaml
local_data_path:
  datasets-origin: '/home/user/data'
  cache: '/home/user/data/.cache'
```

## Step 3: Download Data (Optional, 30-120 minutes)

The data will download automatically when needed, or you can pre-download:

```python
from hydrodataset.camels_us import CamelsUs
from hydrodataset import SETTING

# Initialize dataset (downloads if not present)
data_path = SETTING["local_data_path"]["datasets-origin"]
ds = CamelsUs(data_path, download=True)

print(f"Downloaded {len(ds.read_object_ids())} basins")
```

## Step 4: Create Configuration (1 minute)

Create `quickstart_config.yaml`:

```yaml
data:
  dataset: "camels_us"
  path: "D:/data"  # Or your data path
  basin_ids: ["01013500"]  # Single basin for quick test
  train_period: ["1990-10-01", "1995-09-30"]
  test_period: ["1995-10-01", "2000-09-30"]
  warmup_length: 365
  output_dir: "results"
  experiment_name: "quickstart"

model:
  name: "xaj_mz"
  params:
    source_type: "sources"
    source_book: "HF"

training:
  algorithm: "SCE_UA"
  loss: "RMSE"
  SCE_UA:
    random_seed: 1234
    rep: 1000     # Reduced for quick test
    ngs: 50       # Reduced for quick test
    kstop: 10
    peps: 0.1
    pcento: 0.1

evaluation:
  metrics: ["NSE", "KGE", "RMSE"]
```

## Step 5: Run Calibration (5-10 minutes)

### Using Python API

```python
from hydromodel.trainers.unified_calibrate import calibrate
import yaml

# Load configuration
with open("quickstart_config.yaml", "r") as f:
    config_simple = yaml.safe_load(f)

# Convert to unified format (helper function)
from hydromodel.trainers.config_utils import convert_simple_config
config = convert_simple_config(config_simple)

# Run calibration
print("Starting calibration...")
results = calibrate(config)
print("Calibration complete!")
```

### Using Command-Line Script

If you have the scripts:

```bash
python scripts/run_xaj_calibration.py --config quickstart_config.yaml
```

## Step 6: Check Results (30 seconds)

```python
import pandas as pd

# Load calibration results
results_file = "results/quickstart/01013500_sceua.csv"
df = pd.read_csv(results_file)

# Find best parameters
best_idx = df['like1'].idxmin()
best_params = df.iloc[best_idx]

print(f"Best RMSE: {best_params['like1']:.4f}")
print(f"Best parameters:")
for col in df.columns:
    if col.startswith('p'):
        print(f"  {col}: {best_params[col]:.6f}")
```

## Step 7: Evaluate Model (1 minute)

```python
from hydromodel.trainers.unified_evaluate import evaluate

# Evaluate on test period
eval_results = evaluate(
    config,
    param_dir="results/quickstart",
    eval_period="test"
)

# Print metrics
print("\nTest Period Performance:")
for basin_id, metrics in eval_results['metrics'].items():
    print(f"\n{basin_id}:")
    print(f"  NSE:  {metrics['NSE']:.3f}")
    print(f"  KGE:  {metrics['KGE']:.3f}")
    print(f"  RMSE: {metrics['RMSE']:.3f}")
```

## Step 8: Visualize Results (Optional)

```python
import xarray as xr
import matplotlib.pyplot as plt

# Load simulation results
results_nc = "results/quickstart/evaluation_test/xaj_mz_evaluation_results.nc"
ds = xr.open_dataset(results_nc)

# Extract data
time = ds['time'].values
qobs = ds['qobs'].values[0, :]  # First basin
qsim = ds['qsim'].values[0, :]

# Plot
plt.figure(figsize=(12, 4))
plt.plot(time, qobs, label='Observed', color='blue', linewidth=1)
plt.plot(time, qsim, label='Simulated', color='red', linewidth=1, linestyle='--')
plt.xlabel('Time')
plt.ylabel('Streamflow (m³/s)')
plt.legend()
plt.title('Streamflow Simulation - Test Period')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('quickstart_result.png', dpi=150)
plt.show()

print("Plot saved to quickstart_result.png")
```

## Complete Example Script

Here's everything in one script (`quickstart_complete.py`):

```python
"""Complete quickstart example for hydromodel."""
import yaml
from hydromodel.trainers.unified_calibrate import calibrate
from hydromodel.trainers.unified_evaluate import evaluate

# Configuration
config = {
    "data_cfgs": {
        "data_source_type": "camels_us",
        "data_source_path": "D:/data",
        "basin_ids": ["01013500"],
        "train_period": ["1990-10-01", "1995-09-30"],
        "test_period": ["1995-10-01", "2000-09-30"],
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
        "experiment_name": "quickstart",
        "rep": 1000,
        "ngs": 50,
        "kstop": 10,
        "peps": 0.1,
        "pcento": 0.1,
        "random_seed": 1234,
    },
    "evaluation_cfgs": {
        "metrics": ["NSE", "KGE", "RMSE"],
    },
}

# Step 1: Calibration
print("=" * 60)
print("STEP 1: Model Calibration")
print("=" * 60)
results = calibrate(config)
print("\n✓ Calibration complete!")

# Step 2: Evaluation
print("\n" + "=" * 60)
print("STEP 2: Model Evaluation")
print("=" * 60)
eval_results = evaluate(
    config,
    param_dir="results/quickstart",
    eval_period="test"
)

# Step 3: Print Results
print("\n" + "=" * 60)
print("STEP 3: Results Summary")
print("=" * 60)
for basin_id, metrics in eval_results['metrics'].items():
    print(f"\nBasin: {basin_id}")
    print(f"  NSE:  {metrics['NSE']:.3f}")
    print(f"  KGE:  {metrics['KGE']:.3f}")
    print(f"  RMSE: {metrics['RMSE']:.3f}")

print("\n" + "=" * 60)
print("All done! Check results/ directory for outputs.")
print("=" * 60)
```

Run it:

```bash
python quickstart_complete.py
```

## What's Next?

Now that you've completed the quickstart, here are your next steps:

### 1. Learn More Features

- **Multiple Basins**: Try calibrating multiple basins at once
  ```yaml
  basin_ids: ["01013500", "01022500", "01030500"]
  ```

- **Different Algorithms**: Experiment with GA or scipy optimizers
  ```yaml
  training:
    algorithm: "GA"  # or "scipy"
  ```

- **Custom Periods**: Evaluate on custom time periods
  ```python
  evaluate(config, eval_period="custom",
           custom_period=["2000-10-01", "2005-09-30"])
  ```

### 2. Production-Ready Calibration

For real research, use longer calibration:

```yaml
training:
  SCE_UA:
    rep: 10000   # More iterations
    ngs: 100     # More complexes
    kstop: 50    # Stricter convergence
```

### 3. Explore Documentation

- [Full Usage Guide](usage.md) - Comprehensive documentation
- [XAJ Model Details](models/xaj.md) - Model theory and parameters
- [Examples](examples.md) - More complete examples
- [FAQ](faq.md) - Common questions and solutions

### 4. Try Different Datasets

```python
# Try CAMELS-GB
config["data_cfgs"]["data_source_type"] = "camels_gb"
config["data_cfgs"]["basin_ids"] = ["28015"]  # Example GB basin

# Or your own data
config["data_cfgs"]["data_source_type"] = "selfmadehydrodataset"
config["data_cfgs"]["data_source_path"] = "/path/to/your/data"
```

### 5. Join the Community

- Star the repo: [github.com/OuyangWenyu/hydromodel](https://github.com/OuyangWenyu/hydromodel)
- Report issues: [GitHub Issues](https://github.com/OuyangWenyu/hydromodel/issues)
- Contribute: See [Contributing Guide](contributing.md)

## Common Issues

### "Data not found" Error

Make sure data path is correct:
```python
from hydrodataset import SETTING
print(SETTING["local_data_path"]["datasets-origin"])
```

### Slow Calibration

Reduce iterations for testing:
```yaml
training:
  SCE_UA:
    rep: 500   # Faster but less optimal
    ngs: 20
```

### Memory Issues

Process fewer basins at once:
```yaml
data:
  basin_ids: ["01013500"]  # Single basin
```

## Summary

Congratulations! You've successfully:
- ✓ Installed hydromodel
- ✓ Configured data paths
- ✓ Calibrated an XAJ model
- ✓ Evaluated model performance
- ✓ Visualized results

You're now ready to use hydromodel for your research! 🎉
