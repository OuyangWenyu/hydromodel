# Quick Start Guide

> **Document Purpose**: This guide is designed for **end users** who want to quickly start using hydromodel for hydrological modeling. For developers who need detailed understanding of the code architecture, please refer to [Usage Guide](usage.md).

Get started with hydromodel in 5 minutes! This guide walks you through a complete workflow using command-line scripts - no complex coding required.

---

## What You'll Learn

- ✅ Install hydromodel and prepare data
- ✅ Calibrate a hydrological model
- ✅ Evaluate model performance
- ✅ Run simulations with custom parameters
- ✅ Visualize results

---

## Prerequisites

- Python 3.9 or higher
- Basic command-line knowledge
- Understanding of hydrological modeling concepts

---

## Step 1: Installation (2 minutes)

Install hydromodel with data support:

```bash
pip install hydromodel hydrodataset
```

Or using `uv` (faster):

```bash
uv pip install hydromodel hydrodataset
```

Verify installation:

```bash
python -c "import hydromodel; print(hydromodel.__version__)"
```

---

## Step 2: Get the Scripts (1 minute)

Clone the repository to access example scripts and configs:

```bash
git clone https://github.com/OuyangWenyu/hydromodel.git
cd hydromodel
```

The key files you'll use:
```
hydromodel/
├── scripts/                         # Command-line scripts
│   ├── run_xaj_calibration.py      # Calibrate models
│   ├── run_xaj_evaluate.py         # Evaluate performance
│   ├── run_xaj_simulate.py         # Run simulations
│   └── visualize.py                 # Visualize results
└── configs/                         # Configuration files
    ├── example_config.yaml          # Example calibration config
    ├── example_simulate_config.yaml # Example simulation config
    └── example_xaj_params.yaml      # Example parameters
```

---

## Step 3: Prepare Data (5-30 minutes)

### Option A: Use CAMELS Public Data (Recommended for First Try)

The data downloads automatically on first use:

```python
# Check available basins
from hydrodataset.camels_us import CamelsUs
from hydrodataset import SETTING

data_path = SETTING["local_data_path"]["datasets-origin"]
ds = CamelsUs(data_path, download=True)
basin_ids = ds.read_object_ids()

print(f"Available basins: {len(basin_ids)}")
print(f"Example IDs: {basin_ids[:5]}")
```

**First download may take 30-120 minutes** for the complete CAMELS dataset (~70GB). See [Data Guide](data_guide.md) for details.

### Option B: Use Your Own Data

See [Data Guide - Custom Data Section](data_guide.md#option-2-using-custom-data-hydrodatasource) for preparing your own basin data.

---

## Step 4: Configure Your Experiment (2 minutes)

Edit `configs/example_config.yaml`:

```yaml
# Configuration for model calibration and evaluation
data_cfgs:
  data_source_type: "camels_us"                 # Dataset type
  basin_ids: ["01013500"]                       # Basin(s) to calibrate
  train_period: ["1990-10-01", "2000-09-30"]   # Calibration period
  test_period: ["2000-10-01", "2010-09-30"]    # Evaluation period
  warmup_length: 365                            # Warmup days
  variables: ["precipitation", "potential_evapotranspiration", "streamflow"]

model_cfgs:
  model_name: "xaj_mz"                          # XAJ model with Muskingum routing
  model_params:
    source_type: "sources"
    source_book: "HF"

training_cfgs:
  algorithm_name: "SCE_UA"                      # Calibration algorithm
  algorithm_params:
    rep: 5000                                   # Iterations (use 10000+ for production)
    ngs: 50                                     # Complexes (use 100+ for production)
    random_seed: 1234
  loss_config:
    type: "time_series"
    obj_func: "RMSE"                            # Objective function
  output_dir: "results"
  experiment_name: "quickstart_exp"

evaluation_cfgs:
  metrics: ["NSE", "KGE", "RMSE", "PBIAS"]
```

**Quick Tips:**
- Start with **one basin** for first试验
- Use fewer `rep` (5000) and `ngs` (50) for quick testing
- For production runs, use `rep: 10000` and `ngs: 100`

---

## Step 5: Run Calibration (5-30 minutes)

```bash
python scripts/run_xaj_calibration.py --config configs/example_config.yaml
```

You'll see output like:

```
================================================================================
XAJ Model Calibration using UnifiedCalibrateor
================================================================================

[1/3] Loading configuration...
  Dataset: camels_us
  Basins: ['01013500']
  Model: xaj_mz
  Algorithm: SCE_UA
  Training period: 1990-10-01 to 2000-09-30

[2/3] Loading data...
  Input shape: (3653, 1, 2)
  Qobs shape: (3653, 1, 1)

[3/3] Starting calibration...
  Basin 01013500: SCE-UA optimization
  Progress: [====================] 100% Complete
  Best RMSE: 0.2534

✓ Calibration complete! Results saved to: results/quickstart_exp/
```

**Results location:**
```
results/quickstart_exp/
├── 01013500_sceua.csv              # Calibration history
└── calibration_config.yaml          # Config used (for reproducibility)
```

---

## Step 6: Evaluate Model Performance (1 minute)

Evaluate on the test period:

```bash
python scripts/run_xaj_evaluate.py \
    --calibration-dir results/quickstart_exp \
    --eval-period test
```

Output:

```
================================================================================
XAJ Model Evaluation
================================================================================

[1/3] Loading calibrated parameters...
  Found parameters for 1 basin(s)

[2/3] Running evaluation on test period...
  Basin 01013500: Simulating...
  ✓ Simulation complete

[3/3] Calculating metrics...

================================================================================
Evaluation Results (Test Period: 2000-10-01 to 2010-09-30)
================================================================================
Basin: 01013500
  NSE:   0.756
  KGE:   0.721
  RMSE:  0.312
  PBIAS: -5.23 %

✓ Results saved to: results/quickstart_exp/evaluation_test/
```

**Results include:**
```
results/quickstart_exp/evaluation_test/
├── basins_metrics.csv                  # Performance metrics
├── basins_denorm_params.csv            # Calibrated parameters
└── xaj_mz_evaluation_results.nc        # Full simulation results (NetCDF)
```

---

## Step 7: Run Custom Simulation (1 minute)

**Important**: Simulation does NOT require calibration! You can run simulations with any parameters.

### Option A: Use Calibrated Parameters

```bash
python scripts/run_xaj_simulate.py \
    --param-file results/quickstart_exp/01013500_sceua.csv \
    --plot
```

### Option B: Use Custom Parameters

Edit `configs/example_xaj_params.yaml`:

```yaml
# XAJ model parameters
K: 0.75
B: 0.25
IM: 0.06
UM: 18.0
LM: 80.0
DM: 95.0
C: 0.18
SM: 120.0
EX: 1.5
KI: 0.35
KG: 0.45
A: 0.85
THETA: 0.012
CI: 0.85
CG: 0.95
```

Then run:

```bash
python scripts/run_xaj_simulate.py \
    --config configs/example_simulate_config.yaml \
    --param-file configs/example_xaj_params.yaml \
    --output simulation_results.csv \
    --plot
```

Output:

```
================================================================================
XAJ Model Simulation using UnifiedSimulator
================================================================================

[1/4] Loading configuration from: configs/example_simulate_config.yaml
  Model: xaj_mz
  Basin: 01013500 (index 0)
  Period: ['2000-10-01', '2010-09-30']

[2/4] Loading parameters from: configs/example_xaj_params.yaml
  Parameters:
    K        = 0.750000
    B        = 0.250000
    IM       = 0.060000
    ...

[3/4] Loading data and initializing simulator
  Input shape: (3653, 1, 2)
  Qobs shape: (3653, 1, 1)
  ✓ UnifiedSimulator initialized

[4/4] Running simulation (warmup=365 days)
  ✓ Simulation completed (3288 time steps)

================================================================================
Simulation Results
================================================================================
Basin: 01013500
Time steps: 3288

Performance Metrics:
  NSE        =   0.7234
  KGE        =   0.6912
  RMSE       =   0.3456

✓ Results saved to: simulation_results.csv
```

---

## Step 8: Visualize Results (1 minute)

```bash
python scripts/visualize.py --eval-dir results/quickstart_exp/evaluation_test
```

This creates plots showing:
- Observed vs. simulated streamflow
- Flow duration curves
- Monthly aggregated comparison
- Residual analysis

**Output files:**
```
results/quickstart_exp/evaluation_test/
├── 01013500_timeseries.png          # Time series plot
├── 01013500_flow_duration.png       # FDC plot
└── 01013500_monthly.png             # Monthly comparison
```

---

## Common Workflows

### Workflow 1: Calibration → Evaluation → Visualization

```bash
# 1. Calibrate
python scripts/run_xaj_calibration.py --config configs/example_config.yaml

# 2. Evaluate
python scripts/run_xaj_evaluate.py \
    --calibration-dir results/quickstart_exp \
    --eval-period test

# 3. Visualize
python scripts/visualize.py --eval-dir results/quickstart_exp/evaluation_test
```

### Workflow 2: Simulation Only (No Calibration)

```bash
# Use custom parameters directly
python scripts/run_xaj_simulate.py \
    --config configs/example_simulate_config.yaml \
    --param-file configs/example_xaj_params.yaml \
    --plot
```

### Workflow 3: Multi-Basin Calibration

Edit config to include multiple basins:

```yaml
data_cfgs:
  basin_ids: ["01013500", "01022500", "01030500"]  # Multiple basins
```

Then run:

```bash
python scripts/run_xaj_calibration.py --config configs/multi_basin_config.yaml
```

---

## Tips for Success

### 1. Start Small
- Use **one basin** initially
- Use **shorter periods** for testing
- Use **fewer iterations** (`rep: 1000, ngs: 20`)

### 2. Check Data Quality
```python
# Verify your data before calibration
from hydrodataset.camels_us import CamelsUs

ds = CamelsUs(data_path)
data = ds.read_timeseries(
    gage_id_lst=["01013500"],
    t_range=["1990-10-01", "2000-09-30"],
    var_lst=["precipitation", "streamflow"]
)

print(f"Data shape: {data.shape}")
print(f"Missing values: {data.isna().sum()}")
```

### 3. Monitor Progress
- Calibration saves checkpoints every N iterations
- Check partial results in `results/` directory
- Use `--verbose` flag for detailed output

### 4. Production Settings

For real research, use longer calibration:

```yaml
training_cfgs:
  algorithm_params:
    rep: 10000        # More iterations
    ngs: 100          # More complexes
    kstop: 50         # Stricter convergence
```

---

## Troubleshooting

### Issue 1: "Data not found" Error

**Solution**: Check data path and basin IDs

```python
from hydrodataset import SETTING
from hydrodataset.camels_us import CamelsUs

# Check data path
print(SETTING["local_data_path"]["datasets-origin"])

# Check basin IDs
ds = CamelsUs(SETTING["local_data_path"]["datasets-origin"])
basin_ids = ds.read_object_ids()
print(f"Available basins: {len(basin_ids)}")
```

### Issue 2: Slow Calibration

**Solutions**:
- Use fewer basins initially
- Reduce `rep` and `ngs` for testing
- Use xaj_mz (faster than full XAJ)
- Consider using GA or scipy algorithms

### Issue 3: Poor Model Performance

**Check**:
- Data quality (missing values, outliers)
- Training period length (need ≥5 years)
- Warmup period (use 365 days minimum)
- Parameter ranges (default ranges may not suit all basins)

### Issue 4: Memory Issues

**Solutions**:
- Process fewer basins at once
- Reduce time period length
- Use data caching (see [Data Guide](data_guide.md))

---

## What's Next?

### 1. Explore Advanced Features

- **Different Datasets**: Try CAMELS-GB, CAMELS-AUS, etc.
  ```yaml
  data_cfgs:
    data_source_type: "camels_gb"
    basin_ids: ["28015"]
  ```

- **Different Algorithms**: Try GA or scipy
  ```yaml
  training_cfgs:
    algorithm_name: "GA"
  ```

- **Custom Periods**: Evaluate on different periods
  ```bash
  python scripts/run_xaj_evaluate.py \
      --calibration-dir results/quickstart_exp \
      --eval-period custom \
      --custom-period 2010-01-01 2015-12-31
  ```

### 2. Use Your Own Data

Follow [Data Guide](data_guide.md) to prepare custom basin data.

### 3. Learn Code Architecture

For deeper understanding, read [Usage Guide](usage.md) - the developer documentation.

### 4. API Usage

Transition from scripts to Python API for more flexibility:

```python
from hydromodel.trainers.unified_calibrate import calibrate
from hydromodel.trainers.unified_evaluate import evaluate

# Load configuration
config = {...}

# Run calibration
results = calibrate(config)

# Run evaluation
eval_results = evaluate(config, param_dir="results/exp", eval_period="test")
```

See [Usage Guide](usage.md) for API details.

---

## Summary

Congratulations! You've learned how to:

- ✅ Install hydromodel and prepare data
- ✅ Configure experiments using YAML files
- ✅ Calibrate models using command-line scripts
- ✅ Evaluate model performance on test periods
- ✅ Run simulations with custom parameters
- ✅ Visualize results

**Key Commands:**

```bash
# Calibration
python scripts/run_xaj_calibration.py --config configs/example_config.yaml

# Evaluation
python scripts/run_xaj_evaluate.py --calibration-dir results/exp --eval-period test

# Simulation
python scripts/run_xaj_simulate.py --param-file results/exp/basin_sceua.csv --plot

# Visualization
python scripts/visualize.py --eval-dir results/exp/evaluation_test
```

---

## Additional Resources

- **Usage Guide**: [usage.md](usage.md) - Developer documentation with code architecture details
- **Data Guide**: [data_guide.md](data_guide.md) - Comprehensive data preparation guide
- **FAQ**: [faq.md](faq.md) - Common questions and solutions
- **GitHub**: https://github.com/OuyangWenyu/hydromodel
- **Issues**: https://github.com/OuyangWenyu/hydromodel/issues

Happy modeling! 🌊
