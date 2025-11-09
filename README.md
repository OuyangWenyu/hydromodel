# hydromodel

[![image](https://img.shields.io/pypi/v/hydromodel.svg)](https://pypi.python.org/pypi/hydromodel)
[![image](https://img.shields.io/conda/vn/conda-forge/hydromodel.svg)](https://anaconda.org/conda-forge/hydromodel)
[![image](https://pyup.io/repos/github/OuyangWenyu/hydromodel/shield.svg)](https://pyup.io/repos/github/OuyangWenyu/hydromodel)

**A lightweight Python package for hydrological model calibration and evaluation, featuring the XinAnJiang (XAJ) model.**

- Free software: GNU General Public License v3
- Documentation: https://OuyangWenyu.github.io/hydromodel

## What is hydromodel

`hydromodel` is a Python implementation of conceptual hydrological models, with a focus on the **XinAnJiang (XAJ) model** - one of the most widely-used rainfall-runoff models, especially in China and Asian regions.

**Key Features:**
- **XAJ Model Variants**: Standard XAJ and optimized versions (xaj_mz with Muskingum routing)
- **Multiple Calibration Algorithms**:
  - **SCE-UA**: Shuffled Complex Evolution (robust, recommended)
  - **GA**: Genetic Algorithm with DEAP (flexible, customizable)
  - **scipy**: L-BFGS-B, SLSQP, and other gradient-based methods (fast)
- **Multi-Basin Support**: Efficient calibration and evaluation for multiple basins simultaneously
- **Unified Results Format**: All algorithms save results in standardized JSON + CSV format
- **Comprehensive Evaluation Metrics**: NSE, KGE, RMSE, PBIAS, and more
- **Unified API**: Consistent interfaces for calibration, evaluation, and simulation
- **Flexible Data Integration**: Seamless support for CAMELS datasets via [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) and custom data via [hydrodatasource](https://github.com/OuyangWenyu/hydrodatasource)
- **Configuration-Based Workflow**: YAML configuration for reproducibility
- **Progress Tracking**: Real-time progress display and intermediate results saving

## Why hydromodel?

**For Researchers:**
- Battle-tested XAJ implementations used in published research
- Configuration-based workflow ensures reproducibility
- Easy to extend with new models or calibration algorithms
- Lightweight and fast - perfect for parameter sensitivity studies

**For Practitioners:**
- Simple YAML configuration, minimal coding required
- Handles multi-basin calibration efficiently
- Integration with global CAMELS datasets (11 variants)
- Clear documentation and examples

**Compared to other packages:**
- **vs. SWAT/VIC**: Lighter weight, Python-native, faster iteration
- **vs. pySTREPS**: Focus on conceptual rainfall-runoff models
- **vs. custom scripts**: Well-tested with unified interfaces

## Installation

### For Users

```bash
pip install hydromodel hydrodataset
```

Or using `uv` (faster):

```bash
uv pip install hydromodel hydrodataset
```

### Development Setup

For developers, it is recommended to use `uv` to manage the environment, as this project has local dependencies (e.g., `hydroutils`, `hydrodataset`, `hydrodatasource`).

1. **Clone the repository:**
   ```bash
   git clone https://github.com/OuyangWenyu/hydromodel.git
   cd hydromodel
   ```

2. **Sync the environment with `uv`:**
   This command will install all dependencies, including the local editable packages.
   ```bash
   uv sync --all-extras
   ```

### Configuration

#### Option 1: Use Default Paths (Recommended for Quick Start)

No configuration needed! `hydromodel` automatically uses default paths:

**Default data directory:**
- **Windows:** `C:\Users\YourUsername\hydromodel_data\`
- **macOS/Linux:** `~/hydromodel_data/`

The default structure (aqua_fetch automatically creates uppercase dataset directories):
```
~/hydromodel_data/
├── datasets-origin/
│   ├── CAMELS_US/        # CAMELS US dataset (created by aqua_fetch)
│   ├── CAMELS_AUS/       # CAMELS Australia dataset (if used)
│   └── ...               # Other datasets
├── basins-origin/        # Your custom basin data
└── ...
```

#### Option 2: Custom Paths (For Advanced Users)

Create `~/hydro_setting.yml` to specify custom paths:

```yaml
local_data_path:
  root: 'D:/data'
  datasets-origin: 'D:/data'             # For CAMELS datasets (aqua_fetch adds CAMELS_US automatically)
  datasets-imterim: 'D:/data/my_basins'     # For custom data
```

**Important**: For CAMELS datasets, provide only the `datasets-origin` directory. The system automatically appends the uppercase dataset directory name (e.g., `CAMELS_US`, `CAMELS_AUS`). If your data is in `D:/data/CAMELS_US/`, set `datasets-origin: 'D:/data'`.

## How to Use

### 1. Data Preparation

**Using CAMELS Datasets (hydrodataset):**

Getting public datasets using hydrodataset

```bash
pip install hydrodataset
```
Run the following code to download data to your directory
```python
from hydrodataset.camels_us import CamelsUs

# Auto-downloads if not found. Provide datasets-origin directory (e.g., "D:/data")
# aqua_fetch automatically appends dataset name, creating "D:/data/CAMELS_US/"
ds = CamelsUs(data_path)
basin_ids = ds.read_object_ids()  # Get basin IDs
```

**Note:** First-time download may take some time. The complete CAMELS dataset is approximately 70GB.

**Available datasets:** camels_us, camels_aus, camels_br, camels_ch, camels_cl, camels_gb, camels_de, camels_dk, camels_fr, camels_nz, camels_se

**Using Custom Data (hydrodatasource):**

For your own data to be read using hydrodatasource, it needs to be prepared in the format of  `selfmadehydrodataset` :

```bash
pip install hydrodatasource
```

**Data structure:**
```
/path/to/your_data_root/
    └── my_custom_dataset/              # your dataset name
        ├── attributes/
        │   └── attributes.csv
        ├── shapes/
        │   └── basins.shp
        └── timeseries/
            ├── 1D/                     # One sub folder per time resolution (e.g. 1D/3h/1h)
            │   ├── basin_01.csv
            │   ├── basin_02.csv
            │   └── ...
            └── 1D_units_info.json      # JSON file containing unit information
```

**Required files and formats:**

1. **attributes/attributes.csv**: Basin metadata with required columns
   - `basin_id`: Unique basin identifier (e.g., "basin_001")
   - `area`: Basin area in km² (mapped to `basin_area` internally)
   - Additional columns: Any basin attributes (e.g., elevation, slope)

2. **shapes/basins.shp**: Basin boundary shapefiles (all 4 files required: .shp, .shx, .dbf, .prj)
   - Must contain `BASIN_ID` column (uppercase) matching basin IDs in attributes.csv
   - Geometries: Polygon features defining basin boundaries
   - Coordinate system: Any valid CRS (e.g., EPSG:4326 for WGS84)

3. **timeseries/{time_scale}/{basin_id}.csv**: Time series data for each basin
   - `time`: Datetime column (e.g., "2010-01-01")
   - Variable columns: `prcp`, `PET`, `streamflow` (or your chosen variable names)
   - Format: CSV with header row

4. **timeseries/{time_scale}_units_info.json**: Variable units metadata
   - JSON format: `{"variable_name": "unit"}` (e.g., `{"prcp": "mm/day"}`)
   - Must match variable names in time series files


For detailed format specifications and examples, see:
- [Data Guide](docs/data_guide.md) - Complete guide for both CAMELS and custom data
- [hydrodatasource documentation](https://github.com/OuyangWenyu/hydrodatasource) - Source package
- `configs/example_config_selfmade.yaml` - Complete configuration example for custom datasets

### 2. Quick Start: Calibration, Evaluation, Simulation, and Visualization

**Option 1: Use Command-Line Scripts (Recommended for Beginners)**

We provide ready-to-use scripts for model calibration, evaluation, simulation, and visualization:

```bash
# 1. Calibration (saves config files by default)
python scripts/run_xaj_calibration.py --config configs/example_config.yaml

# 2. Evaluation on test period
python scripts/run_xaj_evaluate.py --calibration-dir results/xaj_mz_SCE_UA 

# 3. Simulation with custom parameters (no calibration required!)
python scripts/run_xaj_simulate.py --config configs/example_simulate_config.yaml --param-file configs/example_xaj_params.yaml --plot

# 4. Visualization (time series plots with precipitation and streamflow)
python scripts/visualize.py --eval-dir results/xaj_mz_SCE_UA/evaluation_test

# Visualize specific basins
python scripts/visualize.py --eval-dir results/xaj_mz_SCE_UA/evaluation_test --basins 01013500

```

**Configuration Files:**

Edit the appropriate configuration file for your data type:
- `configs/example_config.yaml` - For continuous time series data (e.g., CAMELS datasets)
- `configs/example_config_selfmade.yaml` - For custom data and flood event datasets

All configuration options work with the same unified API. For detailed flood event data usage, see [Usage Guide - Flood Event Data](docs/usage.md#flood-event-data).

**Option 2: Use Python API (For Advanced Users)**

```python
from hydromodel.trainers.unified_calibrate import calibrate
from hydromodel.trainers.unified_evaluate import evaluate

config = {
    "data_cfgs": {
        "data_source_type": "camels_us",
        "basin_ids": ["01013500"],
        "train_period": ["1985-10-01", "1995-09-30"],
        "test_period": ["2005-10-01", "2014-09-30"],
        "warmup_length": 365,
        "variables": ["precipitation", "potential_evapotranspiration", "streamflow"]
    },
    "model_cfgs": {
        "model_name": "xaj_mz",
    },
    "training_cfgs": {
        "algorithm_name": "SCE_UA",
        "algorithm_params": {"rep": 5000, "ngs": 1000},
        "loss_config": {"type": "time_series", "obj_func": "RMSE"},
        "output_dir": "results",
        "experiment_name": "my_experiment",
    },
    "evaluation_cfgs": {
        "metrics": ["NSE", "KGE", "RMSE"],
    },
}

results = calibrate(config)  # Calibrate
evaluate(config, param_dir="results/my_experiment", eval_period="test")  # Evaluate
```

Results are saved in the `results/` directory.

## Core API

### Configuration Structure

The unified API uses a configuration dictionary with four main sections:

```python
config = {
    "data_cfgs": {
        "data_source_type": "camels_us",       # Dataset type
        "basin_ids": ["01013500"],             # Basin IDs to calibrate
        "train_period": ["1990-10-01", "2000-09-30"],
        "test_period": ["2000-10-01", "2010-09-30"],
        "warmup_length": 365,                  # Warmup days
        "variables": ["precipitation", "potential_evapotranspiration", "streamflow"],
    },
    "model_cfgs": {
        "model_name": "xaj_mz",                # Model variant
        "model_params": {
            "source_type": "sources",
            "source_book": "HF",
            "kernel_size": 15,                 # Muskingum routing kernel
        },
    },
    "training_cfgs": {
        "algorithm_name": "GA",                # Algorithm: SCE_UA, GA, or scipy

        # Algorithm-specific parameters (choose one based on algorithm_name)

        # For SCE-UA (Shuffled Complex Evolution):
        "SCE_UA": {
            "rep": 1000,                       # Iterations (5000+ recommended)
            "ngs": 1000,                       # Number of complexes
            "kstop": 500,                      # Stop if no improvement
            "peps": 0.1,                       # Parameter convergence
            "pcento": 0.1,                     # Percentage change allowed
            "random_seed": 1234,
        },

        # For GA (Genetic Algorithm):
        "GA": {
            "pop_size": 80,                    # Population size
            "n_generations": 50,               # Generations (100+ recommended)
            "cx_prob": 0.7,                    # Crossover probability
            "mut_prob": 0.2,                   # Mutation probability
            "random_seed": 1234,
        },

        # For scipy (gradient-based optimization):
        "scipy": {
            "method": "SLSQP",                 # L-BFGS-B, SLSQP, TNC, etc.
            "max_iterations": 500,             # Maximum iterations
        },

        "loss_config": {
            "type": "time_series",
            "obj_func": "RMSE",                # RMSE, NSE, or KGE
        },
        "output_dir": "results",
        "experiment_name": "my_exp",
        "save_config": True,                   # Save config files to output directory (default: True)
    },
    "evaluation_cfgs": {
        "metrics": ["NSE", "KGE", "RMSE", "PBIAS"],
    },
}
```
**Configuration for custom datasets:**

See `configs/example_config_selfmade.yaml` for a complete example. Custom datasets require additional parameters:

```python
"data_cfgs": {
  "dataset": "selfmadehydrodataset"     # or "floodevent" for flood event data
  "dataset_name": "my_basin_data"       # Your dataset folder name (REQUIRED)
  "time_unit": ["1D"]                  # Time resolution (e.g., ["1h"], ["3h"], ["1D"])
  "datasource_kwargs":{                # Optional additional parameters
    "offset_to_utc": False             # Whether to convert local time to UTC
    }              
  "is_event_data": True                # Whether floodevent data
  # ... other standard parameters (basin_ids, variables, periods, etc.)
```

**Key differences from CAMELS datasets:**
- `dataset_name`: Specifies your custom dataset folder name (required)
- `time_unit`: Must match the subdirectory names in `timeseries/` folder
- `datasource_kwargs`: Optional parameters for data preprocessing

### Calibration API

```python
from hydromodel.trainers.unified_calibrate import calibrate

results = calibrate(config)
```

**Output:** Calibration results saved to `{output_dir}/{experiment_name}/`

**Saved files:**
```
results/my_exp/
├── calibration_results.json          # Best parameters for all basins (unified format)
├── {basin_id}_sceua.csv              # SCE-UA detailed iteration history
├── {basin_id}_ga.csv                 # GA generation history with parameters
├── {basin_id}_scipy.csv              # scipy iteration history with parameters
├── calibration_config.yaml           # Configuration used (saved if save_config=True)
└── param_range.yaml                  # Parameter ranges for current model only (saved if save_config=True)
```

**Notes:**
- `calibration_results.json`: Always saved, contains best parameters
- `calibration_config.yaml` and `param_range.yaml`: Only saved if `save_config=True` (default)
- `param_range.yaml`: Contains parameter ranges for the current model only (e.g., only `xaj_mz`, not all models)
- In `calibration_config.yaml`, `param_range_file` is set to the actual saved path

**Available algorithms:**
- `SCE_UA` / `sceua`: Shuffled Complex Evolution (recommended for global optimization)
- `GA` / `genetic_algorithm`: Genetic Algorithm with DEAP (flexible, handles complex landscapes)
- `scipy` / `scipy_minimize`: scipy.optimize methods (fast for smooth objectives)

### Evaluation API

```python
from hydromodel.trainers.unified_evaluate import evaluate

# Evaluate on test period
test_results = evaluate(config, param_dir="results/my_exp", eval_period="test")

# Evaluate on training period
train_results = evaluate(config, param_dir="results/my_exp", eval_period="train")

# Evaluate on custom period
custom_results = evaluate(
    config,
    param_dir="results/my_exp",
    eval_period="custom",
    custom_period=["2010-10-01", "2015-09-30"]
)
```

**Output:** Evaluation results in `{param_dir}/evaluation_{period}/`
- `basins_metrics.csv` - Performance metrics
- `basins_norm_params.csv` - Calibrated parameters (normalized [0,1])
- `basins_denorm_params.csv` - Denormalized parameters (physical values)
- `xaj_mz_evaluation_results.nc` - Full simulation results (NetCDF)

**Parameter Loading Priority:**
1. `calibration_results.json` (⭐ Recommended, works for all algorithms)
2. `{basin_id}_ga.csv` (GA algorithm CSV)
3. `{basin_id}_scipy.csv` (scipy algorithm CSV)
4. `{basin_id}_sceua.csv` (SCE-UA algorithm CSV)
5. `{basin_id}_calibrate_params.txt` (Legacy format)

**Available metrics:** NSE, KGE, RMSE, PBIAS, FHV, FLV, FMS

### Understanding Results Format

**calibration_results.json structure:**
```json
{
  "01013500": {
    "convergence": "success",
    "objective_value": 1.234567,
    "best_params": {
      "xaj": {
        "K": 0.567890,
        "B": 0.234567,
        "IM": 0.045678,
        ...
      }
    },
    "algorithm_info": {
      "generations": 50,
      "population_size": 80,
      ...
    }
  }
}
```

**CSV files (GA/scipy) structure:**
```csv
generation,objective_value,param_K,param_B,param_IM,...
0,3.456,0.567,0.234,0.045,...
1,2.345,0.589,0.256,0.047,...
```

**Why two formats?**
- **JSON**: Best parameters only, works with all algorithms, used by evaluation
- **CSV**: Full iteration/generation history, useful for convergence analysis

### Simulation API

**Important:** Simulation does NOT require prior calibration!

`UnifiedSimulator` provides a flexible interface for running model simulations with any parameter values:

```python
from hydromodel.trainers.unified_simulate import UnifiedSimulator
from hydromodel.datasets.unified_data_loader import UnifiedDataLoader

# Load data
data_loader = UnifiedDataLoader(config["data_cfgs"])
p_and_e, qobs = data_loader.load_data()

# Define parameters (from calibration, literature, or custom values)
parameters = {
    "K": 0.75, "B": 0.25, "IM": 0.06,
    "UM": 18.0, "LM": 80.0, "DM": 95.0,
    # ... other parameters
}

# Create simulator
model_config = {
    "model_name": "xaj_mz",
    "parameters": parameters
}
simulator = UnifiedSimulator(model_config, basin_config)

# Run simulation
results = simulator.simulate(
    inputs=p_and_e,
    qobs=qobs,
    warmup_length=365
)

# Extract results
qsim = results["qsim"]  # Simulated streamflow
```

**Command-line usage:**

```bash
# Using custom parameters (works with any parameter values)
python scripts/run_xaj_simulate.py \
    --config configs/example_simulate_config.yaml \
    --param-file configs/example_xaj_params.yaml \
    --output simulation_results.csv \
    --plot

# Using calibrated parameters from SCE-UA (CSV format)
python scripts/run_xaj_simulate.py \
    --param-file results/xaj_mz_SCE_UA/01013500_sceua.csv \
    --plot
```

**Use cases:**
- Parameter sensitivity analysis
- Model comparison
- Scenario testing with custom parameters
- Literature parameter validation

For detailed API documentation and advanced usage, see [Usage Guide - Model Simulation](docs/usage.md#model-simulation).

## Project Structure

```
hydromodel/
├── hydromodel/
│   ├── models/                      # Model implementations
│   │   ├── xaj.py                   # Standard XAJ model
│   │   ├── gr4j.py                  # GR4J model
│   │   └── ...
│   ├── trainers/                    # Calibration, evaluation, and simulation
│   │   ├── unified_calibrate.py     # Calibration API
│   │   ├── unified_evaluate.py      # Evaluation API
│   │   └── unified_simulate.py      # Simulation API
│   └── datasets/                    # Data preprocessing and visualization
│       ├── unified_data_loader.py   # Data loader
│       ├── data_visualize.py        # Visualization functions
│       └── ...
├── scripts/                         # Command-line interface scripts
│   ├── run_xaj_calibration.py       # Calibration script
│   ├── run_xaj_evaluate.py          # Evaluation script
│   ├── run_xaj_simulate.py          # Simulation script
│   └── visualize.py                 # Visualization CLI
├── configs/                         # Configuration files
└── docs/                            # Documentation
```

## Documentation

- **Quick Start**: [docs/quickstart.md](docs/quickstart.md)
- **Usage Guide**: [docs/usage.md](docs/usage.md)
- **API Reference**: https://OuyangWenyu.github.io/hydromodel

## References

**XAJ Model:**
- Zhao, R.J., 1992. The Xinanjiang model applied in China. Journal of Hydrology, 135(1-4), pp.371-381.

**Calibration Algorithms:**
- Duan, Q., et al., 1992. Effective and efficient global optimization for conceptual rainfall-runoff models. Water Resources Research, 28(4), pp.1015-1031. (SCE-UA)

**Related Projects:**
- [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) - CAMELS and other datasets
- [hydrodatasource](https://github.com/OuyangWenyu/hydrodatasource) - Data preparation utilities
- [torchhydro](https://github.com/OuyangWenyu/torchhydro) - PyTorch-based hydrological models

## Citation

If you use hydromodel in your research, please cite:

```bibtex
@software{hydromodel,
  author = {Ouyang, Wenyu},
  title = {hydromodel: A Python Package for Hydrological Model Calibration},
  year = {2025},
  url = {https://github.com/OuyangWenyu/hydromodel}
}
```

## Contributing

Contributions are welcome! For major changes, please open an issue first.

```bash
git clone https://github.com/OuyangWenyu/hydromodel.git
cd hydromodel
uv sync --all-extras
pytest tests/
```

## License

GNU General Public License v3.0 - see [LICENSE](LICENSE) file.

## Contact

- **Author**: Wenyu Ouyang
- **Email**: wenyuouyang@outlook.com
- **GitHub**: https://github.com/OuyangWenyu/hydromodel
- **Issues**: https://github.com/OuyangWenyu/hydromodel/issues
