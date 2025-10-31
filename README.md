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
- **Multiple Calibration Algorithms**: SCE-UA, Genetic Algorithm, and scipy optimizers
- **Comprehensive Evaluation Metrics**: NSE, KGE, RMSE, PBIAS, and more
- **Unified API**: Consistent interfaces for calibration and evaluation
- **Flexible Data Integration**: Seamless support for CAMELS datasets via [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) and custom data via [hydrodatasource](https://github.com/OuyangWenyu/hydrodatasource)
- **Configuration-Based Workflow**: YAML configuration for reproducibility

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

### For Developers

```bash
git clone https://github.com/OuyangWenyu/hydromodel.git
cd hydromodel
uv sync --all-extras
```

### Configuration

#### Option 1: Use Default Paths (Recommended for Quick Start)

No configuration needed! `hydromodel` automatically uses default paths:

**Default data directory:**
- **Windows:** `C:\Users\YourUsername\hydromodel_data\`
- **macOS/Linux:** `~/hydromodel_data/`

The default structure:
```
~/hydromodel_data/
├── datasets-origin/      # CAMELS and other datasets
├── basins-origin/        # Your custom basin data
└── ...
```

#### Option 2: Custom Paths (For Advanced Users)

Create `~/hydro_setting.yml` to specify custom paths:

```yaml
local_data_path:
  root: 'D:/data'
  datasets-origin: 'D:/data/camels'      # For CAMELS datasets
  basins-origin: 'D:/data/my_basins'     # For custom data
```

## How to Use

### 1. Data Preparation

**Using CAMELS Datasets (hydrodataset):**

```bash
pip install hydrodataset
```

```python
from hydrodataset.camels_us import CamelsUs

# Auto-download on first use
ds = CamelsUs(data_path, download=True)
basin_ids = ds.read_object_ids()  # Get basin IDs
```

**Available datasets:** camels_us, camels_aus, camels_br, camels_ch, camels_cl, camels_gb, camels_de, camels_dk, camels_fr, camels_nz, camels_se

**Using Custom Data (hydrodatasource):**

For your own data, use the `selfmadehydrodataset` format:

```bash
pip install hydrodatasource
```

**Data structure:**
```
my_basin_data/
├── attributes/
│   └── attributes.csv              # Basin metadata (required)
├── timeseries/
│   ├── 1D/                         # Daily time series
│   │   ├── basin_001.csv          # One file per basin
│   │   ├── basin_002.csv
│   │   └── ...
│   └── 1D_units_info.json          # Variable units (required)
```

**Required files:**
- `attributes.csv`: Must have `basin_id` and `area` (km²) columns
- `{basin_id}.csv`: Time series with `time` column + variables (`prcp`, `PET`, `streamflow`)
- `{time_scale}_units_info.json`: Units for each variable (e.g., `{"prcp": "mm/day"}`)

**Usage in hydromodel:**
```python
config = {
    "data_cfgs": {
        "data_source_type": "selfmadehydrodataset",  # Use this for custom data
        "data_source_path": "D:/my_basin_data",      # Path to your data
        "basin_ids": ["basin_001"],
        ...
    }
}
```

For detailed format specifications and examples, see:
- [selfmade_data_guide.md](docs/selfmade_data_guide.md) - Complete guide
- [hydrodatasource documentation](https://github.com/OuyangWenyu/hydrodatasource) - Source package

### 2. Quick Start: Calibration, Evaluation, and Visualization

**Option 1: Use Command-Line Scripts (Recommended for Beginners)**

We provide ready-to-use scripts for model calibration, evaluation, and visualization:

```bash
# 1. Calibration
python scripts/run_xaj_calibration.py --config configs/example_config.yaml

# 2. Evaluation on test period
python scripts/run_xaj_evaluate.py --calibration-dir results/xaj_mz_SCE_UA --eval-period test

# 3. Visualization
python scripts/visualize.py --eval-dir results/xaj_mz_SCE_UA/evaluation_test
```

Edit `configs/example_config.yaml` to customize your basin IDs, time periods, and parameters.

**Option 2: Use Python API (For Advanced Users)**

```python
from hydromodel.trainers.unified_calibrate import calibrate
from hydromodel.trainers.unified_evaluate import evaluate

config = {
    "data_cfgs": {"data_source_type": "camels_us", "basin_ids": ["01013500"], ...},
    "model_cfgs": {"model_name": "xaj_mz"},
    "training_cfgs": {"algorithm": "SCE_UA", "loss_func": "RMSE"},
    "evaluation_cfgs": {"metrics": ["NSE", "KGE"]}
}

results = calibrate(config)  # Calibrate
evaluate(config, param_dir="results", eval_period="test")  # Evaluate
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
    },
    "model_cfgs": {
        "model_name": "xaj_mz",                # Model variant
        "source_type": "sources",
        "source_book": "HF",
    },
    "training_cfgs": {
        "algorithm": "SCE_UA",                 # SCE_UA, GA, or scipy
        "loss_func": "RMSE",                   # RMSE, NSE, or KGE
        "output_dir": "results",
        "experiment_name": "my_exp",
        "rep": 10000,                          # Iterations
        "ngs": 100,                            # Complexes (for SCE_UA)
    },
    "evaluation_cfgs": {
        "metrics": ["NSE", "KGE", "RMSE", "PBIAS"],
    },
}
```

### Calibration API

```python
from hydromodel.trainers.unified_calibrate import calibrate

results = calibrate(config)
```

**Output:** Calibration results saved to `{output_dir}/{experiment_name}/`

**Available algorithms:**
- `SCE_UA`: Shuffled Complex Evolution (recommended)
- `GA`: Genetic Algorithm
- `scipy`: scipy.optimize methods

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
- `basins_denorm_params.csv` - Calibrated parameters
- `xaj_mz_evaluation_results.nc` - Full simulation results

**Available metrics:** NSE, KGE, RMSE, PBIAS, FHV, FLV, FMS

## Project Structure

```
hydromodel/
├── hydromodel/
│   ├── models/                    # Model implementations
│   │   ├── xaj.py                 # Standard XAJ model
│   │   └── xaj_mz.py              # XAJ with Muskingum routing
│   ├── trainers/                  # Calibration and evaluation
│   │   ├── unified_calibrate.py   # Calibration API
│   │   └── unified_evaluate.py    # Evaluation API
│   └── datasets/                  # Data preprocessing
├── scripts/                       # Example scripts
└── docs/                          # Documentation
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
