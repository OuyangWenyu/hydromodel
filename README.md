# hydromodel

[![image](https://img.shields.io/pypi/v/hydromodel.svg)](https://pypi.python.org/pypi/hydromodel)
[![image](https://img.shields.io/conda/vn/conda-forge/hydromodel.svg)](https://anaconda.org/conda-forge/hydromodel)
[![image](https://pyup.io/repos/github/OuyangWenyu/hydromodel/shield.svg)](https://pyup.io/repos/github/OuyangWenyu/hydromodel)

**A Python package for hydrological modeling, calibration, and evaluation, with a focus on conceptual models and unified interfaces.**

- Free software: GNU General Public License v3
- Documentation: https://OuyangWenyu.github.io/hydromodel

## Core Philosophy

`hydromodel` provides a unified, production-ready framework for hydrological modeling with conceptual models like XinAnJiang (XAJ). The package has been redesigned to offer:

1. **Unified API**: Consistent interfaces for calibration, evaluation, and simulation across all models
2. **Flexible Data Integration**: Seamless integration with [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) for accessing public datasets
3. **Multiple Calibration Algorithms**: Built-in support for SCE-UA, GA, and scipy optimizers
4. **Reproducible Workflows**: Configuration-based approach with automatic result archiving
5. **Extensibility**: Easy to add new models, algorithms, and datasets

The core workflow is:
1. **Configure**: Define your modeling setup via simple YAML configuration files
2. **Calibrate**: Optimize model parameters using state-of-the-art algorithms
3. **Evaluate**: Assess model performance with comprehensive metrics
4. **Analyze**: Access results in standard formats (CSV, NetCDF, YAML)

## What is hydromodel

**hydromodel is a Python implementation for common hydrological models, with the XinAnJiang (XAJ) model as the flagship implementation.** XAJ is one of the most widely-used conceptual hydrological models, especially in China and across Asian regions.

Key capabilities:
- **XAJ Model Family**: Multiple XAJ variants including the standard version and optimized muskingum routing (xaj_mz)
- **Calibration Framework**: Production-ready parameter optimization with SCE-UA and genetic algorithms
- **Evaluation Tools**: Comprehensive model assessment with NSE, KGE, RMSE, PBIAS, and more
- **Differentiable Models**: PyTorch-based implementations in [torchhydro](https://github.com/OuyangWenyu/torchhydro) for deep learning integration

## Installation

We strongly recommend using a virtual environment to manage dependencies.

### For Users

To install the package from PyPI:

```bash
# Using pip
pip install hydromodel

# Or using uv (recommended for faster installation)
uv pip install hydromodel
```

### For Developers

This project uses [uv](https://github.com/astral-sh/uv) for package and environment management:

```bash
# Clone the repository
git clone https://github.com/OuyangWenyu/hydromodel.git
cd hydromodel

# Create a virtual environment and install dependencies using uv
uv sync --all-extras
```

This will install the base dependencies plus all optional dependencies for development and documentation.

## Quick Start

### 1. Configuration Setup

Create a `hydro_setting.yml` file in your home directory to configure data paths:

**Windows:** `C:\Users\YourUsername\hydro_setting.yml`
**macOS/Linux:** `~/hydro_setting.yml`

```yaml
local_data_path:
  root: 'D:\data\waterism'
  datasets-origin: 'D:\data\waterism\datasets-origin'
  cache: 'D:\data\waterism\cache'
```

### 2. Model Calibration

Create a configuration file for your calibration experiment:

```yaml
# calibration_config.yaml
data:
  dataset: "camels_us"                    # Dataset type
  path: "D:/data/waterism/datasets-origin"  # Data path
  basin_ids: ["01013500"]                 # Basin IDs to calibrate
  train_period: ["1990-10-01", "2000-09-30"]
  test_period: ["2000-10-01", "2010-09-30"]
  warmup_length: 365
  output_dir: "results"
  experiment_name: "xaj_camels_test"

model:
  name: "xaj_mz"                          # Model type
  params:
    source_type: "sources"
    source_book: "HF"
    kernel_size: 15
    time_interval_hours: 24

training:
  algorithm: "SCE_UA"                     # Optimization algorithm
  loss: "RMSE"                            # Objective function
  SCE_UA:
    random_seed: 1234
    rep: 10000                            # Maximum iterations
    ngs: 100                              # Number of complexes
    kstop: 50
    peps: 0.1
    pcento: 0.1

evaluation:
  metrics: ["NSE", "KGE", "RMSE", "PBIAS"]
```

Run calibration from Python:

```python
from hydromodel.trainers.unified_calibrate import calibrate

# Load and run calibration
config = load_config("calibration_config.yaml")
results = calibrate(config)

print(f"Calibration complete! Results saved to {config['training_cfgs']['output_dir']}")
```

Or use the command-line script:

```bash
python scripts/run_xaj_calibration.py --config calibration_config.yaml
```

### 3. Model Evaluation

Evaluate the calibrated model on different time periods:

```python
from hydromodel.trainers.unified_evaluate import evaluate

# Evaluate on test period
results = evaluate(
    config,
    param_dir="results/xaj_camels_test",
    eval_period="test"
)

# Or evaluate on custom period
results = evaluate(
    config,
    param_dir="results/xaj_camels_test",
    eval_period="custom",
    custom_period=["2010-10-01", "2015-09-30"]
)

print("Evaluation metrics:")
for basin_id, metrics in results['metrics'].items():
    print(f"  {basin_id}: NSE={metrics['NSE']:.3f}, KGE={metrics['KGE']:.3f}")
```

Or use the evaluation script:

```bash
python scripts/run_xaj_evaluate.py --exp xaj_camels_test --eval-period test
```

### 4. Access Results

Calibration and evaluation results are saved in structured formats:

```
results/xaj_camels_test/
├── calibration_config.yaml           # Complete configuration
├── param_range.yaml                  # Parameter ranges used
├── 01013500_sceua.csv               # Optimization history
├── evaluation_test/                  # Test period results
│   ├── basins_metrics.csv            # Performance metrics
│   ├── basins_denorm_params.csv      # Calibrated parameters
│   └── xaj_mz_evaluation_results.nc  # Full simulation results
```

## Core API

### Unified Calibration Interface

```python
from hydromodel.trainers.unified_calibrate import calibrate

config = {
    "data_cfgs": {
        "data_source_type": "camels_us",
        "data_source_path": "D:/data/camels",
        "basin_ids": ["01013500", "01022500"],
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

results = calibrate(config)
```

### Unified Evaluation Interface

```python
from hydromodel.trainers.unified_evaluate import evaluate

results = evaluate(
    config,                              # Same config used for calibration
    param_dir="results/my_experiment",   # Directory with calibrated parameters
    eval_period="test",                  # "train", "test", or "custom"
)

# Access metrics
metrics_df = results['metrics_df']
print(metrics_df)
```

### Direct Model Simulation

```python
from hydromodel.models.model_factory import model_factory
import numpy as np

# Create model instance
model = model_factory(
    model_name="xaj_mz",
    source_type="sources",
    source_book="HF"
)

# Prepare input data
# p: precipitation [n_basins, n_time_steps]
# pet: potential evapotranspiration [n_basins, n_time_steps]
# params: model parameters [n_basins, n_params] (normalized [0,1])

q_sim, _ = model.run(p, pet, params)
print(f"Simulated streamflow shape: {q_sim.shape}")
```

## Key Features

### 1. Multiple XAJ Implementations

- **xaj**: Standard XAJ model with traditional routing
- **xaj_mz**: XAJ with Muskingum routing (recommended for easier parameter optimization)

### 2. Flexible Data Sources

Seamlessly work with multiple data sources:
- **CAMELS datasets** via [hydrodataset](https://github.com/OuyangWenyu/hydrodataset)
- **Custom data** via `selfmadehydrodataset` format
- **Any dataset** by implementing the data loader interface

### 3. Production-Ready Calibration

- **SCE-UA**: Shuffled Complex Evolution (recommended for hydrological models)
- **GA**: Genetic Algorithm from DEAP
- **scipy**: Various scipy.optimize methods
- Automatic parameter normalization and denormalization
- Cross-validation support
- Resume from previous calibrations

### 4. Comprehensive Evaluation

- **Hydrological Metrics**: NSE, KGE, RMSE, PBIAS, FHV, FLV, and more
- **Multiple Periods**: Train, test, or custom time periods
- **Multiple Formats**: CSV tables, NetCDF datasets, YAML summaries
- **Visualization Ready**: Results structured for easy plotting

### 5. Extensible Architecture

```python
# Add custom model
from hydromodel.models.base_model import BaseHydroModel

class MyModel(BaseHydroModel):
    def run(self, p, pet, params):
        # Implement your model
        return q_sim, intermediate_states

# Register and use
model_factory.register("my_model", MyModel)
```

## Model Structure

The XAJ model consists of three main components:

![XAJ Model Structure](docs/img/xaj.jpg)

1. **Evapotranspiration Module**: Computes actual evapotranspiration using coefficient K
2. **Runoff Generation**: Tension water storage with various source implementations
3. **Routing Module**: Linear reservoir routing (CS/L method) or Muskingum routing (xaj_mz)

For detailed mathematical formulations, see the documentation.

## Data Format

### For Custom Data

Organize your data in the `selfmadehydrodataset` format:

```
your_data_directory/
├── attributes.nc              # Static basin attributes
└── timeseries/
    ├── basin_001_lump.nc      # Time series for basin 001
    ├── basin_002_lump.nc
    └── ...
```

Or use CSV format and transform with `scripts/prepare_data.py`:

```
your_csv_data/
├── basin_attributes.csv
├── basin_001.csv
└── basin_002.csv
```

Required variables:
- **Time series**: precipitation, potential evapotranspiration, streamflow
- **Attributes**: area, elevation (optional: more catchment attributes)

## Project Status & Future Work

**Current Status**: Production-ready for XAJ model family with unified API architecture.

**Recently Completed**:
- ✅ Unified calibration and evaluation interfaces
- ✅ Configuration-based workflow
- ✅ Comprehensive metrics calculation
- ✅ Integration with hydrodataset package
- ✅ NetCDF result storage

**In Progress**:
- 🔄 Distributed XAJ for large basins
- 🔄 Additional conceptual models (GR4J, HBV, Sacramento)
- 🔄 Multi-objective calibration
- 🔄 Uncertainty quantification tools

**Planned**:
- 📋 GUI interface for non-programmers
- 📋 Real-time forecasting mode
- 📋 Integration with GIS tools
- 📋 Ensemble modeling framework

## Why hydromodel?

When learning about rainfall-runoff processes and making flood forecasts, classic hydrological models like XAJ serve as trusted baselines for engineers and researchers. However, few open-source implementations exist with production-ready code.

**Open-source science** has transformed hydrological modeling (e.g., SWAT, VIC). By making hydromodel public, we aim to:
- Provide a reliable, well-documented implementation of XAJ
- Enable reproducible hydrological research
- Foster community development of conceptual models
- Bridge traditional hydrology with modern ML/DL approaches

XAJ is widely used in practical production across China and Asia. We believe an open, extensible implementation will help inherit and develop this valuable modeling approach.

## References

### Core References

- Zhao, R.J., Zhuang, Y. L., Fang, L. R., Liu, X. R., Zhang, Q. S. (ed) (1980) The Xinanjiang model, Hydrological Forecasting Proc., Oxford Symp., IAHS Publication, Wallingford, U.K.
- Zhao, R.J., 1992. The Xinanjiang model applied in China. J Hydrol 135 (1–4), 371–381.
- 赵人俊，王佩兰 (2013). 流域水文模拟——新安江模型及其应用. 水利水电出版社. [*Watershed Hydrological Modeling*]

### Calibration Methods

- Duan, Q., Sorooshian, S., and Gupta, V. (1992), Effective and efficient global optimization for conceptual rainfall-runoff models, Water Resour. Res., 28(4), 1015–1031, doi:10.1029/91WR02985.
- Houska T, Kraft P, Chamorro-Chavez A, Breuer L (2015) SPOTting Model Parameters Using a Ready-Made Python Package. PLoS ONE 10(12): e0145180. https://doi.org/10.1371/journal.pone.0145180

### Related Projects

- **torchhydro**: Differentiable version of XAJ for deep learning integration
  https://github.com/OuyangWenyu/torchhydro
- **hydrodataset**: Dataset access and management
  https://github.com/OuyangWenyu/hydrodataset
- **HydroDHM**: Applied research using hydromodel
  https://github.com/OuyangWenyu/HydroDHM

### Other XAJ Implementations

- Matlab: https://github.com/wknoben/MARRMoT (m_28_xinanjiang_12p_4s.m)
- Java: https://github.com/wfxr/xaj-hydrological-model
- R/C++: https://github.com/Sibada/XAJ

## Contributing

We welcome contributions! Here's how you can help:

1. **Report Issues**: Found a bug? Have a feature request? Post on [GitHub Issues](https://github.com/OuyangWenyu/hydromodel/issues)
2. **Submit Pull Requests**: Want to add a feature? Create a branch and send a PR
3. **Improve Documentation**: Help make the docs clearer and more comprehensive
4. **Share Use Cases**: Tell us how you're using hydromodel

For major changes, please open an issue first to discuss what you would like to change.

## Citation

If you use hydromodel in your research, please cite:

```bibtex
@software{hydromodel2024,
  author = {Ouyang, Wenyu},
  title = {hydromodel: A Python package for hydrological modeling},
  year = {2024},
  url = {https://github.com/OuyangWenyu/hydromodel},
  version = {0.1.0}
}
```

## License

GNU General Public License v3 - see LICENSE file for details.

## Contact

- **Author**: Wenyu Ouyang
- **GitHub**: https://github.com/OuyangWenyu/hydromodel
- **Issues**: https://github.com/OuyangWenyu/hydromodel/issues
- **Documentation**: https://OuyangWenyu.github.io/hydromodel
