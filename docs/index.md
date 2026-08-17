<!--
 * @Author: Wenyu Ouyang
 * @Date: 2025-11-08
 * @LastEditTime: 2025-11-08
 * @LastEditors: Wenyu Ouyang
 * @Description: hydromodel documentation homepage
 * @FilePath: \hydromodel\docs\index.md
 * Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
-->
# Welcome to hydromodel

[![image](https://img.shields.io/pypi/v/hydromodel.svg)](https://pypi.python.org/pypi/hydromodel)
[![image](https://img.shields.io/conda/vn/conda-forge/hydromodel.svg)](https://anaconda.org/conda-forge/hydromodel)
[![image](https://pyup.io/repos/github/OuyangWenyu/hydromodel/shield.svg)](https://pyup.io/repos/github/OuyangWenyu/hydromodel)

**A lightweight Python package for hydrological model calibration and evaluation, featuring the XinAnJiang (XAJ) model.**

`hydromodel` is a Python implementation of conceptual hydrological models, with a focus on the **XinAnJiang (XAJ) model** - one of the most widely-used rainfall-runoff models, especially in China and Asian regions. The package provides comprehensive tools for model calibration, evaluation, and simulation with a unified API design.

## Key Features

### 🏞️ Hydrological Models
- **XAJ Model Variants**: Standard XAJ, xaj_mz (mizuRoute routing), xaj_slw (Songliao basin variant)
- **GR Models**: GR1A, GR2M, GR3J, GR4J, GR5J, GR6J
- **Other Models**: HYMOD, DHF(Dahuofang model)
- **Extensible Framework**: Easy to add custom models

### 🔧 Calibration Algorithms
- **SCE-UA**: Shuffled Complex Evolution (robust, recommended for global optimization)
- **GA**: Genetic Algorithm with DEAP (flexible, handles complex parameter landscapes)
- **scipy**: L-BFGS-B, SLSQP, and other gradient-based methods (fast for smooth objectives)

### 📊 Evaluation & Analysis
- **Comprehensive Metrics**: NSE, KGE, RMSE, PBIAS, FHV, FLV, FMS
- **Multi-Basin Support**: Efficient calibration and evaluation for multiple basins simultaneously
- **Time Series Analysis**: Flood event extraction and characterization
- **Visualization**: Automated plotting of simulation results and metrics

### 🗄️ Data Integration
- **CAMELS Datasets**: Seamless support for 11 CAMELS variants via [hydrodataset](https://github.com/OuyangWenyu/hydrodataset)
- **Custom Data**: Flexible support for user data via [hydrodatasource](https://github.com/OuyangWenyu/hydrodatasource)
- **Flood Event Data**: Specialized support for discrete flood event datasets
- **Standardized Format**: Unified data interface across all data sources

### 🚀 Developer-Friendly
- **Unified API**: `calibrate(config)` and `simulate(config)` — two functions for all models
- **Configuration-Based**: YAML configuration for reproducibility
- **Progress Tracking**: Real-time progress display and intermediate results saving
- **Standardized Results**: All algorithms save results in unified JSON + CSV format

## Quick Start

```python
from hydromodel import calibrate, simulate, evaluate

# Configuration for calibration
config = {
    "data_cfgs": {
        "dataset": "camels_us",
        "basin_ids": ["01013500"],
        "train_period": ["1985-10-01", "1995-09-30"],
        "test_period": ["2005-10-01", "2014-09-30"],
        "warmup_length": 365,
        "variables": ["precipitation", "potential_evapotranspiration", "streamflow"]
    },
    "model_cfgs": {
        "name": "xaj_mz",
    },
    "training_cfgs": {
        "algorithm": "SCE_UA",
        "SCE_UA": {"rep": 5000, "ngs": 1000},
        "loss_config": {"type": "time_series", "obj_func": "RMSE"},
        "output_dir": "results",
        "experiment_name": "my_experiment",
    },
    "evaluation_cfgs": {
        "metrics": ["NSE", "KGE", "RMSE"],
    },
}

# Calibrate (finds best parameters)
results = calibrate(config)

# Simulate with specific parameters (no calibration needed)
config["model_cfgs"]["parameters"] = {"K": 0.75, "B": 0.25, ...}
sim_results = simulate(config)

# Evaluate on test period
evaluate(config, param_dir="results/my_experiment", eval_period="test")
```

Or use command-line scripts:

```bash
# 1. Calibration
python scripts/run_xaj_calibration.py --config configs/example_config.yaml

# 2. Evaluation
python scripts/run_xaj_evaluate.py --calibration-dir results/xaj_mz_SCE_UA --eval-period test

# 3. Simulation (no calibration required!)
python scripts/run_xaj_simulate.py \
    --config configs/example_simulate_config.yaml \
    --param-file configs/example_xaj_params.yaml \
    --plot

# 4. Visualization
python scripts/visualize.py --eval-dir results/xaj_mz_SCE_UA/evaluation_test
```

## Installation

### Quick Installation

```bash
pip install hydromodel hydrodataset hydrodatasource
```

Or using `uv` (faster):

```bash
uv pip install hydromodel hydrodataset hydrodatasource hydrodatasource
```

### From Source

```bash
git clone https://github.com/OuyangWenyu/hydromodel.git
cd hydromodel
uv sync --all-extras
```

For detailed installation instructions, see the [Installation Guide](installation.md).

## Why hydromodel?

**For Researchers:**
- Battle-tested XAJ implementations used in published research
- Configuration-based workflow ensures reproducibility
- Easy to extend with new models or calibration algorithms
- Lightweight and fast - perfect for parameter sensitivity studies

**For Practitioners:**
- Simple YAML configuration, minimal coding required
- Handles multi-basin calibration efficiently
- Integration with global public datasets (27 registered datasets)
- Clear documentation and examples

**Compared to other packages:**
- **vs. SWAT/VIC**: Lighter weight, Python-native, faster iteration
- **vs. pySTREPS**: Focus on conceptual rainfall-runoff models
- **vs. custom scripts**: Well-tested with unified interfaces

## Documentation Structure

- **[Installation Guide](installation.md)** - Detailed installation instructions for all platforms
- **[Quick Start](quickstart.md)** - Get started in 5 minutes
- **[Usage Guide](usage.md)** - Comprehensive tutorials and examples
- **[Data Guide](data_guide.md)** - How to prepare and use different data sources
- **[API Reference](hydromodel.md)** - Complete API documentation
- **[Model Documentation](models/xaj.md)** - Detailed model descriptions ([XAJ](models/xaj.md), [DHF](models/dhf.md))
- **[Contributing](contributing.md)** - How to contribute to the project
- **[FAQ](faq.md)** - Frequently asked questions
- **[Changelog](changelog.md)** - Version history and updates

## Use Cases

### 1. Model Calibration

Calibrate hydrological models on CAMELS datasets or custom data with various algorithms:

```python
# Use SCE-UA for global optimization
config["training_cfgs"]["algorithm"] = "SCE_UA"
results = calibrate(config)

# Or use GA for flexible optimization
config["training_cfgs"]["algorithm"] = "GA"
results = calibrate(config)
```

### 2. Multi-Basin Evaluation

Efficiently calibrate and evaluate multiple basins:

```python
config["data_cfgs"]["basin_ids"] = ["01013500", "01022500", "01030500"]
results = calibrate(config)
evaluate(config, param_dir="results/my_experiment", eval_period="test")
```

### 3. Parameter Sensitivity Analysis

Run simulations with custom parameter sets:

```python
from hydromodel.trainers.unified_simulate import UnifiedSimulator

# Test different parameter values
for k_value in [0.5, 0.75, 1.0]:
    parameters = {..., "K": k_value, ...}
    simulator = UnifiedSimulator(model_config, basin_config)
    results = simulator.simulate(inputs, qobs, warmup_length=365)
    # Analyze results
```

### 4. Flood Event Analysis

Extract and calibrate on flood events:

```python
config = {
    "data_cfgs": {
        "dataset": "songliao_event",  # registered flood-event dataset
        "time_unit": ["3h"],
        "variables": ["prcp", "PET", "streamflow"],
        ...
    },
    ...
}
results = calibrate(config)
```

## Supported Models

| Model | Description | Parameters | Routing |
|-------|-------------|------------|---------|
| **xaj** | Standard XinAnJiang model | 15 | Linear reservoir |
| **xaj_mz** | XAJ with mizuRoute routing | 15 | Gamma unit hydrograph (mizuRoute) |
| **xaj_slw** | XAJ for Songliao basin (SLW) | 26 | SMS3 + LAG3 |
| **gr1a / gr2m / gr3j** | GR rainfall-runoff models | 1 / 2 / 3 | Unit hydrograph |
| **gr4j** | GR4J rainfall-runoff model | 4 | Unit hydrograph |
| **gr5j / gr6j** | GR rainfall-runoff models | 5 / 6 | Unit hydrograph |
| **hymod** | HYMOD model | 5 | Nash cascade |
| **dhf** | Dahuofang model | 18 | Custom |
| **semi_xaj** | Semi-distributed XAJ variant | - | Custom |
| **unit_hydrograph / categorized_unit_hydrograph** | Unit hydrograph models | - | Unit hydrograph |

For detailed model documentation, see [XAJ Model](models/xaj.md) and [DHF Model](models/dhf.md).

## Calibration Algorithms

| Algorithm | Type | Strengths | Best For |
|-----------|------|-----------|----------|
| **SCE-UA** | Global | Robust, reliable convergence | General purpose, recommended |
| **GA** | Global | Flexible, handles discontinuities | Complex parameter landscapes |
| **scipy** | Local | Fast, gradient-based | Smooth objectives, refinement |

## Data Sources

### CAMELS Datasets

27 public datasets are registered in `hydrodataset`, including the CAMELS series (`camels_us`, `camels_aus`, `camels_br`, `camels_ch`, `camels_cl`, `camels_col`, `camels_de`, `camels_dk`, `camels_fi`, `camels_pe`, `camels_fr`, `camels_gb`, `camels_ind`, `camels_lux`, `camels_nz`, `camels_se`), CAMELSH, CARAVAN, LamaH, and others. See the [Data Guide](data_guide.md) for the full list and local/cloud configuration.

### Custom Data

Use your own data with a `uri` + reader alias (e.g. `selfmade`), or a registered custom dataset id such as `songliao_event`:

```
my_basin_data/
├── attributes/
│   └── attributes.csv
├── shapes/
│   └── basins.shp
├── timeseries/
│   ├── 1D/
│   │   ├── basin_001.csv
│   │   └── basin_002.csv
│   └── 1D_units_info.json
```

See [Data Guide](data_guide.md) for complete specifications.

## Performance

- **Fast calibration**: Optimized algorithms with numba JIT compilation
- **Memory efficient**: Handles large datasets with chunked processing
- **Parallel support**: Multi-basin calibration runs independently
- **Progress tracking**: Real-time monitoring of long-running calibrations

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

## License & Credits

- **License**: GNU General Public License v3.0
- **Author**: Wenyu Ouyang
- **Documentation**: <https://OuyangWenyu.github.io/hydromodel>
- **Source Code**: <https://github.com/OuyangWenyu/hydromodel>

## Getting Help

- **Documentation**: Browse the complete [documentation](https://OuyangWenyu.github.io/hydromodel)
- **Issues**: Report bugs or request features at [GitHub Issues](https://github.com/OuyangWenyu/hydromodel/issues)
- **Discussions**: Ask questions at [GitHub Discussions](https://github.com/OuyangWenyu/hydromodel/discussions)
- **Email**: wenyuouyang@outlook.com

## Contributing

Contributions are welcome! See the [Contributing Guide](contributing.md) for details on:

- Reporting bugs
- Suggesting features
- Submitting pull requests
- Code style and testing guidelines

## Community

Join our growing community:

- ⭐ Star the project on [GitHub](https://github.com/OuyangWenyu/hydromodel)
- 🐛 Report issues and bugs
- 💡 Suggest new features
- 📖 Improve documentation
- 🔧 Contribute code

---

**Ready to get started?** Head to the [Quick Start Guide](quickstart.md) or [Installation Guide](installation.md)!
