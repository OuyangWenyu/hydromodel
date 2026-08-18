# Usage Guide

> **Document Purpose**: This guide provides a **concise overview** of hydromodel's unified API architecture.
> For detailed examples and workflows, see the [Examples](examples/) section.

---

## Overview

### Unified API Design

hydromodel provides a **completely unified interface** for all hydrological models:

```
Data → UnifiedDataLoader → calibrate() / evaluate() / simulate()
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
| `simulate()` | Model simulation (new) | `trainers.unified_simulate` |
| `UnifiedSimulator` | Direct model simulation | `trainers.unified_simulate` |

---

## Architecture

### Code Structure

```
hydromodel/
├── models/              # Model implementations (XAJ, GR, HYMOD, etc.)
│   ├── model_dict.py   # Unified model registry (MODEL_DICT)
│   └── model_config.py # Parameter configuration
├── trainers/           # Calibration algorithms
│   ├── unified_calibrate.py  # Main calibration interface
│   └── unified_evaluate.py   # Main evaluation interface
├── datasets/           # Data loading and processing
│   └── unified_data_loader.py  # Unified data interface
├── configs/            # Configuration system
│   └── config_manager.py  # Configuration management
└── scripts/            # Entry point scripts
```

### Data Flow

```
Config → UnifiedDataLoader → calibrate() → Saved Results
                                  ↓
                          evaluate() → Metrics
                                  ↓
                          simulate() → Simulated Series
```

---

## Configuration System

All APIs use a **consistent configuration format**:

```python
config = {
    "data_cfgs": {
        "dataset": "camels_us",      # Dataset identifier
        "source": "local",           # "local" or "cloud"
        "basin_ids": ["01013500"],
        "warmup_length": 365,
        "variables": ["precipitation", "potential_evapotranspiration", "streamflow"],
        "train_period": ["1985-10-01", "1995-09-30"],
        "test_period": ["2005-10-01", "2014-09-30"],
    },
    "model_cfgs": {
        "name": "xaj",               # Model name from MODEL_DICT
        "params": {                   # Model-specific parameters
            "source_type": "sources",
            "source_book": "HF",
        },
    },
    "training_cfgs": {
        "algorithm": "SCE_UA",        # SCE_UA, GA, or scipy
        "SCE_UA": {                   # Algorithm-specific settings
            "rep": 10000,
            "ngs": 100,
        },
        "loss": "RMSE",              # Objective function
        "output_dir": "results",
        "experiment_name": "my_exp",
    },
    "evaluation_cfgs": {
        "metrics": ["NSE", "KGE", "RMSE"],
    },
}
```

---

## Core APIs

### Calibration

```python
from hydromodel.trainers.unified_calibrate import calibrate

results = calibrate(config)
# Saves: calibration_results.json, {basin_id}_sceua.csv, calibration_config.yaml
```

**Output:** `{output_dir}/{experiment_name}/calibration_results.json`

For detailed examples, see [Calibration Examples](examples/calibration.md).

### Evaluation

```python
from hydromodel.trainers.unified_evaluate import evaluate

results = evaluate(config, param_dir="results/my_exp", eval_period="test")
# Saves: basins_metrics.csv, {model}_evaluation_results.nc
```

For detailed examples, see [Simulation Examples](examples/simulation.md).

### Simulation

```python
from hydromodel import simulate

results = simulate(config)
# Returns: {"simulation": {"qsim": array}, "qobs": array, "parameters": dict, ...}
```

**Return format:**

| Key | Type | Description |
|-----|------|-------------|
| `simulation` | `dict` | Model output arrays (usually `{"qsim": array}`) |
| `qobs` | `ndarray` | Observed streamflow (if available) |
| `parameters` | `dict` | The parameter values used |
| `model_name` | `str` | Model name |
| `basin_ids` | `list` | Basin IDs simulated |

For detailed examples, see [Simulation Examples](examples/simulation.md).

---

## Supported Models

| Model | Name | Parameters | Notes |
|-------|------|-----------|-------|
| **xaj** | Standard XAJ | 15 | Recession constant + lag time routing |
| **xaj_mz** | XAJ with MizuRoute routing | 15 | Gamma unit hydrograph (MizuRoute) |
| **xaj_slw** | XAJ for Songliao basin | 26 | SMS3 + LAG3 routing |
| **gr4j** | GR4J | 4 | Daily lumped model |
| **gr5j** | GR5J | 5 | Extended GR4J |
| **gr6j** | GR6J | 6 | Extended GR5J |
| **hymod** | HYMOD | 5 | Nash cascade |
| **dhf** | Dahuofang model | 18 | Custom |

For detailed model documentation, see [Models](models/).

---

## Supported Datasets

The authoritative runtime registry lives in `hydrodataset` (public datasets) and
`hydrodatasource` (custom datasets), not in hydromodel. You can extend or override
entries with a project-level `configs/datasets.yml`.

**Public datasets (27, via hydrodataset):**
- **CAMELS series (16)**: `camels_us`, `camels_aus`, `camels_br`, `camels_ch`, `camels_cl`, `camels_col`, `camels_de`, `camels_dk`, `camels_fi`, `camels_pe`, `camels_fr`, `camels_gb`, `camels_ind`, `camels_lux`, `camels_nz`, `camels_se`
- **CAMELSH series (2)**: `camelsh`, `camelsh_kr`
- **CARAVAN series (3)**: `caravan`, `caravan_dk`, `grdc_caravan`
- **LamaH series (2)**: `lamah_ce`, `lamah_ice`

For custom data, see [Data Guide](data_guide.md).

---

## Command-Line Scripts

### Calibration

```bash
# Universal calibration (works with all model types)
python scripts/run_xaj_calibration.py --config configs/example_config.yaml

# GR model calibration
python scripts/calibrate_gr.py --config configs/example_config.yaml

# Flood event calibration
python scripts/run_event_calibration.py --config configs/songliao_event_3h.yaml
```

### Simulation

```bash
# XAJ simulation with custom parameters
python scripts/run_xaj_simulate.py \
    --config configs/example_simulate_config.yaml \
    --param-file configs/example_xaj_params.yaml \
    --plot
```

### Evaluation

```bash
# Evaluate calibration results
python scripts/run_xaj_evaluate.py \
    --calibration-dir results/my_experiment \
    --eval-period test
```

### Visualization

```bash
# Visualize evaluation results
python scripts/visualize.py \
    --eval-dir results/my_experiment/evaluation_test
```

---

## Best Practices

1. **Use YAML configs** for reproducibility
2. **Always use warmup** (typically 365 days for daily data)
3. **Evaluate on independent test period** after calibration
4. **Set random seeds** for reproducible results
5. **Version control** your config files

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
data_loader = UnifiedDataLoader(config["data_cfgs"])
p_and_e, qobs = data_loader.load_data()

# Calibration
from hydromodel.trainers.unified_calibrate import calibrate
results = calibrate(config)

# Evaluation
from hydromodel.trainers.unified_evaluate import evaluate
metrics = evaluate(config, param_dir="results/exp", eval_period="test")

# Simulation
from hydromodel import simulate
results = simulate(config)
```

---

## Additional Resources

- **Quick Start**: [quickstart.md](quickstart.md) - End user guide for quick setup
- **Data Guide**: [data_guide.md](data_guide.md) - Data preparation and management
- **FAQ**: [faq.md](faq.md) - Common questions and solutions
- **Examples**: [Examples](examples/) - Detailed examples and workflows
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
