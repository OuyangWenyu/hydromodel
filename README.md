# hydromodel

[![image](https://img.shields.io/pypi/v/hydromodel.svg)](https://pypi.python.org/pypi/hydromodel)
[![image](https://img.shields.io/conda/vn/conda-forge/hydromodel.svg)](https://anaconda.org/conda-forge/hydromodel)
[![image](https://pyup.io/repos/github/OuyangWenyu/hydromodel/shield.svg)](https://pyup.io/repos/github/OuyangWenyu/hydromodel)

**A lightweight Python package for hydrological model calibration, evaluation, and simulation.**

- Free software: GNU General Public License v3
- Documentation: https://OuyangWenyu.github.io/hydromodel

## What is hydromodel

`hydromodel` is a Python implementation of conceptual hydrological models, with a focus on the **XinAnJiang (XAJ) model** - one of the most widely-used rainfall-runoff models, especially in China and Asian regions.

**Registered models (`MODEL_DICT`):**
- **XAJ family**: `xaj`, `xaj_mz` (with mizuRoute routing method), `xaj_slw` (Songliao variant), `semi_xaj`
- **GR family**: `gr1a`, `gr2m`, `gr3j`, `gr4j`, `gr5j`, `gr6j`
- **Others**: `hymod`, `dhf`, `unit_hydrograph`, `categorized_unit_hydrograph`

**Key Features:**
- **Multiple Calibration Algorithms**:
  - **SCE-UA**: Shuffled Complex Evolution with spotpy
  - **GA**: Genetic Algorithm with DEAP
  - **scipy**: L-BFGS-B, SLSQP, and other gradient-based methods
- **Multi-Basin Support**: Calibration/evaluation for multiple basins (one basin at a time)
- **Unified Results Format**: All algorithms save results in standardized JSON + CSV format
- **Comprehensive Evaluation Metrics**: NSE, KGE, RMSE, PBIAS, FHV, FLV, and more
- **Unified API**: Consistent interfaces for calibration, evaluation, and simulation
- **Local & Cloud Data**: Read public datasets (CAMELS series, etc.) and custom data from local disk or OSS/S3 (`source: local` / `source: cloud`)
- **Configuration-Based Workflow**: YAML configuration for reproducibility
- **Parameter Contracts**: Strict validation of `param_range_file` (missing files, unknown/missing parameters, invalid ranges fail fast)

## Why hydromodel?

**For Researchers:**
- Battle-tested XAJ implementations used in published research
- Configuration-based workflow ensures reproducibility
- Easy to extend with new models or calibration algorithms

**For Practitioners:**
- Simple YAML configuration, minimal coding required
- Handles multi-basin calibration
- Integration with global public datasets (27 registered datasets) and custom basin data
- Clear documentation and examples

## Installation

### For Users

```bash
pip install hydromodel hydrodataset hydrodatasource
```

Or using `uv` (faster):

```bash
uv pip install hydromodel hydrodataset hydrodatasource
```

### Development Setup

For developers, it is recommended to use `uv` to manage the environment, as this project has local dependencies (e.g., `hydroutils`, `hydrodataset`, `hydrodatasource`).

1. **Clone the repository:**
   ```bash
   git clone https://github.com/OuyangWenyu/hydromodel.git
   cd hydromodel
   ```

2. **Sync the environment with `uv`:**
   This command installs all dependencies, including the local editable packages declared under `[tool.uv.sources]`.
   ```bash
   uv sync --all-extras
   ```

## Configuration

`hydromodel` no longer resolves data paths itself. Data loading is delegated to
`hydrodatasource` / `hydrodataset` (see `hydromodel/datasets/unified_data_loader.py`),
which read storage settings from `hydro_setting.yml`.

### Storage Configuration

Storage settings live in a `storage:` block in one of two YAML files:

1. `~/hydro_setting.yml` - user-level, shared across projects
2. `{project_root}/.hydro_setting.yml` - project-level; overrides the user-level file key-by-key when present

```yaml
storage:
  default_source: local          # used when data_cfgs.source is omitted
  local:
    root: F:/data                # parent dir containing dataset folders (e.g. CAMELS_US/)
  cache: D:/netcdf               # NetCDF/zarr cache directory
  s3:                            # only needed for source: cloud
    bucket: hydrodataset
    prefix: ''
    access_key_id: your_access_key
    secret_access_key: your_secret_key
    endpoint_url: https://oss-cn-beijing.aliyuncs.com
```

**Important:**
- `storage.local.root` is the **parent** directory that contains per-dataset folders (e.g., `F:/data/CAMELS_US/`). The dataset readers append the dataset folder name automatically. The legacy `local_data_path` block is **not** read by the new resolver.
- Set `data_cfgs.source: cloud` (or `storage.default_source: cloud`) to read from OSS/S3. Cloud reads use zarr caches under `s3://<bucket>/zarr/`; if missing, they are generated from raw files (requires write access).
- `source` is a hard selection: if the chosen backend is unavailable, the run fails rather than silently falling back.

### Dataset Registry

Dataset ids are resolved through a layered registry: built-in defaults (in `hydrodataset`/`hydrodatasource`) first, then an optional project-level `configs/datasets.yml` override. See [Supported Datasets](#supported-datasets) below.

## How to Use

### 1. Data Preparation

**Public datasets (hydrodataset):**

```python
from hydrodatasource.configs.data_resolver import open_dataset

ds = open_dataset("camels_us", source="local")   # or source="cloud"
basin_ids = ds.read_object_ids()                 # e.g. 671 CAMELS-US basins
```

First-time downloads can be large (CAMELS-US is roughly 70 GB including zipped and unzipped files).

**Custom data (hydrodatasource):**

Use a registered custom dataset id, or point directly at your data with `uri` + `reader`:

```python
config = {
    "data_cfgs": {
        "dataset": "songliao_event",   # registered custom dataset (flood events)
        # --- or arbitrary local data ---
        # "dataset": "my_basin",
        # "uri": "D:/data/my_basins",   # explicit path bypasses the registry
        # "reader": "selfmade",         # reader alias (selfmade, floodevent, longterm, ...)
        "source": "local",
        "basin_ids": ["songliao_21401550"],
        "variables": ["rain", "ES", "inflow", "flood_event"],
        "warmup_length": 30,
        "is_event_data": True,
    },
    ...
}
```

Available reader aliases: `floodevent`, `selfmade`, `longterm`, `forecast`, `station`,
`tghydro`, `gages`, `grdc`, `rainfall`, `crd`, `rsvrinflow`.

Additional reader kwargs (`time_unit`, `datasource_kwargs`, ...) are forwarded to the reader constructor. Custom datasets are identified by `data_cfgs.dataset` (registry id) plus `data_cfgs.uri` for explicit paths. See `configs/example_config_selfmade.yaml` for a complete custom-data example.

### 2. Quick Start: Calibration, Evaluation, Simulation, and Visualization

**Option 1: Use Command-Line Scripts (Recommended)**

```bash
# 1. Calibration (saves config files by default)
uv run python scripts/run_xaj_calibration.py --config configs/example_config.yaml

# 2. Evaluation on test period (uses calibration_results.json from the run above)
uv run python scripts/run_xaj_evaluate.py \
    --calibration-dir results/12025000/experiment \
    --eval-period test

# 3. Simulation with custom parameters (no calibration required)
uv run python scripts/run_xaj_simulate.py \
    --config configs/example_simulate_config.yaml \
    --param-file configs/example_xaj_params.yaml \
    --plot

# 4. Visualization (time series, scatter, FDC, monthly plots -> eval_dir/figures)
uv run python scripts/visualize.py --eval-dir results/12025000/experiment/evaluation_test

# Visualize specific basins / plot types
uv run python scripts/visualize.py \
    --eval-dir results/12025000/experiment/evaluation_test \
    --basins 12025000 --plot-types timeseries scatter
```

**Configuration files:**
- `configs/example_config.yaml` - continuous time series data (e.g., CAMELS datasets)
- `configs/example_config_selfmade.yaml` - custom data / flood event datasets
- `configs/example_simulate_config.yaml` - simulation config
- `configs/example_xaj_params.yaml` - example XAJ parameter values (simulation only)

**Option 2: Use Python API (For Advanced Users)**

```python
from hydromodel import calibrate, simulate, evaluate

config = {
    "data_cfgs": {
        "dataset": "camels_us",
        "source": "local",
        "basin_ids": ["01013500"],
        "warmup_length": 365,
        "variables": ["precipitation", "potential_evapotranspiration", "streamflow"],
        "train_period": ["1985-10-01", "1995-09-30"],
        "test_period": ["2005-10-01", "2014-09-30"],
    },
    "model_cfgs": {
        "name": "xaj_mz",
        "params": {
            "source_type": "sources",
            "source_book": "HF",
            "kernel_size": 15,
        },
    },
    "training_cfgs": {
        "algorithm": "SCE_UA",
        "SCE_UA": {"rep": 1000, "ngs": 1000, "random_seed": 1234},
        "loss": "RMSE",
        "output_dir": "results",
        "experiment_name": "my_exp",
    },
    "evaluation_cfgs": {
        "metrics": ["NSE", "KGE", "RMSE", "PBIAS"],
    },
}

results = calibrate(config)                          # Calibrate
evaluate(config, param_dir="results/my_exp", eval_period="test")  # Evaluate
simulate(config)                                     # Simulate with any parameters
```

Results are saved under `{training_cfgs.output_dir}/{training_cfgs.experiment_name}/`.

## Core API

### Configuration Structure

The unified API uses a configuration dictionary with four sections (`data_cfgs`, `model_cfgs`, `training_cfgs`, `evaluation_cfgs`):

```python
config = {
    "data_cfgs": {
        "dataset": "camels_us",        # dataset id from the registry (required)
        "source": "local",             # "local" or "cloud"
        "basin_ids": ["01013500"],     # basins to calibrate
        "warmup_length": 365,          # warmup time steps
        "variables": ["precipitation", "potential_evapotranspiration", "streamflow"],
        "train_period": ["1985-10-01", "1995-09-30"],
        "test_period": ["2005-10-01", "2014-09-30"],
    },
    "model_cfgs": {
        "name": "xaj_mz",              # model name from MODEL_DICT
        "params": {                    # model-specific configuration
            "source_type": "sources",
            "source_book": "HF",
            "kernel_size": 15,
        },
        "output_variable": "qsim",     # optional
    },
    "training_cfgs": {
        "algorithm": "SCE_UA",         # SCE_UA, GA, or scipy
        # Algorithm-specific hyperparameters, keyed by algorithm name:
        "SCE_UA": {
            "rep": 1000,
            "ngs": 1000,
            "kstop": 500,
            "peps": 0.1,
            "pcento": 0.1,
            "random_seed": 1234,
        },
        "GA": {
            "pop_size": 40,
            "n_generations": 20,
            "cx_prob": 0.7,
            "mut_prob": 0.2,
            "random_seed": 1234,
        },
        "scipy": {
            "method": "SLSQP",
            "max_iterations": 500,
        },
        "loss": "RMSE",                # RMSE, NSE, KGE, LOGNSE, ...
        "output_dir": "results",
        "experiment_name": "my_exp",
        "param_range_file": None,      # optional custom parameter ranges
        "save_config": True,
    },
    "evaluation_cfgs": {
        "metrics": ["NSE", "KGE", "RMSE", "PBIAS"],
        "save_results": True,
        "plot_results": True,
    },
}
```

**Notes:**
- `data_cfgs.dataset` is **required**. For custom data, add `uri` (explicit path) and/or `reader`.
- `training_cfgs.loss` (a string) is wrapped internally into a `loss_config`; a full `loss_config` dict is also accepted.
- Optimizers always **minimize**. User objectives `NSE`, `KGE`, and `LOGNSE` are internally mapped to negated objectives (e.g., `KGE -> neg_kge`).
- An explicit `param_range_file` is validated strictly: missing files, unknown parameters, missing parameters, or invalid `[min, max]` ranges fail fast. When omitted, built-in `MODEL_PARAM_DICT` ranges are used.

### Calibration API

```python
from hydromodel import calibrate

results = calibrate(config)
```

**Saved files** (in `{output_dir}/{experiment_name}/`):
```
calibration_results.json          # Best parameters for all basins (unified format)
{basin_id}_sceua.csv              # SCE-UA iteration history (per algorithm)
{basin_id}_ga.csv                 # GA generation history
{basin_id}_scipy.csv              # scipy iteration history
calibration_config.yaml           # Configuration used (if save_config=True)
param_range.yaml                  # Resolved parameter ranges (if save_config=True)
```

**Notes:**
- `calibration_results.json` is always saved and is the primary input for evaluation.
- `best_params` stays normalized `[0,1]`; use `best_params_denormalized` for physical parameter values.
- `param_range_source` / `param_range_source_path` in the JSON record where the ranges came from.

### Evaluation API

```python
from hydromodel import evaluate

test_results = evaluate(config, param_dir="results/my_exp", eval_period="test")
train_results = evaluate(config, param_dir="results/my_exp", eval_period="train")
```

**Output:** `{param_dir}/evaluation_{period}/`
- `basins_metrics.csv` - performance metrics
- `basins_norm_params.csv` / `basins_denorm_params.csv` - calibrated parameters
- `<model>_evaluation_results.nc` - full simulation results (NetCDF)
- `evaluation_info.yaml` - evaluation metadata

**Parameter loading priority:** `calibration_results.json` first, then legacy per-algorithm CSV/txt files.

**Available metrics:** NSE, KGE, RMSE, PBIAS, FHV, FLV, FMS, and more.

### Simulation API

Simulation does **not** require prior calibration. Run a model with any parameter values:

```python
from hydromodel import simulate

config = {
    "data_cfgs": {"dataset": "camels_us", "basin_ids": ["01013500"]},
    "model_cfgs": {
        "name": "xaj",
        "params": {"source_type": "sources", "source_book": "HF"},
        "parameters": {"K": 0.75, "B": 0.25, "IM": 0.06, "UM": 18.0, ...},
    },
}

results = simulate(config)
print(results["simulation"].keys())  # model output arrays (e.g. qsim)
```

**Return format:**
- `results["simulation"]` — model output dict (keys depend on the model, usually `{"qsim": array}`)
- `results["qobs"]` — observed streamflow (if available)
- `results["parameters"]` — the parameter values used
- `results["model_name"]` / `results["basin_ids"]` — metadata

For advanced use (e.g. custom basin configs, multi-step simulation), use `UnifiedSimulator` directly:

```python
from hydromodel.trainers.unified_simulate import UnifiedSimulator

simulator = UnifiedSimulator(model_config, basin_config)
results = simulator.simulate(inputs=p_and_e, qobs=qobs, warmup_length=365)
```

**Command-line usage:**
```bash
# Custom parameters (YAML)
uv run python scripts/run_xaj_simulate.py \
    --config configs/example_simulate_config.yaml \
    --param-file configs/example_xaj_params.yaml \
    --output simulation_results.csv \
    --plot

# Calibrated parameters from SCE-UA CSV (legacy format)
uv run python scripts/run_xaj_simulate.py \
    --param-file results/my_exp/01013500_sceua.csv \
    --plot
```

## Supported Datasets

The authoritative runtime registry lives in `hydrodataset` (public datasets) and
`hydrodatasource` (custom datasets), not in hydromodel. You can extend or override
entries with a project-level `configs/datasets.yml`.

**Public datasets (27, via hydrodataset):**
- **CAMELS series (16)**: `camels_us`, `camels_aus`, `camels_br`, `camels_ch`, `camels_cl`, `camels_col`, `camels_de`, `camels_dk`, `camels_fi`, `camels_pe`, `camels_fr`, `camels_gb`, `camels_ind`, `camels_lux`, `camels_nz`, `camels_se`
- **CAMELSH series (2)**: `camelsh`, `camelsh_kr`
- **CARAVAN series (3)**: `caravan`, `caravan_dk`, `grdc_caravan`
- **LamaH series (2)**: `lamah_ce`, `lamah_ice`
- **Others (4)**: `hysets` (Canada), `bull` (France), `estreams` (Europe), `simbi` (Brazil)

**Custom datasets (via hydrodatasource):**
- `songliao_event` - Songliao flood-event dataset (registered id)
- Arbitrary local/cloud data via `uri` + `reader` aliases (`selfmade`, `floodevent`, `longterm`, `forecast`, `station`, `tghydro`, `gages`, `grdc`, `rainfall`, `crd`, `rsvrinflow`)

Note: hydromodel's `hydromodel/datasets/dataset_dict.py` is a reference mapping only;
some names it lists (e.g., `camels_deby`, `mopex`, `hype`) are not enabled in the
runtime registry and require a `configs/datasets.yml` entry or `uri` to use.

## Project Structure

```
hydromodel/
├── hydromodel/
│   ├── configs/                     # Unified config management & validation
│   │   └── config_manager.py
│   ├── models/                      # Model implementations (registered in MODEL_DICT)
│   │   ├── xaj.py, xaj_slw.py, semi_xaj.py
│   │   ├── gr1a.py ... gr6j.py      # GR family
│   │   ├── hymod.py, dhf.py, unit_hydrograph.py
│   │   ├── model_dict.py            # MODEL_DICT / LOSS_DICT registry
│   │   └── model_config.py          # Parameter contracts & validation
│   ├── trainers/
│   │   ├── unified_calibrate.py     # Calibration API (SCE-UA / GA / scipy)
│   │   ├── unified_evaluate.py      # Evaluation API
│   │   └── unified_simulate.py      # Simulation API
│   ├── datasets/
│   │   ├── unified_data_loader.py   # Unified data loading (delegates to open_dataset)
│   │   ├── dataset_dict.py          # Reference dataset mapping
│   │   └── data_visualize.py        # Plotting functions
│   └── __init__.py                  # Lazy top-level API (list_models, describe_model, ...)
├── scripts/                         # CLI scripts
│   ├── run_xaj_calibration.py
│   ├── run_xaj_evaluate.py
│   ├── run_xaj_simulate.py
│   └── visualize.py
├── configs/                         # Example YAML configs
├── test/                            # Tests (run with: pytest test/)
└── docs/                            # Documentation
```

## Documentation

- **Quick Start**: [docs/quickstart.md](docs/quickstart.md)
- **Usage Guide**: [docs/usage.md](docs/usage.md)
- **Data Guide**: [docs/data_guide.md](docs/data_guide.md)
- **Data Path Resolution (ADR 0001)**: [docs/adr/0001-unified-data-path-resolution.md](docs/adr/0001-unified-data-path-resolution.md)
- **API Reference**: https://OuyangWenyu.github.io/hydromodel

## References

- Allen, R.G., L. Pereira, D. Raes, and M. Smith, 1998. Crop Evapotranspiration, Food and Agriculture Organization of
  the United Nations, Rome, Italy. FAO publication 56. ISBN 92-5-104219-5. 290p.
- Duan, Q., Sorooshian, S., and Gupta, V. (1992), Effective and efficient global optimization for conceptual
  rainfall-runoff models, Water Resour. Res., 28( 4), 1015– 1031, doi:10.1029/91WR02985.
- François-Michel De Rainville, Félix-Antoine Fortin, Marc-André Gardner, Marc Parizeau, and Christian Gagné. 2012.
  DEAP: a python framework for evolutionary algorithms. In Proceedings of the 14th annual conference companion on
  Genetic and evolutionary computation (GECCO '12). Association for Computing Machinery, New York, NY, USA, 85–92.
  DOI:https://doi.org/10.1145/2330784.2330799
- Houska T, Kraft P, Chamorro-Chavez A, Breuer L (2015) SPOTting Model Parameters Using a Ready-Made Python Package.
  PLoS ONE 10(12): e0145180. https://doi.org/10.1371/journal.pone.0145180
- Mizukami, N., Clark, M. P., Sampson, K., Nijssen, B., Mao, Y., McMillan, H., Viger, R. J., Markstrom, S. L., Hay, L.
  E., Woods, R., Arnold, J. R., and Brekke, L. D.: mizuRoute version 1: a river network routing tool for a continental
  domain water resources applications, Geosci. Model Dev., 9, 2223–2238, https://doi.org/10.5194/gmd-9-2223-2016, 2016.
- Zhao, R.J., Zhuang, Y. L., Fang, L. R., Liu, X. R., Zhang, Q. S. (ed) (1980) The Xinanjiang model, Hydrological
  Forecasting Proc., Oxford Symp., IAHS Publication, Wallingford, U.K.
- Zhao, R.J., 1992. The xinanjiang model applied in China. J Hydrol 135 (1–4), 371–381.

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
pytest test/
```

## License

GNU General Public License v3.0 - see [LICENSE](LICENSE) file.

## Contact

- **Author**: Wenyu Ouyang
- **Email**: wenyuouyang@outlook.com
- **GitHub**: https://github.com/OuyangWenyu/hydromodel
- **Issues**: https://github.com/OuyangWenyu/hydromodel/issues