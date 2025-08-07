# HydroModel Configuration Examples

This directory contains example configuration files for different hydrological models and calibration scenarios.

## Available Examples

### Unit Hydrograph Models

1. **`unit_hydrograph_example.yaml`** - Basic unit hydrograph with scipy optimization
2. **`categorized_uh_example.yaml`** - Categorized unit hydrograph with genetic algorithm

### Traditional Models

3. **`xaj_sceua_example.yaml`** - XAJ model with SCE-UA algorithm
4. **`xaj_ga_example.yaml`** - XAJ model with genetic algorithm
5. **`gr4j_example.yaml`** - GR4J model with SCE-UA algorithm

## Quick Start

### Using Configuration Files

```bash
# Unit hydrograph calibration
python run_unit_hydrograph_with_config.py --config configs/examples/unit_hydrograph_example.yaml

# Categorized unit hydrograph calibration
python run_categorized_uh_with_config.py --config configs/examples/categorized_uh_example.yaml

# General calibration (any model)
python run_calibration_with_config.py --config configs/examples/xaj_sceua_example.yaml
```

### Using Quick Setup Mode

```bash
# Unit hydrograph quick setup
python run_unit_hydrograph_with_config.py --quick-setup --station-id songliao_21401550

# Categorized unit hydrograph quick setup
python run_categorized_uh_with_config.py --quick-setup --algorithm genetic_algorithm

# General calibration quick setup (traditional models)
python run_calibration_with_config.py --quick-setup --model xaj --algorithm SCE_UA
```

### Creating Your Own Configuration

```bash
# Create templates
python run_unit_hydrograph_with_config.py --create-template my_uh_config.yaml
python run_categorized_uh_with_config.py --create-template my_categorized_config.yaml
python run_calibration_with_config.py --create-default-config my_general_config.yaml
```

## Configuration Structure

All configurations follow the same four-part structure:

```yaml
data_cfgs:          # Data settings
  data_type: "flood_events"
  data_dir: "path/to/data"
  basin_ids: ["basin_id"]
  warmup_length: 480

model_cfgs:         # Model settings
  model_name: "unit_hydrograph"
  model_params:
    n_uh: 24
    # ... other model parameters

training_cfgs:      # Training/calibration settings
  algorithm_name: "scipy_minimize"
  algorithm_params:
    method: "SLSQP"
    # ... other algorithm parameters

evaluation_cfgs:    # Evaluation settings
  loss_type: "event_based"
  objective_function: "RMSE"
  metrics: ["RMSE", "NSE"]
```

## Parameter Override

You can override any configuration parameter from the command line:

```bash
# Override model parameters
--override model_cfgs.model_params.n_uh=32

# Override algorithm
--override training_cfgs.algorithm_name=genetic_algorithm

# Override multiple parameters
--override model_cfgs.model_params.n_uh=32 --override training_cfgs.algorithm_name=SCE_UA
```

## Supported Models and Algorithms

### Models
- `unit_hydrograph` - Single unit hydrograph
- `categorized_unit_hydrograph` - Categorized unit hydrographs by flood magnitude
- `xaj`, `xaj_mz` - XinAnJiang hydrological model variants
- `gr4j`, `gr1a`, `gr2m`, `gr3j`, `gr5j`, `gr6j` - GR series models
- `hymod` - HyMod conceptual model

### Algorithms
- `scipy_minimize` - SciPy optimization methods
- `SCE_UA` - Shuffled Complex Evolution via spotpy
- `genetic_algorithm` - Genetic algorithm via DEAP (requires `pip install deap`)

## Need Help?

Run any script without arguments to see available options:

```bash
python run_unit_hydrograph_with_config.py
python run_categorized_uh_with_config.py
python run_calibration_with_config.py
```

Or use the `--help` flag for detailed information:

```bash
python run_unit_hydrograph_with_config.py --help
```