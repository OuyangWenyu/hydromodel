## Flood Event Data

### Overview

hydromodel provides specialized support for **flood event datasets**, where data consists of discrete flood events rather than continuous time series. This is particularly useful for:

- Event-based model calibration and validation
- Flood forecasting applications
- Multi-peak flood event analysis
- Studies focusing on extreme hydrological conditions

**Key Features:**

- Multi-basin support with correct time alignment (no padding issues)
- Automatic event grouping for multi-peak floods
- Event-specific visualizations showing only flood periods
- Proper handling of gaps and warmup periods
- Backward compatible with all existing hydromodel APIs

### Data Format

Flood event data uses the `floodevent` reader from `hydrodatasource` (registered example: `songliao_event`):

```python
data_config = {
    "dataset": "songliao_event",  # registered flood-event dataset
    "basin_ids": ["basin_001"],
    # ... other parameters
}
```

**Required Data Structure:**

```
my_flood_data/
├── attributes/
│   └── attributes.csv              # Basin metadata
├── timeseries/
│   ├── 1h/                         # Hourly flood event data
│   │   ├── basin_001.csv          # Time series with marker column
│   │   ├── basin_002.csv
│   │   └── ...
│   └── 1h_units_info.json          # Variable units
```

**Time series CSV format:**

```csv
time,prcp,PET,streamflow,marker,event_id
2020-08-01 00:00,0.5,0.1,10.2,1,25
2020-08-01 01:00,1.2,0.1,12.5,1,25
...
2020-08-05 23:00,0.2,0.05,8.1,1,25
2020-08-06 00:00,0.0,0.0,0.0,0,0
...
2020-09-01 00:00,0.8,0.12,15.3,1,26
```

**Special columns:**

- `marker`:
  - `1` = flood event data (valid)
  - `0` = gap between events (invalid, not used in calibration/evaluation)
  - `NaN` = warmup period (used for model spinup, excluded from metrics)
- `event_id`: Integer identifier grouping related flood peaks together

### Configuration

**YAML Configuration Example:**

```yaml
# configs/songliao_event_3h.yaml
data_cfgs:
  dataset: songliao_event
  basin_ids: ["songliao_21401550", "songliao_21100150"]
  train_period: ["2019-01-01", "2020-12-31"]  # Filter events by date range
  test_period: ["2021-01-01", "2022-12-31"]
  warmup_length: 15  # Hours for event-based data
  time_unit: ["1h"]  # Hourly resolution
  variables: ["prcp", "PET", "streamflow"]

  # Optional: Filter by event_id
  event_ids: [25, 26, 27]  # Only use these events

model_cfgs:
  name: xaj_mz
  params:
    source_type: "sources"
    source_book: "HF"
    kernel_size: 15

training_cfgs:
  algorithm: SCE_UA
  SCE_UA:
    rep: 5000
    ngs: 1000
    random_seed: 1234
  loss_config:
    type: "time_series"
    obj_func: "RMSE"
  output_dir: "results"
  experiment_name: "flood_event_calibration"

evaluation_cfgs:
  metrics: ["NSE", "KGE", "RMSE", "PBIAS"]
```

### Command-Line Scripts

#### Quick Start with Default Configuration

The `run_event_calibration.py` script provides sensible defaults for flood event calibration:

```bash
# Use default configuration
python scripts/run_event_calibration.py --default

# Verify configuration without running
python scripts/run_event_calibration.py --default --dry-run

# Customize basin IDs and algorithm
python scripts/run_event_calibration.py --default \
    --basin-ids songliao_21401550 songliao_21100150 \
    --algorithm GA \
    --output-dir results/flood_ga
```

**Default configuration:**
- Data source: `floodevent`
- Dataset: `songliao_flood_events`
- Basin: `songliao_21401550`
- Algorithm: `SCE_UA`
- Warmup: 15 hours
- Model: `xaj_mz`

#### Using Configuration Files

```bash
# Calibration
python scripts/run_event_calibration.py --config configs/songliao_event_3h.yaml

# Or use standard calibration script (works identically)
python scripts/run_xaj_calibration.py --config configs/songliao_event_3h.yaml

# Evaluation (same as continuous data)
python scripts/run_xaj_evaluate.py \
    --calibration-dir results/flood_event_calibration \
    --eval-period test

# Visualization (event-specific plots)
python scripts/visualize.py \
    --eval-dir results/flood_event_calibration/evaluation_test
```

### Python API Usage

Flood event data works seamlessly with the unified API:

```python
from hydromodel.trainers.unified_calibrate import calibrate
from hydromodel.trainers.unified_evaluate import evaluate

# Configuration
config = {
    "data_cfgs": {
        "dataset": "songliao_event",
        "basin_ids": ["songliao_21401550"],
        "train_period": ["2019-01-01", "2020-12-31"],
        "test_period": ["2021-01-01", "2022-12-31"],
        "warmup_length": 15,
        "time_unit": ["1h"],
        "variables": ["prcp", "PET", "streamflow"],
    },
    "model_cfgs": {
        "name": "xaj_mz",
    },
    "training_cfgs": {
        "algorithm": "SCE_UA",
        "SCE_UA": {"rep": 5000, "ngs": 1000},
        "loss_config": {"type": "time_series", "obj_func": "RMSE"},
        "output_dir": "results",
        "experiment_name": "flood_event_exp",
    },
    "evaluation_cfgs": {
        "metrics": ["NSE", "KGE", "RMSE"],
    },
}

# Calibration
results = calibrate(config)

# Evaluation
metrics = evaluate(config, param_dir="results/flood_event_exp", eval_period="test")
```

### Multi-Basin Time Alignment

**Problem**: Different basins may have flood events at different times, creating challenges for multi-basin calibration.

**Solution**: hydromodel automatically handles time alignment:

```python
# Example: Two basins with different event times
# Basin A: Events on 2020-08-01 to 2020-08-05 (Event 25)
# Basin B: Events on 2020-09-01 to 2020-09-05 (Event 26)

# hydromodel creates a unified time array:
# - Merges all unique timestamps from both basins
# - Maps each basin's data to correct time positions
# - Fills gaps with marker=0 (invalid data, excluded from loss)
# - Preserves event_id for visualization

# No manual intervention required!
```

**Internal Workflow:**

1. Load each basin's event data separately
2. Extract unique timestamps across all basins
3. Create unified sorted time array
4. Remap each basin's data to unified time indices
5. Mark missing periods with `marker=0`, `event_id=0`
6. Calibration/evaluation uses only `marker=1` data

**NetCDF Output Structure:**

```python
import xarray as xr

ds = xr.open_dataset("results/flood_event_exp/evaluation_test/test_evaluation_results.nc")

print(ds.dims)
# {'basin': 2, 'time': 1500}  # Unified time array

print(ds['event_id'])
# Shows event_id for each time step and basin
# event_id=0 indicates gaps (not used in metrics)

print(ds['qsim'])
# Simulated streamflow [time, basin]
# Zero values where marker=0 (gaps)
```

### Event-Specific Visualization

Flood event plots automatically highlight event periods:

```python
from hydromodel.datasets.data_visualize import visualize_evaluation

# Visualize flood events
visualize_evaluation(
    eval_dir="results/flood_event_calibration/evaluation_test",
    basins=["songliao_21401550"]
)
```

**Plot Features:**

- Only shows periods where `marker=1` (flood events)
- Gaps between events are excluded from visualization
- Event IDs displayed in plot titles
- Precipitation and streamflow on separate panels
- Performance metrics (NSE, RMSE, PBIAS) shown in text box

### Multi-Peak Event Grouping

Use `event_id` to group related flood peaks:

```python
# Example: Typhoon with multiple peaks
# Event 25:
#   - Peak 1: 2020-08-01 to 2020-08-03
#   - Gap:    2020-08-04 to 2020-08-05 (marker=0)
#   - Peak 2: 2020-08-06 to 2020-08-08
#   - All marked as event_id=25

# During calibration:
# - Model sees both peaks with correct warmup
# - Gap period (marker=0) excluded from loss calculation
# - event_id preserved for analysis

# During visualization:
# - Both peaks plotted together as "Event 25"
# - Gap period shown but marked differently
```

### Filtering Events

**By Time Range:**

```python
config["data_cfgs"]["train_period"] = ["2019-07-01", "2020-09-30"]
# Only loads events within this date range
```

**By Event ID:**

```python
config["data_cfgs"]["event_ids"] = [25, 26, 27]
# Only loads these specific events
```

**By Basin:**

```python
config["data_cfgs"]["basin_ids"] = ["songliao_21401550"]
# Single basin calibration
```

### Comparison with Continuous Data

| Feature | Continuous Data | Flood Event Data |
|---------|----------------|------------------|
| Data source | CAMELS datasets | songliao_event (flood events) |
| Time structure | Continuous time series | Discrete events with gaps |
| Warmup | Days (typically 365) | Hours (typically 15) |
| marker column | Not used | Required (1=valid, 0=gap) |
| event_id column | Not used | Required for grouping |
| API usage | Identical | Identical |
| Output format | NetCDF with continuous time | NetCDF with unified time array |

### Best Practices

**1. Adequate Warmup:**

```python
# For hourly data, use at least 15 hours warmup
config["data_cfgs"]["warmup_length"] = 15

# For event data with long recessions, increase warmup
config["data_cfgs"]["warmup_length"] = 30
```

**2. Event Selection:**

```python
# Start with well-observed, significant events
config["data_cfgs"]["event_ids"] = [25, 26, 27]  # Major floods only

# Avoid events with missing data or measurement errors
```

**3. Multi-Basin Calibration:**

```python
# Ensure basins have overlapping events for better parameter transfer
basin_ids = ["basin_A", "basin_B", "basin_C"]

# Check event coverage before calibration
from hydrodatasource.configs.data_resolver import open_dataset
ds = open_dataset("floodevent", uri=...)
for basin in basin_ids:
    events = ds.get_events(basin)
    print(f"{basin}: {len(events)} events")
```

**4. Validation:**

```python
# Always validate on independent events
config["data_cfgs"]["train_period"] = ["2018-01-01", "2020-12-31"]
config["data_cfgs"]["test_period"] = ["2021-01-01", "2022-12-31"]

# Or use event-based split
train_events = [20, 21, 22, 23, 24]
test_events = [25, 26, 27]
```

### Troubleshooting

**Issue 1: Time Misalignment in Multi-Basin Results**

**Symptoms:** NetCDF shows incorrect time ranges for events (e.g., Event 26 spanning years instead of days)

**Solution:** This has been fixed in v0.3.0. Ensure you're using the latest version.

**Issue 2: AttributeError in Calibration**

**Symptoms:** `'NoneType' object has no attribute 'shape'`

**Solution:** This has been fixed in v0.3.0. The issue occurred when accessing basin data in separate mode.

**Issue 3: Missing Event Data**

**Symptoms:** Fewer events loaded than expected

**Check:**
```python
# Verify event filtering
print(f"Time range: {config['data_cfgs']['train_period']}")
print(f"Event IDs: {config['data_cfgs'].get('event_ids', 'All')}")

# Check raw data
from hydrodatasource.configs.data_resolver import open_dataset
ds = open_dataset("floodevent", uri=..., time_unit=["1h"])
events = ds.get_events(basin_id)
print(f"Available events: {len(events)}")
```

**Issue 4: Poor Calibration Performance**

**Possible causes:**
- Insufficient warmup period
- Events too short for model spinup
- Mixed event types (e.g., snowmelt and rainfall floods)

**Solutions:**
- Increase warmup: `warmup_length = 30`
- Filter events by type or magnitude
- Check data quality (missing values, outliers)

### Example Workflow

Complete workflow for flood event calibration:

```python
from hydromodel.trainers.unified_calibrate import calibrate
from hydromodel.trainers.unified_evaluate import evaluate
from hydromodel.datasets.data_visualize import visualize_evaluation

# 1. Configuration
config = {
    "data_cfgs": {
        "dataset": "songliao_event",
        "basin_ids": ["songliao_21401550", "songliao_21100150"],
        "train_period": ["2019-01-01", "2020-12-31"],
        "test_period": ["2021-01-01", "2022-12-31"],
        "warmup_length": 15,
        "time_unit": ["1h"],
        "variables": ["prcp", "PET", "streamflow"],
    },
    "model_cfgs": {"name": "xaj_mz"},
    "training_cfgs": {
        "algorithm": "SCE_UA",
        "SCE_UA": {"rep": 5000, "ngs": 1000},
        "loss_config": {"type": "time_series", "obj_func": "RMSE"},
        "output_dir": "results",
        "experiment_name": "flood_2basin",
    },
    "evaluation_cfgs": {"metrics": ["NSE", "KGE", "RMSE", "PBIAS"]},
}

# 2. Calibration
print("Starting calibration...")
results = calibrate(config)
print(f"Calibration completed: {results}")

# 3. Evaluation on test period
print("Evaluating on test period...")
test_metrics = evaluate(
    config,
    param_dir="results/flood_2basin",
    eval_period="test"
)
print(f"Test NSE: {test_metrics}")

# 4. Visualization
print("Creating visualizations...")
visualize_evaluation(
    eval_dir="results/flood_2basin/evaluation_test",
    output_dir="figures/flood_events"
)
print("Done!")
```

### Related Documentation

- **Data Preparation**: [Data Guide](data_guide.md#custom-data) - How to prepare flood event data
- **API Reference**: [Unified API](#unified-calibration-api) - Core API documentation
- **Configuration**: [Configuration System](#configuration-system) - Full configuration options
- **hydrodatasource**: [GitHub](https://github.com/OuyangWenyu/hydrodatasource) - Data source package

---

