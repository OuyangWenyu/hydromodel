# Data Preparation Guide

This guide explains how to prepare and use data with `hydromodel`, covering both public CAMELS datasets and custom data.

## Overview

`hydromodel` supports two main data sources:

1. **Public CAMELS Datasets** - Using [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) package
   - 11 global CAMELS variants (US, GB, AUS, BR, CL, etc.)
   - Automatic download and caching
   - Standardized format and quality-controlled

2. **Custom Data** - Using [hydrodatasource](https://github.com/OuyangWenyu/hydrodatasource) package
   - Your own basin data
   - Flexible data organization
   - Integration with cloud storage

---

## Option 1: Using CAMELS Datasets (hydrodataset)

### Step 1: Install hydrodataset

```bash
pip install hydrodataset
```

### Step 2: Configure Data Path (Optional)

`hydromodel` automatically uses default paths, but you can customize:

**Default paths:**
- Windows: `C:\Users\YourUsername\hydromodel_data\`
- macOS/Linux: `~/hydromodel_data/`

**To customize, create `~/hydro_setting.yml`:**

```yaml
local_data_path:
  root: 'D:/data'
  datasets-origin: 'D:/data'  # CAMELS datasets location
```

**Important**: Provide only the `datasets-origin` directory. The system automatically appends the dataset name (e.g., `CAMELS_US`, `CAMELS_GB`).

Example: If your data is in `D:/data/CAMELS_US/`, set `datasets-origin: 'D:/data'`.

### Step 3: Download Data

The data downloads automatically on first use:

```python
from hydrodataset.camels_us import CamelsUs
from hydrodataset import SETTING

# Initialize dataset (auto-downloads if not present)
data_path = SETTING["local_data_path"]["datasets-origin"]
ds = CamelsUs(data_path, download=True)

# Get available basins
basin_ids = ds.read_object_ids()
print(f"Downloaded {len(basin_ids)} basins")
```

**Note**: First download may take 30-120 minutes depending on dataset size. CAMELS-US is ~70GB.

### Step 4: Use with hydromodel

```python
from hydromodel.trainers.unified_calibrate import calibrate

config = {
    "data_cfgs": {
        "data_source_type": "camels_us",  # Dataset name
        "basin_ids": ["01013500", "01022500"],
        "train_period": ["1990-10-01", "2000-09-30"],
        "test_period": ["2000-10-01", "2010-09-30"],
        "warmup_length": 365,
        "variables": ["precipitation", "potential_evapotranspiration", "streamflow"]
    },
    # ... other configs
}

results = calibrate(config)
```

### Available CAMELS Datasets

| Dataset | Region | Basins | Package Name |
|---------|--------|--------|--------------|
| CAMELS-US | United States | 671 | `camels_us` |
| CAMELS-GB | Great Britain | 671 | `camels_gb` |
| CAMELS-AUS | Australia | 222 | `camels_aus` |
| CAMELS-BR | Brazil | 897 | `camels_br` |
| CAMELS-CL | Chile | 516 | `camels_cl` |
| CAMELS-CH | Switzerland | 331 | `camels_ch` |
| CAMELS-DE | Germany | 1555 | `camels_de` |
| CAMELS-DK | Denmark | 304 | `camels_dk` |
| CAMELS-FR | France | 654 | `camels_fr` |
| CAMELS-NZ | New Zealand | 70 | `camels_nz` |
| CAMELS-SE | Sweden | 54 | `camels_se` |

**Usage example:**

```python
# Use different datasets by changing data_source_type
config["data_cfgs"]["data_source_type"] = "camels_gb"
config["data_cfgs"]["basin_ids"] = ["28015"]  # GB basin ID
```

### CAMELS Data Structure

CAMELS datasets provide standardized variables:

**Time Series Variables:**
- `precipitation` (mm/day or mm/hour)
- `potential_evapotranspiration` (mm/day)
- `streamflow` (mm/day or m³/s)
- `temperature` (°C)
- And more depending on dataset

**Basin Attributes:**
- `area` (km²)
- `elevation` (m)
- `latitude`, `longitude`
- Climate, soil, vegetation attributes

For detailed documentation, see:
- [hydrodataset GitHub](https://github.com/OuyangWenyu/hydrodataset)
- [hydrodataset documentation](https://hydrodataset.readthedocs.io/)

---

## Option 2: Using Custom Data (hydrodatasource)

### Step 1: Install hydrodatasource

```bash
pip install hydrodatasource
```

### Step 2: Organize Your Data

Create a directory with this structure:

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
└── shapes/                         # Basin boundaries (optional)
    └── basins.shp
```

### Step 3: Prepare Required Files

#### 3.1 attributes.csv

Minimum required columns: `basin_id` and `area` (km²)

```csv
basin_id,area,lat,lon,elevation
basin_001,1250.5,30.5,105.2,850
basin_002,856.3,31.2,106.1,920
```

**Important:**
- `basin_id`: String identifier (matches filename)
- `area`: Basin area in km²
- Other columns optional but recommended

#### 3.2 basin_XXX.csv (Time Series)

Required column: `time`. Other columns are your variables.

```csv
time,prcp,PET,streamflow
1990-01-01,5.2,2.1,45.3
1990-01-02,0.0,2.3,42.1
1990-01-03,12.5,1.8,58.7
```

**Important:**
- `time` format: `YYYY-MM-DD` (for daily data)
- Variable names: Lowercase, underscores for multi-word
- Missing values: Use empty cells or `NaN` (not `-9999` or `0`)
- No duplicate time stamps

#### 3.3 1D_units_info.json (Units Definition)

Define physical units for all variables:

```json
{
  "prcp": "mm/day",
  "PET": "mm/day",
  "streamflow": "m^3/s",
  "temp": "degC"
}
```

**Common units:**
- Precipitation/ET: `mm/day` or `mm/hour`
- Streamflow: `m^3/s` or `mm/day`
- Temperature: `degC` or `K`
- Area: `km^2`

### Step 4: Verify Data Structure

```python
from hydrodatasource.reader.data_source import SelfMadeHydroDataset

# Initialize dataset
dataset = SelfMadeHydroDataset(
    data_path="D:/my_basin_data",
    time_unit="1D"
)

# Check basins
basin_ids = dataset.read_object_ids()
print(f"Found {len(basin_ids)} basins: {basin_ids}")

# Check time series
data = dataset.read_timeseries(
    gage_id_lst=["basin_001"],
    t_range=["1990-01-01", "2000-12-31"],
    var_lst=["prcp", "PET", "streamflow"]
)

print(f"Data shape: {data['1D'].shape}")  # [n_basins, n_time, n_vars]
```

### Step 5: Use with hydromodel

```python
from hydromodel.trainers.unified_calibrate import calibrate

config = {
    "data_cfgs": {
        "data_source_type": "selfmadehydrodataset",  # Use custom data
        "data_source_path": "D:/my_basin_data",      # Your data path
        "basin_ids": ["basin_001", "basin_002"],
        "train_period": ["1990-01-01", "2000-12-31"],
        "test_period": ["2001-01-01", "2010-12-31"],
        "warmup_length": 365,
    },
    "model_cfgs": {
        "model_name": "xaj_mz",
    },
    "training_cfgs": {
        "algorithm": "SCE_UA",
        "loss_func": "RMSE",
        "output_dir": "results",
        "experiment_name": "my_basins",
        "rep": 10000,
        "ngs": 100,
    },
    "evaluation_cfgs": {
        "metrics": ["NSE", "KGE", "RMSE"],
    },
}

results = calibrate(config)
```

---

## Option 3: Using Flood Event Data (hydrodatasource)

### Overview

Flood event data is designed for **event-based hydrological modeling** where you focus on specific flood episodes rather than continuous time series. This is particularly useful for:

- Flood forecasting and warning systems
- Peak flow estimation
- Event-based rainfall-runoff analysis
- Unit hydrograph calibration

### Key Differences from Continuous Data

| Feature | Continuous Data | Flood Event Data |
|---------|----------------|------------------|
| **Data Structure** | Complete time series | Individual flood events with gaps |
| **Input Features** | 2D: [prcp, PET] | 4D: [prcp, PET, marker, event_id] |
| **Warmup Handling** | Removed after simulation | Included in each event (NaN markers) |
| **Time Coverage** | Full period | Only flood periods + warmup |
| **Use Case** | Long-term water balance | Flood peak prediction |

### Data Structure and Components

#### 1. Event Format

Each flood event contains:

```python
event = {
    "rain": np.array([...]),           # Precipitation (with NaN in warmup)
    "ES": np.array([...]),             # Evapotranspiration
    "inflow": np.array([...]),         # Streamflow (with NaN in warmup)
    "flood_event_markers": np.array([...]),  # NaN=warmup, 1=flood
    "event_id": 1,                     # Event identifier
    "time": np.array([...])            # Datetime array
}
```

#### 2. Three Key Periods in Each Event

```
[Warmup Period] → [Flood Period] → [GAP]
  marker=NaN        marker=1         marker=0
```

**Warmup Period** (e.g., 30 days before flood):
- Contains NaN values in observations
- Used to initialize model states
- Length specified by `warmup_length` in config
- Extracted from real data before the flood event

**Flood Period** (actual event):
- Contains valid observations (marker=1)
- The period of interest for simulation
- Used for model calibration and evaluation

**GAP Period** (between events):
- Artificial buffer (10 time steps by default)
- Designed for visualization clarity
- **NOT used in simulation** (marker=0, ignored by model)
- Created with: precipitation=0, ET=0.27, flow=0

#### 3. How Events are Concatenated

When loading multiple events, they are combined as:

```
Event1: [warmup-NaN][flood-1][GAP-0]
Event2: [warmup-NaN][flood-1][GAP-0]
Event3: [warmup-NaN][flood-1]
```

**Important**: The final data structure includes GAP periods, but these are **automatically skipped** during simulation based on the marker values.

### Simulation Behavior

#### Event Detection

During simulation (`unified_simulate.py`), the system:

1. Reads the `flood_event_markers` array (3rd feature)
2. Uses `find_flood_event_segments_as_tuples()` to identify events
3. Only processes segments where `marker > 0` (i.e., marker=1)
4. **Skips GAP periods** (marker=0) completely

```python
# Simulation automatically identifies event segments
flood_event_array = inputs[:, basin_idx, 2]  # marker column
event_segments = find_flood_event_segments_as_tuples(
    flood_event_array, warmup_length
)

# Each event is simulated independently
for start, end, orig_start, orig_end in event_segments:
    event_inputs = inputs[start:end+1, :, :3]  # Extract event data
    result = model(event_inputs, ...)  # Simulate this event
```

#### Why GAP Doesn't Affect Results

The GAP period is present in the loaded data but **does not participate in simulation**:

- ✅ GAP helps separate events visually in plots
- ✅ GAP provides clear boundaries between independent floods
- ❌ GAP data is never fed to the hydrological model
- ❌ GAP does not contribute to loss calculation

This design ensures that:
1. Each flood event is simulated independently
2. Model states are reset via warmup for each event
3. Events don't interfere with each other

### Configuration Example

```yaml
data:
  dataset: "floodevent"
  dataset_name: "my_flood_events"
  data_source_path: "D:/flood_data"
  is_event_data: true
  time_unit: ["1D"]

  # Datasource parameters
  datasource_kwargs:
    warmup_length: 30      # Days before flood for warmup
    offset_to_utc: false   # Time zone handling
    version: null

  basin_ids: ["basin_001"]
  train_period: ["2000-01-01", "2020-12-31"]
  test_period: ["2020-01-01", "2023-12-31"]

  variables: ["rain", "ES", "inflow", "flood_event"]
  warmup_length: 30

model:
  name: "xaj"
  params:
    source_type: "sources"
    source_book: "HF"
    kernel_size: 15
```

### Data File Structure

```
my_flood_events/
├── attributes/
│   └── attributes.csv
├── timeseries/
│   ├── 1D/
│   │   ├── basin_001.csv      # Must include 'flood_event' column
│   │   └── basin_002.csv
│   └── 1D_units_info.json
└── shapes/
    └── basins.shp
```

#### basin_001.csv Format

```csv
time,rain,ES,inflow,flood_event
2020-06-01,0.0,3.2,5.1,0
2020-06-02,2.5,3.5,5.8,0
2020-07-01,5.0,4.0,10.2,1    # Flood event starts
2020-07-02,12.5,3.8,25.5,1
2020-07-03,8.0,3.5,30.1,1
2020-07-04,0.0,3.2,20.5,1
2020-07-05,0.0,3.0,12.0,0    # Event ends
```

**Key Points**:
- `flood_event` column: 0=no flood, 1=flood period
- Continuous time series with all periods marked
- System automatically extracts events with warmup

### Best Practices

1. **Warmup Length**:
   - Typical: 30 days for daily data
   - Should be long enough to initialize soil moisture states
   - Too short: poor initial conditions
   - Too long: data availability issues

2. **Event Selection**:
   - Focus on significant floods (peak > threshold)
   - Include complete rising and recession limbs
   - Ensure warmup period has valid data

3. **Data Quality**:
   - Check for missing data in warmup periods
   - Verify flood markers are correctly assigned
   - Ensure precipitation and flow are synchronized

4. **Marker Assignment**:
   ```python
   # Example: Mark floods based on threshold
   threshold = flow.quantile(0.95)
   flood_event = (flow > threshold).astype(int)
   ```

### Common Issues

**1. "Warmup period contains all NaN"**
- Ensure data exists before each flood event
- Check `warmup_length` is not too long
- Verify CSV has continuous time series

**2. "No flood events found"**
- Check `flood_event` column exists
- Verify flood markers are 1 (not True or other values)
- Ensure train_period covers some flood events

**3. "Simulation results are all zeros"**
- Check if events are detected: markers should be 1 for flood periods
- Verify warmup_length matches the actual warmup in data
- Ensure model parameters are physically reasonable

### Advanced: Manual Event Creation

For custom event extraction:

```python
from hydroutils import hydro_event

# Extract events from continuous data
events = hydro_event.extract_flood_events(
    df=continuous_data,
    warmup_length=30,
    flood_event_col="flood_event",
    time_col="time"
)

# Each event includes warmup automatically
for event in events:
    print(f"Event: {event['event_name']}")
    print(f"  Total length: {len(event['data'])}")
    print(f"  Warmup markers: {event['data']['flood_event'].isna().sum()}")
    print(f"  Flood markers: {(event['data']['flood_event']==1).sum()}")
```

---

## Data Requirements for XAJ Model

### Required Variables

| Variable | Description | Unit | Typical Source |
|----------|-------------|------|----------------|
| `prcp` | Precipitation | mm/day | Rain gauge, gridded data (CHIRPS, ERA5) |
| `PET` | Potential Evapotranspiration | mm/day | Penman, Priestley-Taylor, or reanalysis |
| `streamflow` | Observed streamflow | m³/s | Stream gauge |
| `area` | Basin area | km² | GIS analysis |

### Optional Variables

| Variable | Description | Unit | Usage |
|----------|-------------|------|-------|
| `temp` | Temperature | °C | Snow module (if enabled) |
| `elevation` | Basin elevation | m | PET estimation |
| `lat`, `lon` | Coordinates | degrees | Spatial analysis |

### Data Quality Guidelines

1. **Time Resolution**: Daily (1D) is standard for XAJ model

2. **Data Completeness**:
   - Training period: ≥5 years continuous data
   - Warmup period: ≥1 year before training
   - Missing data: <5% acceptable, continuous gaps <7 days

3. **Physical Consistency**:
   - Precipitation ≥ 0
   - Streamflow ≥ 0
   - PET ≥ 0
   - Check water balance: P ≈ Q + ET (within 20%)

4. **Unit Consistency**:
   - Ensure all units match `units_info.json`
   - Use consistent time stamps (no daylight saving shifts)

---

## Advanced Features

### NetCDF Caching (For Large Datasets)

Convert CSV to NetCDF for 10x faster access:

```python
from hydrodatasource.reader.data_source import SelfMadeHydroDataset

dataset = SelfMadeHydroDataset(
    data_path="D:/my_basin_data",
    time_unit="1D"
)

# Cache all data as NetCDF (one-time operation)
dataset.cache_xrdataset(
    gage_id_lst=basin_ids,
    t_range=["1990-01-01", "2010-12-31"],
    var_lst=["prcp", "PET", "streamflow"]
)

# Now access is much faster
data_xr = dataset.read_ts_xrdataset(
    gage_id_lst=["basin_001"],
    t_range=["1990-01-01", "2000-12-31"],
    var_lst=["prcp", "PET", "streamflow"]
)
```

### Multi-Scale Time Series

Support different time scales in one dataset:

```
timeseries/
├── 1h/                     # Hourly data
│   ├── basin_001.csv
│   └── 1h_units_info.json
├── 1D/                     # Daily data (most common)
│   ├── basin_001.csv
│   └── 1D_units_info.json
└── 8D/                     # 8-day data (e.g., MODIS)
    ├── basin_001.csv
    └── 8D_units_info.json
```

Specify in config:
```python
dataset = SelfMadeHydroDataset(
    data_path="D:/my_basin_data",
    time_unit="1h"  # or "1D", "8D"
)
```

### Cloud Storage (MinIO/S3)

For large datasets in the cloud:

```python
from hydrodatasource.reader.data_source import SelfMadeHydroDataset

dataset = SelfMadeHydroDataset(
    data_path="s3://my-bucket/basin-data",
    time_unit="1D",
    minio_paras={
        "endpoint_url": "http://minio.example.com:9000",
        "key_id": "access_key",
        "secret_key": "secret_key"
    }
)
```

---

## Complete Workflow Example

Here's a complete example from raw data to calibration:

```python
import pandas as pd
import json
from hydrodatasource.reader.data_source import SelfMadeHydroDataset
from hydromodel.trainers.unified_calibrate import calibrate

# Step 1: Prepare attributes
attributes = pd.DataFrame({
    'basin_id': ['basin_001', 'basin_002'],
    'area': [1250.5, 856.3],
    'lat': [30.5, 31.2],
    'lon': [105.2, 106.1]
})
attributes.to_csv("my_data/attributes/attributes.csv", index=False)

# Step 2: Prepare time series (assume you have daily_data_001.csv)
# Make sure it has columns: time, prcp, PET, streamflow
daily_data = pd.read_csv("daily_data_001.csv")
daily_data.to_csv("my_data/timeseries/1D/basin_001.csv", index=False)

# Step 3: Create units info
units = {
    "prcp": "mm/day",
    "PET": "mm/day",
    "streamflow": "m^3/s"
}
with open("my_data/timeseries/1D_units_info.json", "w") as f:
    json.dump(units, f, indent=2)

# Step 4: Verify data loads correctly
dataset = SelfMadeHydroDataset(
    data_path="my_data",
    time_unit="1D"
)
print(f"Basins: {dataset.read_object_ids()}")

# Step 5: Cache for faster access (optional)
dataset.cache_xrdataset(
    gage_id_lst=['basin_001'],
    t_range=["1990-01-01", "2010-12-31"],
    var_lst=["prcp", "PET", "streamflow"]
)

# Step 6: Run calibration with hydromodel
config = {
    "data_cfgs": {
        "data_source_type": "selfmadehydrodataset",
        "data_source_path": "my_data",
        "basin_ids": ["basin_001"],
        "train_period": ["1990-01-01", "2000-12-31"],
        "test_period": ["2001-01-01", "2010-12-31"],
        "warmup_length": 365,
    },
    "model_cfgs": {
        "model_name": "xaj_mz",
    },
    "training_cfgs": {
        "algorithm": "SCE_UA",
        "loss_func": "RMSE",
        "output_dir": "results",
        "experiment_name": "my_basin_001",
        "rep": 5000,
        "ngs": 50,
    },
    "evaluation_cfgs": {
        "metrics": ["NSE", "KGE", "RMSE"],
    },
}

results = calibrate(config)
print("Calibration complete!")
```

---

## Troubleshooting

### Common Issues

**1. "Basin ID not found"**
- Check `basin_id` column in `attributes.csv` matches CSV filenames
- Basin IDs must be strings (not numbers)
- Filenames: `{basin_id}.csv` (e.g., `basin_001.csv`)

**2. "Time column not found"**
- CSV must have `time` column (case-sensitive)
- Format: `YYYY-MM-DD` for daily, `YYYY-MM-DD HH:MM` for hourly

**3. "Unit info file not found"**
- Create `{time_unit}_units_info.json` in timeseries folder
- Example: `1D_units_info.json` for daily data

**4. "Variable not found in units info"**
- Every variable in CSV must be in `units_info.json`
- Check spelling matches exactly (case-sensitive)

**5. "Data shape mismatch"**
- All basins should have same variables
- All basins should cover the requested time range

**6. CAMELS data download fails**
- Check internet connection
- Check disk space (CAMELS-US needs ~70GB)
- Try manual download from official sources
- Set `download=False` if data already exists

### Data Validation Checklist

Before using your custom data:

- [ ] `attributes.csv` exists with `basin_id` and `area` columns
- [ ] Time series files named `{basin_id}.csv`
- [ ] All CSV files have `time` column
- [ ] `{time_unit}_units_info.json` exists
- [ ] All variables in CSV are in `units_info.json`
- [ ] No negative precipitation or streamflow values
- [ ] Time series is continuous (no large gaps)
- [ ] Data covers warmup + train + test periods
- [ ] Units are physically reasonable

---

## Data Conversion Tools

### From GIS Shapefile

Extract basin attributes from shapefile:

```python
import geopandas as gpd

# Read shapefile
basins = gpd.read_file("basins.shp")

# Calculate area (convert to km²)
basins['area'] = basins.geometry.area / 1e6

# Export to CSV
basins[['basin_id', 'area', 'lat', 'lon']].to_csv(
    "attributes/attributes.csv",
    index=False
)
```

### From Other Formats

```python
# From Excel
import pandas as pd
df = pd.read_excel("basin_data.xlsx")
df.to_csv("timeseries/1D/basin_001.csv", index=False)

# From NetCDF
import xarray as xr
ds = xr.open_dataset("data.nc")
df = ds.to_dataframe().reset_index()
df.to_csv("timeseries/1D/basin_001.csv", index=False)
```

---

## Summary

### Quick Decision Guide

**Choose CAMELS (hydrodataset) if:**
- ✅ You need quality-controlled data
- ✅ Working with well-studied basins
- ✅ Want standardized format
- ✅ Need consistent attributes

**Choose Custom Data (hydrodatasource) if:**
- ✅ Using your own field data
- ✅ Working with ungauged basins
- ✅ Need specific time periods
- ✅ Have proprietary data

### Key Points

1. **Public Data**: Use `hydrodataset` for CAMELS variants
2. **Custom Data**: Use `hydrodatasource` with `selfmadehydrodataset` format
3. **Data Structure**: Follow standard directory layout
4. **Required Files**: `attributes.csv`, time series CSVs, `units_info.json`
5. **Data Quality**: Check completeness, consistency, and physical validity
6. **Performance**: Use NetCDF caching for large datasets

---

## Additional Resources

- **hydrodataset GitHub**: https://github.com/OuyangWenyu/hydrodataset
- **hydrodataset docs**: https://hydrodataset.readthedocs.io/
- **hydrodatasource GitHub**: https://github.com/OuyangWenyu/hydrodatasource
- **hydromodel docs**: [usage.md](usage.md), [quickstart.md](quickstart.md)
- **CAMELS official sites**:
  - US: https://ral.ucar.edu/solutions/products/camels
  - GB: https://catalogue.ceh.ac.uk/documents/8344e4f3-d2ea-44f5-8afa-86d2987543a9
