# Using Custom Data with hydromodel

This guide explains how to prepare and use your own hydrological data with `hydromodel` using the `selfmadehydrodataset` format from the [hydrodatasource](https://github.com/OuyangWenyu/hydrodatasource) package.

## Overview

`hydrodatasource` provides the `SelfMadeHydroDataset` class for managing custom hydrological data. It offers:
- Standardized data organization and storage
- NetCDF caching for fast access
- Support for multiple time scales (1h, 3h, 1D, 8D)
- Integration with cloud storage (MinIO/S3)
- Direct compatibility with hydromodel

## Quick Start

### 1. Install hydrodatasource

```bash
pip install hydrodatasource
```

### 2. Organize Your Data

Your data directory should follow this structure:

```
my_basin_data/
├── attributes/
│   └── attributes.csv              # Basin attributes (required)
├── timeseries/
│   ├── 1D/                         # Daily time series (most common)
│   │   ├── basin_001.csv
│   │   ├── basin_002.csv
│   │   └── ...
│   └── 1D_units_info.json          # Variable units (required)
└── shapes/                         # Basin boundaries (optional)
    └── basins.shp
```

### 3. Prepare Required Files

#### attributes.csv

Minimum required columns: `basin_id` and `area` (km²)

```csv
basin_id,area,lat,lon,elevation
basin_001,1250.5,30.5,105.2,850
basin_002,856.3,31.2,106.1,920
```

#### basin_001.csv (time series)

Required column: `time`. Other columns are your variables.

```csv
time,prcp,PET,streamflow
1990-01-01,5.2,2.1,45.3
1990-01-02,0.0,2.3,42.1
1990-01-03,12.5,1.8,58.7
```

**Important:**
- `time` format: YYYY-MM-DD (for daily data)
- Variable names: Use lowercase, underscores for multi-word
- Missing values: Use empty cells or NaN (not -9999 or 0)

#### 1D_units_info.json (units definition)

Define physical units for all variables:

```json
{
  "prcp": "mm/day",
  "PET": "mm/day",
  "streamflow": "m^3/s",
  "temp": "degC"
}
```

Common units:
- Precipitation/ET: `mm/day` or `mm/hour`
- Streamflow: `m^3/s` or `mm/day`
- Temperature: `degC` or `K`
- Area: `km^2`

## Using with hydromodel

### Method 1: Direct Integration (Recommended)

Use the unified calibration interface with `selfmadehydrodataset`:

```python
from hydromodel.trainers.unified_calibrate import calibrate
import yaml

config = {
    "data_cfgs": {
        "data_source_type": "selfmadehydrodataset",
        "data_source_path": "D:/my_basin_data",
        "basin_ids": ["basin_001", "basin_002"],
        "train_period": ["1990-01-01", "2000-12-31"],
        "test_period": ["2001-01-01", "2010-12-31"],
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
        "experiment_name": "my_basins",
        "rep": 10000,
        "ngs": 100,
    },
    "evaluation_cfgs": {
        "metrics": ["NSE", "KGE", "RMSE"],
    },
}

# Run calibration
results = calibrate(config)
```

### Method 2: Load Data with hydrodatasource

For more control, load data directly:

```python
from hydrodatasource.reader.data_source import SelfMadeHydroDataset

# Initialize dataset
dataset = SelfMadeHydroDataset(
    data_path="D:/my_basin_data",
    time_unit="1D"
)

# Get basin IDs
basin_ids = dataset.read_object_ids()
print(f"Found {len(basin_ids)} basins")

# Read time series data
data = dataset.read_timeseries(
    gage_id_lst=["basin_001"],
    t_range=["1990-01-01", "2000-12-31"],
    var_lst=["prcp", "PET", "streamflow"]
)

# Data format: numpy array [n_basins, n_time, n_vars]
print(f"Data shape: {data['1D'].shape}")
```

### Method 3: Use NetCDF Cache (Faster)

For large datasets, cache as NetCDF for 10x faster access:

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

## Data Requirements for XAJ Model

The XAJ model requires these variables:

### Required Variables

| Variable | Description | Unit | Source |
|----------|-------------|------|--------|
| `prcp` | Precipitation | mm/day | Rain gauge, gridded data |
| `PET` | Potential Evapotranspiration | mm/day | Penman, Priestley-Taylor, or reanalysis |
| `streamflow` | Observed streamflow | m³/s | Stream gauge |
| `area` | Basin area | km² | GIS analysis |

### Optional Variables

| Variable | Description | Unit | Usage |
|----------|-------------|------|-------|
| `temp` | Temperature | °C | Snow module |
| `elevation` | Basin elevation | m | PET estimation |
| `lat`, `lon` | Coordinates | degrees | Spatial analysis |

### Data Quality Tips

1. **Time Resolution**: Daily (1D) is standard for XAJ
2. **Data Completeness**:
   - Training period: ≥5 years continuous
   - Warmup period: ≥1 year before training
   - Missing data: <5% acceptable
3. **Physical Consistency**:
   - Precipitation ≥ 0
   - Streamflow ≥ 0
   - PET ≥ 0
   - Check water balance: P ≈ Q + ET
4. **Unit Consistency**: Ensure all units match `units_info.json`

## Advanced Features

### Multi-Scale Data

Support different time scales in the same dataset:

```
timeseries/
├── 1h/                    # Hourly data
│   ├── basin_001.csv
│   └── 1h_units_info.json
├── 1D/                    # Daily data
│   ├── basin_001.csv
│   └── 1D_units_info.json
└── 8D/                    # 8-day data (e.g., MODIS)
    ├── basin_001.csv
    └── 8D_units_info.json
```

### Station Data

Include meteorological/hydrological stations:

```
my_basin_data/
├── attributes/
│   ├── attributes.csv
│   └── station_info.csv           # Station metadata
├── timeseries/
│   └── 1D/...
└── stations/
    ├── 1D/
    │   ├── station_001.csv        # Station observations
    │   └── ...
    └── basin_station_map.csv      # Basin-station mapping
```

See [hydrodatasource station guide](https://github.com/OuyangWenyu/hydrodatasource/blob/main/docs/station_dataset_guide.md) for details.

### Cloud Storage

Use MinIO/S3 for large datasets:

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

## Data Conversion Tools

### From CSV to NetCDF

Convert CSV files to efficient NetCDF format:

```python
from hydrodatasource import read_and_save_camels_format

read_and_save_camels_format(
    input_dir="D:/my_basin_data",
    output_dir="D:/my_basin_data_nc",
    basin_id="basin_001"
)
```

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

## Example: Complete Workflow

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

# Step 5: Cache for faster access
dataset.cache_xrdataset(
    gage_id_lst=['basin_001'],
    t_range=["1990-01-01", "2010-12-31"],
    var_lst=["prcp", "PET", "streamflow"]
)

# Step 6: Run calibration
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

## Troubleshooting

### Common Issues

**1. "Basin ID not found"**
- Check `basin_id` column in `attributes.csv` matches filenames
- Ensure basin IDs are strings, not numbers

**2. "Time column not found"**
- CSV must have `time` column (case-sensitive)
- Format: YYYY-MM-DD for daily, YYYY-MM-DD HH:MM for hourly

**3. "Unit info file not found"**
- Create `{time_unit}_units_info.json` in timeseries folder
- Example: `1D_units_info.json` for daily data

**4. "Variable not found in units info"**
- Every variable in CSV must be defined in units_info.json
- Check spelling matches exactly

**5. "Data shape mismatch"**
- All basins should have same variables
- All basins should cover the requested time range

### Data Validation Checklist

Before using your data:

- [ ] `attributes.csv` exists with `basin_id` and `area` columns
- [ ] Time series files named `{basin_id}.csv`
- [ ] All CSV files have `time` column
- [ ] `{time_unit}_units_info.json` exists
- [ ] All variables in CSV are in units_info.json
- [ ] No negative precipitation or streamflow
- [ ] Time series is continuous (no gaps)
- [ ] Data covers warmup + train + test periods

## Additional Resources

- **hydrodatasource GitHub**: https://github.com/OuyangWenyu/hydrodatasource
- **Station Dataset Guide**: [docs/station_dataset_guide.md](https://github.com/OuyangWenyu/hydrodatasource/blob/main/docs/station_dataset_guide.md)
- **Example Scripts**: [scripts/](https://github.com/OuyangWenyu/hydrodatasource/tree/main/scripts)
- **hydromodel Documentation**: [docs/usage.md](usage.md)

## Summary

Key points for using custom data:

1. **Organize**: Follow the standard directory structure
2. **Prepare**: Create `attributes.csv` and `units_info.json`
3. **Validate**: Check data quality and completeness
4. **Cache**: Use NetCDF caching for large datasets
5. **Integrate**: Use `selfmadehydrodataset` in hydromodel config

The `selfmadehydrodataset` format provides a flexible, efficient way to use your own data with hydromodel while maintaining compatibility with the CAMELS-style workflows.
