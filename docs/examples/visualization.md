## Results Visualization

### Overview

The `visualize.py` script provides a simple command-line interface for visualizing evaluation results. It generates time series plots with precipitation and streamflow comparisons.

**Key Features:**
- Time series plots with observed vs simulated streamflow
- Precipitation displayed as inverted bars (top-down)
- Automatic loading from NetCDF evaluation results
- Basin-level or multi-basin visualization

### Command-Line Usage

**Basic usage** (visualize all basins):

```bash
python scripts/visualize.py --eval-dir results/xaj_mz_SCE_UA/evaluation_test
```

**Visualize specific basins:**

```bash
python scripts/visualize.py \
    --eval-dir results/xaj_mz_SCE_UA/evaluation_test \
    --basins 01013500 01022500
```

**Custom output directory:**

```bash
python scripts/visualize.py \
    --eval-dir results/xaj_mz_SCE_UA/evaluation_test \
    --output-dir my_figures
```

### Python API Usage

For programmatic visualization:

```python
from hydromodel.datasets.data_visualize import visualize_evaluation

# Visualize all basins
visualize_evaluation(
    eval_dir="results/xaj_mz_SCE_UA/evaluation_test",
    output_dir="figures",  # Optional, defaults to eval_dir/figures
    basins=None  # Optional, defaults to all basins
)

# Visualize specific basins
visualize_evaluation(
    eval_dir="results/xaj_mz_SCE_UA/evaluation_test",
    basins=["01013500", "01022500"]
)
```

### Input Requirements

The visualization script expects:

1. **NetCDF file**: `*_evaluation_results.nc` containing:
   - `qobs`: Observed streamflow `[time, basin]`
   - `qsim`: Simulated streamflow `[time, basin]`
   - `prcp`: Precipitation `[time, basin]` (optional)
   - `basin`: Basin IDs
   - `time`: Time coordinates

2. **Directory structure**:
   ```
   results/xaj_mz_SCE_UA/
   └── evaluation_test/
       ├── test_evaluation_results.nc
       └── figures/  # Output directory (auto-created)
           ├── 01013500_timeseries.png
           ├── 01022500_timeseries.png
           └── ...
   ```

### Output

For each basin, generates:
- **Time series plot**: `{basin_id}_timeseries.png`
  - Upper panel: Precipitation (inverted bars)
  - Lower panel: Observed vs simulated streamflow

**Plot features:**
- Date formatting (YYYY-MM)
- Dual-axis precipitation/streamflow
- Legend with simulation vs observation
- PNG format with 300 DPI

### Advanced Visualization

For custom plots beyond the CLI tool, use the core plotting functions directly:

```python
from hydromodel.datasets.data_visualize import (
    plot_sim_and_obs,
    plot_sim_and_obs_streamflow,
    plot_precipitation
)
import xarray as xr

# Load evaluation results
ds = xr.open_dataset("results/xaj_mz_SCE_UA/evaluation_test/test_evaluation_results.nc")

# Extract data for a specific basin
basin_idx = 0
time = ds['time'].values
qobs = ds['qobs'].values[:, basin_idx]
qsim = ds['qsim'].values[:, basin_idx]
prcp = ds['prcp'].sel(basin=ds['basin'].values[basin_idx])

# Create custom plot
plot_sim_and_obs(
    date=time,
    prcp=prcp,
    sim=qsim,
    obs=qobs,
    save_fig="my_custom_plot.png",
    ylabel="Streamflow (m³/s)"
)
```

---

