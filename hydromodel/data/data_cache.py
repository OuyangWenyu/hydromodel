import pandas as pd
import xarray as xr
from hydrodataset import CACHE_DIR, HydroDataset


def cache_data_source(camels_us: HydroDataset):
    """Cache data source to xarray dataset (time series data) and pandas feather(static attributes)
    TODO: now only support CAMELS_US

    Parameters
    ----------
    camels_us : HydroDataset
        _description_

    Returns
    -------
    tuple
        streamflow, forcing, attributes
    """
    try:
        streamflow_ds = xr.open_dataset(CACHE_DIR.joinpath("camels_streamflow.nc"))
        forcing_ds = xr.open_dataset(CACHE_DIR.joinpath("camels_daymet_forcing.nc"))
        attrs = pd.read_feather(CACHE_DIR.joinpath("camels_attributes_v2.0.feather"))
    except FileNotFoundError:
        print("cache downloaded data to nc and feather files firstly.")
        camels_us.cache_attributes_feather()
        camels_us.cache_forcing_xrdataset()
        camels_us.cache_streamflow_xrdataset()
        streamflow_ds = xr.open_dataset(CACHE_DIR.joinpath("camels_streamflow.nc"))
        forcing_ds = xr.open_dataset(CACHE_DIR.joinpath("camels_daymet_forcing.nc"))
        attrs = pd.read_feather(CACHE_DIR.joinpath("camels_attributes_v2.0.feather"))
    return streamflow_ds,forcing_ds,attrs