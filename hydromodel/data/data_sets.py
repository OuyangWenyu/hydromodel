import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset


class CamelsDataset(Dataset):
    """Base data set class to load and preprocess CAMELS format data using PyTroch's Dataset"""

    def __init__(
        self,
        cfg: dict,
        loader_type: str,
        data_attr: pd.DataFrame,
        data_forcing: xr.Dataset,
        data_flow: xr.Dataset,
        means: pd.DataFrame = None,
        stds: pd.DataFrame = None,
    ):
        """
        Initialize Dataset containing the data of multiple basins.

        Parameters
        ----------
        basins : list
            _description_
        dates : list
            _description_
        data_attr : pd.DataFrame
            _description_
        data_forcing : xr.Dataset
            _description_
        data_flow : xr.Dataset
            _description_
        loader_type : str, optional
            _description_, by default "train"
        seq_length : int, optional
            _description_, by default 100
        means : pd.DataFrame, optional
            _description_, by default None
        stds : pd.DataFrame, optional
            _description_, by default None

        Raises
        ------
        ValueError
            _description_
        """
        super(CamelsDataset, self).__init__()
        if loader_type in {"train", "valid", "test"}:
            self.loader_type = loader_type
        else:
            raise ValueError(
                " 'loader_type' must be one of 'train', 'valid' or 'test' "
            )
        self.basins = cfg.data_params.object_ids
        if self.loader_type == "train":
            self.dates = cfg.data_params.t_range_train
        elif self.loader_type == "valid":
            self.dates = cfg.data_params.t_range_valid
        elif self.loader_type == "test":
            self.dates = cfg.data_params.t_range_test
        else:
            raise ValueError(
                " 'loader_type' must be one of 'train', 'valid' or 'test' "
            )

        self.seq_length = cfg.training_params.seq_length

        self.means = means
        self.stds = stds

        # Trans str to int if a col is str
        data_attr_numeric = data_attr.apply(
            lambda s: pd.factorize(s)[0] if s.dtype == "object" else s
        )
        # for data_attr, data_forcing, we need to make sure that there are no NaNs
        # in the data. If there are NaNs in data_attr, we need to fill them with the mean of the
        # corresponding column. If there are NaNs in data_forcing, we need to fill them with the
        # interpolated values.
        self.data_attr = data_attr_numeric.fillna(data_attr_numeric.mean())
        self.data_forcing = data_forcing.interpolate_na(dim="time")

        self.data_flow = data_flow

        # load and preprocess data
        self._load_data()

    def __len__(self):
        # self.data_attr.shape[0] means numbers of basins
        return self.num_samples if self.train_mode else self.data_attr.index.size

    def __getitem__(self, item: int):
        if not self.train_mode:
            x = self.x.isel(basin=item).to_array().to_numpy().T
            y = self.y.isel(basin=item).to_array().to_numpy().T
            if self.c is None or self.c.shape[-1] == 0:
                return torch.from_numpy(x).float(), torch.from_numpy(y).float()
            c = self.c.iloc[item, :].values
            c = np.tile(c, (x.shape[0], 1))
            xc = np.concatenate((x, c), axis=1)
            return torch.from_numpy(xc).float(), torch.from_numpy(y).float()
        basin, time = self.lookup_table[item]
        seq_length = self.seq_length
        x = (
            self.x.sel(
                basin=basin,
                time=slice(time, time + np.timedelta64(seq_length - 1, "D")),
            )
            .to_array()
            .to_numpy()
        ).T
        c = self.c.loc[basin].values
        c = np.tile(c, (seq_length, 1))
        xc = np.concatenate((x, c), axis=1)
        y = (
            self.y.sel(
                basin=basin,
                time=slice(time, time + np.timedelta64(seq_length - 1, "D")),
            )
            .to_array()
            .to_numpy()
        ).T
        return torch.from_numpy(xc).float(), torch.from_numpy(y).float()

    def _load_data(self):
        """load data from nc and feather files"""
        if self.loader_type == "train":
            train_mode = True
            df_mean_forcings = self.data_forcing.mean().to_pandas()
            df_std_forcings = self.data_forcing.std().to_pandas()
            df_mean_streamflow = self.data_flow.mean().to_pandas()
            df_std_streamflow = self.data_flow.std().to_pandas()
            # some attributes are strings, convert them to integers
            df_mean_attr = self.data_attr.mean()
            df_std_attr = self.data_attr.std()
            self.means = pd.concat([df_mean_forcings, df_mean_attr, df_mean_streamflow])
            self.stds = pd.concat([df_std_forcings, df_std_attr, df_std_streamflow])
            # for stds, when values are all same, stds will be 0/very small,
            # which will cause NaNs, so we replace small values with 1
            # this means when self.stds > 1e-5, keep the value, otherwise replace with 1
            self.stds = self.stds.where(self.stds > 1e-5, 1)
        else:
            train_mode = False

        # nomalization
        self.x = self._local_normalization(
            self.data_forcing, list(self.data_forcing.keys())
        )
        self.c = self._local_normalization(
            self.data_attr, self.data_attr.columns.values.tolist()
        )
        if train_mode:
            self.y = self._local_normalization(
                self.data_flow, list(self.data_flow.keys())
            )
        else:
            # no normalization for streamflow in valid/test mode
            self.y = self.data_flow
        self.train_mode = train_mode
        self._create_lookup_table()

    def _local_normalization(self, feature, variable) -> np.ndarray:
        """Normalize features with local mean/std."""
        feature = (feature - self.means[variable]) / self.stds[variable]
        return feature

    def _create_lookup_table(self):
        """create a index table for __getitem__ functions"""
        lookup = []
        # list to collect basins ids of basins without a single training sample
        seq_length = self.seq_length
        dates = self.data_flow["time"].to_numpy()
        time_length = len(dates)
        for basin in self.basins:
            lookup.extend(
                (basin, dates[j]) for j in range(time_length - seq_length + 1)
            )
        self.lookup_table = dict(enumerate(lookup))
        self.num_samples = len(self.lookup_table)

    def get_means(self):
        return self.means

    def get_stds(self):
        return self.stds

    def local_denormalization(self, feature, variable="streamflow"):
        """revert the normalization for streaflow"""
        feature = feature * self.stds[variable] + self.means[variable]
        return feature


class CamelsDataset4N2N(CamelsDataset):
    def __init__(
        self,
        data_attr: xr.Dataset,
        data_forcing: xr.Dataset,
        data_flow: xr.Dataset,
        basins: list,
        seq_length: int,
        loader_type: str = "train",
        with_attributes: bool = True,
    ):
        super().__init__(
            data_attr,
            data_forcing,
            data_flow,
            basins,
            seq_length,
            loader_type,
            with_attributes,
        )

    def __getitem__(self, item: int):
        return super().__getitem__(item)

    def __len__(self):
        return super().__len__()


def load_streamflow(ds_flow, ds_attr, basins, time_range):
    """load streamflow data in the time_range and transform its unit from ft3/s to mm/day

    Parameters
    ----------
    ds_flow : _type_
        _description_
    ds_attr : _type_
        _description_
    time_range : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    chosen_streamflow = ds_flow.sel(
        basin=basins, time=slice(time_range[0], time_range[1])
    )
    area = ds_attr["area_gages2"].values
    return (
        0.0283168
        * chosen_streamflow
        * 1000
        * 86400
        / (area.reshape(len(area), 1) * 10**6)
    )
