import collections
import os
import pandas as pd
import numpy as np
from pandas.core.dtypes.common import is_string_dtype, is_numeric_dtype

from src.data.data_base import DatasetBase
from src.explore.stat import cal_fdc
from src.utils.hydro_utils import download_one_zip, t_range_days


class Camels(DatasetBase):
    def __init__(self, data_path, download=False):
        super().__init__(data_path)
        self.dataset_description = self.set_dataset_describe()
        if download:
            self.download_dataset()
        self.camels_sites = self.read_site_info()

    def get_name(self):
        return "CAMELS"

    def set_dataset_describe(self):
        camels_db = self.dataset_dir
        # shp file of basins
        camels_shp_file = os.path.join(camels_db, "basin_set_full_res", "HCDN_nhru_final_671.shp")
        # config of flow data
        flow_dir = os.path.join(camels_db, "basin_timeseries_v1p2_metForcing_obsFlow", "basin_dataset_public_v1p2",
                                "usgs_streamflow")
        # forcing
        forcing_dir = os.path.join(camels_db, "basin_timeseries_v1p2_metForcing_obsFlow", "basin_dataset_public_v1p2",
                                   "basin_mean_forcing")
        forcing_types = ["daymet", "maurer", "nldas"]
        # attr
        attr_dir = os.path.join(camels_db, "camels_attributes_v2.0", "camels_attributes_v2.0")
        gauge_id_file = os.path.join(attr_dir, 'camels_name.txt')

        download_url_lst = [
            "https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/camels_attributes_v2.0.zip",
            "https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/basin_timeseries_v1p2_metForcing_obsFlow.zip",
            "https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/basin_set_full_res.zip"]

        return collections.OrderedDict(CAMELS_DIR=camels_db, CAMELS_FLOW_DIR=flow_dir,
                                       CAMELS_FORCING_DIR=forcing_dir, CAMELS_FORCING_TYPE=forcing_types,
                                       CAMELS_ATTR_DIR=attr_dir, CAMELS_GAUGE_FILE=gauge_id_file,
                                       CAMELS_BASINS_SHP_FILE=camels_shp_file, CAMELS_DOWNLOAD_URL_LST=download_url_lst)

    def download_dataset(self):
        camels_config = self.dataset_description
        if not os.path.isdir(camels_config["CAMELS_DIR"]):
            os.makedirs(camels_config["CAMELS_DIR"])
        [download_one_zip(attr_url, camels_config["CAMELS_DIR"]) for attr_url in
         camels_config["CAMELS_DOWNLOAD_URL_LST"] if
         not os.path.isfile(os.path.join(camels_config["CAMELS_DIR"], attr_url.split("/")[-1]))]
        print("The CAMELS data have been downloaded!")

    def get_constant_cols(self) -> np.array:
        """all readable attrs in CAMELS"""
        data_folder = self.dataset_description["CAMELS_ATTR_DIR"]
        var_dict = dict()
        var_lst = list()
        key_lst = ['topo', 'clim', 'hydro', 'vege', 'soil', 'geol']
        for key in key_lst:
            data_file = os.path.join(data_folder, 'camels_' + key + '.txt')
            data_temp = pd.read_csv(data_file, sep=';')
            var_lst_temp = list(data_temp.columns[1:])
            var_dict[key] = var_lst_temp
            var_lst.extend(var_lst_temp)
        return np.array(var_lst)

    def get_relevant_cols(self) -> np.array:
        # TODO: now only these 7 forcings
        return np.array(['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp'])

    def get_target_cols(self) -> np.array:
        # TODO: now only usgsFlow
        return np.array(["usgsFlow"])

    def get_other_cols(self) -> dict:
        return {"FDC": {"time_range": ["1980-01-01", "2000-01-01"], "quantile_num": 100}}

    def read_site_info(self) -> pd.DataFrame:
        camels_file = self.dataset_description["CAMELS_GAUGE_FILE"]
        data = pd.read_csv(camels_file, sep=';', dtype={"gauge_id": str, "huc_02": str})
        return data

    def read_object_ids(self, object_params=None) -> np.array:
        return self.camels_sites["gauge_id"].values

    def read_basin_area(self, object_ids) -> np.array:
        return self.read_constant_cols(object_ids, ['area_gages2'], is_return_dict=False)

    def read_mean_prep(self, object_ids) -> np.array:
        return self.read_constant_cols(object_ids, ['p_mean'], is_return_dict=False)

    def read_usgs_gage(self, usgs_id, t_range):
        print("reading %s streamflow data", usgs_id)
        gage_id_df = self.camels_sites
        huc = gage_id_df[gage_id_df["gauge_id"] == usgs_id]["huc_02"].values[0]
        usgs_file = os.path.join(self.dataset_description["CAMELS_FLOW_DIR"], huc, usgs_id + '_streamflow_qc.txt')
        data_temp = pd.read_csv(usgs_file, sep=r'\s+', header=None)
        obs = data_temp[4].values
        obs[obs < 0] = np.nan
        t_lst = t_range_days(t_range)
        nt = t_lst.shape[0]
        if len(obs) != nt:
            out = np.full([nt], np.nan)
            df_date = data_temp[[1, 2, 3]]
            df_date.columns = ['year', 'month', 'day']
            date = pd.to_datetime(df_date).values.astype('datetime64[D]')
            [C, ind1, ind2] = np.intersect1d(date, t_lst, return_indices=True)
            out[ind2] = obs[ind1]
        else:
            out = obs
        return out

    def read_target_cols(self, usgs_id_lst=None, t_range=None, target_cols=None, **kwargs):
        # only one output for camels: streamflow, just read it
        nt = t_range_days(t_range).shape[0]
        y = np.empty([len(usgs_id_lst), nt])
        for k in range(len(usgs_id_lst)):
            data_obs = self.read_usgs_gage(usgs_id_lst[k], t_range)
            y[k, :] = data_obs
        return y

    def read_forcing_gage(self, usgs_id, var_lst, t_range_list, dataset='daymet'):
        # dataset = daymet or maurer or nldas
        print("reading %s forcing data", usgs_id)
        gage_id_df = self.camels_sites
        huc = gage_id_df[gage_id_df["gauge_id"] == usgs_id]["huc_02"].values[0]

        data_folder = self.dataset_description["CAMELS_FORCING_DIR"]
        if dataset == 'daymet':
            temp_s = 'cida'
        else:
            temp_s = dataset
        data_file = os.path.join(data_folder, dataset, huc, '%s_lump_%s_forcing_leap.txt' % (usgs_id, temp_s))
        data_temp = pd.read_csv(data_file, sep=r'\s+', header=None, skiprows=4)
        forcing_lst = ["Year", "Mnth", "Day", "Hr", "dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"]
        df_date = data_temp[[0, 1, 2]]
        df_date.columns = ['year', 'month', 'day']
        date = pd.to_datetime(df_date).values.astype('datetime64[D]')

        nf = len(var_lst)
        [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
        nt = c.shape[0]
        out = np.empty([nt, nf])

        for k in range(nf):
            ind = forcing_lst.index(var_lst[k])
            out[:, k] = data_temp[ind].values[ind1]
        return out

    def read_relevant_cols(self, usgs_id_lst=None, t_range=None, var_lst=None, forcing_type="daymet"):
        t_range_list = t_range_days(t_range)
        nt = t_range_list.shape[0]
        x = np.empty([len(usgs_id_lst), nt, len(var_lst)])
        for k in range(len(usgs_id_lst)):
            data = self.read_forcing_gage(usgs_id_lst[k], var_lst, t_range_list, dataset=forcing_type)
            x[k, :, :] = data
        return x

    def read_attr_all(self):
        data_folder = self.dataset_description["CAMELS_ATTR_DIR"]
        f_dict = dict()  # factorize dict
        var_dict = dict()
        var_lst = list()
        out_lst = list()
        key_lst = ['topo', 'clim', 'hydro', 'vege', 'soil', 'geol']
        gage_dict = self.camels_sites
        for key in key_lst:
            data_file = os.path.join(data_folder, 'camels_' + key + '.txt')
            data_temp = pd.read_csv(data_file, sep=';')
            var_lst_temp = list(data_temp.columns[1:])
            var_dict[key] = var_lst_temp
            var_lst.extend(var_lst_temp)
            k = 0
            n_gage = len(gage_dict['gauge_id'].values)
            out_temp = np.full([n_gage, len(var_lst_temp)], np.nan)
            for field in var_lst_temp:
                if is_string_dtype(data_temp[field]):
                    value, ref = pd.factorize(data_temp[field], sort=True)
                    out_temp[:, k] = value
                    f_dict[field] = ref.tolist()
                elif is_numeric_dtype(data_temp[field]):
                    out_temp[:, k] = data_temp[field].values
                k = k + 1
            out_lst.append(out_temp)
        out = np.concatenate(out_lst, 1)
        return out, var_lst, var_dict, f_dict

    def read_constant_cols(self, usgs_id_lst=None, var_lst=None, is_return_dict=False):
        attr_all, var_lst_all, var_dict, f_dict = self.read_attr_all()
        ind_var = list()
        for var in var_lst:
            ind_var.append(var_lst_all.index(var))
        gage_dict = self.camels_sites
        id_lst_all = gage_dict['gauge_id'].values
        c, ind_grid, ind2 = np.intersect1d(id_lst_all, usgs_id_lst, return_indices=True)
        temp = attr_all[ind_grid, :]
        out = temp[:, ind_var]
        if is_return_dict:
            return out, var_dict, f_dict
        else:
            return out

    def read_other_cols(self, object_ids=None, other_cols: dict = None, **kwargs):
        # TODO: FDC for test period should keep same with that in training period
        out_dict = {}
        for key, value in other_cols.items():
            if key == "FDC":
                assert "time_range" in value.keys()
                if "quantile_num" in value.keys():
                    quantile_num = value["quantile_num"]
                    out = cal_fdc(self.read_target_cols(object_ids, value["time_range"], "usgsFlow"),
                                  quantile_num=quantile_num)
                else:
                    out = cal_fdc(self.read_target_cols(object_ids, value["time_range"], "usgsFlow"))
            else:
                raise NotImplementedError("No this item yet!!")
            out_dict[key] = out
        return out_dict
