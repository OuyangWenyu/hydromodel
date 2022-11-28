import collections
import hydrodataset
from hydrodataset import hydro_utils
import os
from typing import Union
import pandas as pd
import numpy as np
from pandas.core.dtypes.common import is_string_dtype, is_numeric_dtype
from tqdm import tqdm


class MyCamels(hydrodataset.Camels):
    def __init__(self, data_path, download=False, region: str = "CC"):
        """
        Initialization for my own CAMELS format dataset

        Parameters
        ----------
        data_path
            where we put the dataset
        download
            if true, download
        region
            remember the name of your own region
        """
        hydrodataset.camels.CAMELS_REGIONS = hydrodataset.camels.CAMELS_REGIONS + [
            region
        ]
        super().__init__(data_path, download, region)

    def set_data_source_describe(self) -> collections.OrderedDict:
        """
        Introduce the files in the dataset and list their location in the file system

        Returns
        -------
        collections.OrderedDict
            the description for a CAMELS dataset
        """
        camels_db = self.data_source_dir
        # shp files of basins
        camels_shp_files_dir = os.path.join(camels_db, "basin_boudaries")
        # attr, flow and forcing data are all in the same dir. each basin has one dir.
        flow_dir = os.path.join(camels_db, "streamflow")
        sm_dir = os.path.join(camels_db, "soil_moisture")
        et_dir = os.path.join(camels_db, "evapotranspiration")
        forcing_dir = os.path.join(camels_db, "basin_mean_forcing")
        attr_dir = os.path.join(camels_db, "attribute")
        # no gauge id file for CAMELS_CC, just read from any attribute file
        gauge_id_file = os.path.join(camels_db, "gage_points.csv")
        attr_key_lst = [
            "climate",
            "geology",
            "land_cover",
            "permeability_porosity",
            "root_depth",
            "soil",
            "topo_elev_slope",
            "topo_shape_factors",
        ]
        return collections.OrderedDict(
            CAMELS_DIR=camels_db,
            CAMELS_FLOW_DIR=flow_dir,
            CAMELS_SM_DIR=sm_dir,
            CAMELS_ET_DIR=et_dir,
            CAMELS_FORCING_DIR=forcing_dir,
            CAMELS_ATTR_DIR=attr_dir,
            CAMELS_ATTR_KEY_LST=attr_key_lst,
            CAMELS_GAUGE_FILE=gauge_id_file,
            CAMELS_BASINS_SHP_DIR=camels_shp_files_dir,
        )

    def read_site_info(self) -> pd.DataFrame:
        """
        Read the basic information of gages in a CAMELS dataset

        Returns
        -------
        pd.DataFrame
            basic info of gages
        """
        camels_file = self.data_source_description["CAMELS_GAUGE_FILE"]
        data = pd.read_csv(camels_file, sep=",", dtype={"gage_id": str})
        return data

    def get_constant_cols(self) -> np.array:
        """
        all readable attrs in CAMELS

        Returns
        -------
        np.array
            attribute types
        """
        data_folder = self.data_source_description["CAMELS_ATTR_DIR"]
        files = np.sort(os.listdir(data_folder))
        attr_types = []
        for file_ in files:
            file = os.path.join(data_folder, file_)
            attr_tmp = pd.read_csv(file, sep=",", dtype={"gage_id": str})
            attr_types = attr_types + attr_tmp.columns[1:].values.tolist()
        return np.array(attr_types)

    def get_relevant_cols(self) -> np.array:
        """
        all readable forcing types

        Returns
        -------
        np.array
            forcing types
        """
        forcing_dir = self.data_source_description["CAMELS_FORCING_DIR"]
        forcing_file = os.path.join(forcing_dir, os.listdir(forcing_dir)[0])
        forcing_tmp = pd.read_csv(forcing_file, sep="\s+", dtype={"gage_id": str})
        return forcing_tmp.columns.values

    def get_target_cols(self) -> np.array:
        """
        For CAMELS, the target vars are streamflows

        Returns
        -------
        np.array
            streamflow types
        """
        # ssm is the surface soil moisture
        return np.array(["Q", "ssm", "ET"])

    def read_object_ids(self, **kwargs) -> np.array:
        """
        read station ids

        Parameters
        ----------
        **kwargs
            optional params if needed

        Returns
        -------
        np.array
            gage/station ids
        """
        return self.camels_sites["gage_id"].values

    def read_target_cols(
        self,
        gage_id_lst: Union[list, np.array] = None,
        t_range: list = None,
        target_cols: Union[list, np.array] = None,
        **kwargs
    ) -> np.array:
        """
        read target values; for all CAMELS, they are streamflows except for CAMELS-CC (inlcude soil moisture)

        default target_cols is an one-value list
        Notice: the unit of target outputs in different regions are not totally same

        Parameters
        ----------
        gage_id_lst
            station ids
        t_range
            the time range, for example, ["1990-01-01", "2000-01-01"]
        target_cols
            the default is None, but we neea at least one default target.
        kwargs
            some other params if needed

        Returns
        -------
        np.array
            streamflow data, 3-dim [station, time, streamflow], unit is m3/s
        """
        if target_cols is None:
            return np.array([])
        else:
            nf = len(target_cols)
        t_range_list = hydro_utils.t_range_days(t_range)
        nt = t_range_list.shape[0]
        y = np.full([len(gage_id_lst), nt, nf], np.nan)
        for j in tqdm(range(len(target_cols)), desc="Read Q/SSM/ET data of CAMELS-CC"):
            for k in tqdm(range(len(gage_id_lst))):
                if target_cols[j] == "ssm":
                    sm_file = os.path.join(
                        self.data_source_description["CAMELS_SM_DIR"],
                        gage_id_lst[k] + "_lump_nasa_usda_smap.txt",
                    )
                    sm_data = pd.read_csv(sm_file, sep=",")
                    df_date = sm_data[["Year", "Mnth", "Day"]]
                    df_date.columns = ["year", "month", "day"]
                    date = pd.to_datetime(df_date).values.astype("datetime64[D]")
                    [c, ind1, ind2] = np.intersect1d(
                        date, t_range_list, return_indices=True
                    )
                    y[k, ind2, j] = sm_data["ssm(mm)"].values[ind1]
                elif target_cols[j] == "ET":
                    et_file = os.path.join(
                        self.data_source_description["CAMELS_ET_DIR"],
                        gage_id_lst[k] + "_lump_modis16a2v006_et.txt",
                    )
                    et_data = pd.read_csv(et_file, sep=",")
                    df_date = et_data[["Year", "Mnth", "Day"]]
                    df_date.columns = ["year", "month", "day"]
                    # all dates in a table
                    date = pd.to_datetime(df_date).values.astype("datetime64[D]")
                    if (
                        np.datetime64(str(date[-1].astype(object).year) + "-12-31")
                        > date[-1]
                        > np.datetime64(str(date[-1].astype(object).year) + "-12-24")
                    ):
                        # the final date in all dates, if it is a date in the end of a year, its internal is 5 or 6
                        final_date = np.datetime64(
                            str(date[-1].astype(object).year + 1) + "-01-01"
                        )
                    else:
                        final_date = date[-1] + np.timedelta64(8, "D")
                    date_all = hydro_utils.t_range_days(
                        hydro_utils.t_days_lst2range([date[0], final_date])
                    )
                    t_range_final = np.intersect1d(date_all, t_range_list)
                    [_, ind3, ind4] = np.intersect1d(
                        date, t_range_final, return_indices=True
                    )

                    days_interval = [y - x for x, y in zip(ind4, ind4[1:])]
                    # get the final range
                    if (
                        t_range_final[-1].item().month == 12
                        and t_range_final[-1].item().day == 31
                    ):
                        final_timedelta = (
                            t_range_final[-1].item() - t_range_final[ind4[-1]].item()
                        )
                        final_day_interval = [final_timedelta.days]
                    else:
                        final_day_interval = [8]
                    days_interval = np.array(days_interval + final_day_interval)
                    # there may be some missing data, so that some interval will be larger than 8
                    days_interval[np.where(days_interval > 8)] = 8
                    # we use mean value rather than sum, because less error when predicting for every day
                    # for example, mean: [1, x, x, 2, x, x, 3] is obs, [1, 1, 1, 2, 2, 2, 3] is pred,
                    # sum: [3, x, x, 6, x, x, 9] is obs, [1, 1, 1, 2, 2, 2, 3] is pred
                    # the final day's error is significant when using sum
                    # although a better way is to extend [1, 1, 1, 2, 2, 2, 3] to [1, 1, 1, 2, 2, 2, 3, 3, 3]
                    y[k, ind4, j] = (
                        et_data["ET(kg/m^2/8day)"][ind3] / days_interval
                    )
                    # More notice: it is only for unified process to divide by 35.314666721489
                    # notice the value's unit is kg/m2/8d and has a scale factor 0.1
                    # more information can be seen here: https://www.ntsg.umt.edu/project/modis/mod16.php
                    # it says: "The users should multiply 0.1 to get the real ET/PET values in mm/8day or mm/month"
                else:
                    # only one streamflow type: Q
                    flow_file = os.path.join(
                        self.data_source_description["CAMELS_FLOW_DIR"],
                        gage_id_lst[k] + ".csv",
                    )
                    flow_data = pd.read_csv(flow_file, sep=",")
                    date = pd.to_datetime(flow_data["DATE"]).values.astype(
                        "datetime64[D]"
                    )
                    [c, ind1, ind2] = np.intersect1d(
                        date, t_range_list, return_indices=True
                    )
                    y[k, ind2, j] = flow_data["Q"].values[ind1]
        return y

    def read_relevant_cols(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        forcing_type="daymet",
    ) -> np.array:
        """
        Read forcing data

        Parameters
        ----------
        gage_id_lst
            station ids
        t_range
            the time range, for example, ["1990-01-01", "2000-01-01"]
        var_lst
            forcing variable types
        forcing_type
            only for CAMELS-US, don't care it
        Returns
        -------
        np.array
            forcing data
        """
        t_range_list = hydro_utils.t_range_days(t_range)
        nt = t_range_list.shape[0]
        x = np.full([len(gage_id_lst), nt, len(var_lst)], np.nan)
        for k in tqdm(range(len(gage_id_lst)), desc="Read forcing data of CAMELS-CC"):
            forcing_file = os.path.join(
                self.data_source_description["CAMELS_FORCING_DIR"],
                gage_id_lst[k] + "_lump_era5_land_forcing.txt",
            )
            forcing_data = pd.read_csv(forcing_file, sep=" ")
            df_date = forcing_data[["Year", "Mnth", "Day"]]
            df_date.columns = ["year", "month", "day"]
            date = pd.to_datetime(df_date).values.astype("datetime64[D]")

            [c, ind1, ind2] = np.intersect1d(date, t_range_list, return_indices=True)
            for j in range(len(var_lst)):
                if "evaporation" in var_lst[j]:
                    # evaporation value are all negative (maybe upward flux is marked as negative)
                    x[k, ind2, j] = forcing_data[var_lst[j]].values[ind1] * -1 * 1e3
                    # unit of prep and pet is m, tran them to mm
                elif "precipitation" in var_lst[j]:
                    prcp = forcing_data[var_lst[j]].values
                    # there are a few negative values for prcp, set them 0
                    prcp[prcp < 0] = 0.0
                    x[k, ind2, j] = prcp[ind1] * 1e3
                else:
                    x[k, ind2, j] = forcing_data[var_lst[j]].values[ind1]
        return x

    def read_attr_all(self):
        data_folder = self.data_source_description["CAMELS_ATTR_DIR"]
        key_lst = self.data_source_description["CAMELS_ATTR_KEY_LST"]
        f_dict = dict()  # factorize dict
        var_dict = dict()
        var_lst = list()
        out_lst = list()
        gage_dict = self.camels_sites
        camels_str = ""
        sep_ = ","
        for key in key_lst:
            data_file = os.path.join(data_folder, key + ".csv")
            data_temp = pd.read_csv(data_file, sep=sep_)
            var_lst_temp = list(data_temp.columns[1:])
            var_dict[key] = var_lst_temp
            var_lst.extend(var_lst_temp)
            k = 0
            gage_id_key = "gage_id"
            n_gage = len(gage_dict[gage_id_key].values)
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

    def read_constant_cols(
        self, gage_id_lst=None, var_lst=None, is_return_dict=False
    ) -> Union[tuple, np.array]:
        """
        Read Attributes data

        Parameters
        ----------
        gage_id_lst
            station ids
        var_lst
            attribute variable types
        is_return_dict
            if true, return var_dict and f_dict for CAMELS_US
        Returns
        -------
        Union[tuple, np.array]
            if attr var type is str, return factorized data.
            When we need to know what a factorized value represents, we need return a tuple;
            otherwise just return an array
        """
        attr_all, var_lst_all, var_dict, f_dict = self.read_attr_all()
        ind_var = [var_lst_all.index(var) for var in var_lst]
        id_lst_all = self.read_object_ids()
        # Notice the sequence of station ids ! Some id_lst_all are not sorted, so don't use np.intersect1d
        ind_grid = [id_lst_all.tolist().index(tmp) for tmp in gage_id_lst]
        temp = attr_all[ind_grid, :]
        out = temp[:, ind_var]
        if is_return_dict:
            return out, var_dict, f_dict
        else:
            return out

    def read_basin_area(self, object_ids) -> np.array:
        return self.read_constant_cols(object_ids, ["Area"], is_return_dict=False)

    def read_mean_prep(self, object_ids) -> np.array:
        return self.read_constant_cols(object_ids, ["p_mean"], is_return_dict=False)
