"""
Author: Wenyu Ouyang
Date: 2022-10-25 21:16:22
LastEditTime: 2022-12-04 14:55:55
LastEditors: Wenyu Ouyang
Description: preprocess data for models in hydro-model-xaj
FilePath: \hydro-model-xaj\hydromodel\data\data_preprocess.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import sys
import os
from pathlib import Path
from collections import OrderedDict

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
import hydrodataset
from hydromodel.utils import hydro_utils
from hydromodel.data import camels_format_data


def trans_camels_format_to_xaj_format(
    camels_data_dir, basin_ids: list, t_range: list, json_file, npy_file
):
    """tranform data with camels format to hydro-model-xaj format

    CAMELS format could be seen here: https://gdex.ucar.edu/dataset/camels/file.html
    download basin_timeseries_v1p2_metForcing_obsFlow.zip and unzip it, you will see the format of data

    hydro-model-xaj format: see README.md file -- https://github.com/OuyangWenyu/hydro-model-xaj

    Parameters
    ----------
    camels_data_dir : str
        the directory of your CAMELS format data
    basin_ids : list
        a list of basins' ids which you choose for modeling
    t_range: list
        for example, ["2014-10-01", "2021-10-01"]
    json_file: str
        where to save the json file
    npy_file: str
        where to save the npy file
    """
    if camels_data_dir.stem == "camels_cc":
        # this is for the author's own data format, for camels we don't need this
        camels = camels_format_data.MyCamels(camels_data_dir)
        p_pet = camels.read_relevant_cols(
            gage_id_lst=basin_ids,
            t_range=t_range,
            var_lst=["total_precipitation", "potential_evaporation"],
        )
        q = camels.read_target_cols(
            gage_id_lst=basin_ids, t_range=t_range, target_cols=["Q"]
        )
    else:
        region = camels_data_dir.stem.split("_")[-1].upper()
        camels = hydrodataset.Camels(camels_data_dir, region=region)
        flow_tag = camels.get_target_cols()
        ft3persec2m3persec = 1 / 35.314666721489
        if region == "US":
            pet = camels.read_camels_us_model_output_data(
                basin_ids, t_range, var_lst=["PET"]
            )
            p = camels.read_relevant_cols(
                gage_id_lst=basin_ids,
                t_range=t_range,
                var_lst=["prcp"],
            )
            p_pet = np.concatenate([p, pet], axis=2)
        else:
            raise NotImplementedError("Only CAMELS-US is supported now.")
        q = camels.read_target_cols(
            gage_id_lst=basin_ids, t_range=t_range, target_cols=flow_tag
        )
        # TODO: camels's streamflow data is in ft3/s, need refactor to unify the unit
        q = q * ft3persec2m3persec
    # generally streamflow's unit is m3/s, we transform it to mm/day
    # basin areas also should be saved,
    # we will use it to transform streamflow's unit to m3/s after we finished predicting
    basin_area = camels.read_basin_area(basin_ids)
    # 1 km2 = 10^6 m2
    km2tom2 = 1e6
    # 1 m = 1000 mm
    mtomm = 1000
    # 1 day = 24 * 3600 s
    daytos = 24 * 3600
    temparea = np.tile(basin_area, (1, q.shape[1]))
    q = np.expand_dims(q[:, :, 0] / (temparea * km2tom2) * mtomm * daytos, axis=2)

    date_lst = [str(t)[:10] for t in hydro_utils.t_range_days(t_range)]
    data_info = OrderedDict(
        {
            "time": date_lst,
            "basin": basin_ids,
            "variable": ["prcp(mm/day)", "pet(mm/day)", "streamflow(mm/day)"],
            "area": basin_area.flatten().tolist(),
        }
    )
    hydro_utils.serialize_json(data_info, json_file)
    hydro_utils.serialize_numpy(
        np.swapaxes(np.concatenate((p_pet, q), axis=2), 0, 1), npy_file
    )


def split_train_test(json_file, npy_file, train_period, test_period):
    """
    Split all data to train and test parts with same format

    Parameters
    ----------
    json_file
        dict file of all data
    npy_file
        numpy file of all data
    train_period
        training period
    test_period
        testing period

    Returns
    -------
    None
    """
    data = hydro_utils.unserialize_numpy(npy_file)
    data_info = hydro_utils.unserialize_json(json_file)
    date_lst = pd.to_datetime(data_info["time"]).values.astype("datetime64[D]")
    t_range_train = hydro_utils.t_range_days(train_period)
    t_range_test = hydro_utils.t_range_days(test_period)
    _, ind1, ind2 = np.intersect1d(date_lst, t_range_train, return_indices=True)
    _, ind3, ind4 = np.intersect1d(date_lst, t_range_test, return_indices=True)
    data_info_train = OrderedDict(
        {
            "time": [str(t)[:10] for t in hydro_utils.t_range_days(train_period)],
            "basin": data_info["basin"],
            "variable": data_info["variable"],
            "area": data_info["area"],
        }
    )
    data_info_test = OrderedDict(
        {
            "time": [str(t)[:10] for t in hydro_utils.t_range_days(test_period)],
            "basin": data_info["basin"],
            "variable": data_info["variable"],
            "area": data_info["area"],
        }
    )
    # unify it with cross validation case, so we add a 'fold0'
    train_json_file = json_file.parent.joinpath(json_file.stem + "_fold0_train.json")
    train_npy_file = json_file.parent.joinpath(npy_file.stem + "_fold0_train.npy")
    hydro_utils.serialize_json(data_info_train, train_json_file)
    hydro_utils.serialize_numpy(data[ind1, :, :], train_npy_file)
    test_json_file = json_file.parent.joinpath(json_file.stem + "_fold0_test.json")
    test_npy_file = json_file.parent.joinpath(npy_file.stem + "_fold0_test.npy")
    hydro_utils.serialize_json(data_info_test, test_json_file)
    hydro_utils.serialize_numpy(data[ind3, :, :], test_npy_file)


def cross_valid_data(json_file, npy_file, period, warmup, cv_fold, time_unit="D"):
    """
    Split all data to train and test parts with same format

    Parameters
    ----------
    json_file
        dict file of all data
    npy_file
        numpy file of all data
    period
        the whole period
    warmup
        warmup period length
    cv_fold
        number of folds

    Returns
    -------
    None
    """
    data = hydro_utils.unserialize_numpy(npy_file)
    data_info = hydro_utils.unserialize_json(json_file)
    date_lst = pd.to_datetime(data_info["time"]).values.astype("datetime64[D]")
    date_wo_warmup = date_lst[warmup:]
    kf = KFold(n_splits=cv_fold, shuffle=False)
    for i, (train, test) in enumerate(kf.split(date_wo_warmup)):
        train_period = date_wo_warmup[train]
        test_period = date_wo_warmup[test]
        train_period_warmup = np.arange(
            train_period[0] - np.timedelta64(warmup, time_unit), train_period[0]
        )
        test_period_warmup = np.arange(
            test_period[0] - np.timedelta64(warmup, time_unit), test_period[0]
        )
        t_range_train = np.concatenate((train_period_warmup, train_period))
        t_range_test = np.concatenate((test_period_warmup, test_period))
        _, ind1, ind2 = np.intersect1d(date_lst, t_range_train, return_indices=True)
        _, ind3, ind4 = np.intersect1d(date_lst, t_range_test, return_indices=True)
        data_info_train = OrderedDict(
            {
                "time": [
                    np.datetime_as_string(d, unit=time_unit) for d in t_range_train
                ],
                "basin": data_info["basin"],
                "variable": data_info["variable"],
                "area": data_info["area"],
            }
        )
        data_info_test = OrderedDict(
            {
                "time": [
                    np.datetime_as_string(d, unit=time_unit) for d in t_range_test
                ],
                "basin": data_info["basin"],
                "variable": data_info["variable"],
                "area": data_info["area"],
            }
        )
        train_json_file = json_file.parent.joinpath(
            json_file.stem + "_fold" + str(i) + "_train.json"
        )
        train_npy_file = json_file.parent.joinpath(
            npy_file.stem + "_fold" + str(i) + "_train.npy"
        )
        hydro_utils.serialize_json(data_info_train, train_json_file)
        hydro_utils.serialize_numpy(data[ind1, :, :], train_npy_file)
        test_json_file = json_file.parent.joinpath(
            json_file.stem + "_fold" + str(i) + "_test.json"
        )
        test_npy_file = json_file.parent.joinpath(
            npy_file.stem + "_fold" + str(i) + "_test.npy"
        )
        hydro_utils.serialize_json(data_info_test, test_json_file)
        hydro_utils.serialize_numpy(data[ind3, :, :], test_npy_file)
