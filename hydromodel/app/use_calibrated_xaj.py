"""use calibrated xaj model to predict"""
import os.path

import numpy as np
import pandas as pd
import datetime as dt
import definitions
from hydromodel.data.data_preprocess import chose_data_by_period
from hydromodel.models.xaj import xaj
from hydromodel.utils import hydro_utils

prameter_file = os.path.join(definitions.ROOT_DIR, "result", "MZ", "basins_params.npy")
param_values = np.load(prameter_file)
data_dir = os.path.join(definitions.ROOT_DIR, "hydromodel", "example")
basins_id = ["60650", "60668", "61239", "61277", "61561", "61716", "62618", "63002",
             "63007", "63486", "63490", "90813", "92353", "92354", "94470", "94560", "94850", "95350"]
test_period = ["2019-10-01", "2021-10-01"]
es = []
for i in range(len(basins_id)):
    one_dir = os.path.join(definitions.ROOT_DIR, "hydromodel", "example", basins_id[i])
    np_file = os.path.join(one_dir, "basins_lump_p_pe_q.npy")
    data = np.load(np_file)
    json_file = os.path.join(one_dir, "data_info.json")
    all_period = hydro_utils.unserialize_json(json_file)["time"]
    all_periods = [dt.datetime.strptime(a_date, "%Y-%m-%d") for a_date in all_period]
    test_data = chose_data_by_period(data, all_periods, test_period)
    qsim, e = xaj(
        test_data[:, :, 0:2],
        param_values[:, i:i + 1].T,
        warmup_length=365,
        route_method="MZ",
    )
    es.append(e.flatten())
all_periods = hydro_utils.t_range_days(["2020-09-30", "2021-10-01"])
df_e = pd.DataFrame(np.array(es), columns=all_periods, index=basins_id)
save_path = os.path.join(definitions.ROOT_DIR, "result", "MZ", "et.txt")
df_e.to_csv(save_path)
