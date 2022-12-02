"""
Author: Wenyu Ouyang
Date: 2022-12-02 15:19:40
LastEditTime: 2022-12-02 17:45:49
LastEditors: Wenyu Ouyang
Description: Some temp script
FilePath: \hydro-model-xaj\hydromodel\app\temp.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
import definitions
from hydromodel.utils import hydro_utils

data_dir = os.path.join(definitions.ROOT_DIR, "hydromodel", "example", "exp003")
data_file = os.path.join(data_dir, "basins_lump_p_pe_q.npy")
data_info_file = os.path.join(data_dir, "data_info.json")
data_info = hydro_utils.unserialize_json_ordered(data_info_file)
p_pet_q = hydro_utils.unserialize_numpy(data_file)
for i in range(p_pet_q.shape[1]):
    pd.DataFrame(
        p_pet_q[:, i, :],
        index=pd.to_datetime(data_info["time"]).values.astype("datetime64[D]"),
    ).to_csv(
        os.path.join(data_dir, "basin_{}.csv".format(data_info["basin"][i])),
        index_label="time",
        header=data_info["variable"],
    )
