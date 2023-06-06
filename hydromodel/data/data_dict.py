"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2023-06-05 21:27:40
LastEditors: Wenyu Ouyang
Description: A dict used for data source and data loader
FilePath: \hydro-model-xaj\hydromodel\data\data_dict.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

from hydrodataset import Camels

from hydromodel.data.data_sets import CamelsDataset


data_sources_dict = {
    "CAMELS": Camels,
}

pytorch_dataset_dict = {
    "CamelsDataset": CamelsDataset,
}
