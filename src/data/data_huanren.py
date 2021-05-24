"""a data downloader and formatter for NID dataset"""
import collections
import os

import numpy as np

from src.data.data_base import DatasetBase


class HuanRen(DatasetBase):
    """Dataset of HuanRen Basin"""

    def __init__(self, data_path, download=False):
        super().__init__(data_path)
        self.dataset_description = self.set_dataset_describe()
        if download:
            self.download_dataset()

    def get_name(self):
        return "HuanRen"

    def set_dataset_describe(self):
        huanren_db = self.dataset_dir
        huanren_flood_dir = os.path.join(huanren_db, "hr-73-floods")
        huanren_constant_cols = ["BASIN_AREA/km^2"]
        # TODO: the basin area is not set yet, ...
        huanren_constant_data = [10400]
        return collections.OrderedDict(HUANREN_DIR=huanren_db, HUANREN_FLOOD_DIR=huanren_flood_dir,
                                       HUANREN_CONSTANT_COLS=huanren_constant_cols,
                                       HUANREN_CONSTANT_DATA=huanren_constant_data)

    def download_dataset(self):
        print("Please manually put the dataset in the data/ directory")
        if not os.path.isdir(self.dataset_description["HUANREN_FLOOD_DIR"]):
            raise NotADirectoryError("No FLOOD DATA for HuanRen!!!")

    def read_object_ids(self, object_params=None) -> np.array:
        pass

    def read_target_cols(self, object_ids=None, t_range_list=None, target_cols=None, **kwargs) -> np.array:
        # no target cols now
        pass

    def read_relevant_cols(self, object_ids=None, t_range_list=None, relevant_cols=None, **kwargs) -> np.array:
        # no relevant cols now
        pass

    def read_constant_cols(self, object_ids=None, constant_cols=None, **kwargs) -> np.array:
        constant_cols_arr = np.array(constant_cols)
        all_constant_cols_arr = np.array(self.dataset_description["HUANREN_CONSTANT_COLS"])
        assert constant_cols_arr in all_constant_cols_arr
        index = np.array([np.where(all_constant_cols_arr == i) for i in constant_cols_arr]).flatten()
        return np.array(self.dataset_description["HUANREN_CONSTANT_DATA"])[index]

    def read_other_cols(self, object_ids=None, other_cols=None, **kwargs):
        pass

    def get_constant_cols(self):
        pass

    def get_relevant_cols(self):
        pass

    def get_target_cols(self):
        pass

    def get_other_cols(self):
        pass
