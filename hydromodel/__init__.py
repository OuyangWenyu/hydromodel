"""
Author: Wenyu Ouyang
Date: 2024-02-09 15:56:48
LastEditTime: 2024-03-28 11:18:55
LastEditors: Wenyu Ouyang
Description: Top-level package for hydromodel
FilePath: \hydromodel\hydromodel\__init__.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
from pathlib import Path
from hydroutils import hydro_file
import yaml

__author__ = """Wenyu Ouyang"""
__email__ = 'wenyuouyang@outlook.com'
__version__ = '0.2.3'


CACHE_DIR = hydro_file.get_cache_dir()
SETTING_FILE = os.path.join(Path.home(), "hydro_setting.yml")


def read_setting(setting_path):
    if not os.path.exists(setting_path):
        raise FileNotFoundError(f"Configuration file not found: {setting_path}")

    with open(setting_path, "r") as file:
        setting = yaml.safe_load(file)

    example_setting = (
        "local_data_path:\n"
        "  root: 'D:\\data\\waterism' # Update with your root data directory\n"
        "  datasets-origin: 'D:\\data\\waterism\\datasets-origin' # datasets-origin is the directory you put downloaded datasets\n"
        "  datasets-interim: 'D:\\data\\waterism\\datasets-interim' # the other choice for the directory you put downloaded datasets\n"
        "  basins-origin: 'D:\\data\\waterism\\basins-origin' # the directory put your own data\n"
        "  basins-interim: 'D:\\data\\waterism\\basins-interim' # the other choice for your own data"
    )

    if setting is None:
        raise ValueError(
            f"Configuration file is empty or has invalid format.\n\nExample configuration:\n{example_setting}"
        )

    # Define the expected structure
    expected_structure = {
        "local_data_path": [
            "root",
            "datasets-origin",
            "datasets-interim",
            "basins-origin",
            "basins-interim",
        ],
    }

    # Validate the structure
    try:
        for key, subkeys in expected_structure.items():
            if key not in setting:
                raise KeyError(f"Missing required key in config: {key}")

            if isinstance(subkeys, list):
                for subkey in subkeys:
                    if subkey not in setting[key]:
                        raise KeyError(f"Missing required subkey '{subkey}' in '{key}'")
    except KeyError as e:
        raise ValueError(
            f"Incorrect configuration format: {e}\n\nExample configuration:\n{example_setting}"
        ) from e

    return setting


try:
    SETTING = read_setting(SETTING_FILE)
except ValueError as e:
    print(e)
except Exception as e:
    print(f"Unexpected error: {e}")
