import os
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
path = Path(ROOT_DIR)
DATASET_DIR = os.path.join(path.parent.parent.absolute(), "data")
HUANREN_DIR = os.path.join(DATASET_DIR, "huanren")
CAMELS_DIR = os.path.join(DATASET_DIR, "camels")
print("Please Check your directory:")
print("ROOT_DIR of the repo: ", ROOT_DIR)
print("DATASET_DIR of the repo: ", DATASET_DIR)
print("HUANREN_DIR of the repo: ", HUANREN_DIR)
print("CAMELS_DIR of the repo: ", CAMELS_DIR)
