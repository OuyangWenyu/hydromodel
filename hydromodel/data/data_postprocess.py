import os


import numpy as np
import pandas as pd
import pathlib

import definitions
from hydromodel.utils import hydro_utils
from collections import OrderedDict
from pandas.core.frame import DataFrame
import re

test_trange=["2019-10-01", "2020-10-01"]
t_range_train = hydro_utils.t_range_days(test_trange)
print(len(t_range_train))
print(t_range_train)