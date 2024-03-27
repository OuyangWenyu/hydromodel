"""
Author: Wenyu Ouyang
Date: 2022-08-06 18:39:15
LastEditTime: 2024-03-27 09:41:06
LastEditors: Wenyu Ouyang
Description: We want to build a trainer class for calibration but not done yet.
FilePath: \hydro-model-xaj\hydromodel\trainers\train.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""

import matplotlib
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))

matplotlib.use("Agg")
exp = "exp61561"
reps = [5000, 10000]
ngs = [1000, 2000]
book = "HF"
source = "sources"


class Trainer:
    # TODO: build a trainer class for calibration
    def __init__(self, exp, book, source, rep, ngs):
        self.exp = exp
        self.warmup_length = 365
        self.model = {
            "name": "xaj_mz",
            "source_type": source,
            "source_book": book,
        }
        self.algorithm = {
            "name": "SCE_UA",
            "random_seed": 1234,
            "rep": rep,
            "ngs": ngs,
            "kstop": int(rep / 5),
            "peps": 0.001,
            "pcento": 0.001,
        }
        self.comment = "{}{}rep{}ngs{}".format(book, source, rep, ngs)


for i in range(len(reps)):
    for j in range(len(ngs)):
        rep = reps[i]
        ng = ngs[j]
        xaj_calibrate = Trainer(exp, book, source, rep, ng)
        # evaluate(xaj_calibrate)
