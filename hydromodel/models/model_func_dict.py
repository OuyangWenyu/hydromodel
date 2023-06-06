"""
Author: Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2023-06-05 20:59:36
LastEditors: Wenyu Ouyang
Description: Dicts including models, losses, and optims
FilePath: \hydro-model-xaj\hydromodel\models\model_dict_function.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
from torch.optim import Adam
from torch.nn import MSELoss
from hydromodel.models.dpl_basic import LSTMLinear

pytorch_model_dict = {
    "LSTMLinear": LSTMLinear,
}

pytorch_loss_dict = {
    "MSELoss": MSELoss,
}

pytorch_opt_dict = {"Adam": Adam}
