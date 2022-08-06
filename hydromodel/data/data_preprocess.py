import numpy as np
import pandas as pd

from hydromodel.utils import hydro_utils

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


def split_train_test(data, train_period, test_period):
    """

    Parameters
    ----------
    data
    train_period
    test_period

    Returns
    -------

    """
    return


def chose_data_by_period(data: pd.DataFrame, a_period: list):
    """
    TODO: not finished

    Parameters
    ----------
    data
        a dataframe of pandas which contains "date" column
    a_period
        a list including start and end date, such as ["2000-01-01", "2010-01-01"]

    Returns
    -------

    """
    date = pd.to_datetime(data['date']).values.astype('datetime64[D]')
    t_range_train = hydro_utils.t_range_days(a_period)
    [C, ind1, ind2] = np.intersect1d(date, t_range_train, return_indices=True)
    data_train = data[ind1]
    return data_train
