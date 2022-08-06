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


def chose_data_by_period(data: np.array, all_periods: list, a_period: list):
    """
    Parameters
    ----------
    data
        a dataframe of pandas which contains "date" column
    all_periods
        a list including all dates of all data, such as ["2000-01-01","2000-01-02","2000-01-03",...]
    a_period
        a list including start and end date, such as ["2000-01-01", "2010-01-01"]

    Returns
    -------

    """
    t_range_train = hydro_utils.t_range_days(a_period)
    [_, ind1, ind2] = np.intersect1d(all_periods, t_range_train, return_indices=True)
    chosen_data = data[ind1]
    return chosen_data
