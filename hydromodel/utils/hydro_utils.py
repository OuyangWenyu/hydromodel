import json
import os
import re
import zipfile
import datetime as dt, datetime
from typing import List
import pickle
from collections import OrderedDict
import numpy as np
import urllib
from urllib import parse

import requests
import matplotlib.pyplot as plt
from itertools import combinations

import threading
import functools

import tqdm

import logging


# -----------------------------------------------logger setting----------------------------------------------------
def get_hydro_logger(log_level_param):
    logger = logging.getLogger(__name__)
    # StreamHandler
    stream_handler = logging.StreamHandler()  # console stream output
    stream_handler.setLevel(level=log_level_param)
    logger.addHandler(stream_handler)
    return logger


log_level = logging.INFO
hydro_logger = get_hydro_logger(log_level)


# ------------------------------------------------progress bar----------------------------------------------------
def provide_progress_bar(
    function, estimated_time, tstep=0.2, tqdm_kwargs={}, args=[], kwargs={}
):
    """
    Tqdm wrapper for a long-running function

    Parameters
    ----------
    function
        function to run
    estimated_time
        how long you expect the function to take
    tstep
        time delta (seconds) for progress bar updates
    tqdm_kwargs
        kwargs to construct the progress bar
    args
        args to pass to the function
    kwargs
        keyword args to pass to the function

    Returns
    -------
    function
    """
    ret = [None]  # Mutable var so the function can store its return value

    def myrunner(function, ret, *args, **kwargs):
        ret[0] = function(*args, **kwargs)

    thread = threading.Thread(
        target=myrunner, args=(function, ret) + tuple(args), kwargs=kwargs
    )
    pbar = tqdm.tqdm(total=estimated_time, **tqdm_kwargs)

    thread.start()
    while thread.is_alive():
        thread.join(timeout=tstep)
        pbar.update(tstep)
    pbar.close()
    return ret[0]


def progress_wrapped(estimated_time, tstep=0.2, tqdm_kwargs={}):
    """Decorate a function to add a progress bar"""

    def real_decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return provide_progress_bar(
                function,
                estimated_time=estimated_time,
                tstep=tstep,
                tqdm_kwargs=tqdm_kwargs,
                args=args,
                kwargs=kwargs,
            )

        return wrapper

    return real_decorator


def setup_log(tag="VOC_TOPICS"):
    # create logger
    logger = logging.getLogger(tag)
    # logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    # logger.handlers = []
    logger.addHandler(ch)
    return logger


# -------------------------------------------------plot utils-------------------------------------------------------
def save_or_show_plot(file_nm: str, save: bool, save_path=""):
    if save:
        plt.savefig(os.path.join(save_path, file_nm))
    else:
        plt.show()


# ----------------------------------------------------data tools--------------------------------------------------------
def unzip_file(dataset_zip, path_unzip):
    """extract a zip file"""
    with zipfile.ZipFile(dataset_zip, "r") as zip_temp:
        zip_temp.extractall(path_unzip)


def unzip_nested_zip(dataset_zip, path_unzip):
    """Extract a zip file including any nested zip files"""
    with zipfile.ZipFile(dataset_zip, "r") as zfile:
        zfile.extractall(path=path_unzip)
    for root, dirs, files in os.walk(path_unzip):
        for filename in files:
            if re.search(r"\.zip$", filename):
                file_spec = os.path.join(root, filename)
                new_dir = os.path.join(root, filename[0:-4])
                unzip_nested_zip(file_spec, new_dir)


def zip_file_name_from_url(data_url, data_dir):
    data_url_str = data_url.split("/")
    filename = parse.unquote(data_url_str[-1])
    zipfile_path = os.path.join(data_dir, filename)
    unzip_dir = os.path.join(data_dir, filename[0:-4])
    return zipfile_path, unzip_dir


def is_there_file(zipfile_path, unzip_dir):
    """if a file has existed"""
    if os.path.isfile(zipfile_path):
        if os.path.isdir(unzip_dir):
            return True
        unzip_nested_zip(zipfile_path, unzip_dir)
        return True


def download_one_zip(data_url, data_dir):
    """download one zip file from url as data_file"""
    zipfile_path, unzip_dir = zip_file_name_from_url(data_url, data_dir)
    if not is_there_file(zipfile_path, unzip_dir):
        if not os.path.isdir(unzip_dir):
            os.mkdir(unzip_dir)
        r = requests.get(data_url, stream=True)
        with open(zipfile_path, "wb") as py_file:
            for chunk in r.iter_content(chunk_size=1024):  # 1024 bytes
                if chunk:
                    py_file.write(chunk)
        unzip_nested_zip(zipfile_path, unzip_dir), download_small_file


def download_small_zip(data_url, data_dir):
    """download zip file and unzip"""
    zipfile_path, unzip_dir = zip_file_name_from_url(data_url, data_dir)
    if not is_there_file(zipfile_path, unzip_dir):
        if not os.path.isdir(unzip_dir):
            os.mkdir(unzip_dir)
        zipfile_path, _ = urllib.request.urlretrieve(data_url, zipfile_path)
        unzip_nested_zip(zipfile_path, unzip_dir)


def download_small_file(data_url, temp_file):
    """download data from url to the temp_file"""
    r = requests.get(data_url)
    with open(temp_file, "w") as f:
        f.write(r.text)


def download_excel(data_url, temp_file):
    """download a excel file according to url"""
    if not os.path.isfile(temp_file):
        urllib.request.urlretrieve(data_url, temp_file)


def download_a_file_from_google_drive(drive, dir_id, download_dir):
    file_list = drive.ListFile(
        {"q": "'" + dir_id + "' in parents and trashed=false"}
    ).GetList()
    for file in file_list:
        print("title: %s, id: %s" % (file["title"], file["id"]))
        file_dl = drive.CreateFile({"id": file["id"]})
        print("mimetype is %s" % file_dl["mimeType"])
        if file_dl["mimeType"] == "application/vnd.google-apps.folder":
            download_dir_sub = os.path.join(download_dir, file_dl["title"])
            if not os.path.isdir(download_dir_sub):
                os.makedirs(download_dir_sub)
            download_a_file_from_google_drive(drive, file_dl["id"], download_dir_sub)
        else:
            # download
            temp_file = os.path.join(download_dir, file_dl["title"])
            if os.path.isfile(temp_file):
                print("file has been downloaded")
                continue
            file_dl.GetContentFile(os.path.join(download_dir, file_dl["title"]))
            print("Downloading file finished")


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def serialize_json_np(my_dict, my_file):
    with open(my_file, "w") as FP:
        json.dump(my_dict, FP, cls=NumpyArrayEncoder)


def serialize_json(my_dict, my_file):
    with open(my_file, "w") as FP:
        json.dump(my_dict, FP, indent=4)


def unserialize_json_ordered(my_file):
    # m_file = os.path.join(my_file, 'master.json')
    with open(my_file, "r") as fp:
        m_dict = json.load(fp, object_pairs_hook=OrderedDict)
    return m_dict


def unserialize_json(my_file):
    with open(my_file, "r") as fp:
        my_object = json.load(fp)
    return my_object


def serialize_pickle(my_object, my_file):
    f = open(my_file, "wb")
    pickle.dump(my_object, f)
    f.close()


def unserialize_pickle(my_file):
    f = open(my_file, "rb")
    my_object = pickle.load(f)
    f.close()
    return my_object


def serialize_numpy(my_array, my_file):
    np.save(my_file, my_array)


def unserialize_numpy(my_file):
    y = np.load(my_file, allow_pickle=True)
    return y


# -------------------------------------------------time & date tools--------------------------------------------------
def t2dt(t, hr=False):
    t_out = None
    if type(t) is int:
        if 30000000 > t > 10000000:
            t = dt.datetime.strptime(str(t), "%Y%m%d").date()
            t_out = t if hr is False else t.datetime()

    if type(t) is dt.date:
        t_out = t if hr is False else t.datetime()

    if type(t) is dt.datetime:
        t_out = t.date() if hr is False else t

    if t_out is None:
        raise Exception("hydroDL.utils.t2dt failed")
    return t_out


def t_range2_array(t_range, *, step=np.timedelta64(1, "D")):
    sd = t2dt(t_range[0])
    ed = t2dt(t_range[1])
    tArray = np.arange(sd, ed, step)
    return tArray


def t_range_days(t_range, *, step=np.timedelta64(1, "D")):
    sd = dt.datetime.strptime(t_range[0], "%Y-%m-%d")
    ed = dt.datetime.strptime(t_range[1], "%Y-%m-%d")
    t_array = np.arange(sd, ed, step)
    return t_array


def t_days_lst2range(t_array):
    if type(t_array[0]) == np.datetime64:
        t0 = t_array[0].astype(datetime.datetime)
        t1 = t_array[-1].astype(datetime.datetime)
    else:
        t0 = t_array[0]
        t1 = t_array[-1]
    sd = t0.strftime("%Y-%m-%d")
    ed = t1.strftime("%Y-%m-%d")
    return [sd, ed]


def t_range_years(t_range):
    """t_range is a left-closed and right-open interval, if t_range[1] is not Jan.1 then end_year should be included"""
    start_year = int(t_range[0].split("-")[0])
    end_year = int(t_range[1].split("-")[0])
    end_month = int(t_range[1].split("-")[1])
    end_day = int(t_range[1].split("-")[2])
    if end_month == 1 and end_day == 1:
        year_range_list = np.arange(start_year, end_year)
    else:
        year_range_list = np.arange(start_year, end_year + 1)
    return year_range_list


def get_year(a_time):
    if isinstance(a_time, datetime.date):
        return a_time.year
    elif isinstance(a_time, np.datetime64):
        return a_time.astype("datetime64[Y]").astype(int) + 1970
    else:
        return int(a_time[0:4])


def intersect(t_lst1, t_lst2):
    C, ind1, ind2 = np.intersect1d(t_lst1, t_lst2, return_indices=True)
    return ind1, ind2


def date_to_julian(a_time):
    if type(a_time) == str:
        fmt = "%Y-%m-%d"
        dt = datetime.datetime.strptime(a_time, fmt)
    else:
        dt = a_time
    tt = dt.timetuple()
    julian_date = tt.tm_yday
    return julian_date


def t_range_to_julian(t_range):
    t_array = t_range_days(t_range)
    t_array_str = np.datetime_as_string(t_array)
    julian_dates = [date_to_julian(a_time[0:10]) for a_time in t_array_str]
    return julian_dates


# --------------------------------------------------MATH CALCULATION---------------------------------------------------
def subset_of_dict(dict, chosen_keys):
    """make a new dict from key-values of chosen keys in a list"""
    return {key: value for key, value in dict.items() if key in chosen_keys}


def pair_comb(combine_attrs):
    if len(combine_attrs) == 1:
        values = list(combine_attrs[0].values())[0]
        key = list(combine_attrs[0].keys())[0]
        results = []
        for value in values:
            d = dict()
            d[key] = value
            results.append(d)
        return results
    items_all = list()
    for dict_item in combine_attrs:
        list_temp = list(dict_item.values())[0]
        items_all = items_all + list_temp
    all_combs = list(combinations(items_all, 2))

    def is_in_same_item(item1, item2):
        for dict_item in combine_attrs:
            list_now = list(dict_item.values())[0]
            if item1 in list_now and item2 in list_now:
                return True
        return False

    def which_dict(item):
        for dict_item in combine_attrs:
            list_now = list(dict_item.values())[0]
            if item in list_now:
                return list(dict_item.keys())[0]

    combs = [comb for comb in all_combs if not is_in_same_item(comb[0], comb[1])]
    combs_dict = [
        {which_dict(comb[0]): comb[0], which_dict(comb[1]): comb[1]} for comb in combs
    ]
    return combs_dict


def flat_data(x):
    xArrayTemp = x.flatten()
    xArray = xArrayTemp[~np.isnan(xArrayTemp)]
    xSort = np.sort(xArray)
    return xSort


def interpNan(x, mode="linear"):
    if len(x.shape) == 1:
        ngrid = 1
        nt = x.shape[0]
    else:
        ngrid, nt = x.shape
    for k in range(ngrid):
        xx = x[k, :]
        xx = interpNan1d(xx, mode)
    return x


def interpNan1d(x, mode="linear"):
    i0 = np.where(np.isnan(x))[0]
    i1 = np.where(~np.isnan(x))[0]
    if len(i1) > 0:
        if mode == "linear":
            x[i0] = np.interp(i0, i1, x[i1])
        if mode == "pre":
            x0 = x[i1[0]]
            for k in range(len(x)):
                if k in i0:
                    x[k] = x0
                else:
                    x0 = x[k]

    return x


def is_any_elem_in_a_lst(lst1, lst2, return_index=False, include=False):
    do_exist = False
    idx_lst = []
    for j in range(len(lst1)):
        if include:
            for lst2_elem in lst2:
                if lst1[j] in lst2_elem:
                    idx_lst.append(j)
                    do_exist = True
        else:
            if lst1[j] in lst2:
                idx_lst.append(j)
                do_exist = True
    if return_index:
        return do_exist, idx_lst
    return do_exist


def random_choice_no_return(arr, num_lst):
    """sampling without replacement multi-times, and the num of each time is in num_lst"""
    num_lst_arr = np.array(num_lst)
    num_sum = num_lst_arr.sum()
    if type(arr) == list:
        arr = np.array(arr)
    assert num_sum <= arr.size
    results = []
    arr_residue = np.arange(arr.size)
    for num in num_lst_arr:
        idx_chosen = np.random.choice(arr_residue.size, num, replace=False)
        chosen_idx_in_arr = np.sort(arr_residue[idx_chosen])
        results.append(arr[chosen_idx_in_arr])
        arr_residue = np.delete(arr_residue, idx_chosen)
    return results


def find_integer_factors_close_to_square_root(integer):
    start = int(np.sqrt(integer))
    factor = integer / start
    while not is_integer(factor):
        start += 1
        factor = integer / start
    return int(factor), start


def is_integer(number):
    if int(number) == number:
        return True
    else:
        return False


def random_index(ngrid, nt, dim_subset):
    batch_size, rho = dim_subset
    i_grid = np.random.randint(0, ngrid, [batch_size])
    i_t = np.random.randint(0, nt - rho, [batch_size])
    return i_grid, i_t


def flatten_list_function(input_list: List):
    return [item for sublist in input_list for item in sublist]


def nanlog(x):
    if x != x:
        return np.nan
    else:
        return np.log(x)
