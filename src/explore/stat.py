import itertools

import numpy as np
import scipy.stats
from scipy.stats import wilcoxon

from src.utils.hydro_utils import hydro_logger


def KGE(xs, xo):
    """
    Kling Gupta Efficiency (Gupta et al., 2009, http://dx.doi.org/10.1016/j.jhydrol.2009.08.003)
    input:
        xs: simulated
        xo: observed
    output:
        KGE: Kling Gupta Efficiency
    """
    r = np.corrcoef(xo, xs)[0, 1]
    alpha = np.std(xs) / np.std(xo)
    beta = np.mean(xs) / np.mean(xo)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge


def statError(target, pred):
    ngrid, nt = pred.shape
    # Bias
    Bias = np.nanmean(pred - target, axis=1)
    # RMSE
    RMSE = np.sqrt(np.nanmean((pred - target) ** 2, axis=1))
    # ubRMSE
    predMean = np.tile(np.nanmean(pred, axis=1), (nt, 1)).transpose()
    targetMean = np.tile(np.nanmean(target, axis=1), (nt, 1)).transpose()
    predAnom = pred - predMean
    targetAnom = target - targetMean
    ubRMSE = np.sqrt(np.nanmean((predAnom - targetAnom) ** 2, axis=1))
    # rho R2 NSE
    Corr = np.full(ngrid, np.nan)
    R2 = np.full(ngrid, np.nan)
    NSE = np.full(ngrid, np.nan)
    KGe = np.full(ngrid, np.nan)
    PBiaslow = np.full(ngrid, np.nan)
    PBiashigh = np.full(ngrid, np.nan)
    PBias = np.full(ngrid, np.nan)
    num_lowtarget_zero = 0
    for k in range(0, ngrid):
        x = pred[k, :]
        y = target[k, :]
        ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
        if ind.shape[0] > 0:
            xx = x[ind]
            yy = y[ind]
            # percent bias
            PBias[k] = np.sum(xx - yy) / np.sum(yy) * 100
            if ind.shape[0] > 1:
                # Theoretically at least two points for correlation
                Corr[k] = scipy.stats.pearsonr(xx, yy)[0]
                yymean = yy.mean()
                SST = np.sum((yy - yymean) ** 2)
                SSReg = np.sum((xx - yymean) ** 2)
                SSRes = np.sum((yy - xx) ** 2)
                R2[k] = 1 - SSRes / SST
                NSE[k] = 1 - SSRes / SST
                KGe[k] = KGE(xx, yy)
            # FHV the peak flows bias 2%
            # FLV the low flows bias bottom 30%, log space
            pred_sort = np.sort(xx)
            target_sort = np.sort(yy)
            indexlow = round(0.3 * len(pred_sort))
            indexhigh = round(0.98 * len(pred_sort))
            lowpred = pred_sort[:indexlow]
            highpred = pred_sort[indexhigh:]
            lowtarget = target_sort[:indexlow]
            hightarget = target_sort[indexhigh:]
            if np.sum(lowtarget) == 0:
                num_lowtarget_zero = num_lowtarget_zero + 1
            PBiaslow[k] = np.sum(lowpred - lowtarget) / np.sum(lowtarget) * 100
            PBiashigh[k] = np.sum(highpred - hightarget) / np.sum(hightarget) * 100
            outDict = dict(Bias=Bias, RMSE=RMSE, ubRMSE=ubRMSE, Corr=Corr, R2=R2, NSE=NSE, KGE=KGe,
                           FHV=PBiashigh, FLV=PBiaslow)
    hydro_logger.debug("The CDF of BFLV will not reach 1.0 because some basins have all zero flow observations for the "
                       "30% low flow interval, the percent bias can be infinite\n" + "The number of these cases is "
                       + str(num_lowtarget_zero))
    return outDict


def cal_4_stat_inds(b):
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]


def cal_stat(x):
    a = x.flatten()
    b = a[~np.isnan(a)]
    if b.size == 0:
        # if b is [], then give it a 0 value
        b = np.array([0])
    return cal_4_stat_inds(b)


def cal_stat_gamma(x):
    """for daily streamflow and precipitation"""
    a = x.flatten()
    b = a[~np.isnan(a)]  # kick out Nan
    b = np.log10(np.sqrt(b) + 0.1)  # do some tranformation to change gamma characteristics
    return cal_4_stat_inds(b)


def cal_stat_basin_norm(x, basinarea, meanprep):
    """for daily streamflow normalized by basin area and precipitation
    basinarea = readAttr(gageDict['id'], ['area_gages2'])
    meanprep = readAttr(gageDict['id'], ['p_mean'])
    """
    # meanprep = readAttr(gageDict['id'], ['q_mean'])
    temparea = np.tile(basinarea, (1, x.shape[1]))
    tempprep = np.tile(meanprep, (1, x.shape[1]))
    flowua = (x * 0.0283168 * 3600 * 24) / (
            (temparea * (10 ** 6)) * (tempprep * 10 ** (-3)))  # unit (m^3/day)/(m^3/day)
    return cal_stat_gamma(flowua)


def trans_norm(x, var_lst, stat_dict, *, to_norm):
    """normalization，including denormalization code
    :parameter
        x：2d or 3d data
            2d：1st-sites，2nd-var type
            3d：1st-sites，2nd-time, 3rd-var type
    """
    if type(var_lst) is str:
        var_lst = [var_lst]
    out = np.zeros(x.shape)
    for k in range(len(var_lst)):
        var = var_lst[k]
        stat = stat_dict[var]
        if to_norm is True:
            if len(x.shape) == 3:
                out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]
            elif len(x.shape) == 2:
                out[:, k] = (x[:, k] - stat[2]) / stat[3]
        else:
            if len(x.shape) == 3:
                out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
            elif len(x.shape) == 2:
                out[:, k] = x[:, k] * stat[3] + stat[2]
    return out


def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n + 1) / n
    return (x, y)


def wilcoxon_t_test(xs, xo):
    """Wilcoxon t test"""
    diff = xs - xo  # same result when using xo-xs
    w, p = wilcoxon(diff)
    return w, p


def wilcoxon_t_test_for_lst(x_lst, rnd_num=2):
    """Wilcoxon t test for every two array in a 2-d array"""
    arr_lst = np.asarray(x_lst)
    w, p = [], []
    arr_lst_pair = list(itertools.combinations(arr_lst, 2))
    for arr_pair in arr_lst_pair:
        wi, pi = wilcoxon_t_test(arr_pair[0], arr_pair[1])
        w.append(round(wi, rnd_num))
        p.append(round(pi, rnd_num))
    return w, p


def cal_fdc(data: np.array, quantile_num=100):
    # data = n_grid * n_day
    n_grid, n_day = data.shape
    fdc = np.full([n_grid, quantile_num], np.nan)
    for ii in range(n_grid):
        temp_data0 = data[ii, :]
        temp_data = temp_data0[~np.isnan(temp_data0)]
        # deal with no data case for some gages
        if len(temp_data) == 0:
            temp_data = np.full(n_day, 0)
        # sort from large to small
        temp_sort = np.sort(temp_data)[::-1]
        # select quantile_num quantile points
        n_len = len(temp_data)
        ind = (np.arange(quantile_num) / quantile_num * n_len).astype(int)
        fdc_flow = temp_sort[ind]
        if len(fdc_flow) != quantile_num:
            raise Exception('unknown assimilation variable')
        else:
            fdc[ii, :] = fdc_flow

    return fdc