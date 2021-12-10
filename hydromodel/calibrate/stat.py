import numpy as np
import scipy.stats

from hydromodel.utils.hydro_utils import hydro_logger


def KGE(xs, xo) -> float:
    """
    Kling Gupta Efficiency (Gupta et al., 2009, http://dx.doi.org/10.1016/j.jhydrol.2009.08.003)

    Parameters
    ----------
    xs
        simulated
    xo
        observed

    Returns
    -------
    float
        KGE: Kling Gupta Efficiency
    """
    r = np.corrcoef(xo, xs)[0, 1]
    alpha = np.std(xs) / np.std(xo)
    beta = np.mean(xs) / np.mean(xo)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge


def NSE(xs, xo) -> float:
    """
    Calculate NSE

    Parameters
    ----------
    xs
        simulated
    xo
        observed

    Returns
    -------
    float
        NSE: Nash-Sutcliffe model efficiency coefficient
    """
    x_mean = xo.mean()
    SST = np.sum((xo - x_mean) ** 2)
    SSRes = np.sum((xo - xs) ** 2)
    nse = 1 - SSRes / SST
    return nse


def statNse(target, pred):
    """
    Calculate NSE for two dimensional array

    Parameters
    ----------
    target
        observation
    pred
        prediction

    Returns
    -------
    np.array
        one NSE for one element in the 1-dim result
    """
    ngrid, nt = pred.shape
    NSe = np.full(ngrid, np.nan)
    for k in range(0, ngrid):
        x = pred[k, :]
        y = target[k, :]
        ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
        if ind.shape[0] > 0:
            xx = x[ind]
            yy = y[ind]
            NSe[k] = NSE(xx, yy)
    return NSe


def statRmse(target, pred, axis=0):
    """
    Calculate RMSE for multi-dim arrays

    Parameters
    ----------
    target
        observation
    pred
        prediction
    axis
        calculate through which axis

    Returns
    -------
    np.array
        RMSE
    """
    return np.sqrt(np.nanmean((pred - target) ** 2, axis=axis))


def statError(target, pred):
    """
    Calculate multiple statistics indicators:

    Parameters
    ----------
    target
        observation
    pred
        prediction

    Returns
    -------
    dict
        (Bias=Bias, RMSE=RMSE, ubRMSE=ubRMSE, Corr=Corr, R2=R2, NSE=NSE, KGE=KGe, FHV=PBiashigh, FLV=PBiaslow)
    """
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
