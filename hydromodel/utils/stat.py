import numpy as np
import scipy.stats
from scipy.stats import wilcoxon


def KGE(simulation, evaluation):
    """
    Kling Gupta Efficiency (Gupta et al., 2009, http://dx.doi.org/10.1016/j.jhydrol.2009.08.003)
    input:
        simulated
        observed
    output:
        KGE: Kling Gupta Efficiency
    """
    r = np.corrcoef(evaluation, simulation)[0, 1]
    alpha = np.std(simulation) / np.std(evaluation)
    beta = np.mean(simulation) / np.mean(evaluation)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge


def statError(evaluation, simulation):
    ngrid, nt = simulation.shape
    # Bias
    Bias = np.nanmean(simulation - evaluation, axis=1)
    # RMSE
    RMSE = np.sqrt(np.nanmean((simulation - evaluation) ** 2, axis=1))
    # ubRMSE
    simulationMean = np.tile(np.nanmean(simulation, axis=1), (nt, 1)).transpose()
    evaluationMean = np.tile(np.nanmean(evaluation, axis=1), (nt, 1)).transpose()
    simulationAnom = simulation - simulationMean
    evaluationAnom = evaluation - evaluationMean
    ubRMSE = np.sqrt(np.nanmean((simulationAnom - evaluationAnom) ** 2, axis=1))
    # rho R2 NSE
    Corr = np.full(ngrid, np.nan)
    R2 = np.full(ngrid, np.nan)
    NSE = np.full(ngrid, np.nan)
    KGe = np.full(ngrid, np.nan)
    PBiaslow = np.full(ngrid, np.nan)
    PBiashigh = np.full(ngrid, np.nan)
    PBias = np.full(ngrid, np.nan)
    num_lowevaluation_zero = 0
    for k in range(0, ngrid):
        x = simulation[k, :]
        y = evaluation[k, :]
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
            simulation_sort = np.sort(xx)
            evaluation_sort = np.sort(yy)
            indexlow = round(0.3 * len(simulation_sort))
            indexhigh = round(0.98 * len(simulation_sort))
            lowsimulation = simulation_sort[:indexlow]
            highsimulation = simulation_sort[indexhigh:]
            lowevaluation = evaluation_sort[:indexlow]
            highevaluation = evaluation_sort[indexhigh:]
            if np.sum(lowevaluation) == 0:
                num_lowevaluation_zero = num_lowevaluation_zero + 1
            PBiaslow[k] = np.sum(lowsimulation - lowevaluation) / np.sum(lowevaluation) * 100
            PBiashigh[k] = np.sum(highsimulation - highevaluation) / np.sum(highevaluation) * 100
            outDict = dict(Bias=Bias, RMSE=RMSE, ubRMSE=ubRMSE, Corr=Corr, R2=R2, NSE=NSE, KGE=KGe,
                           FHV=PBiashigh, FLV=PBiaslow)
    # hydro_logger.debug("The CDF of BFLV will not reach 1.0 because some basins havye all zero flow observations for the "
    #                    "30% low flow interval, the percent bias can be infinite\n" + "The number of these cases is "
    #                    + str(num_lowevaluation_zero))
    return outDict


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