"""新安江模型核心计算程序"""
import pandas as pd


def initial_soil_moisture(xaj_params, w0_initial, day_precip, day_evapor):
    """计算初始土壤含水量。

    Parameters
    ----------
    xaj_params: 新安江模型参数
    w0_initial: 流域初始土壤含水量，包括上层wu0，下层wl0和深层wd0
    day_precip: 流域日降雨
    day_evapor: 流域日蒸发

    Returns
    -------
    out : array
        w0:模型计算使用的流域前期土壤含水量
    """

    w0 = w0_initial
    for i in len(day_evapor):
        eu, el, ed, wu, wl, wd = evapor_single_period(xaj_params, w0, day_precip[i], day_evapor[i])
        w0 = pd.Series([wu, wl, wd], index=['WU', 'WL', 'WD'])
    return w0


def evapor_single_period(evapor_params, initial_conditions, precip, evapor):
    """每时段蒸发计算，三层蒸发模型

    Parameters
    ----------
    evapor_params: 三层蒸发模型计算所需参数
    initial_conditions: 三层蒸发模型计算所需初始条件
    precip: 流域时段面平均降雨量
    evapor: 流域时段蒸散发

    Returns
    -------
    out : float,float,float
        eu,el,ed:流域时段三层蒸散发
        wu,wl,wd:流域时段三层含水量
    """
    # K: 蒸发系数
    # IMP: 流域不透水系数
    # B: 流域蓄水容量曲线的方次
    # WM: 流域平均蓄水容量
    # WUM: 流域上层土壤平均蓄水容量
    # WLM: 流域下层土壤平均蓄水容量
    # C: 深层蒸散发折算系数
    k = evapor_params['K']
    imp = evapor_params['IMP']
    b = evapor_params['B']
    wm = evapor_params['WM']
    wum = evapor_params['WUM']
    wlm = evapor_params['WLM']
    c = evapor_params['C']
    # evapor是实际从蒸发皿测得的水面蒸发数据，也即水面蒸发能力。乘以系数k得到流域蒸散发能力em
    # e就是em，值得注意的是最后流域实际蒸发量不一定和e相等？
    e = k * evapor
    p = precip

    wu0 = initial_conditions['WU']
    wl0 = initial_conditions['WL']
    wd0 = initial_conditions['WD']

    wdm = wm - wum - wlm

    if p - e > 0:
        # 当pe>0时，说明流域蓄水量增加，首先补充上层土壤水，然后补充下层，最后补充深层
        r = runoff_generation_single_period(evapor_params, initial_conditions, precip, evapor)
        eu = e
        el = 0
        ed = 0
        if wu0 + p - e - r < wum:
            wu = wu0 + p - e - r
            wl = wl0
            wd = wd0
        else:
            if wu0 + wl0 + p - e - r < wum + wlm:
                wu = wum
                wl = wu0 + wl0 + p - e - r - wum
                wd = wd0
            else:
                wu = wum
                wl = wlm
                wd = wm + p - e - r - wu - wl
    else:
        if wu0 + p - e >= 0:
            eu = e
            el = 0
            ed = 0
            wu = wu0 + p - e
            wl = wl0
            wd = wd0
        else:
            eu = wu0 + p
            wu = 0
            if wl0 >= c * wlm:
                el = (e - eu) * wl0 / wlm
                ed = 0
                wl = wl0 - el
                wd = wd0
            else:
                if wl0 >= c * (e - p - eu):
                    el = c * (e - p - eu)
                    ed = 0
                    wl = wl0 - el
                    wd = wd0
                else:
                    el = wl0
                    ed = c * (e - p - eu) - wl0
                    wl = 0
                    wd = wd0 - ed
    if wu < 0:
        wu = 0
    if wl < 0:
        wl = 0
    if wd < 0:
        wd = 0
    if wu > wum:
        wu = wum
    if wl > wlm:
        wl = wlm
    if wd > wdm:
        wd = wdm
    return eu, el, ed, wu, wl, wd


def runoff_generation(gene_params, w0_first, precips, evapors):
    """产流计算模型

    Parameters
    ----------
    gene_params: 新安江模型产流参数
    w0_first: 流域三层初始土壤含水量
    precips: 流域各时段面平均降雨量
    evapors: 流域各时段蒸散发

    Returns
    -------
    out : array
        runoff:流域各时段产流量
        runoff_imp:流域各时段不透水面积上的产流量
    """
    # 时段循环计算
    w0 = w0_first
    runoff = pd.Series()
    runoff_imp = pd.Series()
    for i in len(evapors):
        eu, el, ed, wu, wl, wd = evapor_single_period(gene_params, w0, precips[i], evapors[i])
        r, r_imp = runoff_generation_single_period(gene_params, w0, precips[i], evapors[i])
        w0 = pd.Series([wu, wl, wd], index=['WU', 'WL', 'WD'])
        runoff.append(r)
        runoff_imp.append(r_imp)
    return runoff, runoff_imp


def runoff_generation_single_period(gene_params, initial_params, precip, evapor):
    """单时段流域产流计算模型——蓄满产流

    Parameters
    ----------
    gene_params: 新安江模型产流参数
    initial_params: 时段初计算条件
    precip: 该时段面平均降雨量
    evapor: 流域该时段蒸散发

    Returns
    -------
    out : float
        runoff:流域时段产流量
    """
    # K: 蒸发系数
    # IMP: 流域不透水系数
    # B: 流域蓄水容量曲线的方次
    # WM: 流域平均蓄水容量
    # WUM: 流域上层土壤平均蓄水容量
    # WLM: 流域下层土壤平均蓄水容量
    # C: 深层蒸散发折算系数
    k = gene_params['K']
    imp = gene_params['IMP']
    b = gene_params['B']
    wm = gene_params['WM']

    wu0 = initial_params['WU']
    wl0 = initial_params['WL']
    wd0 = initial_params['WD']
    p = precip
    e = k * evapor

    wmm = wm * (1 + b) / (1 - imp)
    w0 = wu0 + wl0 + wd0
    a = wmm * (1 - (1 - w0 / wmm) ** (1 / (1 + b)))

    if p - e >= 0:
        if p - e + a < wmm:
            r = p - e - (wm - w0) + (wm * (1 - (a + p - e) / wmm) ** (1 + b))
        else:
            r = p - e - (wm - w0)
        r_imp = (p - e) * imp
    else:
        r = 0
        r_imp = 0
    return r, r_imp


def different_sources():
    """分水源计算"""
    return


def land_route():
    """坡面汇流计算"""
    return


def network_route():
    """河网汇流计算"""
    return


def river_route():
    """河道汇流计算"""
    return
