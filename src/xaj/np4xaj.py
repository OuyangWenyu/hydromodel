"""xaj model with jax"""
from functools import partial

import numpy as np


def calculate_evap(lm, c,
                   wu0, wl0,
                   prcp, pet):
    """时段蒸发计算，三层蒸发模型 from <<SHUIWEN YUBAO>> the fifth version. Page 22-23

    Parameters
    ----------
    lm, c: 三层蒸发模型计算所需参数
    wu0, wl0: 三层蒸发模型计算所需初始条件
    prcp, pet: 流域面平均降雨量, potential evapotranspiration

    Returns
    -------
    out : float
        eu,el,ed:流域时段三层蒸散发
    """
    eu = np.where(wu0 + prcp >= pet, pet, wu0 + prcp)
    ed = np.where((wl0 < c * lm) & (wl0 < c * (pet - eu)), c * (pet - eu) - wl0, 0)
    el = np.where(wu0 + prcp >= pet,
                  0,
                  np.where(wl0 >= c * lm, (pet - eu) * wl0 / lm,
                           np.where(wl0 >= c * (pet - eu), c * (pet - eu), wl0)))
    return eu, el, ed


def calculate_prcp_runoff(b, im, wm,
                          w0,
                          pe):
    """Calculates the amount of runoff generated from rainfall after entering the underlying surface
    Parameters
    ----------
    b, im, wm:
        计算所需参数
    w0:
        计算所需初始条件
    pe:
        net precipitation

    Returns
    -------
    out :
       r, r_im: runoff
    """
    wmm = wm * (1 + b) / (1 - im)
    a = wmm * (1 - (1 - w0 / wm) ** (1 / (1 + b)))
    r_cal = np.where(pe + a < wmm,
                     pe - (wm - w0) + wm * (1 - (a + pe) / wmm) ** (1 + b),
                     pe - (wm - w0))
    r = np.maximum(r_cal, 0)
    r_im_cal = pe * im
    r_im = np.maximum(r_im_cal, 0)
    return r, r_im


def calculate_w_storage(um, lm, dm,
                        wu0, wl0, wd0, eu, el, ed,
                        p, r):
    """update the w values of the three layers
       according to the runoff-generation equation 2.60, dW = dPE - dR,
       which means that for one period: the change of w = pe - r
    Parameters
    ----------
    um, lm, dm: 计算所需参数
    wu0, wl0, wd0, eu, el, ed: 计算所需state variables
    p, r: 流域面平均降雨量, and runoff

    Returns
    -------
    out : float
        eu,el,ed:流域时段三层蒸散发
        wu,wl,wd:流域时段三层含水量
    """
    e = eu + el + ed
    # net precipitation
    pe = np.maximum(p - e, 0)
    # 当pe>0时，说明流域蓄水量增加，首先补充上层土壤水，然后补充下层，最后补充深层
    # pe<=0: no additional water, just remove evapotranspiration
    wu = np.where(pe > 0, np.where(wu0 + pe - r < um, wu0 + pe - r, um), wu0 - eu)
    # calculate wd before wl because it is easier to cal using where statement
    wd = np.where(pe > 0, np.where(wu0 + wl0 + pe - r > um + lm, wu0 + wl0 + wd0 + pe - r - um - lm, wd0), wd0 - ed)
    # water balance (equation 2.2 in Page 13, also shown in Page 23)
    wl = np.where(pe > 0, wu0 + wl0 + wd0 + pe - r - wu - wd, wl0 - el)
    # 可能有计算误差使得数据略超出合理范围，应该规避掉，如果明显超出范围，则可能计算有误，应仔细检查计算过程
    wu_ = np.clip(wu, 0, um)
    wl_ = np.clip(wl, 0, lm)
    wd_ = np.clip(wd, 0, dm)
    return wu_, wl_, wd_


def calculate_different_sources(sm, ex, ki, kg,
                                s0, fr0,
                                pe, r):
    """分水源计算  from <<SHUIWEN YUBAO>> the fifth version. Page 40-41 and 150-151
        the procedures in <<GONGCHENG SHUIWENXUE>> the third version are different.
        Here we used the former. We'll add the latter in the future.
    Parameters
    ------------
    sm, ex, ki, kg: required parameters
    s0, fr0: 计算所需初始条件 initial_conditions
    pe: net precipitation
    r: 产流
    Return
    ------------
    rs,ri,rg:
        除不透水面积以外的面积上划分水源得到的地表径流，壤中流和地下径流，最后将水深值从不透水面积折算到流域面积

    """
    # 流域最大点自由水蓄水容量深
    ms = sm * (1 + ex)
    # FR of this period  equation 5.24. However, we should notice that when pe=0,
    # we think no change occurred in S, so fr = fr0 and s = s0
    fr = np.where(pe == 0, fr0, r / pe)

    # we don't know how the equation 5.32 was derived, so we don't divide the Runoff here.
    # equation 2.84
    au = ms * (1 - (1 - (fr0 * s0 / fr) / sm) ** (1 / (1 + ex)))
    rs = np.where(pe + au < ms,
                  # equation 2.85
                  fr * (pe + (fr0 * s0 / fr) - sm + sm * (1 - (pe + au) / ms) ** (ex + 1)),
                  # equation 2.86
                  fr * (pe + (fr0 * s0 / fr) - sm))
    # equation 2.87
    s = (fr0 * s0 / fr) + (r - rs) / fr
    # equation 2.88
    # We either don't know how the equations 5.33 and 5.34 were derived, and we just perform daily calculation.
    # Also, we believed the parameters could be tuned during calibration. Hence, we directly use ki and kg.
    ri = ki * s * fr
    rg = kg * s * fr
    s1 = s * (1 - ki - kg)
    return rs, ri, rg, fr, s1


def streamflow_step(params, state_variables, forcings):
    """Logic for simulating a single timestep of streamflow from GR4J within Jax.
    This function is usually used as an argument to lax.scan as the inner function for a loop.

    Parameters
    ----------
    params : the parameters in the model
    state_variables : these variables will be used in next period
    forcings : Current timestep's value for precipitation, evapotranspiration input.

    Returns
    -------
    S : scalar tensor
        Storage reservoir level at the end of the timestep
    qt_history : 1D tensor
        Past timesteps' stream input values
    R : scalar tensor
        Routing reservoir level at the end of the timestep
    Q : scalar tensor
        Resulting streamflow
    """
    # params
    b = params[0]
    im = params[1]
    um = params[2]
    lm = params[3]
    dm = params[4]
    c = params[5]
    sm = params[6]
    ex = params[7]
    ki = params[8]
    kg = params[9]
    ci = params[10]
    cg = params[11]
    l = params[12]
    cs = params[13]

    # state_variables
    wu0 = state_variables[0]
    wl0 = state_variables[1]
    wd0 = state_variables[2]
    s0 = state_variables[3]
    fr0 = state_variables[4]
    qi0 = state_variables[5]
    qg0 = state_variables[6]
    # 1d tensor
    qt_history = state_variables[7]
    q0 = state_variables[8]

    # forcings
    prcp = forcings[0]
    pet = forcings[1]
    area = forcings[2]

    # 为了防止后续计算出现不符合物理意义的情况，这里要对p和e的取值进行范围限制
    prcp = np.maximum(prcp, 0)
    pet = np.maximum(pet, 0)
    # wm
    wm = um + lm + dm

    w0_ = wu0 + wl0 + wd0
    # 注意计算a时，开方运算，偶数方次时，根号下不能为负数，所以需要限制w0的取值，这也是物理意义上的要求
    w0 = np.where(w0_ > wm, wm, w0_)

    # Calculate the amount of evaporation from storage
    eu, el, ed = calculate_evap(lm, c,
                                wu0, wl0,
                                prcp, pet)
    e = eu + el + ed

    # Calculate the runoff generated by net precipitation
    prcp_difference = prcp - e
    pe = np.maximum(prcp_difference, 0)
    r, rim = calculate_prcp_runoff(b, im, wm,
                                   w0,
                                   pe)
    # Update wu, wl, wd
    wu, wl, wd = calculate_w_storage(um, lm, dm,
                                     wu0, wl0, wd0, eu, el, ed,
                                     prcp, r)

    # calculate different sources runoff
    rs, ri, rg, fr, s = calculate_different_sources(sm, ex, ki, kg,
                                                    s0, fr0,
                                                    pe, r)

    # the routing scheme mainly comes from the figure 5.1
    u = area / (3.6 * 24)
    # The surface runoff include runoff_imp and rs from surface runoff source and directly enter into river network
    qs = (rim + rs) * u
    # equation 5.43
    qi = qi0 * ci + (1 - ci) * ri * u
    # equation 5.38
    qg = qg0 * cg + (1 - cg) * rg * u
    # equation 5.44
    qt = qs + qi + qg

    # river network routing
    # this is a inverted order array. we roll the qt_history
    # so that we can put the current value into the first index
    qt_history = np.roll(qt_history, 1)
    qt_history[0] = qt
    # not sure if this could be used with "grad" for l.
    q = cs * q0 + (1 - cs) * qt_history[l]
    # q = np.convolve(qt_history, UH)

    # The order of the returned values is important because it must correspond
    # up with the order of the kwarg list argument 'outputs_info' to lax.scan.
    return [wu, wl, wd, s, fr, qi, qg, qt_history, q], q


def simulate_streamflow(prcp, pet, area,
                        wu0, wl0, wd0, s0, fr0, qi0, qg0, qt_history, q0,
                        b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg, l, cs):
    """Simulates streamflow over time using the model logic from XAJ as implemented in Jax.
    This function can be used in PyMC3 or other Theano-based libraries to
    offer up the functionality of GR4J with added gradient information.
    Parameters
    ----------
    prcp : 1D tensor
        Time series of precipitation
    pet : 1D tensor
        Time series of potential evapotranspiration
    area: 1D tensor
        repeat for basin areas to be concatenated with P and E
    wu0: scalar tensor
        流域上层张力水蓄量初值（三层蒸发模型计算使用的参数），
    wl0: scalar tensor
        流域下层张力水蓄量初值
    wd0: scalar tensor
        流域深层张力水蓄量初值
    s0:
        自由水蓄量初值
    fr0:
        分水源计算的产流面积初值、
    qi0:
        inter-streamflow of last period
    qg0:
        groundwater streamflow of last period
      Initial levels of streamflow input. Needed for routing streamflow.
      If this is nonzero, then it is implied that there is initially
      some streamflow which must be routed in the first few timesteps.
    qt_history: 1d tensor
        history values of flow into river network
    q0:
        history forecasted streamflow
    b:
        B 流域蓄水容量曲线的方次
    im:
        IMP 流域不透水系数
    um:
        WUM 流域上层土壤平均蓄水容量
    lm:
        WLM 流域下层土壤平均蓄水容量
    dm:
        WDM 流域 deep 土壤平均蓄水容量
    c:
        C 深层蒸散发折算系数
    sm:
        SM 表层自由水蓄水容量
    ex:
        EX 表层自由水蓄水容量曲线指数
    ki:
        KSS 壤中流出流系数
    kg:
        KG 地下水出流系数  Generally, KSS+KG=0.7, but it's a empirical value, so we don't use it at first
    ci:
        KKSS 壤中流消退系数
    cg:
        KKG 地下水消退系数
    l:
        L 河网汇流迟滞时间
    cs:
        CR 河网蓄水消退系数

    Returns
    -------
    streamflow : 1D tensor
      Time series of simulated streamflow
    """
    area = np.full(len(prcp), area)
    # sequence-first variables are needed for loop calculation
    forcings = np.moveaxis(np.array([prcp, pet, area]), 1, 0)
    parameters = [b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg, l, cs]
    state_variables = [wu0, wl0, wd0, s0, fr0, qi0, qg0, qt_history, q0]
    assert l < len(qt_history)
    streamflow = []
    for i in range(len(prcp)):
        state_variables, streamflow_ = streamflow_step(parameters, state_variables, forcings[i])
        streamflow.append(streamflow_)
        print("the  " + str(i) + "th iteration: " + str(streamflow_))
    return streamflow
