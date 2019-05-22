"""新安江模型核心计算程序"""
import numpy as np
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
    out : pandas.Series
        w0:模型计算使用的流域前期土壤含水量
    """

    w0 = w0_initial
    for i in range(day_evapor.size):
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
    out : float
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
        # 当pe>0时，说明流域蓄水量增加，首先补充上层土壤水，然后补充下层，最后补充深层，不透水层产生的径流不补充张力水，因此直接不参与接下来计算
        r, r_imp = runoff_generation_single_period(evapor_params, initial_conditions, precip, evapor)
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
    # 可能有计算误差使得数据略超出合理范围，应该规避掉，如果明显超出范围，则可能计算有误，应仔细检查计算过程
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
    out : list 数组
        runoff:流域各时段产流量
        runoff_imp:流域各时段不透水面积上的产流量
    """
    # 时段循环计算
    w0 = w0_first
    runoff = []
    runoff_imp = []
    for i in range(evapors.size):
        eu, el, ed, wu, wl, wd = evapor_single_period(gene_params, w0, precips[i], evapors[i])
        r, r_imp = runoff_generation_single_period(gene_params, w0, precips[i], evapors[i])
        w0 = pd.Series([wu, wl, wd], index=['WU', 'WL', 'WD'])
        runoff.append(r)
        runoff_imp.append(r_imp)
    return runoff, runoff_imp


def runoff_generation_single_period(gene_params, initial_conditions, precip, evapor):
    """单时段流域产流计算模型——蓄满产流

    Parameters
    ----------
    gene_params: 新安江模型产流参数
    initial_conditions: 时段初计算条件
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

    wu0 = initial_conditions['WU']
    wl0 = initial_conditions['WL']
    wd0 = initial_conditions['WD']
    p = precip
    e = k * evapor
    # 为了防止后续计算出现不符合物理意义的情况，这里要对p和e的取值进行范围限制
    if e < 0:
        e = 0
    if p < 0:
        p = 0

    # 这里的imp是不是流域不透水面积/总面积的意思？个人认为应该不是，这里应该只是把wmm做了一个处理，
    # 后面计算的时候实际上只是在透水面积上做的计算，并没有真的把不透水面积比例考虑进产流模型中
    wmm = wm * (1 + b) / (1 - imp)
    w0 = wu0 + wl0 + wd0
    # 注意计算a时，开方运算，偶数方次时，根号下不能为负数，所以需要限制w0的取值，这也是物理意义上的要求
    if w0 > wm:
        w0 = wm
    a = wmm * (1 - (1 - w0 / wm) ** (1 / (1 + b)))

    if p - e >= 0:
        if p - e + a < wmm:
            r = p - e - (wm - w0) + wm * (1 - (a + p - e) / wmm) ** (1 + b)
        else:
            r = p - e - (wm - w0)
        r_imp = (p - e) * imp
    else:
        r = 0
        r_imp = 0
    return r, r_imp


def different_sources(diff_source_params, config, initial_conditions, precips, evapors, runoffs):
    """分水源计算

    Parameters
    ------------
    diff_source_params:分水源计算所需参数
    config: 计算配置条件
    initial_conditions:计算所需初始条件
    precips:各时段降雨
    evapors:各时段蒸发
    runoffs:各时段产流

    Return
    ------------
    rs_s,rss_s,rg_s: np.array 数组
        除不透水面积以外的面积上划分水源得到的地表径流，壤中流和地下径流，最后将水深值从不透水面积折算到流域面积

    """
    # 取参数值
    k = diff_source_params['K']
    sm = diff_source_params['SM']
    ex = diff_source_params['EX']
    kss = diff_source_params['KSS']
    # kg由结构性约束获取，KG+KSS=0.7
    kg = 0.7 - kss
    imp = diff_source_params['IMP']

    # 为便于后面计算，这里以数组形式给出s和fr
    s_s = []
    fr_s = []
    # 取初始值
    s0 = initial_conditions['S0']
    fr0 = initial_conditions['FR0']
    s_s.append(s0)
    fr_s.append(fr0)

    # 计算前因为产流结果是整个流域上的，而接下来的公式推导都是在透水面积上的，因此需要先把产流值折算到透水面积上
    runoffs = np.array(runoffs)
    runoffs = runoffs / (1 - imp)

    # 由于Ki、Kg、Ci、Cg都是以24小时为时段长定义的，需根据时段长转换
    time_interval_hours = config['time_interval/h']
    hours_per_day = 24
    # 非整除情况，时段+1
    residue_temp = hours_per_day % time_interval_hours
    if residue_temp != 0:
        residue_temp = 1
    period_num_1d = int(hours_per_day / time_interval_hours) + residue_temp
    # 当kss+kg>1时，根式为偶数运算时，kss_period会成为复数，这里会报错；另外注意分母可能为0，kss不可取0
    # 对kss+kg的取值进行限制，也是符合物理意义的，地下水出流不能超过自身的蓄水。
    kss_period = (1 - (1 - (kss + kg)) ** (1 / period_num_1d)) / (1 + kg / kss)
    kg_period = kss_period * kg / kss

    # 流域最大点自由水蓄水容量深
    smm = sm * (1 + ex)

    rs_s, rss_s, rg_s = [], [], []
    for i in range(precips.size):
        fr0 = fr_s[i]
        p = precips[i]
        e = k * evapors[i]
        if p - e <= 0:
            # 如果没有净雨，没有产流，地表径流应该为0
            rs = 0
            # 没有产流也要往s_s和fr_s数组里放置数字，否则计算无法继续，依据是产流面积不增加，水量平衡原则，计算时段初的S应该与上时段末一致。
            if i == 0:
                s = s0
            else:
                s = s_s[i - 1]
            # 那么根据S×FR=S0×FR0，有FR=FR0
            fr = fr0
            if fr < 0:
                raise ArithmeticError("检查runoff值是否有负数！")
            # 如果出现fr=0的情况，需要稍作修正，因为可能导致后续计算出现分母为0的情况
            if fr == 0:
                fr = 0.001
            if fr > 1:
                fr = 1
            fr_s.append(fr)
            # 地下水库线性泄流
            rss = kss_period * s * fr
            rg = kg_period * s * fr
            # 计算时段末自由水蓄水量
            s = s - (kss_period + kg_period) / fr
            if s < 0:
                s = 0
            if s > sm:
                s = sm
            s_s.append(s)
        else:
            # 首先要给定时段出的fr0，以根据fr0*s0=fr*s计算时段计算中所需的s值
            # 计算当前时段的fr，后面分5mm之后，每个时段还要单独算，但是fr必须得先算出来，后面才能算fr_d，所以这里不能少fr
            fr = runoffs[i] / (p - e)
            if fr < 0:
                raise ArithmeticError("检查runoff值是否有负数！")
            # 如果出现fr=0的情况，需要稍作修正，因为可能导致后续计算出现分母为0的情况
            if fr == 0:
                fr = 0.001
            if fr > 1:
                fr = 1

            # 计算当前时段计算中的s值，后面分5mm段之后，s还要重新计算，所以这里算不算都行
            s0 = s_s[i]
            s = s0 * fr0 / fr

            # 净雨分5mm一段进行计算，因为计算时在FS/FR ~ SMF'关系图上开展，即计算在产流面积上开展，所以用PE做净雨.分段为了差分计算更精确。
            residue_temp = (p - e) % 5
            if residue_temp != 0:
                residue_temp = 1
            n = int((p - e) / 5) + residue_temp
            # 整除了就是5mm，不整除就少一些，差分每段小了也挺好
            pe = (p - e) / n
            kss_d = (1 - (1 - (kss_period + kg_period)) ** (1 / n)) / (1 + kg_period / kss_period)
            kg_d = kss_d * kg_period / kss_period

            rs = rss = rg = 0

            s_ds = []
            fr_ds = []
            s_ds.append(s0)
            fr_ds.append(fr0)

            for j in range(n):
                # 因为产流面积随着自由水蓄水容量的变化而变化，每5mm净雨对应的产流面积肯定是不同的，因此fr是变化的
                fr0_d = fr_ds[j]
                s0_d = s_ds[j]
                fr_d = 1 - (1 - fr) ** (1 / n)  # 按《水文预报》书上公式计算
                s_d = fr0_d * s0_d / fr_d

                smmf = smm * (1 - (1 - fr_d) ** (1 / ex))
                smf = smmf / (1 + ex)
                # 如果出现s_d>smf的情况，说明s_d = fr0_d * s0_d / fr_d导致的计算误差不合理，需要进行修正。
                if s_d > smf:
                    s_d = smf

                au = smmf * (1 - (1 - s_d / smf) ** (1 / (1 + ex)))

                if pe + au >= smmf:
                    rs_j = (pe + s_d - smf) * fr_d
                    rss_j = smf * kss_d * fr_d
                    rg_j = smf * kg_d * fr_d
                    s_d = smf - (rss_j + rg_j) / fr_d
                elif 0 < pe + au < smmf:
                    rs_j = (pe - smf + s_d + smf * (1 - (pe + au) / smmf) ** (ex + 1)) * fr_d
                    rss_j = (pe - rs_j / fr_d + s_d) * kss_d * fr_d
                    rg_j = (pe - rs_j / fr_d + s_d) * kg_d * fr_d
                    s_d = s_d + pe - (rs_j + rss_j + rg_j) / fr_d
                else:
                    rs_j = rss_j = rg_j = s_d = 0
                rs = rs + rs_j
                rss = rss + rss_j
                rg = rg + rg_j
                # 赋值s_d和fr_d到数组中，以给下一段做初值
                s_ds.append(s_d)
                fr_ds.append(fr_d)
                if j == n - 1:
                    # 最后一个净雨段，把fr_d和s_d值赋给s0和fr0作为下一个时段计算初值
                    fr_s.append(fr_d)
                    s_s.append(s_d)
        rs_s.append(rs)
        rss_s.append(rss)
        rg_s.append(rg)
    # 从透水面积折算到流域面积上，为了方便，利用numpy的广播进行计算
    rs_s = np.array(rs_s)
    rss_s = np.array(rss_s)
    rg_s = np.array(rg_s)
    rs_s = rs_s * (1 - imp)
    rss_s = rss_s * (1 - imp)
    rg_s = rg_s * (1 - imp)
    return rs_s, rss_s, rg_s


def split_flow(floods_data, auto=None):
    """分割流量过程，有时候，会手动进行分割，所以可能不需要调用次函数
    Parameters
    ------------
    floods_data:多组观测的出口断面流量过程线，np.array组成的list
    auto: 是程序自动计算退水曲线还是已经手动计算好了

    Return
    ------------
    floods: np.array组成的list
            去掉前期洪水尚未退完的部分水量和非本次降雨补给的流量过程线

    """
    # TODO: 求退水曲线
    # 首先取出各个洪水过程退水段
    floods = []
    for flood_data in floods_data:
        temp = flood_data.loc[:, "flood_quant"]
        floods.append(np.array(temp))
    # 然后取出退水段各点，组成向量

    # 最小二乘法计算

    # 利用计算好的退水曲线分割观测的流量过程线

    if auto is None:
        return floods
    return floods


def divide_source(floods, auto=None):
    """从观测的场次洪水径流中分割地表地下径流
    Parameters
    ------------
    floods:多组观测的已经去掉前期洪水尚未退完的部分水量和非本次降雨补给的出口断面流量过程线，np.array组成的list
    auto: 是程序自动计算分割流量还是已经手动计算好了

    Return
    ------------
    rs_s,rg_s: np.array组成的list
            地表径流流量过程线和地下径流过程线

    """
    # TODO: 分割径流
    if auto is None:
        return floods
    return floods


def iuh_recognise(runoffs, flood_data, linear_reservoir=None, linear_canal=None, isochrone=None):
    """瞬时单位线的识别，目前以计算Nash单位线为主
    Parameters
    ------------
    runoffs:多组各时段净雨（地表），矩阵表示
    flood_data:多组出口断面流量过程线，矩阵表示
    linear_reservoir:
    linear_canal:线性渠个数
    isochrone:等流时线个数

    Return
    ------------
    n,k:float
        nash单位线两参数
    """
    # TODO: 这个函数没想好怎么写，能很好地把串并联结构和三个基础模型搭配在一起，目前以n个线性水库串联为主

    return


def route_linear_reservoir(route_params, basin_property, config, rss, rg):
    """运用瞬时单位线进行汇流计算
    Parameters
    ------------
    route_params:汇流参数
    basin_property: 流域属性条件
    config: 配置条件
    rss: 壤中流净雨
    rg:地下径流净雨

    Return
    ------------
    q_rss,q_rg: np.array
        汇流计算结果——流量过程线
    """
    area = basin_property['basin_area/km^2']
    time_in = config['time_interval/h']
    u = area / (3.6 * time_in)

    kkss = route_params['KKSS']
    kkg = route_params['KKG']

    q_rss = np.zeros(rss.size)
    q_rg = np.zeros(rg.size)

    q_rss[0] = rss[0] * (1 - kkss) * u
    q_rg[0] = rg[0] * (1 - kkg) * u

    for i in range(1, rss.size):
        q_rss[i] = rss[i] * (1 - kkss) * u + q_rss[i - 1] * kkss
        q_rg[i] = rg[i] * (1 - kkg) * u + q_rg[i - 1] * kkg

    return q_rss, q_rg


def uh_recognise(runoffs, flood_data):
    """时段单位线的识别，先以最小二乘法为主。
    Parameters
    ------------
    runoffs:各场次洪水对应的各时段净雨，np.array组成的list
    flood_data:多组出口断面流量过程线，np.array组成的list

    Return
    ------------
    uh:array
        时段单位线，每个数据的单位为m^3/s
    """
    # 最小二乘法计算针对每场次洪水得到一条单位线，多场次洪水，按照书上的意思，可以取平均。如果曲线之间差异较大，需要进行分类，目前先求平均。
    uh_s = []
    uh_sum = np.array([])
    for i in range(len(runoffs)):
        ht = []
        # q表示实际径流
        q = flood_data[i]
        l = len(flood_data[i])
        m = len(runoffs[i])
        n = l - m + 1
        for j in range(n):
            # numpy默认为行向量
            h_row = np.zeros(l)
            for k in range(j, j + m):
                h_row[k] = runoffs[i][k - j]
            ht.append(h_row)
        h = np.transpose(ht)
        ht_h = np.dot(ht, h)
        ht_q = np.dot(ht, q)
        # 求得一条单位线
        uh_temp = np.linalg.solve(ht_h, ht_q)
        # 每场次洪水均有一条单位线
        uh_s.append(uh_temp)
        # 求和，因为单位线长度可能不一致，所以需要进行补0对齐
        if i == 0:
            uh_sum = uh_temp
        else:
            # 当维度不同的向量要对齐时，需在不足处补0
            length_zero = max(uh_sum.size, uh_temp.size) - min(uh_sum.size, uh_temp.size)
            zeros_need = np.zeros(length_zero)
            if uh_sum.size > uh_temp.size:
                arr_new = np.hstack([uh_temp, zeros_need])
                uh_sum = uh_sum + arr_new
            else:
                arr_new = np.hstack([uh_sum, zeros_need])
                uh_sum = uh_temp + arr_new
    # 广播运算
    uh = uh_sum / len(runoffs)
    return uh


def uh_forecast(runoffs, uh):
    """运用时段单位线进行汇流计算
    Parameters
    ------------
    runoffs:场次洪水对应的各时段净雨，数组
    uh:单位线各个时段数值

    Return
    ------------
    q:array
        汇流计算结果——流量过程线
    """
    q = np.convolve(runoffs, uh)
    return q


def network_route(runoffs, route_params):
    """河网汇流计算，新安江模型里一般采用线性水库或滞后演算法。这里使用滞后演算法
     Parameters
    ------------
    runoffs:坡面汇流计算结果，数组
    route_params:模型参数

    Return
    ------------
    q:array
        汇流计算结果——流量过程线
    """
    # 取整后面才能计算
    t = int(route_params['L'])
    cr = route_params['CR']
    # 初始化
    qr = runoffs
    q = np.zeros(runoffs.size)
    if t <= 0:
        for i in range(runoffs.size):
            if i == 0:
                q[0] = (1 - cr) * qr[0]
            else:
                q[i] = cr * q[i - 1] + (1 - cr) * qr[i]
    else:
        for i in range(runoffs.size):
            if i == 0:
                q[0] = 0
            elif i < t:
                q[i] = cr * q[i - 1]
            else:
                q[i] = cr * q[i - 1] + (1 - cr) * qr[i - t]
    return q


def river_route(config, route_params, qf):
    """河道汇流计算，新安江模型一般采用马斯京根法
     Parameters
    ------------
    config:计算设置条件
    route_params:模型参数
    qf:河网汇流计算结果，数组

    Return
    ------------
    q: np.array
        汇流计算结果——流量过程线
    """
    ke = route_params['KE']
    xe = route_params['XE']
    time_interval = config['time_interval/h']

    q = np.zeros(qf.size)

    c0 = (0.5 * time_interval - ke * xe) / (0.5 * time_interval + ke - ke * xe)
    c1 = (0.5 * time_interval + ke * xe) / (0.5 * time_interval + ke - ke * xe)
    c2 = 1 - c0 - c1

    q[0] = qf[0]
    if c0 >= 0 and c2 >= 0:
        for i in range(1, q.size):
            q[i] = c0 * qf[i] + c1 * qf[i - 1] + c2 * q[i - 1]
    else:
        # 当马斯京根不适用时，暂未处理
        q = qf
    return q
