"""新安江模型程序入口"""
import numpy as np
from core import initial_soil_moisture, runoff_generation, different_sources, uh_forecast, route_linear_reservoir, \
    network_route, river_route, uh_recognise, split_flow, divide_source


def xaj_runoff_generation(config, initial_conditions, day_rain_evapor, flood_data, xaj_params):
    """场次洪水新安江模型产流部分调用入口
    Parameters
    ------------
    basin_property: 流域属性条件
    config: 配置条件
    initial_conditions: 场次洪水流域初始计算条件
    day_rain_evapor:该场次洪水前期降雨蒸发数据
    xaj_params:新安江模型参数

    Return
    ------------
    rs+r_imp,rss,rg: np.array
        场次洪水产流计算结果——地表径流深，壤中流径流深，地下水径流深
    """
    # 计算前期土壤含水量
    w0_initial = initial_conditions.iloc[:3]
    day_precip = day_rain_evapor.loc[:, 'day_precip']
    day_evapor = day_rain_evapor.loc[:, 'day_evapor']
    w0 = initial_soil_moisture(xaj_params, w0_initial, day_precip, day_evapor)
    # 流域产流计算
    precips = flood_data.loc[:, 'flood_precip']
    evapors = flood_data.loc[:, 'flood_evapor']
    runoff, runoff_imp = runoff_generation(xaj_params, w0, precips, evapors)
    # 水源划分计算
    rs, rss, rg = different_sources(xaj_params, config, initial_conditions, precips, evapors, runoff)
    rs_imp = np.array(runoff_imp)
    return rs + rs_imp, rss, rg


def xaj_routing(basin_property, config, xaj_params, uh, rs, rss, rg):
    """场次洪水汇流计算，首先是地表径流汇流，一般如果有单位线，就直接运用计算即可。
    场次洪水新安江模型产流部分调用入口
    Parameters
    ------------
    basin_property: 流域属性条件
    config: 配置条件
    xaj_params:新安江模型参数
    uh: 时段单位线
    rs:场次洪水产流计算结果——地表径流深
    rss:壤中流径流深
    rg: 地下水径流深

    Return
    ------------
    q: np.array
        汇流计算结果
    """
    qs = uh_forecast(rs, uh)
    qi, qg = route_linear_reservoir(xaj_params, basin_property, config, rss, rg)
    q = qs + qi + qg
    # 单元面积河网汇流计算
    q = network_route(q, xaj_params)
    # 单元面积以下河道汇流
    q = river_route(config, xaj_params, q)
    return q


def xaj(basin_property, config, initial_conditions, days_rain_evapor, floods_data, xaj_params, uh=None):
    """多场次洪水新安江模型调用
    Parameters
    ------------
    basin_property: 流域属性条件
    config: 配置条件
    initial_conditions:计算初始条件
    days_rain_evapor:各场次洪水前期降雨蒸发
    floods_data:各场次洪水
    xaj_params:新安江模型参数
    uh: 流域单元时段单位线

    Return
    ------------
    q: np.array组成的list
        各场次洪水对应的模型计算的出口断面流量
    """
    # 分别对每场场次洪水进行计算
    rs_s = []
    ri_s = []
    rg_s = []
    for i in range(len(floods_data)):
        rs, ri, rg = xaj_runoff_generation(config, initial_conditions, days_rain_evapor[i], floods_data[i], xaj_params)
        rs_s.append(rs)
        ri_s.append(ri)
        rg_s.append(rg)
    if uh is None:
        floods = split_flow(floods_data)
        flood_data = divide_source(floods)
        uh = uh_recognise(rs_s, flood_data)
    qs = []
    for i in range(len(floods_data)):
        q = xaj_routing(basin_property, config, xaj_params, uh, rs_s[i], ri_s[i], rg_s[i])
        qs.append(q)
    return qs
