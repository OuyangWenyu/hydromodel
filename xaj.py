"""总程序入口"""
from core import initial_soil_moisture, runoff_generation, different_sources, uh_forecast, iuh_forecast, network_route, \
    river_route


def xaj(property, config, initial_conditions, day_rain_evapor, flood_data, xaj_params):
    """新安江模型调用入口"""
    # 计算前期土壤含水量
    w0_initial = initial_conditions[:3]
    day_precip = day_rain_evapor[:, 1]
    day_evapor = day_rain_evapor[:, 2]
    w0 = initial_soil_moisture(xaj_params, w0_initial, day_precip, day_evapor)
    # 流域产流计算
    precips = flood_data[:, 'FloodPrecip']
    evapors = flood_data[:, 'FloodEvapor']
    runoff, runoff_imp = runoff_generation(xaj_params, w0, precips, evapors)
    # 水源划分计算
    rs, rss, rg = different_sources(xaj_params, initial_conditions, precips, evapors, runoff)
    # 汇流计算，首先是地表径流汇流，一般如果有单位线，就直接运用计算即可。
    qs = uh_forecast(rs, flood_data)
    qi = iuh_forecast(rss, flood_data)
    qg = iuh_forecast(rg, flood_data)
    q = qs + qi + qg
    # 单元面积河网汇流计算
    q = network_route(q)
    # 单元面积以下河道汇流
    q = river_route(q)
    return q
