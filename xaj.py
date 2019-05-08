"""总程序入口"""
from core import initial_soil_moisture, runoff_generation, different_sources


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
    rs, rss, rg = different_sources()
    return
