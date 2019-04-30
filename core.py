"""新安江模型核心计算程序"""


def initial_soil_moisture(xaj_params, w0, day_precip_evapor):
    """计算初始土壤含水量"""
    # 首先计算WDM，WMM等可直接由参数计算得的值
    wm = xaj_params['WM']
    b = xaj_params['B']
    imp = xaj_params['IMP']
    dm = wm - xaj_params['WUM'] - xaj_params['WLM']
    wmm = wm * (1 + b) / (1 - imp)

    # 最后计算结果更新存储到w0中
    return w0


def runoff_generation():
    """产流计算模型"""
    return


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
