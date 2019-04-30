"""设置或读取新安江模型的输入数据，"""
import pandas as pd


def init_parameters(basin, flood_ids, ):
    """输入数据，初始化参数"""
    # 流域属性值的读取，包括流域面积
    properties = basin.area
    # 模型计算的简单配置，包括场次洪水洪号，计算的时间步长等
    configs = pd.Series({'FloodIds': flood_ids})
    # 初始化一部分参数初值，包括流域上层、下层、深层张力水蓄量初值（三层蒸发模型计算使用的参数），分水源计算的产流面积初值、自由水蓄量初值
    w0 = pd.Series([0, 1, 1], index=['WUM', 'WLM', 'WDM'])  # 如何初始化？
    # 然后读取场次洪水数据和每场次洪水数据前若干日的日降雨和蒸发数据（计算前期影响雨量作为初始土壤含水量依据）
    precip_day, evapor_day = pd.read_excel("input.xlsx")
    # 初始化模型参数值，才可使用模型进行计算，新安江模型有16个参数值，包括：K,IMP,B,WM,WUM,WLM,C,SM,EX,KG,KSS,KKG,KKSS,UH,KE,XE,
    # 为便于命名及使用，用pandas的series数据结构存储。
    # K:蒸发系数，
    xaj_params = pd.Series(0, index=['K', 'IMP', 'WM', 'B', 'WUM', 'WLM', 'C', 'SM', 'EX', 'KSS', 'KG', 'KKSS', 'KKG',
                                     'UH', 'KE', 'XE'])
    print(xaj_params)
    return w0, precip_day, evapor_day, xaj_params


def write_netcdf():
    """把数据写入netcdf文件"""

    return
