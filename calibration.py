"""率定参数，调用python moea框架——Platypus（该框架和java的moea框架出自同一作者） 对模型参数进行优化"""
from data_process import init_parameters
from xaj import xaj


def certainty_coefficient(config, property, simulated_flow, flood_data):
    pass


def flood_quality(simulated_flow, flood_data):
    pass


def nash_coefficient(simulated_flow, flood_data):
    pass


def cal_fitness():
    """统计预报误差等，计算模型fitness，也便于后面进行参数率定"""
    # 构造输入数据
    property, config, initial_conditions, day_rain_evapor, flood_data, xaj_params = init_parameters()
    # 调用模型计算，得到输出
    simulated_flow = xaj(property, config, initial_conditions, day_rain_evapor, flood_data, xaj_params)
    # 计算适应度
    fitness = []
    # 确定性系数
    fitness[0] = certainty_coefficient(config, property, simulated_flow, flood_data)
    # 洪水总量、洪峰值、峰现时间按许可误差统计合格率
    fitness[1] = flood_quality(simulated_flow, flood_data);
    # nash系数
    fitness[2] = nash_coefficient(simulated_flow, flood_data);
    return fitness


def calibrate():
    """调用优化计算模型进行参数优选"""
    return
