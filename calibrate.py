"""率定参数，调用python moea框架——Platypus（该框架和java的moea框架出自同一作者） 对模型参数进行优化"""
from hydroeval import *
from platypus import Problem, Real, NSGAII, GAOperator, SBX, PM
from data_process import init_parameters
from xaj import xaj

import matplotlib.pyplot as plt


class XajCalibrate(Problem):
    def __init__(self):
        # 定义决策变量个数（这里就是参与寻优的xaj模型参数的个数），目标个数（要计算的适应度，暂时设为2个）
        # 约束书本上给出了一个结构性约束KG+KSS=0.7
        super(XajCalibrate, self).__init__(16, 2, 1)
        # 新安江模型直接给定参数范围，所有参数均给出取值范围，先按照书上的建议给出取值范围
        # 按层次顺序给出各参数范围
        # 第一层：K/WUM/WLM/C
        # 第二层：WM/B/IMP
        # 第三层：SM/EX/KG/KSS
        # 第四层：KKSS/KKG/KKS/L/XE  KE直接取时段数，就不优化了
        self.types[:] = [Real(0, 1.5), Real(10, 20), Real(60, 90), Real(0.10, 0.20), Real(120, 200), Real(0.1, 0.4),
                         Real(0.01, 0.04), Real(0, 100), Real(1.0, 1.5), Real(0, 1), Real(0, 1), Real(0, 1),
                         Real(0, 1), Real(0, 4), Real(0, 0.5)]
        self.constraints[:] = "=0"

    def evaluate(self, solution):
        kss = solution.variables[9]  # KG
        kg = solution.variables[10]  # KSS
        params = solution.variables
        solution.objectives[:] = cal_fitness(params)
        solution.constraints[:] = [kss + kg - 0.7]


def cal_nse(simulated_flow, flood_data):
    """计算NSE，也就是确定性系数"""
    return evaluator(nse, simulated_flow, flood_data)


def cal_mare(simulated_flow, flood_data):
    """计算mare，即平均绝对误差"""
    return evaluator(mare, simulated_flow, flood_data)


def cal_fitness(xaj_params):
    """统计预报误差等，计算模型fitness，也便于后面进行参数率定"""
    # 构造输入数据
    basin_property, config, initial_conditions, day_rain_evapor, flood_data, xaj_params = init_parameters(xaj_params)
    # 调用模型计算，得到输出
    simulated_flow = xaj(basin_property, config, initial_conditions, day_rain_evapor, flood_data, xaj_params)
    # 计算适应度，先只以
    return [cal_mare(simulated_flow, flood_data), cal_nse(simulated_flow, flood_data)]


def calibrate():
    """调用优化计算模型进行参数优选"""
    algorithm = NSGAII(XajCalibrate(), population_size=500, variator=GAOperator(SBX(0.95, 20.0), PM(2, 25.0)))
    algorithm.run(10000)

    # plot the results using matplotlib
    # 如果目标超过三个，可视化方面需要进行降维操作，这里先以两个目标为例进行分析
    plt.scatter([s.objectives[0] for s in algorithm.result],
                [s.objectives[1] for s in algorithm.result])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("$mare$")
    plt.ylabel("$nse$")
    plt.show()
    return
