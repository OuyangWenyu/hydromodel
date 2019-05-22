"""率定参数，调用python moea框架——Platypus（该框架和java的moea框架出自同一作者） 对模型参数进行优化"""
from hydroeval import *
from platypus import Problem, Real, NSGAII, GAOperator, SBX, PM, nondominated
from data_process import init_parameters
from xaj import xaj

import numpy as np
import matplotlib.pyplot as plt


class XajCalibrate(Problem):
    def __init__(self):
        # 定义决策变量个数（这里就是参与寻优的xaj模型参数的个数），目标个数（要计算的适应度，暂时设为2个）
        # 约束书本上给出了一个结构性约束KG+KSS=0.7，因为moea框架里是软约束，且KG+KSS一定需要小于1的，所以直接在运算中定义
        super(XajCalibrate, self).__init__(15, 2)
        # 新安江模型直接给定参数范围，所有参数均给出取值范围，先按照书上的建议给出取值范围
        # 按层次顺序给出各参数范围
        # 第一层：K/WUM/WLM/C
        # 第二层：WM/B/IMP
        # 第三层：SM/EX/KG和KSS两者定义1个即可，另一个在计算中使用结构性约束KG+KSS=0.7
        # 第四层：KKSS/KKG/CR/L/XE  KE直接取时段数，就不优化了
        self.types[:] = [Real(0, 1.5), Real(10, 20), Real(60, 90), Real(0.10, 0.20), Real(120, 200), Real(0.1, 0.4),
                         Real(0.01, 0.04), Real(0, 100), Real(1.0, 1.5), Real(0.000001, 1), Real(0, 1),
                         Real(0, 1), Real(0, 0.99), Real(0, 5), Real(0, 0.5)]
        # 误差是越小越好，而nash系数是越大越好，因此定义目标函数方向
        self.directions[:] = [Problem.MINIMIZE, Problem.MAXIMIZE]

    def evaluate(self, solution):
        params = solution.variables[:]
        solution.objectives[:] = cal_fitness(params)


def cal_nse(simulated_flow, flood_data):
    """计算NSE，也就是确定性系数。
    evaluator运算对象是np.array，即ndarray，因此需要对list每个元素计算，Nash系数直接计算所有数据，所以把list中的array拼接起来。
    Parameters
    ----------
    simulated_flow: 各模拟径流ndarray组成的list
    flood_data: 实测径流ndarray组成的list

    Returns
    -------
    nse: float
        所有模拟数据与实测数据之间的nash系数
    """
    simulated_flow_array = np.array([])
    flood_data_array = np.array([])
    for an_array1 in simulated_flow:
        simulated_flow_array = np.hstack([simulated_flow_array, an_array1])
    for an_array2 in flood_data:
        flood_data_array = np.hstack([flood_data_array, an_array2])
    result = evaluator(nse, simulated_flow_array, flood_data_array)
    if result == np.nan:
        raise ArithmeticError("检查计算错误！")
    return result


def cal_mare(simulated_flow, flood_data):
    """计算mare，即平均绝对误差
    evaluator运算对象是np.array，即ndarray，因此需要对list每个元素计算，Nash系数直接计算所有数据，所以把list中的array拼接起来
    Parameters
    ----------
    simulated_flow: 各模拟径流ndarray组成的list
    flood_data: 实测径流ndarray组成的list

    Returns
    -------
    nse: float
        所有模拟数据与实测数据之间的平均绝对误差
    """
    simulated_flow_array = np.array([])
    flood_data_array = np.array([])
    for an_array1 in simulated_flow:
        simulated_flow_array = np.hstack([simulated_flow_array, an_array1])
    for an_array2 in flood_data:
        flood_data_array = np.hstack([flood_data_array, an_array2])
    result = evaluator(mare, simulated_flow_array, flood_data_array)
    if result == np.nan:
        raise ArithmeticError("检查计算错误！")
    return result


def cal_fitness(xaj_params):
    """统计预报误差等，计算模型fitness，也便于后面进行参数率定"""
    print("----------------------------------------一次径流模拟开始-------------------------------------------------")
    # 构造输入数据
    basin_property, config, initial_conditions, day_rain_evapor, flood_data, xaj_params = init_parameters(xaj_params)
    # 调用模型计算，得到输出
    simulated_flow = xaj(basin_property, config, initial_conditions, day_rain_evapor, flood_data, xaj_params)
    # 计算适应度，先只以nse和mare为例
    # 因为实测径流flood_data数据结构是由pd.DataFrame组成的list，所以需要转换为ndarray组成的list，以便于后续计算。
    flood_data_array_list = []
    for a_dataframe in flood_data:
        # DataFrame转ndarray
        flood_data_array_list.append(np.array(a_dataframe['flood_quant']))
    # 输出各个指标计算结果
    mare = cal_mare(simulated_flow, flood_data_array_list)
    nse = cal_nse(simulated_flow, flood_data_array_list)
    print("-----------------本次模拟绝对误差为：" + str(mare) + "-----------------------")
    print("-----------------本次模拟纳什系数为：" + str(nse) + "------------------------")
    print("----------------------------------------一次径流模拟结束！-------------------------------------------------")
    print(" ")
    return [mare, nse]


def calibrate(run_counts):
    """调用优化计算模型进行参数优选
    Parameters
    ----------
    run_counts: int   运行次数

    Returns
    ---------
    optimal_params: list
          非劣解集
    """
    algorithm = NSGAII(XajCalibrate(), population_size=500, variator=GAOperator(SBX(0.95, 20.0), PM(2, 25.0)))
    algorithm.run(run_counts)

    # We could also get only the non-dominated solutions，这里只展示非劣解集
    nondominated_solutions = nondominated(algorithm.result)

    # plot the results using matplotlib
    # 如果目标超过三个，可视化方面需要进行降维操作，这里先以两个目标为例进行分析
    plt.scatter([s.objectives[0] for s in nondominated_solutions],
                [s.objectives[1] for s in nondominated_solutions])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("$mare$")
    plt.ylabel("$nse$")
    plt.show()

    # 返回最优参数
    optimal_params = []
    for nondominated_solution in nondominated_solutions:
        optimal_params.append(nondominated_solution.variables)
    return optimal_params
