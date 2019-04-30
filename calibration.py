"""率定参数，调用python moea框架——Platypus（该框架和java的moea框架出自同一作者） 对模型参数进行优化"""


def fitness():
    """统计预报误差等，计算模型fitness，也便于后面进行参数率定"""
    return