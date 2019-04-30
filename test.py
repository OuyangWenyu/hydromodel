from data_process import init_parameters
from xaj import xaj

# 构造输入数据
input = init_parameters()
# 调用模型计算，得到输出
simulated_flow = xaj(input)
