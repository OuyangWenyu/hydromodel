from data_process import init_parameters
from xaj import xaj

# 构造输入数据
property, config, model_initial_params, day_rain_evapor, flood_data, xaj_params = init_parameters()
# 调用模型计算，得到输出
simulated_flow = xaj(property, config, model_initial_params, day_rain_evapor, flood_data, xaj_params)
