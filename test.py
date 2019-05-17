"""使用python自带的单元测试库unittest进行测试"""
import datetime

from calibrate import calibrate
from data_process import init_parameters
from xaj import xaj

import unittest


class TestXaj(unittest.TestCase):

    def test_init(self):
        # 构造输入数据
        property, config, initial_conditions, day_rain_evapor, flood_data, xaj_params = init_parameters()
        # 调用模型计算，得到输出
        simulated_flow = xaj(property, config, initial_conditions, day_rain_evapor, flood_data, xaj_params)
        print(simulated_flow)

    def test_key(self):
        starttime = datetime.datetime.now()
        calibrate()
        endtime = datetime.datetime.now()
        print("率定完毕，耗费时间为 " + (endtime - starttime).seconds + " s")
