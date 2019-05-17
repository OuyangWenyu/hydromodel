"""使用python自带的单元测试库unittest进行测试"""
import datetime

from calibrate import calibrate
from data_process import init_parameters
from xaj import xaj

import unittest


class TestXaj(unittest.TestCase):
    """每个单元测试类都必须继承unittest.TestCase"""

    def test_xaj(self):
        """每个测试用例都必须以test开头"""
        # 构造输入数据
        property, config, initial_conditions, day_rain_evapor, flood_data, xaj_params = init_parameters()
        # 调用模型计算，得到输出
        simulated_flow = xaj(property, config, initial_conditions, day_rain_evapor, flood_data, xaj_params)
        print(simulated_flow)

    def test_xaj_calibrate(self):
        starttime = datetime.datetime.now()
        calibrate()
        endtime = datetime.datetime.now()
        print("率定完毕，耗费时间为 " + (endtime - starttime).seconds + " s")


if __name__ == '__main__':
    # 测试流程：写好TestCase（类TestXaj），加载TestCase到TestSuite，由TextTestRunner运行TestSuite，运行结果保存在TextTestResult中
    suite = unittest.TestSuite()
    tests = [TestXaj('test_xaj'), TestXaj('test_xaj_calibrate'), ]
    suite.addTests(tests)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
