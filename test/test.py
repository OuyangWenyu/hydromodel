"""使用python自带的单元测试库unittest进行测试"""
import datetime

import unittest

from src.calibrate import calibrate
from src.data_process import init_parameters
from src.xaj import xaj


class TestXaj(unittest.TestCase):
    """每个单元测试类都必须继承unittest.TestCase"""

    def test_xaj(self):
        """每个测试用例都必须以test开头"""
        # 构造输入数据
        property, config, initial_conditions, days_rain_evapor, floods_data, xaj_params = init_parameters()
        # 调用模型计算，得到输出
        simulated_flow = xaj(property, config, initial_conditions, days_rain_evapor, floods_data, xaj_params)
        print(simulated_flow)

    def test_xaj_calibrate(self):
        starttime = datetime.datetime.now()
        run_counts = 100
        optimal_params = calibrate(run_counts)
        print("本次优化的计算结果，即优选参数集为：")
        print(optimal_params)
        endtime = datetime.datetime.now()
        print("率定完毕，耗费时间为 " + str((endtime - starttime).seconds) + " s")


if __name__ == '__main__':
    # 测试流程：写好TestCase（类TestXaj），加载TestCase到TestSuite，由TextTestRunner运行TestSuite，运行结果保存在TextTestResult中
    suite = unittest.TestSuite()
    tests = [TestXaj('test_xaj_calibrate')]
    suite.addTests(tests)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
