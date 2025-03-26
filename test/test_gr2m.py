"""
Author: zhuanglaihong
Date: 2025-03-27 00:48:22
LastEditTime: 2025-03-27 01:01:08
LastEditors: zhuanglaihong
Description: test gr2m model
FilePath: /zlh/hydromodel/test/test_gr2m.py
Copyright: Copyright (c) 2021-2024 zhuanglaihong. All rights reserved.
"""

import pytest
import numpy as np
from hydromodel.models.gr2m import gr2m, production, routing


@pytest.fixture
def setup_data():
    """准备测试数据"""
    time_length = 5
    basin_num = 2
    var_num = 2
    p_and_e = np.ones((time_length, basin_num, var_num))

    p_and_e[:, :, 0] = np.array(
        [[100, 120], [110, 130], [90, 110], [105, 125], [95, 115]]  # 月降水量
    )

    p_and_e[:, :, 1] = np.array(
        [[80, 85], [82, 87], [78, 83], [81, 86], [79, 84]]  # 月蒸发量
    )

    parameters = np.array([[0.5, 0.5], [0.6, 0.4]])  # 两个参数x1和x2，两个流域

    return {
        "p_and_e": p_and_e,
        "parameters": parameters,
        "time_length": time_length,
        "basin_num": basin_num,
    }


def test_production():
    """测试产流函数"""

    inputs = np.array([[100, 80], [120, 85]])  # [basin, variable]
    x1 = np.array([400, 450])
    s0 = np.array([200, 225])

    # 运行产流函数
    p3, et, s = production(inputs, x1, s0)

    # 检查结果维度
    assert p3.shape == (2,)
    assert et.shape == (2,)
    assert s.shape == (2,)

    # 检查结果范围
    assert np.all(p3 >= 0)
    assert np.all(et >= 0)
    assert np.all(s >= 0)
    assert np.all(s <= x1)

    # 检查水量平衡
    # 由于GR2M是概念模型，水量平衡不是严格的线性关系，但可以检查大致范围
    assert np.all(p3 + et + (s - s0) <= inputs[:, 0] * 1.01)  # 允许1%的误差


def test_routing():
    """测试汇流函数"""

    p3 = np.array([50, 60])
    x2 = np.array([0.8, 0.7])
    r0 = np.array([100, 120])

    # 运行汇流函数
    q, r = routing(p3, x2, r0)

    # 检查结果维度
    assert q.shape == (2,)
    assert r.shape == (2,)

    # 检查结果范围
    assert np.all(q >= 0)
    assert np.all(r >= 0)

    # 检查水量平衡
    assert np.allclose(q + r, x2 * (r0 + p3))


def test_gr2m_no_warmup(setup_data):
    """测试无预热期的GR2M模型"""
    data = setup_data

    # 运行模型
    result, ets = gr2m(
        data["p_and_e"], data["parameters"], warmup_length=0, return_state=False
    )

    # 检查结果维度
    assert result.shape == (data["time_length"], data["basin_num"], 1)
    assert ets.shape == (data["time_length"], data["basin_num"])

    # 检查结果范围
    assert np.all(result >= 0)
    assert np.all(ets >= 0)

    # 只检查基本属性和合理性

    # 确保结果小于输入降水量
    assert np.all(result[:, :, 0] <= data["p_and_e"][:, :, 0])

    # 检查径流量随时间的变化是否合理
    # 第一个时间步应该有径流（因为初始状态s0=0.5*x1）
    assert np.all(result[0, :, 0] > 0)


def test_gr2m_with_warmup(setup_data):
    """测试有预热期的GR2M模型"""
    data = setup_data
    warmup_length = 2

    result, ets = gr2m(
        data["p_and_e"],
        data["parameters"],
        warmup_length=warmup_length,
        return_state=False,
    )

    # 检查结果维度
    expected_length = data["time_length"] - warmup_length
    assert result.shape == (expected_length, data["basin_num"], 1)
    assert ets.shape == (expected_length, data["basin_num"])


def test_gr2m_return_state(setup_data):
    """测试返回状态的GR2M模型"""
    data = setup_data

    # 运行模型
    result, ets, s, r = gr2m(
        data["p_and_e"], data["parameters"], warmup_length=0, return_state=True
    )

    # 检查结果维度
    assert result.shape == (data["time_length"], data["basin_num"], 1)
    assert ets.shape == (data["time_length"], data["basin_num"])
    assert s.shape == (data["basin_num"],)
    assert r.shape == (data["basin_num"],)

    # 检查状态值范围
    x1 = 140 + data["parameters"][:, 0] * 1860  # 参数范围[140, 2000]
    assert np.all(s >= 0)
    assert np.all(s <= x1)
    assert np.all(r >= 0)


def test_gr2m_custom_params(setup_data):
    """测试自定义参数范围"""
    data = setup_data

    # 自定义参数范围
    custom_params = {
        "param_range": {
            "x1": [100, 1000],  # 自定义x1参数范围
            "x2": [0.3, 0.9],  # 自定义x2参数范围
        }
    }

    # 运行模型
    result1, _ = gr2m(
        data["p_and_e"], data["parameters"], warmup_length=0, return_state=False
    )

    result2, _ = gr2m(
        data["p_and_e"],
        data["parameters"],
        warmup_length=0,
        return_state=False,
        gr2m=custom_params,
    )

    # 结果应该不同，因为使用了不同的参数范围
    assert not np.array_equal(result1, result2)


def test_gr2m_extreme_values():
    """测试极端值情况"""

    time_length = 3
    basin_num = 1
    var_num = 2

    # 极端降水和蒸发
    p_and_e = np.ones((time_length, basin_num, var_num))
    p_and_e[:, :, 0] = np.array([[1000], [0], [0.1]])  # 极大、零和极小降水
    p_and_e[:, :, 1] = np.array([[0], [500], [0.1]])  # 零、极大和极小蒸发

    parameters = np.array([[0.5, 0.5]])

    # 运行模型
    result, ets = gr2m(p_and_e, parameters, warmup_length=0)

    # 检查结果
    assert np.all(result >= 0)  # 径流应该非负
    assert np.all(ets >= 0)  # 蒸发应该非负

    # 零降水应该产生较小的径流（来自水库）
    assert result[1, 0, 0] < result[0, 0, 0]


def test_gr2m_consecutive_runs(setup_data):
    """测试连续运行的一致性"""
    data = setup_data

    # 完整运行
    full_result, _, full_s, full_r = gr2m(
        data["p_and_e"], data["parameters"], warmup_length=0, return_state=True
    )

    # 分段运行
    split_point = 3
    first_part = data["p_and_e"][:split_point]
    second_part = data["p_and_e"][split_point:]

    # 运行第一部分
    _, _, s_mid, r_mid = gr2m(
        first_part, data["parameters"], warmup_length=0, return_state=True
    )

    # 使用第一部分的最终状态作为第二部分的初始状态
    with pytest.MonkeyPatch().context() as mp:

        def mock_production(inputs, x1, s0_unused):
            return production(inputs, x1, s_mid)

        def mock_routing(p3, x2, r0_unused):
            return routing(p3, x2, r_mid)

        mp.setattr("hydromodel.models.gr2m.production", mock_production)
        mp.setattr("hydromodel.models.gr2m.routing", mock_routing)

        # 只运行第一个时间步，因为我们只能控制第一个时间步的初始状态
        second_result, _ = gr2m(
            second_part[:1], data["parameters"], warmup_length=0, return_state=False
        )

    # 检查第一个时间步的结果是否一致
    assert np.allclose(second_result[0, :, 0], full_result[split_point, :, 0])


if __name__ == "__main__":
    pytest.main(["-v"])
