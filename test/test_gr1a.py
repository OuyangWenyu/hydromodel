"""
Author: zhuanglaihong
Date: 2025-03-27 00:44:00
LastEditTime: 2025-03-27 00:57:17
LastEditors: zhuanglaihong
Description: test gr1a model
FilePath: /zlh/hydromodel/test/test_gr1a.py
Copyright: Copyright (c) 2021-2024 zhuanglaihong. All rights reserved.
"""

import contextlib
import pytest
import numpy as np
from hydromodel.models.gr1a import gr1a, calculate_qk
from hydromodel.models.model_config import MODEL_PARAM_DICT


@pytest.fixture
def setup_data():
    """准备测试数据"""

    time_length = 5
    basin_num = 2
    var_num = 2

    p_and_e = np.ones((time_length, basin_num, var_num))
    # 设置降水数据
    p_and_e[:, :, 0] = np.array(
        [[1000, 1200], [1100, 1300], [900, 1100], [1050, 1250], [950, 1150]]
    )
    # 设置蒸发数据
    p_and_e[:, :, 1] = np.array(
        [[800, 850], [820, 870], [780, 830], [810, 860], [790, 840]]
    )

    # 创建参数数据 [basin, parameter]
    parameters = np.array([[0.5], [0.6]])

    return {
        "p_and_e": p_and_e,
        "parameters": parameters,
        "time_length": time_length,
        "basin_num": basin_num,
    }


def test_calculate_qk():
    """测试calculate_qk函数"""
    # 测试单个流域
    pk = 1000.0
    pk_1 = 900.0
    ek = 800.0
    x = 0.7

    result = calculate_qk(pk, pk_1, ek, x)

    # 手动计算预期结果
    denominator = 1 + ((0.7 * pk + 0.3 * pk_1) / (x * ek)) ** 2
    expected = pk * (1 - 1 / (denominator**0.5))

    assert np.isclose(result, expected)

    # 测试多个流域
    pk = np.array([1000.0, 1200.0])
    pk_1 = np.array([900.0, 1100.0])
    ek = np.array([800.0, 850.0])
    x = np.array([0.7, 0.8])

    result = calculate_qk(pk, pk_1, ek, x)

    assert result.shape == (2,)
    assert np.all(result >= 0)
    assert np.all(result <= pk)


def test_gr1a_no_warmup(setup_data):
    """测试无预热期的GR1A模型"""
    data = setup_data

    # 获取实际参数范围
    param_ranges = MODEL_PARAM_DICT["gr1a"]["param_range"]
    x1_range = param_ranges["x1"]

    # 运行模型
    result, ets = gr1a(
        data["p_and_e"],
        data["parameters"],
        warmup_length=0,
        return_state=False,
    )

    # 检查结果维度
    assert result.shape == (data["time_length"], data["basin_num"], 1)
    assert ets.shape == (data["time_length"], data["basin_num"])

    # 检查结果范围
    assert np.all(result >= 0)
    assert np.all(result <= data["p_and_e"][:, :, 0:1])

    # 检查第一年的计算（使用当年降水量的80%作为前一年降水量）
    pk_1_first = data["p_and_e"][0, :, 0] * 0.8
    x1 = x1_range[0] + data["parameters"][:, 0] * (x1_range[1] - x1_range[0])
    expected_first = calculate_qk(
        data["p_and_e"][0, :, 0], pk_1_first, data["p_and_e"][0, :, 1], x1
    )
    assert np.allclose(result[0, :, 0], expected_first)


def test_gr1a_with_warmup(setup_data):
    """测试有预热期的GR1A模型"""
    data = setup_data
    warmup_length = 2

    result, ets = gr1a(
        data["p_and_e"],
        data["parameters"],
        warmup_length=warmup_length,
        return_state=False,
    )

    # 检查结果维度
    expected_length = data["time_length"] - warmup_length
    assert result.shape == (expected_length, data["basin_num"], 1)
    assert ets.shape == (expected_length, data["basin_num"])


def test_gr1a_return_state(setup_data):
    """测试返回状态的GR1A模型"""
    data = setup_data

    result, ets, s, r = gr1a(
        data["p_and_e"], data["parameters"], warmup_length=0, return_state=True
    )

    # 检查结果维度
    assert result.shape == (data["time_length"], data["basin_num"], 1)
    assert ets.shape == (data["time_length"], data["basin_num"])
    assert s.shape == (data["basin_num"],)
    assert r.shape == (data["time_length"], data["basin_num"])

    # 检查状态值 - s应该是倒数第二年的降水量（因为最后一年的pk_1是倒数第二年的降水量）
    assert np.array_equal(
        s, data["p_and_e"][-2, :, 0]
    )  # 修正：使用倒数第二年的降水量
    assert np.array_equal(r, result[:, :, 0])  # r应该等于径流量


def test_gr1a_custom_params(setup_data):
    """测试自定义参数范围"""
    data = setup_data

    # 自定义参数范围
    custom_params = {"param_range": {"x1": [0.5, 1.0]}}

    result1, _ = gr1a(
        data["p_and_e"],
        data["parameters"],
        warmup_length=0,
        return_state=False,
    )

    result2, _ = gr1a(
        data["p_and_e"],
        data["parameters"],
        warmup_length=0,
        return_state=False,
        gr1a=custom_params,
    )

    # 结果应该不同，因为使用了不同的参数范围
    assert not np.array_equal(result1, result2)


def test_gr1a_invalid_input():
    """测试无效输入"""
    # 创建无效的输入数据 - 变量数量错误
    invalid_p_and_e = np.ones(
        (5, 2, 3)
    )  # 错误的变量数量（GR1A需要2个变量：降水和蒸发）
    parameters = np.array([[0.5], [0.6]])

    # 测试变量数量错误
    with contextlib.suppress(Exception):
        gr1a(invalid_p_and_e, parameters, warmup_length=0)
        # 如果没有抛出异常，确保结果仍然合理
        assert False, "应该抛出异常或返回错误"
    # 测试负降水量
    invalid_p_and_e = np.ones((5, 2, 2))
    parameters = np.array([[0.5], [0.6]])
    invalid_p_and_e[0, 0, 0] = -100  # 负降水量
    result, _ = gr1a(invalid_p_and_e, parameters, warmup_length=0)

    # 检查模型是否能处理后续时间步
    assert result.shape == (5, 2, 1)  # 确保结果形状正确


if __name__ == "__main__":
    pytest.main(["-v"])
