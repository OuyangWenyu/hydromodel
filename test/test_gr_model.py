"""
Author: zhuanglaihong
Date: 2025-03-27 00:50:14
LastEditTime: 2025-03-27 01:26:58
LastEditors: zhuanglaihong
Description: test GR4J GR5J GR6J model
FilePath: /zlh/hydromodel/test/test_gr_model.py
Copyright: Copyright (c) 2021-2024 zhuanglaihong. All rights reserved.
"""

import pytest
import numpy as np
from hydromodel.models.gr_model import GRModel
from hydromodel.models.model_config import MODEL_PARAM_DICT


@pytest.fixture
def setup_data():
    """准备测试数据"""
    time_length = 5
    basin_num = 2
    var_num = 2

    p_and_e = np.ones((time_length, basin_num, var_num))

    p_and_e[:, :, 0] = np.array(
        [[10, 12], [11, 13], [9, 11], [10.5, 12.5], [9.5, 11.5]]  # 日降水量
    )

    p_and_e[:, :, 1] = np.array(
        [[4, 4.5], [4.2, 4.7], [3.8, 4.3], [4.1, 4.6], [3.9, 4.4]]  # 日蒸发量
    )

    return {"p_and_e": p_and_e, "time_length": time_length, "basin_num": basin_num}


def test_gr4j_model(setup_data):
    """测试GR4J模型"""
    model = GRModel(model_type="gr4j")
    data = setup_data

    parameters = np.array(
        [[0.5, 0.5, 0.5, 0.5], [0.6, 0.4, 0.7, 0.6]]  # x1, x2, x3, x4
    )

    # 运行模型
    result, ets = model.run(
        data["p_and_e"], parameters, warmup_length=0, return_state=False
    )

    # 检查结果维度
    assert result.shape == (data["time_length"], data["basin_num"], 1)
    assert ets.shape == (data["time_length"], data["basin_num"])

    # 检查结果范围
    assert np.all(result >= 0)
    assert np.all(ets >= 0)


def test_gr5j_model(setup_data):
    """测试GR5J模型"""
    model = GRModel(model_type="gr5j")
    data = setup_data

    parameters = np.array(
        [[0.5, 0.5, 0.5, 0.5, 0.5], [0.6, 0.4, 0.7, 0.6, 0.4]]  # x1, x2, x3, x4, x5
    )

    # 运行模型
    result, ets = model.run(
        data["p_and_e"], parameters, warmup_length=0, return_state=False
    )

    # 检查结果维度
    assert result.shape == (data["time_length"], data["basin_num"], 1)
    assert ets.shape == (data["time_length"], data["basin_num"])

    # 检查结果范围
    assert np.all(result >= 0)
    assert np.all(ets >= 0)


def test_gr6j_model(setup_data):
    """测试GR6J模型"""
    model = GRModel(model_type="gr6j")
    data = setup_data

    parameters = np.array(
        [
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # x1, x2, x3, x4, x5, x6
            [0.6, 0.4, 0.7, 0.6, 0.4, 0.5],
        ]
    )

    # 运行模型
    result, ets = model.run(
        data["p_and_e"], parameters, warmup_length=0, return_state=False
    )

    # 检查结果维度
    assert result.shape == (data["time_length"], data["basin_num"], 1)
    assert ets.shape == (data["time_length"], data["basin_num"])

    # 检查结果范围
    assert np.all(result >= 0)
    assert np.all(ets >= 0)


def test_model_state_variables():
    """测试模型状态变量"""
    time_length = 3
    basin_num = 1
    var_num = 2
    p_and_e = np.ones((time_length, basin_num, var_num))

    # 测试GR4J状态变量
    model_gr4j = GRModel("gr4j")
    parameters_gr4j = np.array([[0.5, 0.5, 0.5, 0.5]])
    _, _, s4, r4 = model_gr4j.run(p_and_e, parameters_gr4j, return_state=True)
    assert s4.shape == (basin_num,)
    assert r4.shape == (basin_num,)

    # 测试GR5J状态变量
    model_gr5j = GRModel("gr5j")
    parameters_gr5j = np.array([[0.5, 0.5, 0.5, 0.5, 0.5]])
    _, _, s5, r5 = model_gr5j.run(p_and_e, parameters_gr5j, return_state=True)
    assert s5.shape == (basin_num,)
    assert r5.shape == (basin_num,)

    # 测试GR6J状态变量
    model_gr6j = GRModel("gr6j")
    parameters_gr6j = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
    result = model_gr6j.run(p_and_e, parameters_gr6j, return_state=True)

    # 检查返回值数量
    assert len(result) >= 3  # 至少返回径流、蒸发和一个状态变量

    # 如果返回4个值，则解包为径流、蒸发、s和r
    if len(result) == 4:
        _, _, s6, r6 = result
        assert s6.shape == (basin_num,)
        assert r6.shape == (basin_num,)
    # 如果返回5个值，则解包为径流、蒸发、s、r和n
    elif len(result) == 5:
        _, _, s6, r6, n6 = result
        assert s6.shape == (basin_num,)
        assert r6.shape == (basin_num,)
        assert n6.shape == (basin_num,)


def test_model_warmup():
    """测试模型预热期"""
    time_length = 5
    basin_num = 1
    var_num = 2
    p_and_e = np.ones((time_length, basin_num, var_num))
    warmup_length = 2

    # 测试GR4J预热期
    model_gr4j = GRModel("gr4j")
    parameters_gr4j = np.array([[0.5, 0.5, 0.5, 0.5]])
    result_gr4j, _ = model_gr4j.run(
        p_and_e, parameters_gr4j, warmup_length=warmup_length
    )
    assert result_gr4j.shape == (time_length - warmup_length, basin_num, 1)

    # 测试GR5J预热期
    model_gr5j = GRModel("gr5j")
    parameters_gr5j = np.array([[0.5, 0.5, 0.5, 0.5, 0.5]])
    result_gr5j, _ = model_gr5j.run(
        p_and_e, parameters_gr5j, warmup_length=warmup_length
    )
    assert result_gr5j.shape == (time_length - warmup_length, basin_num, 1)

    # 测试GR6J预热期
    model_gr6j = GRModel("gr6j")
    parameters_gr6j = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
    result_gr6j, _ = model_gr6j.run(
        p_and_e, parameters_gr6j, warmup_length=warmup_length
    )
    assert result_gr6j.shape == (time_length - warmup_length, basin_num, 1)


def test_model_parameter_ranges():
    """测试模型参数范围"""
    for model_type in ["gr4j", "gr5j", "gr6j"]:
        model = GRModel(model_type)
        param_ranges = MODEL_PARAM_DICT[model_type]["param_range"]

        # 检查参数数量
        expected_params = (
            4 if model_type == "gr4j" else (5 if model_type == "gr5j" else 6)
        )
        assert len(param_ranges) == expected_params

        # 检查参数范围有效性
        for param_name, param_range in param_ranges.items():
            assert len(param_range) == 2
            assert param_range[0] < param_range[1]


def test_invalid_model_type():
    """测试无效的模型类型"""
    with pytest.raises(ValueError):
        GRModel("invalid_model")


def test_model_extreme_values(setup_data):
    """测试极端值情况"""
    data = setup_data
    data["p_and_e"][0, 0, 0] = 1000  # 极大降水
    data["p_and_e"][1, 0, 0] = 0  # 零降水
    data["p_and_e"][2, 0, 1] = 100  # 极大蒸发

    for model_type in ["gr4j", "gr5j", "gr6j"]:
        model = GRModel(model_type)
        num_params = 4 if model_type == "gr4j" else (5 if model_type == "gr5j" else 6)
        parameters = np.full((data["basin_num"], num_params), 0.5)

        # 运行模型
        result, ets = model.run(data["p_and_e"], parameters, warmup_length=0)

        # 检查结果
        assert np.all(result >= 0)  # 径流应该非负
        assert np.all(ets >= 0)  # 蒸发应该非负
        assert not np.any(np.isnan(result))  # 不应该有NaN值
        assert not np.any(np.isnan(ets))  # 不应该有NaN值


if __name__ == "__main__":
    pytest.main(["-v"])
