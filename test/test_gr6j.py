import numpy as np
import pytest
from hydromodel.models.gr6j import (
    production,
    uh_gr6j,
    routing_store,
    exponential_store,
    gr6j,
)


@pytest.fixture
def setup_data():
    """准备测试数据"""
    time_length = 5
    basin_num = 2
    var_num = 2
    
   
    p_and_e = np.ones((time_length, basin_num, var_num))
    
    p_and_e[:, :, 0] = np.array([
        [10, 12],  # 日降水量
        [11, 13],
        [9, 11],
        [10.5, 12.5],
        [9.5, 11.5]
    ])
   
    p_and_e[:, :, 1] = np.array([
        [4, 4.5],  # 日蒸发量
        [4.2, 4.7],
        [3.8, 4.3],
        [4.1, 4.6],
        [3.9, 4.4]
    ])
    
    # 创建参数数据 [basin, parameter]
    parameters = np.array([
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # x1, x2, x3, x4, x5, x6
        [0.6, 0.4, 0.7, 0.6, 0.4, 0.5]
    ])
    
    return {
        'p_and_e': p_and_e,
        'parameters': parameters,
        'time_length': time_length,
        'basin_num': basin_num
    }


def test_production():
    """测试产流模块"""
    p_and_e = np.array([[10, 4], [12, 4.5]]) 
    x1 = np.array([100, 120])
    s_level = np.array([50, 60])
    
    pr, et, s = production(p_and_e, x1, s_level)
    
    assert pr.shape == (2,)
    assert et.shape == (2,)
    assert s.shape == (2,)
    assert np.all(pr >= 0)
    assert np.all(et >= 0)
    assert np.all(s >= 0)
    assert np.all(s <= x1)


def test_uh_gr6j():
    """测试单位线模块"""
    x4 = np.array([1.5, 2.0])
    
    uh1, uh2 = uh_gr6j(x4)
    
    assert len(uh1) == 2
    assert len(uh2) == 2
    assert len(uh1[0]) == 2  # ceil(1.5)
    assert len(uh2[0]) == 3  # ceil(2 * 1.5)
    assert len(uh1[1]) == 2  # ceil(2.0)
    assert len(uh2[1]) == 4  # ceil(2 * 2.0)


def test_routing_store():
    """测试汇流模块"""
    q9 = np.array([5, 6])
    q1 = np.array([2, 3])
    x2 = np.array([3, 4])
    x3 = np.array([100, 120])
    x5 = np.array([0.5, 0.6])
    r1 = np.array([70, 84])  # 0.7 * x3
    
    q, r1_updated = routing_store(q9, q1, x2, x3, x5, SC=0.4, r1=r1)
    
    assert q.shape == (2,)
    assert r1_updated.shape == (2,)
    assert np.all(q >= 0)
    assert np.all(r1_updated >= 0)
    assert np.all(r1_updated <= x3)


def test_exponential_store():
    """测试指数型水库模块"""
    q9 = np.array([5, 6])
    x3 = np.array([100, 120])
    x6 = np.array([10, 12])
    r2 = np.array([30, 36])  # 0.3 * x3
    
    qr2, r2_updated = exponential_store(q9, x3, x6, SC=0.2, r2=r2)
    
    assert qr2.shape == (2,)
    assert r2_updated.shape == (2,)
    assert np.all(qr2 >= 0)
    assert np.all(r2_updated >= 0)


def test_gr6j_model(setup_data):
    """测试完整的GR6J模型"""
    data = setup_data
    
    # 运行模型
    result, ets = gr6j(
        data['p_and_e'],
        data['parameters'],
        warmup_length=0,
        return_state=False
    )
    
    # 检查结果维度
    assert result.shape == (data['time_length'], data['basin_num'], 1)
    assert ets.shape == (data['time_length'], data['basin_num'])
    
    # 检查结果范围
    assert np.all(result >= 0)
    assert np.all(ets >= 0)


def test_gr6j_warmup(setup_data):
    """测试GR6J模型的预热期功能"""
    data = setup_data
    warmup_length = 2
    
    # 运行模型（带预热期）
    result = gr6j(
        data['p_and_e'],
        data['parameters'],
        warmup_length=warmup_length,
        return_state=True
    )
    
    # 检查返回值数量
    assert len(result) == 5  # streamflow, ets, s, r1, r2
    
    # 检查结果维度
    streamflow = result[0]
    ets = result[1]
    assert streamflow.shape == (data['time_length'] - warmup_length, data['basin_num'], 1)
    assert ets.shape == (data['time_length'] - warmup_length, data['basin_num'])


if __name__ == '__main__':
    pytest.main(['-v'])