"""
Test script for XAJ Songliao model using real data from json file
主要测试总径流量和河道出口流量
"""

import numpy as np
import json
import sys
import os
from datetime import datetime
from typing import Tuple

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hydromodel.models.xaj_songliao import xaj_songliao, load_xaj_data_from_json

def test_xaj_with_real_data():
    """使用实际数据测试XAJ松辽模型"""
    print("=" * 80)
    print("新安江松辽模型测试 - 使用实际数据")
    print("=" * 80)
    
    try:
        # 加载数据和参数
        json_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'xaj_data.json')
        p_and_e, parameters = load_xaj_data_from_json(json_file)
        
        # 读取流域面积
        with open(json_file, 'r') as f:
            data = json.load(f)
            area = float(data['F'])  # 流域面积
        
        print("\n输入数据信息:")
        print(f"时间序列长度: {len(p_and_e)} 步")
        print(f"流域面积: {area} km²")
        print(f"时间步长: 1.0 小时")
        
        # 运行模型 - 获取所有状态变量
        print("\n运行模型...")
        q_sim, runoff_sim, rs, ri, rg, pe, wu, wl, wd = xaj_songliao(
            p_and_e=p_and_e,
            parameters=parameters,
            warmup_length=0,  # 不使用预热期，因为数据量较小
            return_state=True,
            normalized_params=False,  # 参数已经是原始尺度
            time_interval_hours=1.0,
            area=area,
        )
        
        # 解析时间序列
        dt = [datetime.fromisoformat(t.replace('Z', '+00:00')) 
              for t in json.loads(data['dt'])]
        
        # 创建结果DataFrame并保存到CSV
        import pandas as pd
        
        # 准备数据
        results_data = []
        for i in range(len(dt)):
            results_data.append({
                '时间': dt[i].strftime('%Y-%m-%d %H:%M'),
                '降雨量': p_and_e[i,0,0],
                '蒸发量': p_and_e[i,0,1],
                '总径流': runoff_sim[i,0,0],
                '地表径流': rs[i,0,0],
                '壤中流': ri[i,0,0],
                '地下径流': rg[i,0,0],
                '出口流量': q_sim[i,0,0]
            })
        
        # 创建DataFrame
        results_df = pd.DataFrame(results_data)
        
        # 保存到CSV文件
        csv_filename = "xaj_songliao_results.csv"
        results_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        
        print(f"\n结果已保存到: {csv_filename}")
        print(f"总记录数: {len(results_data)}")
        
        # 显示前几行数据
        print("\n前5行结果预览:")
        print(results_df.head())
        
        # 计算水量平衡
        total_rain = np.sum(p_and_e[:,:,0])
        total_evap = np.sum(p_and_e[:,:,1])
        total_runoff = np.sum(runoff_sim)
        total_outflow = np.sum(q_sim)
        
        print("\n水量平衡分析:")
        print(f"总降雨量: {total_rain:.2f} mm")
        print(f"总蒸发量: {total_evap:.2f} mm")
        print(f"总径流量: {total_runoff:.2f} mm")
        print(f"总出流量: {total_outflow:.2f} m³/s")
        
        # 分析三水源比例
        total_rs = np.sum(rs)
        total_ri = np.sum(ri)
        total_rg = np.sum(rg)
        total_components = total_rs + total_ri + total_rg
        
        print("\n三水源分量分析:")
        print(f"地表径流: {total_rs:.2f} mm ({total_rs/total_components*100:.1f}%)")
        print(f"壤中流: {total_ri:.2f} mm ({total_ri/total_components*100:.1f}%)")
        print(f"地下径流: {total_rg:.2f} mm ({total_rg/total_components*100:.1f}%)")
        
        # 基本验证
        print("\n基本验证检查:")
        checks = {
            '出口流量非负': np.all(q_sim >= 0),
            '总径流非负': np.all(runoff_sim >= 0),
            '三水源非负': np.all(rs >= 0) and np.all(ri >= 0) and np.all(rg >= 0),
            '土壤含水量非负': np.all(wu >= 0) and np.all(wl >= 0) and np.all(wd >= 0),
        }
        
        all_valid = True
        for name, check in checks.items():
            if check:
                print(f"✓ {name}")
            else:
                print(f"✗ {name}")
                all_valid = False
        
        print("\n" + "=" * 80)
        if all_valid:
            print("🎉 测试通过！模型运行正常。")
        else:
            print("❌ 测试失败！请检查模型实现。")
        print("=" * 80)
        
        return all_valid
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_xaj_with_real_data()