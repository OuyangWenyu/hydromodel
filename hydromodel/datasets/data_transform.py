'''
Author: zhuanglaihong
Date: 2025-03-12 13:41:30
LastEditTime: 2025-03-17 10:20:51
LastEditors: zhuanglaihong
Description: Convert various time scales of data
FilePath: /zlh/hydromodel/hydromodel/datasets/data_transform.py
Copyright: Copyright (c) 2021-2024 zhuanglaihong. All rights reserved.
'''

import pandas as pd
from pathlib import Path
import sys
import os
import argparse

current_script_path = Path(os.path.realpath(__file__))
repo_root_dir = current_script_path.parent.parent
sys.path.append(str(repo_root_dir))

def tran_csv_hour_to_day(data_path):
    """
    Transform hourly data to daily data
    
    return :
        filepath to daily_csv
    """

    # 读取数据
    df = pd.read_csv(data_path)
    
    input_path = Path(data_path)
    output_path = input_path.parent / f"{input_path.stem}_daily{input_path.suffix}"

    # 转换时间列，尝试不同的时间格式
    time_formats = [
        '%d/%m/%Y %H:%M',  # 31/12/2023 23:59
        '%Y-%m-%d %H:%M',  # 2023-12-31 23:59
        '%m/%d/%Y %H:%M',  # 12/31/2023 23:59
        '%Y/%m/%d %H:%M',  # 2023/12/31 23:59
    ]
    
    for time_format in time_formats:
        try:
            df['time'] = pd.to_datetime(df['time'], format=time_format)
            break
        except ValueError:
            continue
    else:
        raise ValueError("无法解析时间列，请检查时间格式是否正确")


    # 设置时间索引
    df.set_index('time', inplace=True)

    # 重采样到日尺度
    daily_data = pd.DataFrame()
    # 降水和蒸散发累加
    daily_data['prcp(mm/day)'] = df['prcp(mm/hour)'].resample('D').sum()
    daily_data['pet(mm/day)'] = df['pet(mm/hour)'].resample('D').sum()
    # 流量取平均
    daily_data['flow(m^3/s)'] = df['flow(m^3/s)'].resample('D').mean()

    # 重置索引，让时间成为一列
    daily_data.reset_index(inplace=True)

    # 调整列顺序
    daily_data = daily_data[['time',  'prcp(mm/day)','pet(mm/day)', 'flow(m^3/s)']]

    # 保存结果
    daily_data.to_csv(output_path, index=False)
    
    return output_path

def tran_csv_hour_to_month(data_path):
    """
    Transform hourly data to monthly data
    
    return :
        filepath to monthly_csv
    """
    # 读取数据
    df = pd.read_csv(data_path)
    
    input_path = Path(data_path)
    output_path = input_path.parent / f"{input_path.stem}_monthly{input_path.suffix}"

    # 转换时间列，尝试不同的时间格式
    time_formats = [
        '%d/%m/%Y %H:%M',  # 31/12/2023 23:59
        '%Y-%m-%d %H:%M',  # 2023-12-31 23:59
        '%m/%d/%Y %H:%M',  # 12/31/2023 23:59
        '%Y/%m/%d %H:%M',  # 2023/12/31 23:59
    ]
    
    for time_format in time_formats:
        try:
            df['time'] = pd.to_datetime(df['time'], format=time_format)
            break
        except ValueError:
            continue
    else:
        raise ValueError("无法解析时间列，请检查时间格式是否正确")

    # 设置时间索引
    df.set_index('time', inplace=True)

    # 重采样到月尺度
    monthly_data = pd.DataFrame()
    # 降水和蒸散发累加
    monthly_data['prcp(mm/month)'] = df['prcp(mm/hour)'].resample('M').sum()
    monthly_data['pet(mm/month)'] = df['pet(mm/hour)'].resample('M').sum()
    # 流量取平均
    monthly_data['flow(m^3/s)'] = df['flow(m^3/s)'].resample('M').mean()

    # 重置索引，让时间成为一列
    monthly_data.reset_index(inplace=True)

    # 修改时间格式为 YYYY-MM
    monthly_data['time'] = monthly_data['time'].dt.strftime('%Y-%m-01')

    # 调整列顺序
    monthly_data = monthly_data[['time',  'prcp(mm/month)','pet(mm/month)', 'flow(m^3/s)']]

    # 保存结果
    monthly_data.to_csv(output_path, index=False)
    
    return output_path
    
def tran_csv_hour_to_year(data_path):
    """
    Transform hourly data to yearly data
    
    return :
        filepath to yearly_csv
    """
    # 读取数据
    df = pd.read_csv(data_path)
    
    input_path = Path(data_path)
    output_path = input_path.parent / f"{input_path.stem}_yearly{input_path.suffix}"

    # 转换时间列，尝试不同的时间格式
    time_formats = [
        '%d/%m/%Y %H:%M',  # 31/12/2023 23:59
        '%Y-%m-%d %H:%M',  # 2023-12-31 23:59
        '%m/%d/%Y %H:%M',  # 12/31/2023 23:59
        '%Y/%m/%d %H:%M',  # 2023/12/31 23:59
    ]
    
    for time_format in time_formats:
        try:
            df['time'] = pd.to_datetime(df['time'], format=time_format)
            break
        except ValueError:
            continue
    else:
        raise ValueError("无法解析时间列，请检查时间格式是否正确")

    # 设置时间索引
    df.set_index('time', inplace=True)

    # 重采样到年尺度
    yearly_data = pd.DataFrame()
    # 降水和蒸散发累加
    yearly_data['prcp(mm/year)'] = df['prcp(mm/hour)'].resample('YE').sum()
    yearly_data['pet(mm/year)'] = df['pet(mm/hour)'].resample('YE').sum()
    # 流量取平均
    yearly_data['flow(m^3/s)'] = df['flow(m^3/s)'].resample('YE').mean()

    # 重置索引，让时间成为一列
    yearly_data.reset_index(inplace=True)

    # 修改时间格式为 YYYY-MM
    yearly_data['time'] = yearly_data['time'].dt.strftime('%Y-01-01')

    # 调整列顺序
    yearly_data = yearly_data[['time', 'prcp(mm/year)', 'pet(mm/year)', 'flow(m^3/s)']]

    # 保存结果
    yearly_data.to_csv(output_path, index=False)
    
    return output_path

if __name__ == "__main__":
    
    data_path="/home/zlh/hydromodel/data/biliuhe/basin_21401550.csv",
    tran_csv_hour_to_day(data_path)