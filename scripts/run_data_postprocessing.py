"""
Author: Wenyu Ouyang
Date: 2025-07-31 16:25:50
LastEditTime: 2025-07-31 16:25:50
LastEditors: Wenyu Ouyang
Description: 数据后处理示例 - 将增强数据与预热期数据拼接成长时间序列
FilePath: \hydromodel\scripts\run_data_postprocessing.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import os

from hydrodatasource.configs.config import SETTING
from hydromodel.models.floodevent import FloodEventDatasource


def main():
    """演示如何使用数据后处理功能"""

    # 1. 设置参数
    data_path = os.path.join(
        SETTING["local_data_path"]["datasets-interim"], "songliaorrevent"
    )
    station_id = "songliao_21401550"
    augmented_files_dir = "results/real_data_augmentation_shared"

    # 2. 创建FloodEventDatasource实例
    print("🔄 初始化FloodEventDatasource...")
    dataset = FloodEventDatasource(
        data_path=data_path,
        dataset_name="songliaorrevents",
        flow_unit="mm/3h",
        trange4cache=["1960-01-01 02", "2024-12-31 23"],
    )

    # 3. 生成要处理的增强文件编号列表（示例：处理前10个文件）
    print("📋 生成增强文件编号列表...")
    file_indices = dataset.generate_augmented_file_indices(
        start_idx=1, end_idx=10, step=1
    )
    print(f"   要处理的文件编号: {file_indices}")

    # 4. 批量处理增强文件，生成长时间序列数据
    print("🔄 批量处理增强文件...")
    cache_file_path = dataset.process_augmented_files_to_timeseries(
        station_id=station_id,
        augmented_file_indices=file_indices,
        augmented_files_dir=augmented_files_dir,
        warmup_hours=240,  # 10天预热期
        time_unit="3h",
    )

    if cache_file_path:
        print(f"✅ 成功生成增强数据缓存文件: {cache_file_path}")

        # 5. 测试读取增强数据
        print("🔄 测试读取增强数据...")
        augmented_data = dataset.read_ts_xrdataset_augmented(
            gage_id_lst=[station_id],
            t_range=["2027-01-01", "2028-01-01"],  # 示例时间范围
            var_lst=["inflow", "net_rain"],
        )

        if "3h" in augmented_data:
            ds = augmented_data["3h"]
            print(f"   增强数据集形状: {ds.dims}")
            print(
                f"   时间范围: {ds.time.min().values} 到 {ds.time.max().values}"
            )
            print(f"   变量: {list(ds.data_vars.keys())}")
            print("✅ 增强数据读取成功!")
        else:
            print("❌ 增强数据读取失败")
    else:
        print("❌ 增强数据处理失败")


if __name__ == "__main__":
    main()
