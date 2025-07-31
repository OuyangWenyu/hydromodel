# 数据后处理功能说明

本文档介绍了新增的数据后处理功能，用于将增强数据与预热期数据拼接成长时间序列。

## 功能概述

该功能实现以下主要任务：
1. 解析增强文件的元信息，提取原始场次信息
2. 读取原始场次前面的预热期数据
3. 调整预热期数据的时间（年份改为增强数据的年份）
4. 拼接预热期数据和增强场次数据
5. 批量处理多个增强文件
6. 转换为xarray Dataset格式并保存到cache
7. 提供统一的读取接口

## 新增方法

### FloodEventDatasource类新增方法

#### 1. `parse_augmented_file_metadata(augmented_file_path: str) -> Dict`
解析增强文件的元信息，提取源场次信息。

#### 2. `get_warmup_period_data(original_start_time: str, original_end_time: str, station_id: str, warmup_hours: int = 240) -> Optional[pd.DataFrame]`
获取原始场次前面的预热期数据（默认240小时=10天）。

#### 3. `adjust_warmup_time_to_augmented_year(warmup_df: pd.DataFrame, augmented_start_time: str) -> pd.DataFrame`
调整预热期数据的年份到增强数据的年份，保持数值不变。

#### 4. `concatenate_warmup_and_augmented_data(warmup_df: pd.DataFrame, augmented_file_path: str) -> pd.DataFrame`
拼接预热期数据和增强场次数据。

#### 5. `process_augmented_files_to_timeseries(station_id: str, augmented_file_indices: List[int], augmented_files_dir: str, warmup_hours: int = 240, time_unit: str = "3h") -> Optional[str]`
批量处理增强文件的主要方法。

#### 6. `create_xarray_dataset_from_timeseries(df: pd.DataFrame, station_id: str, time_unit: str = "3h") -> xr.Dataset`
将DataFrame转换为xarray Dataset格式。

#### 7. `save_augmented_timeseries_to_cache(ds: xr.Dataset, station_id: str, time_unit: str = "3h") -> str`
保存增强数据到cache目录，文件名格式：`{dataset_name}_dataaugment_timeseries_{time_unit}_batch_{station_id}_{station_id}.nc`

#### 8. `read_ts_xrdataset_augmented(gage_id_lst: Optional[List[str]] = None, t_range: Optional[List[str]] = None, var_lst: Optional[List[str]] = None, time_unit: str = "3h", **kwargs) -> Dict`
读取增强数据的新接口，优先从dataaugment缓存读取，失败时回退到原始数据。

#### 9. `generate_augmented_file_indices(start_idx: int = 1, end_idx: int = 100, step: int = 1) -> List[int]`
生成要处理的增强文件编号列表的辅助方法。

## 使用示例

```python
from hydromodel_dev.floodevent import FloodEventDatasource

# 1. 创建数据源实例
dataset = FloodEventDatasource(
    data_path="/path/to/data",
    dataset_name="songliaorrevents",
    flow_unit="mm/3h"
)

# 2. 生成要处理的文件编号列表
file_indices = dataset.generate_augmented_file_indices(
    start_idx=1, 
    end_idx=10, 
    step=1
)

# 3. 批量处理增强文件
cache_file_path = dataset.process_augmented_files_to_timeseries(
    station_id="songliao_21401550",
    augmented_file_indices=file_indices,
    augmented_files_dir="results/real_data_augmentation_shared",
    warmup_hours=240  # 10天预热期
)

# 4. 读取增强数据
augmented_data = dataset.read_ts_xrdataset_augmented(
    gage_id_lst=["songliao_21401550"],
    t_range=["2027-01-01", "2028-01-01"],
    var_lst=["inflow", "net_rain"]
)
```

## 数据格式

### 增强文件格式
增强文件包含头部元信息和数据部分：
```
# Augmented Event: event_2027081520_2027081805_aug_0001.csv
# Source: event_1994081520_1994081805.csv
# Scale Factor: 1.0
# Start Time: 2027081520
# End Time: 2027081805
time,net_rain,gen_discharge,obs_discharge
2027-08-15 20:00:00,12.000000,241.254552,20.000000
...
```

### 输出数据格式
处理后的数据保存为NetCDF格式，包含：
- 变量：`inflow`, `net_rain`
- 维度：`time`, `basin`
- 坐标：时间序列和流域ID

## 缓存文件命名规则

缓存文件名格式：`{dataset_name}_dataaugment_timeseries_{time_unit}_batch_{station_id}_{station_id}.nc`

例如：`songliaorrevents_dataaugment_timeseries_3h_batch_songliao_21401550_songliao_21401550.nc`

## 注意事项

1. **预热期长度**：默认240小时（10天），可根据需要调整
2. **时间处理**：预热期数据的数值保持不变，但年份会调整为增强数据的年份
3. **文件编号**：增强文件编号从0001开始，支持批量选择处理
4. **向后兼容**：现有的`read_ts_xrdataset`接口仍然可用
5. **错误处理**：如果增强数据不可用，会自动回退到原始数据

## 完整示例脚本

参见 `scripts/example_data_postprocessing.py` 文件，包含完整的使用示例。