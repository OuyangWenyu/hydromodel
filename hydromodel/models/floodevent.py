"""
Author: Wenyu Ouyang
Date: 2025-01-19 18:05:00
LastEditTime: 2025-08-01 14:26:02
LastEditors: Wenyu Ouyang
Description: 流域场次数据处理类 - 继承自SelfMadeHydroDataset
FilePath: \hydromodel\hydromodel\models\floodevent.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import pandas as pd
import numpy as np
import os
import xarray as xr
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from hydrodatasource.utils.utils import streamflow_unit_conv
from hydrodatasource.reader.data_source import SelfMadeHydroDataset
from hydrodatasource.configs.config import CACHE_DIR
from hydromodel.models.consts import OBS_FLOW, NET_RAIN
from hydromodel.models.common_utils import (
    read_basin_area_safe,
)


class FloodEventDatasource(SelfMadeHydroDataset):
    """
    Flood event dataset processing class

    Inherits from SelfMadeHydroDataset, specifically designed for
    processing individual flood event data, including event extraction functions.
    """

    def __init__(
        self,
        data_path: str,
        dataset_name: str = "songliaorrevents",
        time_unit: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize the flood event dataset.

        Parameters
        ----------
        data_path : str
            Path to the data.
        dataset_name : str, optional
            Name of the dataset.
        time_unit : list of str, optional
            List of time units, default is ["3h"].
        **kwargs
            Additional keyword arguments passed to the parent class.
        """
        if time_unit is None:
            time_unit = ["3h"]
        super().__init__(
            data_path=data_path,
            download=False,
            time_unit=time_unit,
            dataset_name=dataset_name,
            **kwargs,
        )

    def extract_flood_events(
        self, df: pd.DataFrame
    ) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """
        从数据框中提取洪水事件，返回净雨、径流数组和洪峰日期

        Args:
            df: 站点数据框
            station_id: 站点ID（用于打印信息）

        Returns:
            List[Tuple[np.ndarray, np.ndarray, str]]: (净雨数组, 径流数组, 洪峰日期) 列表
        """
        events: List[Tuple[np.ndarray, np.ndarray, str]] = []
        # 找到连续的flood_event > 0区间
        flood_mask = df["flood_event"] > 0

        if not flood_mask.any():
            return events

        # 找连续区间
        in_event = False
        start_idx = None

        for idx, is_flood in enumerate(flood_mask):
            if is_flood and not in_event:
                start_idx = idx
                in_event = True
            elif not is_flood and in_event:
                # 事件结束，提取数据
                event_data = df.iloc[start_idx:idx]
                net_rain = event_data["net_rain"].values
                inflow = event_data["inflow"].values
                event_times = event_data["time"].values

                # 基本验证
                if (
                    len(net_rain) > 0
                    and len(inflow) > 0
                    and np.nansum(inflow) > 1e-6
                ):
                    # 获取场次开始和结束时间
                    start_time = event_times[0]
                    end_time = event_times[-1]

                    # 转换为十位数字格式 (YYYYMMDDHH)
                    def time_to_ten_digits(time_obj):
                        """将时间对象转换为十位数字格式 YYYYMMDDHH"""
                        if isinstance(time_obj, np.datetime64):
                            # 如果是numpy datetime64对象
                            return (
                                time_obj.astype("datetime64[h]")
                                .astype(str)
                                .replace("-", "")
                                .replace("T", "")
                                .replace(":", "")
                            )
                        elif hasattr(time_obj, "strftime"):
                            # 如果是datetime对象
                            return time_obj.strftime("%Y%m%d%H")
                        else:
                            # 如果是字符串，尝试解析
                            try:
                                from datetime import datetime

                                if isinstance(time_obj, str):
                                    dt = datetime.fromisoformat(
                                        time_obj.replace("Z", "+00:00")
                                    )
                                    return dt.strftime("%Y%m%d%H")
                                else:
                                    return "0000000000"  # 默认值
                            except Exception:
                                return "0000000000"  # 默认值

                    start_digits = time_to_ten_digits(start_time)
                    end_digits = time_to_ten_digits(end_time)

                    # 组合成场次名称：起始时间_结束时间
                    event_name = f"{start_digits}_{end_digits}"

                    events.append((net_rain, inflow, event_name))

                in_event = False
        return events

    def create_event_dict(
        self,
        net_rain: np.ndarray,
        inflow: np.ndarray,
        event_name: str,
        include_peak_obs: bool = True,
    ) -> Optional[Dict]:
        """
        将净雨和径流数组转换为标准事件字典格式

        Parameters
        ----------
        net_rain: np.ndarray
            净雨数组
        inflow: np.ndarray
            径流数组
        event_name: str
            洪峰日期（8位数字格式）
        include_peak_obs: bool
            是否包含洪峰观测值

        Returns
        -------
            Dict: 标准格式的事件字典，与uh_utils.py完全兼容
        """
        try:
            # 计算有效降雨时段数
            valid_rain_mask = ~np.isnan(net_rain) & (net_rain > 0)
            m_eff = np.sum(valid_rain_mask)

            if m_eff == 0:
                return None

            # 验证径流数据
            if np.nansum(inflow) < 1e-6:
                return None

            # 创建标准格式字典（与uh_utils.py期望的key完全一致）
            event_dict = {
                NET_RAIN: net_rain,  # 有效降雨（净雨）
                OBS_FLOW: inflow,  # 观测径流
                "m_eff": m_eff,  # 有效降雨时段数
                "n_specific": len(net_rain),  # 单位线长度
                "filepath": f"event_{event_name}.csv",  # 避免KeyError
            }

            # 添加洪峰观测值
            if include_peak_obs:
                peak_flow = np.nanmax(inflow)
                if peak_flow < 1e-6:
                    return {}
                event_dict["peak_obs"] = peak_flow

            return event_dict

        except Exception:
            return {}

    def _load_1basin_flood_events(
        self,
        station_id: Optional[str] = None,
        flow_unit: str = "mm/3h",
        include_peak_obs: bool = True,
        verbose: bool = True,
    ) -> Optional[List[Dict]]:
        """
        加载洪水事件数据

        Parameters
        ----------
        station_id:
            指定站点ID，如果为None则处理所有站点
        flow_unit
            Unit of streamflow, default is "mm/3h".
        include_peak_obs:
            是否包含洪峰观测值
        verbose:
            是否打印详细信息

        Returns
        -------
            List[Dict]: 标准格式的事件字典列表，与现有算法完全兼容
        """
        # 获取流域面积
        basin_area_km2 = None

        if station_id:
            basin_area_km2 = read_basin_area_safe(self, station_id, verbose)
        else:
            basin_area_km2 = None

        try:
            if verbose:
                print("🔄 正在加载洪水事件数据...")
                if station_id:
                    print(f"   指定站点: {station_id}")

            all_events = []
            total_events = 0

            xr_ds = self.read_ts_xrdataset(
                gage_id_lst=[station_id],
                t_range=["1960-01-01", "2024-12-31"],
                var_lst=["inflow", "net_rain", "flood_event"],
                # recache=True,
            )["3h"]

            xr_ds["inflow"] = streamflow_unit_conv(
                xr_ds[["inflow"]],
                target_unit=flow_unit,
                area=basin_area_km2,
            )["inflow"]
            df = xr_ds.to_dataframe()
            if df is None:
                return None

            # 提取洪水事件
            flood_events = self.extract_flood_events(
                df.loc[station_id].reset_index()
            )

            if not flood_events:
                if verbose:
                    print(f"  ⚠️  {station_id}: 没有找到有效洪水事件")
                return None

            # 转换为标准格式
            station_event_count = 0
            for net_rain, inflow, event_name in flood_events:
                event_dict = self.create_event_dict(
                    net_rain, inflow, event_name, include_peak_obs
                )
                if event_dict is not None:
                    all_events.append(event_dict)
                    station_event_count += 1

            if verbose and station_event_count > 0:
                print(
                    f"  ✅ {station_id}: 成功处理 {station_event_count} 个洪水事件"
                )
                total_events += station_event_count

            if not all_events:
                if verbose:
                    print("❌ 没有成功处理的洪水事件数据")
                return None

            if verbose:
                print(f"✅ 总共成功加载 {len(all_events)} 个洪水事件")

            return all_events

        except Exception as e:
            if verbose:
                print(f"❌ 加载洪水事件数据时发生错误: {str(e)}")
            return None

    def parse_augmented_file_metadata(self, augmented_file_path: str) -> Dict:
        """
        解析增强文件的元信息

        Parameters
        ----------
        augmented_file_path : str
            增强文件的路径

        Returns
        -------
        Dict
            包含源场次信息的字典，包括起始时间、结束时间、源文件名等
        """
        metadata = {}

        with open(augmented_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    if "Source:" in line:
                        source_file = line.split("Source:")[1].strip()
                        metadata["source_file"] = source_file
                        # 从源文件名提取起始时间
                        if "event_" in source_file and ".csv" in source_file:
                            time_part = source_file.replace(
                                "event_", ""
                            ).replace(".csv", "")
                            if "_" in time_part:
                                start_time_str, end_time_str = time_part.split(
                                    "_"
                                )
                                metadata["original_start_time"] = (
                                    start_time_str
                                )
                                metadata["original_end_time"] = end_time_str
                    elif "Start Time:" in line:
                        metadata["augmented_start_time"] = line.split(
                            "Start Time:"
                        )[1].strip()
                    elif "End Time:" in line:
                        metadata["augmented_end_time"] = line.split(
                            "End Time:"
                        )[1].strip()
                    elif "Scale Factor:" in line:
                        metadata["scale_factor"] = float(
                            line.split("Scale Factor:")[1].strip()
                        )
                    elif "Sample ID:" in line:
                        metadata["sample_id"] = int(
                            line.split("Sample ID:")[1].strip()
                        )
                else:
                    break

        return metadata

    def get_warmup_period_data(
        self,
        original_start_time: str,
        original_end_time: str,
        station_id: str,
        warmup_hours: int = 240,
    ) -> Optional[pd.DataFrame]:
        """
        获取原始场次前面的预热期数据

        Parameters
        ----------
        original_start_time : str
            原始场次起始时间 (YYYYMMDDHH格式)
        original_end_time : str
            原始场次结束时间 (YYYYMMDDHH格式)
        station_id : str
            站点ID
        warmup_hours : int, optional
            预热期小时数，默认240小时(10天)

        Returns
        -------
        Optional[pd.DataFrame]
            预热期数据，包含time, net_rain, inflow列
        """
        try:
            # 解析时间
            start_dt = datetime.strptime(original_start_time, "%Y%m%d%H")
            warmup_start = start_dt - timedelta(hours=warmup_hours)
            warmup_end = start_dt - timedelta(hours=3)

            # 读取预热期数据
            xr_ds = self.read_ts_xrdataset(
                gage_id_lst=[station_id],
                t_range=[
                    warmup_start.strftime("%Y-%m-%d %H"),
                    warmup_end.strftime("%Y-%m-%d %H"),
                ],
                var_lst=["inflow", "net_rain"],
            )["3h"]

            if xr_ds is None:
                return None

            # 转换为DataFrame
            df = xr_ds.to_dataframe().reset_index()
            df = df[df["basin"] == station_id].copy()

            # 重命名列以匹配增强文件格式
            df = df.rename(columns={"inflow": "obs_discharge"})
            df["gen_discharge"] = df["obs_discharge"]

            return df[["time", "net_rain", "gen_discharge", "obs_discharge"]]
        except Exception as e:
            print(f"获取预热期数据失败: {e}")
            return None

    def adjust_warmup_time_to_augmented_year(
        self, warmup_df: pd.DataFrame, augmented_start_time: str
    ) -> pd.DataFrame:
        """
        调整预热期数据的年份到增强数据的年份

        Parameters
        ----------
        warmup_df : pd.DataFrame
            预热期数据
        augmented_start_time : str
            增强数据的起始时间 (YYYYMMDDHH格式)

        Returns
        -------
        pd.DataFrame
            调整年份后的预热期数据
        """
        df = warmup_df.copy()

        # 获取增强数据的年份
        aug_year = int(augmented_start_time[:4])

        # 调整时间列的年份
        df["time"] = pd.to_datetime(df["time"])
        df["time"] = df["time"].apply(lambda x: x.replace(year=aug_year))

        return df

    def concatenate_warmup_and_augmented_data(
        self, warmup_df: pd.DataFrame, augmented_file_path: str
    ) -> pd.DataFrame:
        """
        拼接预热期数据和增强场次数据

        Parameters
        ----------
        warmup_df : pd.DataFrame
            预热期数据
        augmented_file_path : str
            增强文件路径

        Returns
        -------
        pd.DataFrame
            拼接后的完整数据
        """
        # 读取增强数据
        aug_df = pd.read_csv(augmented_file_path, comment="#")
        aug_df["time"] = pd.to_datetime(aug_df["time"])

        # 拼接数据
        combined_df = pd.concat([warmup_df, aug_df], ignore_index=True)
        combined_df = combined_df.sort_values("time").reset_index(drop=True)

        return combined_df

    def process_augmented_files_to_timeseries(
        self,
        station_id: str,
        augmented_file_indices: List[int],
        augmented_files_dir: str,
        warmup_hours: int = 240,
        time_unit: str = "3h",
    ) -> Optional[str]:
        """
        批量处理增强文件，生成长时间序列数据并保存为nc文件

        Parameters
        ----------
        station_id : str
            站点ID
        augmented_file_indices : List[int]
            要处理的增强文件编号列表
        augmented_files_dir : str
            增强文件所在目录
        warmup_hours : int, optional
            预热期小时数，默认240小时
        time_unit : str, optional
            时间单位，默认"3h"

        Returns
        -------
        Optional[str]
            生成的nc文件路径，如果失败返回None
        """
        all_timeseries_data = []

        # 获取目录下所有增强文件
        aug_files = [
            f
            for f in os.listdir(augmented_files_dir)
            if f.endswith(".csv") and "aug_" in f
        ]
        aug_files.sort()

        # 筛选指定编号的文件
        selected_files = []
        for idx in augmented_file_indices:
            matching_files = [
                f for f in aug_files if f"aug_{idx:04d}.csv" in f
            ]
            selected_files.extend(matching_files)

        if not selected_files:
            print(f"未找到指定编号的增强文件: {augmented_file_indices}")
            return None

        print(f"处理 {len(selected_files)} 个增强文件...")

        for file_name in selected_files:
            file_path = os.path.join(augmented_files_dir, file_name)

            try:
                # 解析元信息
                metadata = self.parse_augmented_file_metadata(file_path)

                if "original_start_time" not in metadata:
                    print(f"跳过文件 {file_name}: 无法解析原始时间信息")
                    continue

                # 获取预热期数据
                warmup_df = self.get_warmup_period_data(
                    metadata["original_start_time"],
                    metadata["original_end_time"],
                    station_id,
                    warmup_hours,
                )

                if warmup_df is None:
                    print(f"跳过文件 {file_name}: 无法获取预热期数据")
                    continue

                # 调整预热期时间
                warmup_df = self.adjust_warmup_time_to_augmented_year(
                    warmup_df, metadata["augmented_start_time"]
                )

                # 拼接数据
                combined_df = self.concatenate_warmup_and_augmented_data(
                    warmup_df, file_path
                )

                all_timeseries_data.append(combined_df)

            except Exception as e:
                print(f"处理文件 {file_name} 时出错: {e}")
                continue

        if not all_timeseries_data:
            print("没有成功处理的数据")
            return None

        # 合并所有时间序列数据
        full_timeseries = pd.concat(all_timeseries_data, ignore_index=True)
        full_timeseries = full_timeseries.sort_values("time").reset_index(
            drop=True
        )

        # 转换为xarray Dataset
        xr_ds = self.create_xarray_dataset_from_timeseries(
            full_timeseries, station_id, time_unit
        )

        # 保存到cache目录
        cache_file_path = self.save_augmented_timeseries_to_cache(
            xr_ds, station_id, time_unit
        )

        return cache_file_path

    def create_xarray_dataset_from_timeseries(
        self, df: pd.DataFrame, station_id: str, time_unit: str = "3h"
    ) -> xr.Dataset:
        """
        将时间序列DataFrame转换为xarray Dataset格式

        Parameters
        ----------
        df : pd.DataFrame
            时间序列数据
        station_id : str
            站点ID
        time_unit : str, optional
            时间单位，默认"3h"

        Returns
        -------
        xr.Dataset
            xarray格式的数据集
        """
        # 创建xarray Dataset
        ds = xr.Dataset(
            {
                "inflow": (
                    ["time", "basin"],
                    df[["obs_discharge"]].values.reshape(-1, 1),
                ),
                "net_rain": (
                    ["time", "basin"],
                    df[["net_rain"]].values.reshape(-1, 1),
                ),
            },
            coords={"time": df["time"].values, "basin": [station_id]},
        )

        # 添加属性
        ds.attrs["description"] = "Augmented hydrological time series data"
        ds.attrs["station_id"] = station_id
        ds.attrs["time_unit"] = time_unit
        ds.attrs["creation_time"] = datetime.now().isoformat()

        return ds

    def save_augmented_timeseries_to_cache(
        self, ds: xr.Dataset, station_id: str, time_unit: str = "3h"
    ) -> str:
        """
        将增强时间序列数据保存到cache目录

        Parameters
        ----------
        ds : xr.Dataset
            要保存的数据集
        station_id : str
            站点ID
        time_unit : str, optional
            时间单位，默认"3h"

        Returns
        -------
        str
            保存的文件路径
        """
        # 构造文件名，参考原有的命名规则，加上dataaugment前缀
        prefix = f"{self.dataset_name}_dataaugment_"
        cache_file_name = f"{prefix}timeseries_{time_unit}_batch_{station_id}_{station_id}.nc"
        cache_file_path = os.path.join(CACHE_DIR, cache_file_name)

        # 保存数据
        ds.to_netcdf(cache_file_path)

        print(f"增强时间序列数据已保存到: {cache_file_path}")
        return cache_file_path

    def read_ts_xrdataset_augmented(
        self,
        gage_id_lst: Optional[List[str]] = None,
        t_range: Optional[List[str]] = None,
        var_lst: Optional[List[str]] = None,
        time_unit: str = "3h",
        **kwargs,
    ) -> Dict:
        """
        读取增强的时间序列数据，优先从dataaugment缓存文件读取

        Parameters
        ----------
        gage_id_lst : Optional[List[str]], optional
            站点ID列表
        t_range : Optional[List[str]], optional
            时间范围
        var_lst : Optional[List[str]], optional
            变量列表
        time_unit : str, optional
            时间单位，默认"3h"
        **kwargs
            其他参数

        Returns
        -------
        Dict
            包含增强数据的字典，格式与read_ts_xrdataset一致
        """
        if gage_id_lst is None or len(gage_id_lst) == 0:
            return self.read_ts_xrdataset(
                gage_id_lst, t_range, var_lst, **kwargs
            )

        station_id = gage_id_lst[0]

        # 构造增强数据缓存文件路径
        prefix = f"{self.dataset_name}_dataaugment_"
        cache_file_name = f"{prefix}timeseries_{time_unit}_batch_{station_id}_{station_id}.nc"
        cache_file_path = os.path.join(CACHE_DIR, cache_file_name)

        # 检查增强数据文件是否存在
        if os.path.exists(cache_file_path):
            try:
                # 读取增强数据
                ds = xr.open_dataset(cache_file_path)

                # 应用时间范围过滤
                if t_range is not None and len(t_range) >= 2:
                    start_time = pd.to_datetime(t_range[0])
                    end_time = pd.to_datetime(t_range[1])
                    ds = ds.sel(time=slice(start_time, end_time))

                # 应用变量过滤
                if var_lst is not None:
                    available_vars = [
                        var for var in var_lst if var in ds.data_vars
                    ]
                    if available_vars:
                        ds = ds[available_vars]

                print(f"成功从增强数据缓存读取: {cache_file_path}")
                return {time_unit: ds}

            except Exception as e:
                print(f"读取增强数据缓存失败，回退到原始数据: {e}")

        # 如果增强数据不存在或读取失败，回退到原始数据
        return self.read_ts_xrdataset(gage_id_lst, t_range, var_lst, **kwargs)

    def generate_augmented_file_indices(
        self, start_idx: int = 1, end_idx: int = 100, step: int = 1
    ) -> List[int]:
        """
        生成要处理的增强文件编号列表

        Parameters
        ----------
        start_idx : int, optional
            起始编号，默认1
        end_idx : int, optional
            结束编号，默认100
        step : int, optional
            步长，默认1

        Returns
        -------
        List[int]
            文件编号列表
        """
        return list(range(start_idx, end_idx + 1, step))


def _calculate_event_characteristics(
    event: Dict, delta_t_hours: float = 3.0
) -> Dict:
    """
    计算洪水事件的详细特征指标，用于画图和分析

    Parameters
    ----------
        event: dict
            事件字典，包含 'P_eff' (净雨) 和 'Q_obs_eff' (径流) 数组
        delta_t_hours: float
            时段长度（小时），默认3小时

    Returns
    -------
        Dict: 包含计算出的水文特征指标

    Calculated metrics:
        - peak_obs: 洪峰流量 (m³/s)
        - runoff_volume_m3: 洪量 (m³)
        - runoff_duration_hours: 洪水历时 (小时)
        - total_net_rain: 总净雨量 (mm)
        - lag_time_hours: 洪峰雨峰延迟 (小时)
    """
    try:
        # 提取数据
        net_rain = event.get(NET_RAIN, [])
        direct_runoff = event.get(OBS_FLOW, [])

        net_rain = np.array(net_rain)
        direct_runoff = np.array(direct_runoff)

        # 转换为秒
        delta_t_seconds = delta_t_hours * 3600.0

        # 1. 计算洪峰流量
        peak_obs = np.max(direct_runoff)
        if peak_obs < 1e-6:
            return None

        # 2. 计算洪量 (m³)
        runoff_volume_m3 = np.sum(direct_runoff) * delta_t_seconds

        # 3. 计算洪水历时 (小时)
        runoff_indices = np.where(direct_runoff > 1e-6)[0]
        if len(runoff_indices) < 2:
            return None
        runoff_duration_hours = (
            runoff_indices[-1] - runoff_indices[0] + 1
        ) * delta_t_hours

        # 4. 计算总净雨量 (mm)
        total_net_rain = np.sum(net_rain)

        # 5. 计算洪峰雨峰延迟 (小时)
        t_peak_flow_idx = np.argmax(direct_runoff)
        t_peak_rain_idx = np.argmax(net_rain)
        lag_time_hours = (t_peak_flow_idx - t_peak_rain_idx) * delta_t_hours

        # 6. 计算有效降雨时段数
        m_eff = len(net_rain)

        # 7. 计算径流时段数
        n_obs = len(direct_runoff)

        # 8. 计算单位线长度
        n_specific = n_obs - m_eff + 1

        # 返回计算结果
        characteristics = {
            "peak_obs": peak_obs,  # 洪峰流量 (m³/s)
            "runoff_volume_m3": runoff_volume_m3,  # 洪量 (m³)
            "runoff_duration_hours": runoff_duration_hours,  # 洪水历时 (小时)
            "total_net_rain": total_net_rain,  # 总净雨量 (mm)
            "lag_time_hours": lag_time_hours,  # 洪峰雨峰延迟 (小时)
            "m_eff": m_eff,  # 有效降雨时段数
            "n_obs": n_obs,  # 径流时段数
            "n_specific": n_specific,  # 单位线长度
            "delta_t_hours": delta_t_hours,  # 时段长度
        }

        return characteristics

    except Exception as e:
        print(f"计算事件特征时出错: {e}")
        return None


def calculate_events_characteristics(
    events: List[Dict], delta_t_hours: float = 3.0
) -> List[Dict]:
    """
    批量计算多个洪水事件的特征指标

    Args:
        events: 事件列表，每个事件包含 'P_eff' 和 'Q_obs_eff' 数组
        delta_t_hours: 时段长度（小时），默认3小时

    Returns:
        List[Dict]: 包含计算出的水文特征指标的事件列表
    """
    enhanced_events = []

    for i, event in enumerate(events):
        # 计算特征指标
        characteristics = _calculate_event_characteristics(
            event, delta_t_hours
        )

        if characteristics:
            # 将特征指标添加到原事件字典中
            enhanced_event = event.copy()
            enhanced_event.update(characteristics)
            enhanced_events.append(enhanced_event)
        else:
            print(f"⚠️ 事件 {i+1} 特征计算失败，跳过")

    return enhanced_events


def load_and_preprocess_events_unified(
    data_dir: str,
    station_id: Optional[str] = None,
    include_peak_obs: bool = True,
    verbose: bool = True,
    flow_unit: str = "mm/3h",
) -> Optional[List[Dict]]:
    """
    Unified backward-compatible interface function.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.
    station_id : Optional[str], optional
        Basin station ID (default is None).
    include_peak_obs : bool, optional
        Whether to include observed flood peak values (default is True).
    verbose : bool, optional
        Whether to print detailed information (default is True).
    flow_unit : str, optional
        Unit of flow data (default is "mm/3h").

    Returns
    -------
    Optional[List[Dict]]
        List of event dictionaries in standard format, fully compatible with existing unit hydrograph algorithms.
    """
    # 创建数据集实例
    dataset = FloodEventDatasource(
        data_dir,
        flow_unit=flow_unit,
        trange4cache=["1960-01-01 02", "2024-12-31 23"],
    )
    return dataset._load_1basin_flood_events(
        station_id, flow_unit, include_peak_obs, verbose
    )


def check_event_data_nan(all_event_data: List[Dict]):
    """
    检查所有洪水事件数据中的降雨和径流是否有空值，若有则报错并打印详细信息。
    Args:
        all_event_data: 事件字典列表（每个字典包含P_eff、Q_obs_eff、filepath等）
    Raises:
        ValueError: 如果发现空值，抛出异常并打印详细信息
    """
    for event in all_event_data:
        event_name = event.get("filepath", "unknown")
        p_eff = event.get(NET_RAIN)
        q_obs = event.get(OBS_FLOW)
        # 检查降雨
        if p_eff is not None and np.any(np.isnan(p_eff)):
            nan_idx = np.where(np.isnan(p_eff))[0]
            print(f"❌ 场次 {event_name} 的 P_eff 存在空值，索引: {nan_idx}")
            raise ValueError(
                f"Event {event_name} has NaN in P_eff at index {nan_idx}"
            )
        # 检查径流
        if q_obs is not None and np.any(np.isnan(q_obs)):
            nan_idx = np.where(np.isnan(q_obs))[0]
            print(
                f"❌ 场次 {event_name} 的 {OBS_FLOW} 存在空值，索引: {nan_idx}"
            )
            raise ValueError(
                f"Event {event_name} has NaN in {OBS_FLOW} at index {nan_idx}"
            )
