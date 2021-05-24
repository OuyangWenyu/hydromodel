"""设置或读取新安江模型的输入数据，"""
from typing import Union
import numpy as np
import pandas as pd

from src.data.data_base import DatasetBase
from src.pet.pet4daymet import pm_fao56
from src.utils.hydro_utils import t_range_days


def initialize_condition(basin_ids: list,
                         wu_range: Union[list, float] = 0.,
                         wl_range: Union[list, float] = 1.,
                         wd_range: Union[list, float] = 20.,
                         fr0_range: Union[list, float] = 0.001,
                         s0_range: Union[list, float] = 0.) -> pd.DataFrame:
    # 初始化一部分模型所需的参数初值，包括流域上层、下层、深层张力水蓄量初值（三层蒸发模型计算使用的参数），
    #                               分水源计算的产流面积初值、自由水蓄量初值
    if type(wu_range) is list:
        wu_range = np.random.uniform(wu_range[0], wu_range[1], len(basin_ids))
    else:
        wu_range = np.full(len(basin_ids), wu_range)
    if type(wl_range) is list:
        wl_range = np.random.uniform(wl_range[0], wl_range[1], len(basin_ids))
    else:
        wl_range = np.full(len(basin_ids), wl_range)
    if type(wd_range) is list:
        wd_range = np.random.uniform(wd_range[0], wd_range[1], len(basin_ids))
    else:
        wd_range = np.full(len(basin_ids), wd_range)
    if type(fr0_range) is list:
        fr0_range = np.random.uniform(fr0_range[0], fr0_range[1], len(basin_ids))
    else:
        fr0_range = np.full(len(basin_ids), fr0_range)
    if type(s0_range) is list:
        s0_range = np.random.uniform(s0_range[0], s0_range[1], len(basin_ids))
    else:
        s0_range = np.full(len(basin_ids), s0_range)
    initial_conditions = pd.DataFrame(np.array([wu_range, wl_range, wd_range, fr0_range, s0_range]).T,
                                      index=basin_ids, columns=['WU', 'WL', 'WD', 'FR0', 'S0'])
    return initial_conditions


def init_parameters(basin_ids: list,
                    wum_range: Union[list, float] = 27.764,
                    wlm_range: Union[list, float] = 84.393,
                    wdm_range: Union[list, float] = 70.358,
                    b_range: Union[list, float] = .4,
                    imp_range: Union[list, float] = .04,
                    c_range: Union[list, float] = .2,
                    sm_range: Union[list, float] = 51.634,
                    ex_range: Union[list, float] = 1.002,
                    ki_range: Union[list, float] = .379,
                    kg_range: Union[list, float] = .321,
                    ci_range: Union[list, float] = .986,
                    cg_range: Union[list, float] = .284,
                    cr_range: Union[list, float] = .766,
                    l_range: Union[list, float] = 1.791) -> pd.DataFrame:
    """输入数据，初始化参数
    Here, include 13 params: WUM,WLM,WM,B,IMP,C,SM,EX,KSS,KKSS,KKG,CR,L.

    Parameters
    ----------
    l_range # L: 河网汇流迟滞时间
    cr_range # CR: 河网蓄水消退系数
    cg_range # KKG: 地下水消退系数
    ci_range # KKSS: 壤中流消退系数
    ki_range # KSS: 壤中流出流系数
    kg_range # KG: 地下水出流系数  Generally, KSS+KG=0.7
    ex_range # EX: 表层自由水蓄水容量曲线指数
    sm_range  # SM: 表层自由水蓄水容量
    c_range # C: 深层蒸散发折算系数
    imp_range # IMP: 流域不透水系数
    b_range # B: 流域蓄水容量曲线的方次
    wdm_range # WDM: 流域deep layer平均蓄水容量
    wlm_range  # WLM: 流域下层土壤平均蓄水容量
    wum_range # WUM: 流域上层土壤平均蓄水容量
    We don't use the following params:
     # K: 蒸发系数.
     # XE: 单元河段马斯京根模型参数X值
     # KE: 单元河段马斯京根模型参数K值
    We directly use potential et, pet = K * water surface et; don't use river routing module either.

    Returns
    -------
    xaj_params:Series 新安江模型参数
    """
    if type(wum_range) is list:
        wum_range = np.random.uniform(wum_range[0], wum_range[1], len(basin_ids))
    else:
        wum_range = np.full(len(basin_ids), wum_range)
    if type(wlm_range) is list:
        wlm_range = np.random.uniform(wlm_range[0], wlm_range[1], len(basin_ids))
    else:
        wlm_range = np.full(len(basin_ids), wlm_range)
    if type(wdm_range) is list:
        wdm_range = np.random.uniform(wdm_range[0], wdm_range[1], len(basin_ids))
    else:
        wdm_range = np.full(len(basin_ids), wdm_range)
    if type(b_range) is list:
        b_range = np.random.uniform(b_range[0], b_range[1], len(basin_ids))
    else:
        b_range = np.full(len(basin_ids), b_range)
    if type(imp_range) is list:
        imp_range = np.random.uniform(imp_range[0], imp_range[1], len(basin_ids))
    else:
        imp_range = np.full(len(basin_ids), imp_range)
    if type(c_range) is list:
        c_range = np.random.uniform(c_range[0], c_range[1], len(basin_ids))
    else:
        c_range = np.full(len(basin_ids), c_range)
    if type(sm_range) is list:
        sm_range = np.random.uniform(sm_range[0], sm_range[1], len(basin_ids))
    else:
        sm_range = np.full(len(basin_ids), sm_range)
    if type(ex_range) is list:
        ex_range = np.random.uniform(ex_range[0], ex_range[1], len(basin_ids))
    else:
        ex_range = np.full(len(basin_ids), ex_range)
    if type(ki_range) is list:
        ki_range = np.random.uniform(ki_range[0], ki_range[1], len(basin_ids))
    else:
        ki_range = np.full(len(basin_ids), ki_range)
    if type(kg_range) is list:
        kg_range = np.random.uniform(kg_range[0], kg_range[1], len(basin_ids))
    else:
        kg_range = np.full(len(basin_ids), kg_range)
    if type(ci_range) is list:
        ci_range = np.random.uniform(ci_range[0], ci_range[1], len(basin_ids))
    else:
        ci_range = np.full(len(basin_ids), ci_range)
    if type(cg_range) is list:
        cg_range = np.random.uniform(cg_range[0], cg_range[1], len(basin_ids))
    else:
        cg_range = np.full(len(basin_ids), cg_range)
    if type(cr_range) is list:
        cr_range = np.random.uniform(cr_range[0], cr_range[1], len(basin_ids))
    else:
        cr_range = np.full(len(basin_ids), cr_range)
    if type(l_range) is list:
        l_range = np.random.uniform(l_range[0], l_range[1], len(basin_ids))
    else:
        l_range = np.full(len(basin_ids), l_range)
    xaj_params = pd.DataFrame(
        np.array([wum_range, wlm_range, wdm_range, b_range, imp_range, c_range, sm_range, ex_range, ki_range, kg_range,
                  ci_range, cg_range, cr_range, l_range]).T,
        columns=['WUM', 'WLM', 'WDM', 'B', 'IMP', 'C', 'SM', 'EX', 'KI', 'KG', 'CI', 'CG', 'CR', 'L'],
        index=basin_ids)
    return xaj_params


def read_data(dataset: DatasetBase,
              basin_ids: list,
              t_range: list):
    """
    Parameters
    ----------
    dataset: dataset for the model
    basin_ids: the ids of basins
    t_range: the time range

    Returns
    -------
    basin_area, pet, rainfall and streamflow
    """
    var_lst = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
    forcing_data = dataset.read_relevant_cols(basin_ids, t_range, var_lst)
    t_min = forcing_data[:, :, 5]
    t_max = forcing_data[:, :, 4]
    # average over the daylight period of the day, W/m^2 -> average over the day, MJ m-2 day-1
    r_surf = forcing_data[:, :, 2] * forcing_data[:, :, 0] * 1e-6
    attr_lst = ["gauge_lat", "elev_mean", "area_gages2"]
    attr_data = dataset.read_constant_cols(basin_ids, attr_lst)
    lat = attr_data[:, 0]
    # ° -> rad
    phi = lat * np.pi / 180.0
    elevation = attr_data[:, 1]
    dtype = t_min.dtype
    time_range_lst = t_range_days(t_range)
    doy_values = pd.to_datetime(time_range_lst).dayofyear.astype(dtype).values
    doy = np.expand_dims(doy_values, axis=0).repeat(len(basin_ids), axis=0)
    e_a = forcing_data[:, :, 6] * 1e-3

    phi_ = np.expand_dims(phi, axis=1).repeat(len(time_range_lst), axis=1)
    elev_ = np.expand_dims(elevation, axis=1).repeat(len(time_range_lst), axis=1)
    pet_fao56 = pm_fao56(t_min, t_max, r_surf, phi_, elev_, doy, e_a=e_a)
    basin_areas = attr_data[:, 2]
    rainfalls = forcing_data[:, :, 1]

    streamflows = dataset.read_target_cols(basin_ids, t_range, ["usgsFlow"])
    return basin_areas, rainfalls, pet_fao56, streamflows
