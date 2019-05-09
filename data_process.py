"""设置或读取新安江模型的输入数据，"""
import pandas as pd
import netCDF4
import numpy as np
import os
import logging


def init_parameters():
    """输入数据，初始化参数"""
    # 流域属性值的读取，包括流域面积
    property = pd.Series({'basin_area/km^2': 343})
    # 模型计算的简单配置，包括场次洪水洪号，计算的时间步长等
    config = pd.Series({'flood_ids': [19980531], 'time_interval/h': 1})
    # 初始化一部分模型所需的参数初值，包括流域上层、下层、深层张力水蓄量初值（三层蒸发模型计算使用的参数），
    #                               分水源计算的产流面积初值、自由水蓄量初值
    initial_conditions = pd.Series([0, 1, 20, 0.001, 0.00], index=['WU', 'WL', 'WD', 'FR0', 'S0'])  # 如何取值？
    # 然后读取场次洪水数据和每场次洪水数据前若干日的日降雨和蒸发数据（计算前期影响雨量作为初始土壤含水量依据）
    day_rain_evapor = pd.read_csv('data/example_day_rain_evapor.txt', sep='\t')
    flood_data = pd.read_csv("data/example_flood.txt", sep='\t')
    # 初始化模型参数值，才可使用模型进行计算，新安江模型有16个参数值，包括：K,IMP,B,WM,WUM,WLM,C,SM,EX,KG,KSS,KKG,KKSS,UH,KE,XE.
    # 初值如何选？
    # K: 蒸发系数
    # IMP: 流域不透水系数
    # B: 流域蓄水容量曲线的方次
    # WM: 流域平均蓄水容量
    # WUM: 流域上层土壤平均蓄水容量
    # WLM: 流域下层土壤平均蓄水容量
    # C: 深层蒸散发折算系数
    # SM: 表层自由水蓄水容量
    # EX: 表层自由水蓄水容量曲线指数
    # KG: 地下水出流系数
    # KSS: 壤中流出流系数
    # KKG: 地下水消退系数
    # KKSS: 壤中流消退系数
    # UH: 单元流域上地面径流单位线
    # KE: 单元河段马斯京根模型参数K值
    # XE: 单元河段马斯京根模型参数X值
    xaj_params = pd.Series([.998, .040, 182.515, .400, 27.764, 84.393, .200, 51.634, 1.002, .379, .986, .284, .766,
                            1.791, 1.001, .029],
                           index=['K', 'IMP', 'WM', 'B', 'WUM', 'WLM', 'C', 'SM', 'EX', 'KSS', 'KG', 'KKSS', 'KKG',
                                  'UH', 'KE', 'XE'])
    return property, config, initial_conditions, day_rain_evapor, flood_data, xaj_params


def txt2nc():
    """把ASCII数据写入netcdf文件。txt格式固定：数据第一行为列标题，下面各行为数据。很难写通用的，因此特例特写。以下是个例子。。

    Parameters
    ----------

    Returns
    -------
    out :
    """

    # 读取txt数据
    PATH = '.'
    LL = np.loadtxt('%s/global_soils_default.txt' % PATH);
    LL = LL[:, 2:4]
    OBS = np.loadtxt('%s/VIC_GRDC_Monthly_Climatology.txt' % PATH, delimiter=',', skiprows=1)

    # NC file setup
    root_grp = netCDF4.Dataset('vic_LHS_climatology.nc', 'w', format='NETCDF4')
    root_grp.description = 'Results from VIC 10K Latin Hypercube ensemble, 60-year simulation on Blue Waters'

    # dimensions
    root_grp.createDimension('lat', 180)
    root_grp.createDimension('lon', 360)
    root_grp.createDimension('month', 12)
    ensemble = root_grp.createDimension('ensemble', 10000)

    # variables
    latitudes = root_grp.createVariable('latitude', 'f4', ('lat',))
    longitudes = root_grp.createVariable('longitude', 'f4', ('lon',))
    vic_runoff = root_grp.createVariable('vic_runoff', 'f4', ('lat', 'lon', 'ensemble', 'month',), fill_value=-9999.0)
    obs_runoff = root_grp.createVariable('obs_runoff', 'f4', ('lat', 'lon', 'month'), fill_value=-9999.0)

    vic_runoff.units = 'mm/month'
    obs_runoff.units = 'mm/month'

    # set the variables we know first
    latitudes = np.arange(-90.5, 89.5, 1.0)
    longitudes = np.arange(0.5, 360.5, 1.0)
    cellsdone = 0

    # set the variables we know first
    latitudes = np.arange(-90.5, 89.5, 1.0)
    longitudes = np.arange(0.5, 360.5, 1.0)
    cellsdone = 0

    for lati, lat in enumerate(latitudes):
        for loni, lon in enumerate(longitudes):

            # grab the index of the 0-15836 list of grid cells
            i = np.where((np.floor(LL[:, 0]) == np.floor(lat)) & (np.floor(LL[:, 1]) == np.floor(lon)))

            # if this is one of our land surface grid cells...
            if (np.size(i) > 0):
                current_obs = OBS[
                              np.where((np.floor(OBS[:, 0]) == np.floor(lat)) & (np.floor(OBS[:, 1]) == np.floor(lon))),
                              2:15]
                current_obs = np.squeeze(current_obs)

                if (current_obs.size > 0):
                    obs_runoff[lati, loni, :] = current_obs

                # keep values in memory until ready to write a big chunk to NC file
                tempstore = np.zeros((len(ensemble), 12), float)

                for filenum in range(0, 200):

                    output_filename = '%s' % PATH + '/txt/file_' + '%d' % filenum + '/txt/hcube_lat_' + '%.6f' % LL[
                        i, 0] + '_long_' + '%.6f' % LL[i, 1] + '.txt'

                    if (os.path.isfile(output_filename)):
                        try:
                            output = np.loadtxt(output_filename)
                        except:
                            pass

                        # write the VIC output data
                        startix = filenum * 50
                        tempstore[startix:(startix + output.shape[0]), :] = output  # ensembles x months

                vic_runoff[lati, loni, :, :] = tempstore  # write all ensembles to netcdf
                cellsdone = cellsdone + 1
                print
                cellsdone

    root_grp.close()
    return
