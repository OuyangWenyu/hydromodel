"""
Author: Wenyu Ouyang
Date: 2022-10-25 21:16:22
LastEditTime: 2022-12-08 11:21:22
LastEditors: Wenyu Ouyang
Description: Plots for calibration and testing results
FilePath: \hydro-model-xaj\hydromodel\visual\pyspot_plots.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import spotpy
from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np
from hydromodel.utils import hydro_constant, stat
from hydromodel.utils import hydro_utils
from hydromodel.visual.hydro_plot import plot_sim_and_obs, plot_train_iteration


def show_calibrate_result(
    spot_setup,
    sceua_calibrated_file,
    warmup_length,
    save_dir,
    basin_id,
    train_period,
    result_unit="mm/hour",
    basin_area=None,
    prcp=None,

):
    """
    Plot all year result to see the effect of optimized parameters

    Parameters
    ----------
    spot_setup
        Spotpy's setup class instance
    sceua_calibrated_file
        the result file saved after optimizing
    basin_id
        id of the basin
    train_period
        the period of training data
    result_unit
        the unit of result, default is mm/day, we will convert it to m3/s
    basin_area
        the area of the basin, its unit must be km2

    Returns
    -------
    None
    """
    # Load the results gained with the sceua sampler, stored in SCEUA_xaj.csv
    
    # results = []
    # for chunk in pd.read_csv(sceua_calibrated_file, chunksize=100000 ):
    #  results.append(chunk)
    # results = pd.concat(results)
    results = spotpy.analyser.load_csv_results(sceua_calibrated_file)#读取结果
    # Plot how the objective function was minimized during sampling
    if not os.path.exists(save_dir):  #绘制采样过程中目标函数的最小化情况
        os.makedirs(save_dir)
    plot_train_iteration(    
        results["like1"], os.path.join(save_dir, "train_iteration.png")  #绘制迭代中的RMSE
    )
    # Plot the best model run
    # Find the run_id with the minimal objective function value
    bestindex, bestobjf = spotpy.analyser.get_minlikeindex(results)  #绘制最佳模型图并找到run—id
    # Select best model run
    best_model_run = results[bestindex]   #选择最佳模型结果
    # Filter results for simulation results #最佳模型模拟结果
    fields = [word for word in best_model_run.dtype.names if word.startswith("sim")]
    best_simulation = list(best_model_run[fields]) 
    convert_unit_sim = hydro_constant.convert_unit(
        np.array(best_simulation).reshape(1, -1),
        # np.array(list(map(float, best_simulation)), dtype=float).reshape(1, -1),
        result_unit,
        hydro_constant.unit["streamflow"],
        basin_area=basin_area,
    )
    convert_unit_obs = hydro_constant.convert_unit(
        np.array(spot_setup.evaluation()).reshape(1, -1),   #转换实测的流量为一行
        result_unit,
        hydro_constant.unit["streamflow"],
        basin_area=basin_area,
    )

    # save calibrated results of calibration period      #保存率定的结果
    train_result_file = os.path.join(
        save_dir,
        "train_qsim_" + spot_setup.model["name"] + "_" + str(basin_id) + ".csv",
    )
    pd.DataFrame(convert_unit_sim.reshape(-1, 1)).to_csv(
        train_result_file,
        sep=",",
        index=False,
        header=False,
    )
    # calculation rmse、nashsutcliffe and bias for training period    #计算rmse、nash、bias
    stat_error = stat.statError(
        convert_unit_obs,
        convert_unit_sim,
    )
    print("Training Metrics:", basin_id, stat_error)
    hydro_utils.serialize_json_np(
        stat_error, os.path.join(save_dir, "train_metrics.json")  #计算的误差保存为json格式
    )

    #循还画图
   # time = pd.read_excel('/home/ldaning/code/biye/hydro-model-xaj/hydromodel/example/zhandian/洪水率定时间1.xlsx')
   # calibrate_starttime = pd.to_datetime("2014-7-10 0:00")
   # calibrate_endtime = pd.to_datetime("2020-6-24 0:00")
    #basin_area = float(basin_area)
    #best_simulation = [x * (basin_area*1000000/1000/3600) for x in best_simulation]
    #obs = [x * (basin_area*1000000/1000/3600) for x in spot_setup.evaluation()]
    #for i in range(len(time)):
     ##   if(time.iloc[i,0]<calibrate_endtime):
     #           start_num = (time.iloc[i,0]-calibrate_starttime-pd.Timedelta(hours=warmup_length))/pd.Timedelta(hours=1)   
     #           end_num = (time.iloc[i,1]-calibrate_starttime-pd.Timedelta(hours=warmup_length))/pd.Timedelta(hours=1)
     #           start_period = (time.iloc[i,0]-calibrate_starttime)/pd.Timedelta(hours=1)
     #           end_period = (time.iloc[i,1]-calibrate_starttime)/pd.Timedelta(hours=1)
     #           start_period = int(start_period)
     #           end_period = int(end_period)
     #           start_num = int(start_num)
     #           end_num = int(end_num)
     #           t_range_train_changci = pd.to_datetime(train_period[start_period:end_period]).values.astype("datetime64[h]")
     #           save_fig = os.path.join(save_dir, "train_results"+str(i)+".png")
     #           best_simulation_changci = best_simulation[start_num:end_num]
     #           plot_sim_and_obs(t_range_train_changci, best_simulation_changci, obs[start_num:end_num],prcp[start_num:end_num],save_fig)
                
    t_range_train = pd.to_datetime(train_period[warmup_length:]).values.astype(
        "datetime64[h]"
    )
    save_fig = os.path.join(save_dir, "train_results.png")   #生成结果图
    plot_sim_and_obs(t_range_train, best_simulation, spot_setup.evaluation(),prcp[:],save_fig)
    


def show_test_result(basin_id, test_date, qsim, obs, save_dir,warmup_length,prcp):
    stat_error = stat.statError(obs.reshape(1, -1), qsim.reshape(1, -1))
    print("Test Metrics:", basin_id, stat_error)
    hydro_utils.serialize_json_np(
        stat_error, os.path.join(save_dir, "test_metrics.json")
    )
   # time = pd.read_excel('/home/ldaning/code/biye/hydro-model-xaj/hydromodel/example/zhandian/洪水率定时间1.xlsx')
   #  test_starttime = pd.to_datetime("2020-6-27 0:00")
   #  test_endtime = pd.to_datetime("2021-9-3 0:00")
   #  for i in range(len(time)):
   #      if(test_starttime<time.iloc[i,0]<test_endtime):
   #              start_num = (time.iloc[i,0]-test_starttime-pd.Timedelta(hours=warmup_length))/pd.Timedelta(hours=1)   
   #            end_num = (time.iloc[i,1]-test_starttime-pd.Timedelta(hours=warmup_length))/pd.Timedelta(hours=1)
   #              start_period = (time.iloc[i,0]-test_starttime)/pd.Timedelta(hours=1)
   # #              end_period = (time.iloc[i,1]-test_starttime)/pd.Timedelta(hours=1)
   #              start_period = int(start_period)
   #              end_period = int(end_period)
    #             start_num = int(start_num)
   #  #             end_num = int(end_num)
   #  #             t_range_test_changci = pd.to_datetime(test_date[start_period:end_period]).values.astype("datetime64[h]")
    #             save_fig = os.path.join(save_dir, "test_results"+str(i)+".png")
    #             print(qsim.flatten()[start_num:end_num].shape,obs.flatten()[start_num:end_num].shape, prcp[start_num:end_num].shape)
    #             plot_sim_and_obs(t_range_test_changci, qsim.flatten()[start_num:end_num],obs.flatten()[start_num:end_num], prcp[start_num:end_num],save_fig)
        
    save_fig = os.path.join(save_dir, "test_results.png")
    plot_sim_and_obs(
        test_date[365:],
        qsim.flatten(),
        obs.flatten(),
        prcp[:],
        save_fig,
    )
