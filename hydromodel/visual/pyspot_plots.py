import spotpy
from matplotlib import pyplot as plt
import pandas as pd
import definitions
import os
import numpy as np
from spotpy.objectivefunctions import rmse
from spotpy.objectivefunctions import nashsutcliffe
from spotpy.objectivefunctions import bias
from spotpy.objectivefunctions import correlationcoefficient
from hydromodel.utils import stat

def show_calibrate_result(spot_setup, result_file_name, warmup_length,basin_id,split_train_test,flow_unit="mm day-1"):
    """
    Plot all year result to see the effect of optimized parameters

    Parameters
    ----------
    spot_setup
        Spotpy's setup class instance
    result_file_name
        the result file saved after optimizing
    flow_unit
        unit of streamflow

    Returns
    -------
    None
    """
    # Load the results gained with the sceua sampler, stored in SCEUA_xaj.csv
    results = spotpy.analyser.load_csv_results(result_file_name)
    # Plot how the objective function was minimized during sampling
    fig = plt.figure(1, figsize=(9, 6))
    plt.plot(results['like1'])
    plt.ylabel('RMSE')
    plt.xlabel('Iteration')
    # Plot the best model run
    # Find the run_id with the minimal objective function value
    bestindex, bestobjf = spotpy.analyser.get_minlikeindex(results)
    # Select best model run
    best_model_run = results[bestindex]
    # Filter results for simulation results
    fields = [word for word in best_model_run.dtype.names if word.startswith('sim')]
    best_simulation = list(best_model_run[fields])
    # calculation train‘s rmse、nashsutcliffe and bias
    RMSE = rmse(spot_setup.evaluation(), best_simulation)
    NSE = nashsutcliffe(spot_setup.evaluation(), best_simulation)
    BIAS = bias(spot_setup.evaluation(), best_simulation)
    CORRE = correlationcoefficient(spot_setup.evaluation(), best_simulation)
    print("Train's RMSE:", RMSE)
    print("Train's NSE:", NSE)
    print("Train's BIAS:", BIAS)
    print("Train's CORRE:",CORRE)
    test_data = pd.read_csv(os.path.join(definitions.ROOT_DIR, "data",str(basin_id),str(basin_id)+'_lump_p_pe_q.txt'))
    date=pd.to_datetime(test_data['date']).dt.year
    year_unique = date[warmup_length:split_train_test[0]].unique()
    for i in year_unique:
        year_index = np.where(date[warmup_length:split_train_test[0]] == i)
        fig = plt.figure(figsize=(9, 6))
        ax = plt.subplot(1, 1, 1)
        # TODO: now we  plot all year's data
        ax.plot(best_simulation[year_index[0][0]:year_index[0][-1]], color='black', linestyle='solid',
                label='Best objf.=' + str(bestobjf))
        ax.plot(spot_setup.evaluation()[year_index[0][0]:year_index[0][-1]], 'r.', markersize=3,
                label='Observation data')
        plt.xlabel('Number of Observation Points')
        plt.ylabel('Discharge [' + flow_unit + ']')
        plt.legend(loc='upper right')
        plt.title(i)
        plt.tight_layout()
        plt.savefig('..\\data\\'+str(basin_id)+'\\'+ str(i) + '.png', bbox_inches='tight')

def show_test_result(qsim,obs,warmup_length,basin_id):
    eva=obs[warmup_length:,:,:]
    RMSE = rmse(eva,qsim)
    NSE= nashsutcliffe(eva,qsim)
    BIAS=bias(eva,qsim)
    print("Test’s RMSE:", RMSE)
    print("Test’s NSE:", NSE)
    print("Test’s BIAS:", BIAS)
    StatError=stat.statError(eva.reshape(1,-1),qsim.reshape(1,-1))
    print(StatError)
    f=open (r'..\\data\\'+str(basin_id)+'\\'+str(basin_id)+'_zhibiao.txt','w')
    print(StatError, file=f)
    f.close()
    fig = plt.figure(1, figsize=(9, 6))
    ax = plt.subplot(1, 1, 1)
    ax.plot(qsim.flatten(), color='black', linestyle='solid', label='simulation data')
    ax.plot(eva.flatten(), 'r.', markersize=3, label='Observation data')
    plt.legend(loc='upper right')
    # plt.text(50, 16, 'CORR=' + str(Corr), fontsize=15, color='g')
    # plt.text(50, 35, 'CORR=' + str(StatError.get('Corr'))[1:-1], fontsize=15, color='g')
    # plt.text(50, 30, 'RMSE='+str(StatError.get('RMSE'))[1:-1], fontsize=15, color='g')
    # plt.text(50, 25, 'NSE=' + str(StatError.get('NSE'))[1:-1], fontsize=15, color='g')
    # plt.text(50, 20, 'BIAS=' + str(StatError.get('Bias'))[1:-1], fontsize=15, color='g')

    plt.savefig('..\\data\\'+str(basin_id)+'\\test.png', bbox_inches='tight')


