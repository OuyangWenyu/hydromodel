import spotpy
from matplotlib import pyplot as plt
import os
import definitions
import pandas as pd
import numpy as np

def show_calibrate_result(spot_setup, warmup_length,result_file_name, flow_unit="mm day-1"):
    """
    Plot one year result to see the effect of optimized parameters

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
    plt.savefig('..\\hydromodel\\result\\01013500\\result.png', bbox_inches='tight')
    # Plot the best model run
    # Find the run_id with the minimal objective function value
    bestindex, bestobjf = spotpy.analyser.get_minlikeindex(results)
    # Select best model run
    best_model_run = results[bestindex]
    # Filter results for simulation results
    fields = [word for word in best_model_run.dtype.names if word.startswith('sim')]
    best_simulation =list(best_model_run[fields])
    #Add date column for simulation and evaluation result
    test_data_con=pd.read_csv(os.path.join(definitions.ROOT_DIR, "hydromodel", "example", '01013500_lump_p_pe_q.txt'))
    test_data_con['year']=pd.to_datetime(test_data_con['date']).dt.year
    year_unique=test_data_con['year'][warmup_length:].unique()
    for i in year_unique:
        year_index=np.where(test_data_con['year'][warmup_length:]== i)
        fig = plt.figure(figsize=(9, 6))
        ax = plt.subplot(1, 1, 1)
        # TODO: now we just plot all year's data
        ax.plot(best_simulation[year_index[0][0]:year_index[0][-1]], color='black', linestyle='solid', label='Best objf.=' + str(bestobjf))
        ax.plot(spot_setup.evaluation()[year_index[0][0]:year_index[0][-1]], 'r.', markersize=3, label='Observation data')
        plt.xlabel('Number of Observation Points')
        plt.ylabel('Discharge [' + flow_unit + ']')
        plt.legend(loc='upper right')
        plt.title(i)
        plt.tight_layout()
        plt.savefig('..\\hydromodel\\result\\01013500\\' + str(i) + '.png',bbox_inches='tight')


