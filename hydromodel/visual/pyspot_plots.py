import spotpy
from matplotlib import pyplot as plt
import pandas as pd
import definitions
import os
import numpy as np

def show_calibrate_result(spot_setup, result_file_name, warmup_length,flow_unit="mm day-1"):
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
    test_data = pd.read_csv(os.path.join(definitions.ROOT_DIR, "hydromodel", "example",'hymod_input.csv'), sep=";")
    date=pd.to_datetime(test_data['Date']).dt.year
    year_unique = date[warmup_length:].unique()
    for i in year_unique:
        year_index = np.where(date[warmup_length:] == i)
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


