import spotpy
from matplotlib import pyplot as plt


def show_calibrate_result(spot_setup, result_file_name, flow_unit="mm day-1"):
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

    fig = plt.figure(figsize=(9, 6))
    ax = plt.subplot(1, 1, 1)
    ax.plot(best_simulation[365:730], color='black', linestyle='solid', label='Best objf.=' + str(bestobjf))
    ax.plot(spot_setup.evaluation()[365:730], 'r.', markersize=3, label='Observation data')
    plt.xlabel('Number of Observation Points')
    plt.ylabel('Discharge [' + flow_unit + ']')
    plt.legend(loc='upper right')
