import logging

import definitions
from xaj.configuration import read_config
from xaj.xaj_bmi import xajBmi

logging.basicConfig(level=logging.INFO)

import pandas as pd
import os
from pathlib import Path
import numpy as np
import fnmatch
import socket
from datetime import datetime

from hydromodel.utils import hydro_utils
from hydromodel.data.data_preprocess import (
    cross_valid_data,
    split_train_test,
)
from xaj.calibrate_sceua_xaj_bmi import calibrate_by_sceua
from xaj.calibrate_ga_xaj_bmi import (
    calibrate_by_ga,
    show_ga_result,
)
from hydromodel.visual.pyspot_plots import show_calibrate_result, show_test_result
from hydromodel.data.data_postprocess import (
    renormalize_params,
    read_save_sceua_calibrated_params,
    save_streamflow,
    summarize_metrics,
    summarize_parameters,
)
from hydromodel.utils import hydro_constant


def test_bmi():
    '''
    model = xajBmi()
    print(model.get_component_name())
    model.initialize("runxaj.yaml")
    print("Start time:", model.get_start_time())
    print("End time:", model.get_end_time())
    print("Current time:", model.get_current_time())
    print("Time step:", model.get_time_step())
    print("Time units:", model.get_time_units())
    print(model.get_input_var_names())
    print(model.get_output_var_names())

    discharge = []
    ET = []
    time = []
    while model.get_current_time() <= model.get_end_time():
        time.append(model.get_current_time())
        model.update()

    discharge=model.get_value("discharge")
    ET=model.get_value("ET")

    results = pd.DataFrame({
                    'discharge': discharge.flatten(),
                    'ET': ET.flatten(),
                })
    results.to_csv('/home/wangjingyi/code/hydro-model-xaj/scripts/xaj.csv')
    model.finalize()
    '''
    # 模型率定
    config = read_config(os.path.relpath("runxaj.yaml"))
    forcing_data = Path(str(definitions.ROOT_DIR) + str(config['forcing_data']))
    train_period = config['train_period']
    test_period = config['test_period']
    # period = config['period']
    json_file = Path(str(definitions.ROOT_DIR) + str(config['json_file']))
    npy_file = Path(str(definitions.ROOT_DIR) + str(config['npy_file']))
    cv_fold = config['cv_fold']
    warmup_length = config['warmup_length']
    # model_info
    model_name = config['model_name']
    source_type = config['source_type']
    source_book = config['source_book']
    # algorithm
    algorithm_name = config['algorithm_name']

    if not (cv_fold > 1):
        # no cross validation
        periods = np.sort(
            [train_period[0], train_period[1], test_period[0], test_period[1]]
        )
    if cv_fold > 1:
        cross_valid_data(json_file, npy_file, periods, warmup_length, cv_fold)
    else:
        split_train_test(json_file, npy_file, train_period, test_period)

    kfold = [
        int(f_name[len("data_info_fold"): -len("_test.json")])
        for f_name in os.listdir(os.path.dirname(forcing_data))
        if fnmatch.fnmatch(f_name, "*_fold*_test.json")
    ]
    kfold = np.sort(kfold)
    for fold in kfold:
        print(f"Start to calibrate the {fold}-th fold")
        train_data_info_file = os.path.join(
            os.path.dirname(forcing_data), f"data_info_fold{str(fold)}_train.json"
        )
        train_data_file = os.path.join(
            os.path.dirname(forcing_data), f"basins_lump_p_pe_q_fold{str(fold)}_train.npy"
        )
        test_data_info_file = os.path.join(
            os.path.dirname(forcing_data), f"data_info_fold{str(fold)}_test.json"
        )
        test_data_file = os.path.join(
            os.path.dirname(forcing_data), f"basins_lump_p_pe_q_fold{str(fold)}_test.npy"
        )
        if (
                os.path.exists(train_data_info_file) is False
                or os.path.exists(train_data_file) is False
                or os.path.exists(test_data_info_file) is False
                or os.path.exists(test_data_file) is False
        ):
            raise FileNotFoundError(
                "The data files are not found, please run datapreprocess4calibrate.py first."
            )
        data_train = hydro_utils.unserialize_numpy(train_data_file)
        data_test = hydro_utils.unserialize_numpy(test_data_file)
        data_info_train = hydro_utils.unserialize_json_ordered(train_data_info_file)
        data_info_test = hydro_utils.unserialize_json_ordered(test_data_info_file)
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        # one directory for one model + one hyperparam setting and one basin
        save_dir = os.path.join(os.path.dirname(forcing_data), current_time + "_" + socket.gethostname() + "_fold" + str(fold))
        if algorithm_name == "SCE_UA":
            random_seed = config['random_seed']
            rep = config['rep']
            ngs = config['ngs']
            kstop = config['kstop']
            peps = config['peps']
            pcento = config['pcento']
            for i in range(len(data_info_train["basin"])):
                basin_id = data_info_train["basin"][i]
                basin_area = data_info_train["area"][i]
                # one directory for one model + one hyperparam setting and one basin
                spotpy_db_dir = os.path.join(
                    save_dir,
                    basin_id,
                )

                if not os.path.exists(spotpy_db_dir):
                    os.makedirs(spotpy_db_dir)
                db_name = os.path.join(spotpy_db_dir, "SCEUA_" + model_name)
                sampler = calibrate_by_sceua(
                    data_train[:, i: i + 1, 0:2],
                    data_train[:, i: i + 1, -1:],
                    db_name,
                    warmup_length=warmup_length,
                    model={
                        'name': model_name,
                        'source_type': source_type,
                        'source_book': source_book
                    },
                    algorithm={
                        'name': algorithm_name,
                        'random_seed': random_seed,
                        'rep': rep,
                        'ngs': ngs,
                        'kstop': kstop,
                        'peps': peps,
                        'pcento': pcento
                    },
                )

                show_calibrate_result(
                    sampler.setup,
                    db_name,
                    warmup_length=warmup_length,
                    save_dir=spotpy_db_dir,
                    basin_id=basin_id,
                    train_period=data_info_train["time"],
                    basin_area=basin_area,
                )
                params = read_save_sceua_calibrated_params(
                    basin_id, spotpy_db_dir, db_name
                )

                model = xajBmi()
                model.initialize(os.path.relpath("runxaj.yaml"), params, data_test[:, i: i + 1, 0:2])
                j = 0
                while j <= len(data_info_test["time"]):
                    model.update()
                    j += 1
                q_sim = model.get_value("discharge")

                qsim = hydro_constant.convert_unit(
                    q_sim,
                    unit_now="mm/day",
                    unit_final=hydro_constant.unit["streamflow"],
                    basin_area=basin_area,
                )

                qobs = hydro_constant.convert_unit(
                    data_test[warmup_length:, i: i + 1, -1:],
                    unit_now="mm/day",
                    unit_final=hydro_constant.unit["streamflow"],
                    basin_area=basin_area,
                )
                test_result_file = os.path.join(
                    spotpy_db_dir,
                    "test_qsim_" + model_name + "_" + str(basin_id) + ".csv",
                )
                pd.DataFrame(qsim.reshape(-1, 1)).to_csv(
                    test_result_file,
                    sep=",",
                    index=False,
                    header=False,
                )
                test_date = pd.to_datetime(
                    data_info_test["time"][warmup_length:]
                ).values.astype("datetime64[D]")
                show_test_result(
                    basin_id, test_date, qsim, qobs, save_dir=spotpy_db_dir
                )
        elif algorithm_name == "GA":
            random_seed = config['random_seed']
            run_counts = config['run_counts']
            pop_num = config['pop_num']
            cross_prob = config['cross_prob']
            mut_prob = config['mut_prob']
            save_freq = config['save_freq']
            for i in range(len(data_info_train["basin"])):
                basin_id = data_info_train["basin"][i]
                basin_area = data_info_train["area"][i]
                # one directory for one model + one hyperparam setting and one basin
                deap_db_dir = os.path.join(save_dir, basin_id)

                if not os.path.exists(deap_db_dir):
                    os.makedirs(deap_db_dir)
                calibrate_by_ga(
                    data_train[:, i: i + 1, 0:2],
                    data_train[:, i: i + 1, -1:],
                    deap_db_dir,
                    warmup_length=warmup_length,
                    model={
                        'name': model_name,
                        'source_type': source_type,
                        'source_book': source_book
                    },
                    ga_param={
                        'name': algorithm_name,
                        'random_seed': random_seed,
                        'run_counts': run_counts,
                        'pop_num': pop_num,
                        'cross_prob': cross_prob,
                        'mut_prob': mut_prob,
                        'save_freq': save_freq
                    },
                )
                show_ga_result(
                    deap_db_dir,
                    warmup_length=warmup_length,
                    basin_id=basin_id,
                    the_data=data_train[:, i: i + 1, :],
                    the_period=data_info_train["time"],
                    basin_area=basin_area,
                    model_info={
                        'name': model_name,
                        'source_type': source_type,
                        'source_book': source_book
                    },
                    train_mode=True,
                )
                show_ga_result(
                    deap_db_dir,
                    warmup_length=warmup_length,
                    basin_id=basin_id,
                    the_data=data_test[:, i: i + 1, :],
                    the_period=data_info_test["time"],
                    basin_area=basin_area,
                    model_info={
                        'name': model_name,
                        'source_type': source_type,
                        'source_book': source_book
                    },
                    train_mode=False,
                )
        else:
            raise NotImplementedError(
                "We don't provide this calibrate method! Choose from 'SCE_UA' or 'GA'!"
            )
        summarize_parameters(save_dir, model_info={
            'name': model_name,
            'source_type': source_type,
            'source_book': source_book
        })
        renormalize_params(save_dir, model_info={
            'name': model_name,
            'source_type': source_type,
            'source_book': source_book
        })
        summarize_metrics(save_dir, model_info={
            'name': model_name,
            'source_type': source_type,
            'source_book': source_book
        })
        save_streamflow(save_dir, model_info={
            'name': model_name,
            'source_type': source_type,
            'source_book': source_book,
        }, fold=fold)
        print(f"Finish calibrating the {fold}-th fold")
