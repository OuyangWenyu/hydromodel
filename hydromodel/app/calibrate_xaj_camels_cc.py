import argparse
import os
import sys
import pandas as pd
from pathlib import Path

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
import definitions
from hydromodel.calibrate.calibrate_sceua import calibrate_by_sceua
from hydromodel.utils import hydro_utils
from hydromodel.data.data_postprocess import (
    mm_per_day_to_m3_per_sec,
    save_sceua_calibrated_params,
)
from hydromodel.visual.pyspot_plots import show_calibrate_result, show_test_result
from hydromodel.models.xaj import xaj
from hydromodel.calibrate.calibrate_ga import calibrate_by_ga


def main(args):
    algo = args.algorithm
    warmup = args.warmup_length
    route_method = args.route_method
    model = args.model_name
    data_dir = args.data_dir
    hyperparam_file = args.hyperparam_file
    algo_param_file = os.path.join(data_dir, hyperparam_file)
    algo_param = hydro_utils.unserialize_json_ordered(algo_param_file)
    train_data_info_file = os.path.join(data_dir, "data_info_train.json")
    train_data_file = os.path.join(data_dir, "basins_lump_p_pe_q_train.npy")
    test_data_info_file = os.path.join(data_dir, "data_info_test.json")
    test_data_file = os.path.join(data_dir, "basins_lump_p_pe_q_test.npy")
    data_train = hydro_utils.unserialize_numpy(train_data_file)
    data_test = hydro_utils.unserialize_numpy(test_data_file)
    data_info_train = hydro_utils.unserialize_json_ordered(train_data_info_file)
    data_info_test = hydro_utils.unserialize_json_ordered(test_data_info_file)
    if algo == "SCE_UA":
        for i in range(len(data_info_train["basin"])):
            basin_id = data_info_train["basin"][i]
            basin_area = data_info_train["area"][i]
            hyper_param = {}
            for key, value in algo_param.items():
                if key == "basin":
                    assert value[i] == basin_id
                    continue
                hyper_param[key] = value[i]
            # one directory for one model + one hyperparam setting and one basin
            spotpy_db_dir = os.path.join(
                definitions.ROOT_DIR,
                "hydromodel",
                "example",
                model + "_" + hyperparam_file[:-5],
                basin_id,
            )
            if not os.path.exists(spotpy_db_dir):
                os.makedirs(spotpy_db_dir)
            db_name = os.path.join(spotpy_db_dir, "SCEUA_" + model)
            sampler = calibrate_by_sceua(
                data_train[:, i : i + 1, 0:2],
                data_train[:, i : i + 1, -1:],
                db_name,
                warmup_length=warmup,
                model=model,
                **hyper_param
            )

            show_calibrate_result(
                sampler.setup,
                db_name,
                warmup_length=warmup,
                save_dir=spotpy_db_dir,
                basin_id=basin_id,
                train_period=data_info_train["time"],
            )

            params = save_sceua_calibrated_params(basin_id, spotpy_db_dir, db_name)
            # _ is et which we didn't use here
            qsim, _ = xaj(
                data_test[:, i : i + 1, 0:2],
                params,
                warmup_length=warmup,
                route_method=route_method,
            )

            qsim = mm_per_day_to_m3_per_sec(basin_area, qsim)
            qobs = mm_per_day_to_m3_per_sec(
                basin_area, data_test[warmup:, i : i + 1, -1:]
            )
            test_result_file = os.path.join(
                spotpy_db_dir, "test_qsim_" + model + "_" + str(basin_id) + ".csv"
            )
            pd.DataFrame(qsim.reshape(-1, 1)).to_csv(
                test_result_file,
                sep=",",
                index=False,
                header=False,
            )
            test_date = data_info_test["time"][warmup:]
            show_test_result(basin_id, test_date, qsim, qobs, save_dir=spotpy_db_dir)

    elif algo == "GA":
        # TODO: not finished
        for i in range(len(data_info_train["basin"])):
            basin_id = data_info_train["basin"][i]
            hyper_param = {}
            for key, value in algo_param.items():
                if key == "basin":
                    assert value[i] == basin_id
                    continue
                hyper_param[key] = value[i]
            calibrate_by_ga(
                data_train[:, i : i + 1, 0:2],
                data_train[:, i : i + 1, -1:],
                warmup_length=warmup,
                model=model,
                **hyper_param
            )
    else:
        raise NotImplementedError(
            "We don't provide this calibrate method! Choose from 'SCE_UA' or 'GA'!"
        )


# before run this command, you should run data_preprocess.py file to save your data as hydro-model-xaj data format
# you also need to write a hyperparam file for algorithm: SCE_UA -- hyperparam_SCE_UA.json or GA -- hyperparam_GA.json
# TODO: an example file could be found in example directory
# python calibrate_xaj.py --data_dir "D:\\code\\hydro-model-xaj\\hydromodel\\example" --warmup_length 60
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate XAJ model by SCE-UA.")
    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        help="the data directory for XAJ model",
        default="C:\\Users\\wenyu\\.hydrodataset\\cache",
        type=str,
    )
    parser.add_argument(
        "--warmup_length",
        dest="warmup_length",
        help="the length of warmup period for XAJ model",
        default=365,
        type=int,
    )
    parser.add_argument(
        "--route_method",
        dest="route_method",
        help="ethod from mizuRoute for XAJ model",
        default="MZ",
        type=str,
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        help="different implementations for XAJ: 'xaj' or 'xaj_mz'",
        default="xaj_mz",
        type=str,
    )
    parser.add_argument(
        "--algorithm",
        dest="algorithm",
        help="calibrate algorithm: SCE_UA (default) or GA",
        default="SCE_UA",
        type=str,
    )
    parser.add_argument(
        "--hyperparam_file",
        dest="hyperparam_file",
        help="hyperparam_file used for calibrating algorithm. its parent dir is data_dir",
        default="hyperparam_SCE_UA_rep1000_ngs1000.json",
        type=str,
    )
    the_args = parser.parse_args()
    main(the_args)
