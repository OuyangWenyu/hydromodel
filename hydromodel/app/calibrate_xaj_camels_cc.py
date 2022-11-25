import argparse
import json
import socket
from datetime import datetime
import pandas as pd
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
import definitions
from hydromodel.calibrate.calibrate_sceua import calibrate_by_sceua
from hydromodel.utils import hydro_utils
from hydromodel.data.data_postprocess import (
    mm_per_day_to_m3_per_sec,
    renormalize_params,
    save_sceua_calibrated_params,
    save_streamflow,
    summarize_metrics,
    summarize_parameters,
)
from hydromodel.visual.pyspot_plots import show_calibrate_result, show_test_result
from hydromodel.models.xaj import xaj
from hydromodel.calibrate.calibrate_ga import calibrate_by_ga


def main(args):
    exp = args.exp
    warmup = args.warmup_length
    model_info = args.model
    algo_info = args.algorithm
    comment = args.comment
    data_dir = os.path.join(definitions.ROOT_DIR, "hydromodel", "example", exp)
    train_data_info_file = os.path.join(data_dir, "data_info_train.json")
    train_data_file = os.path.join(data_dir, "basins_lump_p_pe_q_train.npy")
    test_data_info_file = os.path.join(data_dir, "data_info_test.json")
    test_data_file = os.path.join(data_dir, "basins_lump_p_pe_q_test.npy")
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
    save_dir = os.path.join(
        data_dir, current_time + "_" + socket.gethostname() + comment
    )
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    if algo_info["name"] == "SCE_UA":
        hydro_utils.serialize_json(vars(args), os.path.join(save_dir, "args.json"))
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
            db_name = os.path.join(spotpy_db_dir, "SCEUA_" + model_info["name"])
            sampler = calibrate_by_sceua(
                data_train[:, i : i + 1, 0:2],
                data_train[:, i : i + 1, -1:],
                db_name,
                warmup_length=warmup,
                model=model_info,
                algorithm=algo_info,
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
                data_test[:, i : i + 1, 0:2], params, warmup_length=warmup, **model_info
            )

            qsim = mm_per_day_to_m3_per_sec(basin_area, qsim)
            qobs = mm_per_day_to_m3_per_sec(
                basin_area, data_test[warmup:, i : i + 1, -1:]
            )
            test_result_file = os.path.join(
                spotpy_db_dir,
                "test_qsim_" + model_info["name"] + "_" + str(basin_id) + ".csv",
            )
            pd.DataFrame(qsim.reshape(-1, 1)).to_csv(
                test_result_file,
                sep=",",
                index=False,
                header=False,
            )
            test_date = data_info_test["time"][warmup:]
            show_test_result(basin_id, test_date, qsim, qobs, save_dir=spotpy_db_dir)
        summarize_parameters(save_dir, model_info)
        renormalize_params(save_dir, model_info)
        summarize_metrics(save_dir, model_info)
        save_streamflow(save_dir, model_info)

    elif algo_info["name"] == "GA":
        # TODO: not finished
        for i in range(len(data_info_train["basin"])):
            calibrate_by_ga(
                data_train[:, i : i + 1, 0:2],
                data_train[:, i : i + 1, -1:],
                warmup_length=warmup,
                model=model_info,
                algorithm=algo_info,
            )
    else:
        raise NotImplementedError(
            "We don't provide this calibrate method! Choose from 'SCE_UA' or 'GA'!"
        )


# NOTE: Before run this command, you should run data_preprocess.py file to save your data as hydro-model-xaj data format,
# the exp must be same as the exp in data_preprocess.py
# python calibrate_xaj_camels_cc.py --exp exp001 --warmup_length 365 --model {\"name\":\"xaj_mz\",\"source_type\":\"sources5mm\"} --algorithm {\"name\":\"SCE_UA\",\"random_seed\":1234,\"rep\":2,\"ngs\":100,\"kstop\":50,\"peps\":0.001,\"pcento\":0.001}
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate a hydrological model.")
    parser.add_argument(
        "--exp",
        dest="exp",
        help="An exp is corresponding to a data plan from data_preprocess.py",
        default="exp001",
        type=str,
    )
    parser.add_argument(
        "--warmup_length",
        dest="warmup_length",
        help="the length of warmup period for hydro model",
        default=365,
        type=int,
    )
    parser.add_argument(
        "--model",
        dest="model",
        help="which hydro model you want to calibrate and the parameters setting for model function, note: not hydromodel parameters but function's parameters",
        default={
            "name": "xaj_mz",
            "source_type": "sources5mm",
        },
        type=json.loads,
    )
    parser.add_argument(
        "--algorithm",
        dest="algorithm",
        help="algorithm and its parameters used for calibrating algorithm",
        default={
            "name": "SCE_UA",
            "random_seed": 1234,
            "rep": 2,
            "ngs": 100,
            "kstop": 50,
            "peps": 0.001,
            "pcento": 0.001,
        },
        type=json.loads,
    )
    parser.add_argument(
        "--comment",
        dest="comment",
        help="directory name",
        default="",
        type=str,
    )
    the_args = parser.parse_args()
    main(the_args)
