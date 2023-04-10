import argparse
import json
import socket
import fnmatch
from datetime import datetime
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
import definitions
from hydromodel.calibrate.calibrate_sceua import calibrate_by_sceua
from hydromodel.utils import hydro_utils
from hydromodel.data.data_postprocess import (
    renormalize_params,
    read_save_sceua_calibrated_params,
    save_streamflow,
    summarize_metrics,
    summarize_parameters,
)
from hydromodel.visual.pyspot_plots import show_calibrate_result, show_test_result
from hydromodel.models.xaj import xaj
from hydromodel.calibrate.calibrate_ga import calibrate_by_ga, show_ga_result
from hydromodel.utils import hydro_constant


def calibrate(args):
    exp = args.exp
    warmup = args.warmup_length
    model_info = args.model
    algo_info = args.algorithm
    comment = args.comment
    data_dir = os.path.join(definitions.ROOT_DIR, "hydromodel", "example", exp)
    kfold = [
        int(f_name[len("data_info_fold") : -len("_test.json")])
        for f_name in os.listdir(data_dir)
        if fnmatch.fnmatch(f_name, "*_fold*_test.json")
    ]
    kfold = np.sort(kfold)
    for fold in kfold:
        print(f"Start to calibrate the {fold}-th fold")
        train_data_info_file = os.path.join(
            data_dir, f"data_info_fold{str(fold)}_train.json"
        )
        train_data_file = os.path.join(
            data_dir, f"basins_lump_p_pe_q_fold{str(fold)}_train.npy"
        )
        test_data_info_file = os.path.join(
            data_dir, f"data_info_fold{str(fold)}_test.json"
        )
        test_data_file = os.path.join(
            data_dir, f"basins_lump_p_pe_q_fold{str(fold)}_test.npy"
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
        save_dir = os.path.join(
            data_dir,
            current_time
            + "_"
            + socket.gethostname()
            + "_fold"
            + str(fold)
            + "_"
            + comment,
        )
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
        hydro_utils.serialize_json(vars(args), os.path.join(save_dir, "args.json"))
        if algo_info["name"] == "SCE_UA":
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
                    basin_area=basin_area,
                )

                params = read_save_sceua_calibrated_params(
                    basin_id, spotpy_db_dir, db_name
                )
                # _ is et which we didn't use here
                qsim, _ = xaj(
                    data_test[:, i : i + 1, 0:2],
                    params,
                    warmup_length=warmup,
                    **model_info,
                )

                qsim = hydro_constant.convert_unit(
                    qsim,
                    unit_now="mm/day",
                    unit_final=hydro_constant.unit["streamflow"],
                    basin_area=basin_area,
                )
                qobs = hydro_constant.convert_unit(
                    data_test[warmup:, i : i + 1, -1:],
                    unit_now="mm/day",
                    unit_final=hydro_constant.unit["streamflow"],
                    basin_area=basin_area,
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
                test_date = pd.to_datetime(
                    data_info_test["time"][warmup:]
                ).values.astype("datetime64[D]")
                show_test_result(
                    basin_id, test_date, qsim, qobs, save_dir=spotpy_db_dir
                )
        elif algo_info["name"] == "GA":
            for i in range(len(data_info_train["basin"])):
                basin_id = data_info_train["basin"][i]
                basin_area = data_info_train["area"][i]
                # one directory for one model + one hyperparam setting and one basin
                deap_db_dir = os.path.join(
                    save_dir,
                    basin_id,
                )
                if not os.path.exists(deap_db_dir):
                    os.makedirs(deap_db_dir)
                calibrate_by_ga(
                    data_train[:, i : i + 1, 0:2],
                    data_train[:, i : i + 1, -1:],
                    deap_db_dir,
                    warmup_length=warmup,
                    model=model_info,
                    ga_param=algo_info,
                )
                show_ga_result(
                    deap_db_dir,
                    warmup_length=warmup,
                    basin_id=basin_id,
                    the_data=data_train[:, i : i + 1, :],
                    the_period=data_info_train["time"],
                    basin_area=basin_area,
                    model_info=model_info,
                    train_mode=True,
                )
                show_ga_result(
                    deap_db_dir,
                    warmup_length=warmup,
                    basin_id=basin_id,
                    the_data=data_test[:, i : i + 1, :],
                    the_period=data_info_test["time"],
                    basin_area=basin_area,
                    model_info=model_info,
                    train_mode=False,
                )
        else:
            raise NotImplementedError(
                "We don't provide this calibrate method! Choose from 'SCE_UA' or 'GA'!"
            )
        summarize_parameters(save_dir, model_info)
        renormalize_params(save_dir, model_info)
        summarize_metrics(save_dir, model_info)
        save_streamflow(save_dir, model_info, fold=fold)
        print(f"Finish calibrating the {fold}-th fold")


# NOTE: Before run this command, you should run data_preprocess.py file to save your data as hydro-model-xaj data format,
# the exp must be same as the exp in data_preprocess.py
# python calibrate_xaj.py --exp exp201 --warmup_length 365 --model {\"name\":\"xaj_mz\",\"source_type\":\"sources\",\"source_book\":\"HF\"} --algorithm {\"name\":\"SCE_UA\",\"random_seed\":1234,\"rep\":2000,\"ngs\":20,\"kstop\":3,\"peps\":0.1,\"pcento\":0.1}
# python calibrate_xaj.py --exp exp61561 --warmup_length 365 --model {\"name\":\"xaj_mz\",\"source_type\":\"sources\",\"source_book\":\"HF\"} --algorithm {\"name\":\"GA\",\"random_seed\":1234,\"run_counts\":50,\"pop_num\":50,\"cross_prob\":0.5,\"mut_prob\":0.5,\"save_freq\":1}
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate a hydrological model.")
    parser.add_argument(
        "--exp",
        dest="exp",
        help="An exp is corresponding to a data plan from data_preprocess.py",
        default="exp004",
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
            "source_type": "sources",
            "source_book": "HF",
        },
        type=json.loads,
    )
    parser.add_argument(
        "--algorithm",
        dest="algorithm",
        help="algorithm and its parameters used for calibrating algorithm. "
        "Here are some advices for the algorithmic parameter settings:"
        "rep is the maximum number of calling hydro-model, it is mainly impacted by ngs, if ngs is 30, one population need about 900 evaluations, at this time, 10000 maybe a good choice;"
        "ngs is the number of complex, better larger than your hydro-model-params number (nopt) but not too large, because the number of population's individuals is ngs * (2*nopt+1), larger ngs need more evaluations;"
        "kstop is the number of evolution (not evaluation) loops, some small numbers such as 2, 3, 5, ... are recommended, if too large it is hard to finish optimizing;"
        "peps and pcento are two loop-stop criterion, 0.1 (its unit is %, 0.1 means a relative change of 1/1000) is a good choice",
        # default={
        #     "name": "SCE_UA",
        #     "random_seed": 1234,
        #     "rep": 5000,
        #     "ngs": 20,
        #     "kstop": 3,
        #     "peps": 0.1,
        #     "pcento": 0.1,
        # },
        default={
            "name": "GA",
            "random_seed": 1234,
            "run_counts": 2,
            "pop_num": 50,
            "cross_prob": 0.5,
            "mut_prob": 0.5,
            "save_freq": 1,
        },
        type=json.loads,
    )
    parser.add_argument(
        "--comment",
        dest="comment",
        help="A tag for a plan, we will use it when postprocessing results",
        default="HFsources",
        type=str,
    )
    the_args = parser.parse_args()
    calibrate(the_args)
