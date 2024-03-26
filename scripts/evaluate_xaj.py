import argparse
import socket
import fnmatch
from datetime import datetime
import numpy as np
import pandas as pd
import yaml
import os
import sys
from pathlib import Path
from hydroutils import hydro_file

repo_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
sys.path.append(repo_path)
from hydromodel.trainers.calibrate_sceua import calibrate_by_sceua
from hydromodel.datasets.data_postprocess import (
    renormalize_params,
    read_save_sceua_calibrated_params,
    save_streamflow,
    summarize_metrics,
    summarize_parameters,
)
from hydromodel.trainers.train_utils import show_calibrate_result, show_test_result
from hydromodel.models.xaj import xaj
from hydromodel.trainers.calibrate_ga import calibrate_by_ga, show_ga_result


def read_yaml_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def evaluate(args):
    exp = args.exp
    warmup = args.warmup_length
    cali_dir = Path(os.path.join(repo_path, "result", exp))
    cali_config = read_yaml_config(os.path.join(cali_dir, "config.yaml"))
    kfold = np.sort(kfold)
    for fold in kfold:
        print(f"Start to calibrate the {fold}-th fold")
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        save_dir = os.path.join(
            cali_dir,
            current_time
            + "_"
            + socket.gethostname()
            + "_fold"
            + str(fold)
        )
        # 读输入文件
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
        hydro_file.serialize_json(vars(args), os.path.join(save_dir, "args.json"))
        if algo_info["name"] == "SCE_UA":
            for i in range(len(data_info_train["basin"])):
                basin_id = data_info_train["basin"][i]
                basin_area = data_info_train["area"][i]
                # one directory for one model + one hyperparam setting and one basin
                spotpy_db_dir = os.path.join(  # 一个模型一个文件夹
                    save_dir,
                    basin_id,
                )
                if not os.path.exists(spotpy_db_dir):
                    os.makedirs(spotpy_db_dir)
                db_name = os.path.join(spotpy_db_dir, "SCEUA_" + model_info["name"])
                show_calibrate_result(  # 展示率定结果
                    sampler.setup,
                    db_name,
                    warmup_length=warmup,
                    save_dir=spotpy_db_dir,
                    basin_id=basin_id,
                    train_period=data_info_train["time"],
                    basin_area=basin_area,
                    prcp=data_train[365:, i : i + 1, 0:1].flatten(),
                )

                params = read_save_sceua_calibrated_params(  # 保存率定的参数文件
                    basin_id, spotpy_db_dir, db_name
                )
                # _ is et which we didn't use here
                qsim, _ = xaj(  # 计算模拟结果
                    data_test[:, i : i + 1, 0:2],
                    params,
                    warmup_length=0,
                    **model_info,
                )

                qsim = units.convert_unit(
                    qsim,
                    # TODO: to unify "mm/hour"
                    unit_now="mm/day",
                    unit_final=units.unit["streamflow"],
                    basin_area=basin_area,
                )
                qobs = units.convert_unit(
                    data_test[warmup:, i : i + 1, -1:],
                    # TODO: to unify "mm/hour"
                    unit_now="mm/day",
                    unit_final=units.unit["streamflow"],
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
                test_date = pd.to_datetime(data_info_test["time"][:]).values.astype(
                    "datetime64[h]"
                )
                show_test_result(
                    basin_id,
                    test_date,
                    qsim,
                    qobs,
                    save_dir=spotpy_db_dir,
                    warmup_length=warmup,
                    prcp=data_test[365:, i : i + 1, 0:1].flatten(),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate a calibrated hydrological model."
    )
    parser.add_argument(
        "--exp",
        dest="exp",
        help="An exp is corresponding to a data plan from calibrate_xaj.py",
        default="expcamels001",
        type=str,
    )
    parser.add_argument(
        "--warmup_length",
        dest="warmup_length",
        help="the length of warmup period for hydro model",
        default=365,
        type=int,
    )
    the_args = parser.parse_args()
    evaluate(the_args)
