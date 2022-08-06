import argparse
import json
import os
import sys
import pandas as pd


sys.path.append("../..")
import spotpy
from hydromodel.calibrate.calibrate_sceua import calibrate_by_sceua, SpotSetup
from hydromodel.utils import hydro_utils
# from hydromodel.data.data_preprocess import split_train_data,split_test_data,Unit_Conversion
from hydromodel.data.data_preprocess import split_train_data,split_test_data,calibrate_params,Unit_Conversion
from hydromodel.visual.pyspot_plots import show_calibrate_result,show_test_result
from hydromodel.models.xaj import xaj
from hydromodel.calibrate.calibrate_ga import calibrate_by_ga

def main(args):
    algo = args.algorithm
    basin_id=args.basin_id
    train_period=args.train_period
    test_period = args.test_period
    warmup = args.warmup_length
    route_method = args.route_method
    model = args.model_name
    algo_param = args.algorithm_param
    data = hydro_utils.unserialize_numpy(
        os.path.join(args.data_dir,basin_id,"basins_lump_p_pe_q.npy")
    )
    txt_file=pd.read_csv(os.path.join(
        args.data_dir,basin_id,basin_id+'_lump_p_pe_q.txt'))
    data_train=split_train_data(data,txt_file,train_period)
    data_test = split_test_data(data, txt_file, test_period)
    if algo == "SCE_UA":

        calibrate_by_sceua(
            data_train[:, :, 0:2],
            data_train[:, :, -1:],
            warmup_length=warmup,
            model=model,
            **algo_param
        )

        spot_setup = SpotSetup(
            data_train[:, :, 0:2],
            data_train[:, :, -1:],
            warmup_length=warmup,
            obj_func=spotpy.objectivefunctions.rmse
        )

        show_calibrate_result(
            spot_setup,
            "SCEUA_xaj_mz",
            warmup_length=warmup,
            basin_id=basin_id,
            train_period=train_period
        )

        params= calibrate_params(
            basin_id,
           "SCEUA_xaj_mz"
        )

        qsim = xaj(
            data_test[:, :, 0:2],
            params,
            warmup_length=warmup,
            route_method=route_method,
        )

        qsim,qobs=Unit_Conversion(
            basin_id,
            qsim,
            data_test[:, :, -1:]
        )
        pd.DataFrame(qsim.reshape(-1,1)).to_csv('..\\example\\' + str(basin_id) + '\\' + str(basin_id) + '_qsim.txt', sep=',',
                index=False, header=True)

        show_test_result(
            qsim,
            qobs,
            warmup_length=warmup,
            basin_id=basin_id
        )

    elif algo == "GA":
        calibrate_by_ga(
            data[:, :, 0:2],
            data[:, :, -1:],
            warmup_length=warmup,
            model=model,
            **algo_param
        )
    else:
        raise NotImplementedError(
            "We don't provide this calibrate method! Choose from 'SCE_UA' or 'GA'!"
        )


# python calibrate_xaj.py --data_dir "D:\\code\\hydro-model-xaj\\hydromodel\\example" --warmup_length 60
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate XAJ model by SCE-UA.")
    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        help="the data directory for XAJ model",
        default="../example",
        type=str,
    )
    parser.add_argument(
        "--basin_id",
        dest="basin_id",
        help="the basin for XAJ model",
        default="92354",
        type=str,
    )
    parser.add_argument(
        "--train_period",
        dest="train_period",
        help="the train period for XAJ model",
        default=["2014-10-01","2020-10-01"],
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--test_period",
        dest="test_period",
        help="the test period for XAJ model",
        default=["2019-10-01", "2021-10-01"],
        type=str,
        nargs="+",
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
        "--algorithm_param",
        dest="algorithm_param",
        help="parameters set for calibrate algorithm",
        default={
            "random_seed": 1234,
            "rep":20,
            "ngs":5000,
            "kstop":10000,
            "peps": 0.01,
            "pcento": 0.01,
        },
        type=json.loads,
    )
    the_args = parser.parse_args()
    main(the_args)
