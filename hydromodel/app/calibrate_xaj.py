import argparse
import json
import os
import sys

from hydromodel.calibrate.calibrate_ga import calibrate_by_ga

sys.path.append("../..")
from hydromodel.calibrate.calibrate_sceua import calibrate_by_sceua
from hydromodel.utils import hydro_utils


def main(args):
    data = hydro_utils.unserialize_numpy(
        os.path.join(args.data_dir, "basins_lump_p_pe_q.npy")
    )
    algo = args.algorithm
    warmup = args.warmup_length
    model = args.model_name
    algo_param = args.algorithm_param
    if algo == "SCE_UA":
        calibrate_by_sceua(
            data[:, :, 0:2],
            data[:, :, -1:],
            warmup_length=warmup,
            model=model,
            **algo_param
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
        "--warmup_length",
        dest="warmup_length",
        help="the length of warmup period for XAJ model",
        default=30,
        type=int,
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        help="different implementations for XAJ: 'xaj' or 'xaj_mz'",
        default="xaj",
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
            "random_seed": 2000,
            "rep": 5000,
            "ngs": 7,
            "kstop": 3,
            "peps": 0.1,
            "pcento": 0.1,
        },
        type=json.loads,
    )
    the_args = parser.parse_args()
    main(the_args)
