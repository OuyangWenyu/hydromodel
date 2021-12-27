import argparse
import os
import sys

sys.path.append("../..")
from hydromodel.calibrate.calibrate_sceua import calibrate_by_sceua
from hydromodel.utils import hydro_utils


def main(args):
    data = hydro_utils.unserialize_numpy(os.path.join(args.data_dir, "basins_lump_p_pe_q.npy"))
    calibrate_by_sceua(data[:, :, 0:2], data[:, :, -1:], warmup_length=args.warmup_length)


# python calibrate_xaj.py --data_dir "D:\\code\\hydro-model-xaj\\hydromodel\\example" --warmup_length 60
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibrate XAJ model by SCE-UA.')
    parser.add_argument('--data_dir', dest='data_dir', help='the data directory for XAJ model', default="../example",
                        type=str)
    parser.add_argument('--warmup_length', dest='warmup_length', help='the length of warmup period for XAJ model',
                        default=30, type=int)
    the_args = parser.parse_args()
    main(the_args)
