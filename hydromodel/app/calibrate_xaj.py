import argparse
import os
import sys
import numpy as np
import pandas as pd

import definitions
from hydromodel.calibrate.calibrate_xaj_sceua import calibrate_xaj_sceua

sys.path.append("../..")


def main():
    """
    Main function which is called from the command line. Entrypoint for training all ML models.
    """
    root_dir = definitions.ROOT_DIR
    data = pd.read_csv(os.path.join(root_dir, "hydromodel", "example", 'hymod_input.csv'), sep=";")
    p_and_e_df = data[['rainfall[mm]', 'TURC [mm d-1]']]
    p_and_e = np.expand_dims(p_and_e_df.values, axis=1)
    km2tom2 = 1e6
    # 1 m = 1000 mm
    mtomm = 1000
    # 1 day = 24 * 3600 s
    daytos = 24 * 3600
    qobs_ = np.expand_dims(data[['Discharge[ls-1]']].values, axis=1)
    basin_area = 1.783
    # trans l/s to mm/day
    qobs = qobs_ * 1e-3 / (basin_area * km2tom2) * mtomm * daytos
    calibrate_xaj_sceua(p_and_e, qobs, warmup_length=30)


# python script_args.py --year_range 1990 2011
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibrate XAJ model by SCE-UA.')
    the_args = parser.parse_args()
    main()
