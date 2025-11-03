"""
Author: Wenyu Ouyang
Date: 2024-03-25 09:21:56
LastEditTime: 2024-09-12 08:46:51
LastEditors: Wenyu Ouyang
Description: preprocess data in an exp before training
FilePath: \hydromodel\scripts\preprocess.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

from pathlib import Path
import sys
import os
import argparse
from hydrodatasource.processor.dmca_esr import get_rr_events, plot_rr_events

current_script_path = Path(os.path.realpath(__file__))
repo_path = current_script_path.parent.parent
sys.path.append(str(repo_path))
from hydromodel.datasets.data_preprocess import (
    get_basin_area,
    get_ts_from_diffsource,
)


def main(args):
    data_path = args.data_dir
    result_dir = args.result_dir
    data_type = args.data_type
    basin_ids = args.basin_id
    periods = args.period
    exp = args.exp
    rr_event = args.rr_event
    where_save = Path(os.path.join(result_dir, exp))
    if os.path.exists(where_save) is False:
        os.makedirs(where_save)
    if rr_event > 0:
        ts_data = get_ts_from_diffsource(
            data_type, data_path, periods, basin_ids
        )
        basin_area = get_basin_area(basin_ids, data_type, data_path)
        rr_events = get_rr_events(ts_data["prcp"], ts_data["flow"], basin_area)
        for basin, event in rr_events.items():
            basin_rr_dir = os.path.join(where_save, f"{basin}_rr_events")
            plot_rr_events(
                event,
                ts_data["prcp"].sel(basin=basin),
                ts_data["flow"].sel(basin=basin),
                basin_rr_dir,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data.")
    parser.add_argument(
        "--data_type",
        dest="data_type",
        help="CAMELS dataset or your own data, such as 'camels' in datasource_dict.keys() or 'owndata'",
        # default="selfmadehydrodataset",
        default="owndata",
        type=str,
    )
    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        help="The directory of the CAMELS dataset or your own data, for CAMELS,"
        + " as we use SETTING to set the data path, you can directly choose camels_us;"
        + " for your own data, you should set the absolute path of your data directory",
        # default="C:\\Users\\wenyu\\OneDrive\\data\\FD_sources",
        default="C:\\Users\\wenyu\\OneDrive\\data\\biliuhe",
        # default="C:\\Users\\wenyu\\Downloads\\biliuhe",
        type=str,
    )
    parser.add_argument(
        "--result_dir",
        dest="result_dir",
        help="The root directory of results",
        default=os.path.join(repo_path, "result"),
        type=str,
    )
    parser.add_argument(
        "--exp",
        dest="exp",
        help="An exp is corresponding to one data setting",
        # default="expcamels001",
        # default="expselfmadehydrodataset001",
        default="expbiliuhe001",
        # default="expbiliuhetest001",
        type=str,
    )
    parser.add_argument(
        "--basin_id",
        dest="basin_id",
        help="The basins' ids",
        # default=["01439500", "06885500", "08104900", "09510200"],
        default=["21401550"],
        # default=["songliao_21401550"],
        nargs="+",
    )
    parser.add_argument(
        "--period",
        dest="period",
        help="The whole period",
        # default=["2007-01-01", "2014-01-01"],
        default=["2012-06-10 00:00", "2022-08-31 23:00"],
        # default=["2010-01-01 08:00", "2013-09-14 02:00"],
        nargs="+",
    )
    parser.add_argument(
        "--rr_event",
        dest="rr_event",
        help="if split the rr events, 0 for no, 1 for yes",
        default=0,
        type=int,
    )
    args = parser.parse_args()
    main(args)
