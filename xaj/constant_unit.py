"""
Author: Wenyu Ouyang
Date: 2022-12-08 09:24:54
LastEditTime: 2022-12-08 09:51:54
LastEditors: Wenyu Ouyang
Description: some constant for hydro model
FilePath: /hydro-model-xaj/hydromodel/utils/hydro_constant.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
# unify the unit of each variable
unit = {"streamflow": "m3/s"}
def convert_unit(data, unit_now, unit_final, **kwargs):
    """
    convert unit of variable

    Parameters
    ----------
    data
        data to be converted
    unit_now
        unit of variable now
    unit_final
        unit of variable after conversion
    **kwargs
        other parameters required for conversion

    Returns
    -------
    data
        data after conversion
    """
    if unit_now == "mm/day" and unit_final == "m3/s":
        result = mm_per_day_to_m3_per_sec(basin_area=kwargs["basin_area"], q=data)
    else:
        raise ValueError("unit conversion not supported")
    return result


def mm_per_day_to_m3_per_sec(basin_area, q):
    """
    trans mm/day to m3/s for xaj models

    Parameters
    ----------
    basin_area
        we need to know the area of a basin so that we can perform this transformation
    q
        original streamflow data

    Returns
    -------

    """
    # 1 ft3 = 0.02831685 m3
    # ft3tom3 = 2.831685e-2
    # 1 km2 = 10^6 m2
    km2tom2 = 1e6
    # 1 m = 1000 mm
    mtomm = 1000
    # 1 day = 24 * 3600 s
    daytos = 24 * 3600
    q_trans = q * basin_area * km2tom2 / (mtomm * daytos)
    return q_trans
