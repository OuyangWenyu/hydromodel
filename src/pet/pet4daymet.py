"""calculate potential evapotranspiration according to forcing data used in CAMELS"""
from typing import Union

import numpy as np
import xarray as xr
from src.pet.meteo_utils import calc_press, calc_psy, calc_vpc, calc_lambda_, calc_ea, calc_es
from src.pet.rad_utils import calc_rad_short, calc_rad_long


def priestley_taylor(t_min: Union[np.ndarray, xr.DataArray],
                     t_max: Union[np.ndarray, xr.DataArray],
                     s_rad: Union[np.ndarray, xr.DataArray],
                     lat: Union[np.ndarray, xr.DataArray],
                     elevation: Union[np.ndarray, xr.DataArray],
                     doy: Union[np.ndarray, xr.DataArray],
                     e_a: Union[np.ndarray, xr.DataArray] = None) -> Union[np.ndarray, xr.DataArray]:
    """Evaporation calculated according to [priestley_and_taylor_1965]_.

    Parameters
    ----------
    t_max:
        maximum day temperature [°C]
    t_min:
        minimum day temperature [°C]
    s_rad:
        incoming solar radiation [MJ m-2 d-1]
    lat:
        the site latitude [rad]
    elevation:
        the site elevation [m]
    doy:
        Day of the year
    e_a:
        Actual vapor pressure [kPa].
    Returns
    -------
        the calculated evaporation [mm day-1]

    Examples
    --------
    >>> pt = priestley_taylor(t_min, t_max, s_rad, lat, elevation, doy, e_a)

    Notes
    -----

    .. math:: PE = \\frac{\\alpha_{PT} \\Delta (R_n-G)}
        {\\lambda(\\Delta +\\gamma)}

    References
    ----------
    .. [priestley_and_taylor_1965] Priestley, C. H. B., & TAYLOR, R. J. (1972).
       On the assessment of surface heat flux and evaporation using large-scale
       parameters. Monthly weather review, 100(2), 81-92.

    """
    #  tmean: average day temperature [°C]
    t_mean = 0.5 * (t_min + t_max)
    pressure = calc_press(elevation)
    gamma = calc_psy(pressure)
    dlt = calc_vpc(t_mean)
    _lambda = calc_lambda_(t_mean)
    albedo = 0.23
    rns = calc_rad_short(s_rad=s_rad, alpha=albedo)  # [MJ/m2/d]
    # a: empirical coefficient for Net Long-Wave radiation [-]
    a = 1.35
    # b: empirical coefficient for Net Long-Wave radiation [-]
    b = -0.35
    rnl = calc_rad_long(s_rad, doy, t_mean=t_mean, t_max=t_max, t_min=t_min, elevation=elevation, lat=lat, a=a, b=b,
                        ea=e_a)
    # The total daily value for Rn is almost always positive over a period of 24 hours, except in extreme conditions
    # at high latitudes. Page43 in [allen_1998]
    rn = rns - rnl
    # g: soil heat flux [MJ m-2 d-1], for daily calculation, it equals to 0
    g = 0
    # alpha: calibration coeffiecient [-]
    alpha = 1.26
    return (alpha * dlt * (rn - g)) / (_lambda * (dlt + gamma))


def pm_fao56(t_min: Union[np.ndarray, xr.DataArray],
             t_max: Union[np.ndarray, xr.DataArray],
             s_rad: Union[np.ndarray, xr.DataArray],
             lat: Union[np.ndarray, xr.DataArray],
             elevation: Union[np.ndarray, xr.DataArray],
             doy: Union[np.ndarray, xr.DataArray],
             e_a: Union[np.ndarray, xr.DataArray] = None) -> Union[np.ndarray, xr.DataArray]:
    """Evaporation calculated according to [allen_1998]_.
    Parameters
    ----------
    t_max:
        maximum day temperature [°C]
    t_min:
        minimum day temperature [°C]
    s_rad:
        incoming solar radiation [MJ m-2 d-1]
    lat:
        the site latitude [rad]
    elevation:
        the site elevation [m]
    doy:
        Day of the year
    e_a:
        Actual vapor pressure [kPa].
    Returns
    -------
        pandas.Series containing the calculated evaporation

    Examples
    --------
    >>> et_fao56 = pm_fao56(t_min, t_max, s_rad, lat, elevation, doy, e_a)

    Notes
    -----
    .. math:: PE = \\frac{0.408 \\Delta (R_{n}-G)+\\gamma \\frac{900}{T+273}
        (e_s-e_a) u_2}{\\Delta+\\gamma(1+0.34 u_2)}

    """
    # t_mean: average day temperature [°C]
    t_mean = (t_max + t_min) / 2
    # pressure: atmospheric pressure [kPa]
    pressure = calc_press(elevation)
    gamma = calc_psy(pressure)
    dlt = calc_vpc(t_mean)

    # wind: mean day wind speed [m/s]
    # Where no wind data are available within the region, a value of 2 m/s can be used as a
    # temporary estimate. This value is the average over 2 000 weather stations around the globe. Page63 in [allen_1998]
    wind = 2
    gamma1 = (gamma * (1 + 0.34 * wind))
    if e_a is None:
        e_a = calc_ea(t_mean=t_mean, t_max=t_max, t_min=t_min)
    e_s = calc_es(t_mean=t_mean, t_max=t_max, t_min=t_min)
    albedo = 0.23
    rns = calc_rad_short(s_rad=s_rad, alpha=albedo)  # [MJ/m2/d]
    # a: empirical coefficient for Net Long-Wave radiation [-]
    a = 1.35
    # b: empirical coefficient for Net Long-Wave radiation [-]
    b = -0.35
    rnl = calc_rad_long(s_rad=s_rad, doy=doy, t_mean=t_mean, t_max=t_max, t_min=t_min, elevation=elevation, lat=lat,
                        a=a, b=b, ea=e_a)  # [MJ/m2/d]
    rn = rns - rnl

    den = dlt + gamma1
    # g: soil heat flux [MJ m-2 d-1]
    g = 0
    num1 = (0.408 * dlt * (rn - g)) / den
    num2 = (gamma * (e_s - e_a) * 900 * wind / (t_mean + 273)) / den
    return num1 + num2
