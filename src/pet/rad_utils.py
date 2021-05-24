from typing import Union

import numpy as np
import xarray as xr
from src.pet.meteo_utils import calc_ea


def calc_rad_short(s_rad: Union[np.ndarray, xr.DataArray] = None,
                   doy: Union[np.ndarray, xr.DataArray] = None,
                   lat: Union[np.ndarray, xr.DataArray] = None,
                   alpha=0.23,
                   n=None,
                   nn=None) -> Union[np.ndarray, xr.DataArray]:
    """Net shortwave radiation [MJ m-2 d-1].

    Parameters
    ----------
    s_rad:
        incoming solar radiation [MJ m-2 d-1]
    doy:
        day of the year
    lat:
        the site latitude [rad]
    alpha: float, optional
        surface albedo [-]
    n: pandas.Series/float, optional
        actual duration of sunshine [hour]
    nn: pandas.Series/float, optional
        maximum possible duration of sunshine or daylight hours [hour]


    Returns
    -------
    net shortwave radiation [MJ m-2 d-1]

    Notes
    -----
    Based on equation 38 in [allen_1998]_.
    """
    if s_rad is not None:
        return (1 - alpha) * s_rad
    else:
        return (1 - alpha) * calc_rad_sol_in(doy, lat, n=n, nn=nn)


def calc_rad_sol_in(doy: Union[np.ndarray, xr.DataArray],
                    lat: Union[np.ndarray, xr.DataArray],
                    as1=0.25,
                    bs1=0.5,
                    n=None,
                    nn=None) -> Union[np.ndarray, xr.DataArray]:
    """Incoming solar radiation [MJ m-2 d-1].

    Parameters
    ----------
    doy: day of the year
    lat:
        the site latitude [rad]
    as1:
        regression constant,  expressing the fraction of extraterrestrial
        reaching the earth on overcast days (n = 0) [-]
    bs1:
        empirical coefficient for extraterrestrial radiation [-]
    n:
        actual duration of sunshine [hour]
    nn:
        maximum possible duration of sunshine or daylight hours [hour]

    Returns
    -------
    net shortwave radiation [MJ m-2 d-1]

    Notes
    -----
    Based on equation 35 in [allen_1998]_.
    """
    ra = extraterrestrial_r(doy, lat)
    if n is None:
        n = daylight_hours(doy, lat)
    return (as1 + bs1 * n / nn) * ra


def extraterrestrial_r(doy: Union[np.ndarray, xr.DataArray],
                       lat: Union[np.ndarray, xr.DataArray], ) -> Union[np.ndarray, xr.DataArray]:
    """Extraterrestrial daily radiation [MJ m-2 d-1].

    Parameters
    ----------
    doy: day of the year
    lat: float
        the site latitude [rad]

    Returns
    -------
    extraterrestrial radiation

    Notes
    -----
    Based on equation 21 in [allen_1998]_.
    """
    dr = relative_distance(doy)
    sol_dec = solar_declination(doy)

    omega = sunset_angle(sol_dec, lat)
    xx = np.sin(sol_dec) * np.sin(lat)
    yy = np.cos(sol_dec) * np.cos(lat)
    return (24 * 60) / np.pi * 0.082 * dr * (omega * xx + yy * np.sin(omega))


def relative_distance(doy: Union[np.ndarray, xr.DataArray]) -> Union[np.ndarray, xr.DataArray]:
    """Inverse relative distance between earth and sun from day of the year.

    Parameters
    ----------
    doy: array.py
        day of the year (1-365)
    Returns
    -------
    relative distance between earth and sun.

    Notes
    -------
    Based on equations 23 in [allen_1998]_.
    """
    return 1 + 0.033 * np.cos(2. * np.pi / 365. * doy)


def solar_declination(doy: Union[np.ndarray, xr.DataArray]) -> Union[np.ndarray, xr.DataArray]:
    """Solar declination from day of year [rad].

    Parameters
    ----------
    doy: array.py
        day of the year (1-365)
    Returns
    -------
    solar declination [rad].

    Notes
    -------
    Based on equations 24 in [allen_1998]_.
    """
    return 0.409 * np.sin(2. * np.pi / 365. * doy - 1.39)


def sunset_angle(sol_dec: Union[np.ndarray, xr.DataArray],
                 lat: Union[np.ndarray, xr.DataArray]) -> Union[np.ndarray, xr.DataArray]:
    """Sunset hour angle from latitude and solar declination - daily [rad].

    Parameters
    ----------
    sol_dec:
        solar declination [rad]
    lat: float
        the site latitude [rad]

    Returns
    -------
    sunset hour angle - daily [rad]

    Notes
    -----
    Based on equations 25 in [allen_1998]_.
    """
    return np.arccos(-np.tan(sol_dec) * np.tan(lat))


def daylight_hours(doy: Union[np.ndarray, xr.DataArray],
                   lat: Union[np.ndarray, xr.DataArray]) -> Union[np.ndarray, xr.DataArray]:
    """Daylight hours [hour].

    Parameters
    ----------
    doy: day of the year
    lat: float
        the site latitude [rad]

    Returns
    -------
    daylight hours [hour]

    Notes
    -----
    Based on equation 34 in [allen_1998]_.
    """
    sol_dec = solar_declination(doy)
    sangle = sunset_angle(sol_dec, lat)
    return 24 / np.pi * sangle


def calc_rad_long(s_rad,
                  doy,
                  hour=None,
                  t_mean=None,
                  t_max=None,
                  t_min=None,
                  rh_max=None,
                  rh_min=None,
                  rh=None,
                  elevation=None,
                  lat=None,
                  rso=None,
                  a=1.35,
                  b=-0.35,
                  ea=None,
                  freq="D"):
    """Net longwave radiation [MJ m-2 d-1].

    Parameters
    ----------
    s_rad:
        incoming solar radiation [MJ m-2 d-1]
    doy:
        day of the year
    hour:
        hour of the day
    t_mean:
        average day temperature [°C]
    t_max:
        maximum day temperature [°C]
    t_min:
        minimum day temperature [°C]
    rh_max:
        maximum daily relative humidity [%]
    rh_min:
        mainimum daily relative humidity [%]
    rh:
        mean daily relative humidity [%]
    elevation: float, optional
        the site elevation [m]
    lat: float, optional
        the site latitude [rad]
    rso:
        clear-sky solar radiation [MJ m-2 day-1]
    a: float, optional
        empirical coefficient for Net Long-Wave radiation [-]
    b: float, optional
        empirical coefficient for Net Long-Wave radiation [-]
    ea:
        actual vapor pressure [kPa]
    freq: string, optional
        "D" => daily estimation
        "H" => hourly estimation

    Returns
    -------
        net longwave radiation

    Notes
    -----
    Based on equation 39 in [allen_1998]_.

    References
    ----------
    """
    if ea is None:
        ea = calc_ea(t_mean=t_mean, t_max=t_max, t_min=t_min, rh_max=rh_max, rh_min=rh_min, rh=rh)

    if freq == "H":
        # TODO: not tested
        if rso is None:
            ra = extraterrestrial_r_hour(doy, hour, lat=lat)
            rso = calc_rso(ra=ra, elevation=elevation)
        solar_rat = np.clip(s_rad / rso, 0.3, 1)
        # Stefan Boltzmann constant - hourly [MJm-2K-4h-1]
        STEFAN_BOLTZMANN_HOUR = 2.042 * 10 ** -10
        tmp1 = STEFAN_BOLTZMANN_HOUR * (t_mean + 273.16) ** 4
    else:
        if rso is None:
            ra = extraterrestrial_r(doy=doy, lat=lat)
            rso = calc_rso(ra=ra, elevation=elevation)
        # The ratio varies between about 0.33 (dense cloud cover) and 1 (clear sky)  Page 42 in [allen_1998]
        # people always clip with [0.3, 1]:
        # https://github.com/WSWUP/RefET/blob/7a0a6723a95e540141bf32cf6f650c77a191dd13/refet/calcs.py#L613
        # https://github.com/phydrus/pyet/blob/ecc5df23e067faeead6d523002247fe20b99b24b/pyet/rad_utils.py#L78
        solar_rat = np.clip(s_rad / rso, 0.3, 1)
        # Stefan Boltzmann constant - daily [MJm-2K-4d-1]
        STEFAN_BOLTZMANN_DAY = 4.903 * 10 ** -9
        if t_max is not None:
            tmp1 = STEFAN_BOLTZMANN_DAY * ((t_max + 273.16) ** 4 + (t_min + 273.16) ** 4) / 2
        else:
            tmp1 = STEFAN_BOLTZMANN_DAY * (t_mean + 273.16) ** 4

    tmp2 = 0.34 - 0.14 * np.sqrt(ea)
    # don't know why there is a clip here, this clip comes from:
    # https://github.com/phydrus/pyet/blob/ecc5df23e067faeead6d523002247fe20b99b24b/pyet/rad_utils.py#L86
    # tmp2 = np.clip(tmp2, 0.05, 1)
    tmp3 = a * solar_rat + b
    return tmp1 * tmp2 * tmp3


def extraterrestrial_r_hour(doy, hour, lat, lz=0, lm=0):
    """Extraterrestrial hourly radiation [MJ m-2 h-1].

    Parameters
    ----------
    doy:
        day of the year
    hour:
        hour of the day
    lat: float
        the site latitude [rad]
    lz: float, optional
        longitude of the centre of the local time zone (0° for Greenwich) [°]
    lm: float, optional
        longitude of the measurement site [degrees west of Greenwich] [°]

    Returns
    -------
    pandas.Series containing the calculated extraterrestrial radiation

    Notes
    -----
    Based on equation 28 in [allen_1998]_.

    """
    dr = relative_distance(doy)
    sol_dec = solar_declination(doy)

    omega2, omega1 = sunset_angle_hour(doy, hour, lz=lz, lm=lm, lat=lat,
                                       sol_dec=sol_dec)
    xx = np.sin(sol_dec) * np.sin(lat)
    yy = np.cos(sol_dec) * np.cos(lat)
    gsc = 4.92
    return 12 / np.pi * gsc * dr * ((omega2 - omega1) * xx + yy *
                                    (np.sin(omega2) - np.sin(omega1)))


def sunset_angle_hour(doy, hour, sol_dec, lat, lz, lm):
    """Sunset hour angle from latitude and solar declination - hourly [rad].

    Parameters
    ----------
    doy:
        day of the year
    hour:
        hour of the day
    sol_dec: pandas.Series
        solar declination [rad]
    lat: float
        the site latitude [rad]
    lz: float
        longitude of the local time zone [°]
    lm: float
        longitude of the measurement site [°]

    Returns
    -------
    pandas.Series containing the calculated sunset hour angle - hourly [rad]

    Notes
    -----
    Based on equations 29, 30, 31, 32, 33 in [allen_1998]_.
    """
    b = 2 * np.pi * (doy - 81) / 364
    sc = 0.1645 * np.sin(2 * b) - 0.1255 * np.cos(b) - 0.025 * np.sin(b)
    t = hour + 0.5
    sol_t = t + 0.06667 * (lz - lm) + sc - 12  # equation 31
    omega = np.pi / 12 * sol_t

    omega1 = omega - np.pi / 24
    omega2 = omega + np.pi / 24

    omegas = np.arccos(-np.tan(lat) * np.tan(sol_dec))

    omega1 = np.clip(omega1, -omegas, omegas)
    omega2 = np.clip(omega2, -omegas, omegas)
    omega1 = np.maximum(omega1, omega1, )
    omega1 = np.clip(omega1, -100000000, omega2)

    return omega2, omega1


def calc_rso(ra, elevation):
    """Clear-sky solar radiation [MJ m-2 day-1].

    Parameters
    ----------
    ra:
        Extraterrestrial daily radiation [MJ m-2 d-1]
    elevation:
        the site elevation [m]

    Returns
    -------
    Clear-sky solar radiation

    Notes
    -----
    Based on equation 37 in [allen_1998]_.

    """
    return (0.75 + (2 * 10 ** -5) * elevation) * ra
