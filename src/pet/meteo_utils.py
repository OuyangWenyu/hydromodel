from typing import Union

import numpy as np
import xarray as xr


def calc_press(elevation: Union[np.ndarray, xr.DataArray]) -> Union[np.ndarray, xr.DataArray]:
    """Atmospheric pressure [kPa].

    Parameters
    ----------
    elevation:
        the site elevation [m]

    Returns
    -------
        atmospheric pressure [kPa].

    Examples
    --------
    >>> pressure = calc_press(elevation)

    Notes
    -----
    Based on equation 7 in [allen_1998]_.
    """
    return 101.3 * ((293 - 0.0065 * elevation) / 293) ** 5.26


def calc_psy(pressure: Union[np.ndarray, xr.DataArray],
             t_mean: Union[np.ndarray, xr.DataArray] = None) -> Union[np.ndarray, xr.DataArray]:
    """Psychrometric constant [kPa °C-1].

    Parameters
    ----------
    pressure: float
        atmospheric pressure [kPa].
    t_mean: float, optional
        average day temperature [°C].

    Returns
    -------
        the Psychrometric constant [kPa °C-1].

    Examples
    --------
    >>> psy = calc_psy(pressure, t_mean)

    Notes
    -----
    if tmean is none:
        Based on equation 8 in [allen_1998]_.
    elif rh is None:
        From FAO (1990), ANNEX V, eq. 4.

    References
    ----------
    .. [allen_1998] Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998).
       Crop evapotranspiration-Guidelines for computing crop water
       requirements-FAO Irrigation and drainage paper 56. Fao, Rome, 300.
       (http://www.fao.org/3/x0490e/x0490e06.htm#TopOfPage).
    """
    if t_mean is None:
        return 0.000665 * pressure
    else:
        lambd = calc_lambda_(t_mean)  # MJ kg-1
        # Specific heat of air [MJ kg-1 °C-1]
        CP = 1.013 * 10 ** -3
        return CP * pressure / (0.622 * lambd)


def calc_lambda(t_mean: Union[np.ndarray, xr.DataArray]) -> Union[np.ndarray, xr.DataArray]:
    """ Latent Heat of Vaporization [MJ kg-1].

    Parameters
    ----------
    t_mean:
        average day temperature [°C]

    Returns
    -------
    Latent Heat of Vaporization [MJ kg-1].

    Examples
    --------
    >>> lambd = calc_lambda_(t_mean)

    Notes
    -----
    Based on equation (3-1) in [allen_1998]_.
    """
    return 2.501 - 0.002361 * t_mean


def calc_vpc(t_mean: Union[np.ndarray, xr.DataArray]) -> Union[np.ndarray, xr.DataArray]:
    """Slope of saturation vapour pressure curve at air Temperature [kPa °C-1].

    Parameters
    ----------
    t_mean:
        average day temperature [°C]

    Returns
    -------
        Saturation vapour pressure
        [kPa °C-1].

    Examples
    --------
    >>> vpc = calc_vpc(t_mean)

    Notes
    -----
    Based on equation 13. in [allen_1998]_.
    """
    es = calc_e0(t_mean)
    return 4098 * es / (t_mean + 237.3) ** 2


def calc_e0(t_mean: Union[np.ndarray, xr.DataArray]) -> Union[np.ndarray, xr.DataArray]:
    """ Saturation vapor pressure at the air temperature T [kPa].

    Parameters
    ----------
    t_mean:
        average day temperature [°C]

    Returns
    -------
    saturation vapor pressure at the air temperature tmean [kPa].

    Examples
    --------
    >>> e0 = calc_e0(t_mean)

    Notes
    -----
    Based on equation 11 in [allen_1998]_.
    """
    return 0.6108 * np.exp(17.27 * t_mean / (t_mean + 237.3))


def calc_lambda_(t_mean: Union[np.ndarray, xr.DataArray]) -> Union[np.ndarray, xr.DataArray]:
    """ Latent Heat of Vaporization [MJ kg-1].

    Parameters
    ----------
    t_mean:
        average day temperature [°C]

    Returns
    -------
    Latent Heat of Vaporization
        [MJ kg-1].

    Examples
    --------
    >>> _lambda = calc_lambda_(t_mean)

    Notes
    -----
    Based on equation (3-1) in [allen_1998]_.
    """
    return 2.501 - 0.002361 * t_mean


def calc_ea(t_mean=None, t_max=None, t_min=None, rh_max=None, rh_min=None, rh=None):
    """ Actual vapor pressure [kPa].

    Parameters
    ----------
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

    Returns
    -------
    actual vapor pressure [kPa].

    Examples
    --------
    >>> ea = calc_ea(t_mean, rh)

    Notes
    -----
    Based on equation 17, 19, 48 in [allen_1998]_.
    """
    if rh_max is not None:  # equation 17
        es_max = calc_e0(t_max)
        es_min = calc_e0(t_min)
        return (es_min * rh_max / 200) + (es_max * rh_min / 200)
    elif rh is not None:
        if t_max is not None:  # equation 19
            es = calc_es(t_max=t_max, t_min=t_min)
        else:
            # equation 3-8
            es = calc_e0(t_mean)
        return rh / 100 * es
    else:
        # equation 48
        return 0.611 * np.exp((17.27 * t_min) / (t_min + 237.3))


def calc_es(t_mean=None, t_max=None, t_min=None):
    """ Saturation vapor pressure [kPa].

    Parameters
    ----------
    t_mean:
        average day temperature [°C]
    t_max:
        maximum day temperature [°C]
    t_min:
        minimum day temperature [°C]

    Returns
    -------
    saturation vapor pressure [kPa].

    Examples
    --------
    >>> es = calc_es(t_mean)

    Notes
    -----
    Based on equation 11, 12 in [allen_1998]_.
    """
    if t_max is not None:
        ea_max = calc_e0(t_max)
        ea_min = calc_e0(t_min)
        return (ea_max + ea_min) / 2
    else:
        return calc_e0(t_mean)
