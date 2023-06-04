"""
The method comes from this paper: https://doi.org/10.1038/s41467-021-26107-z
It use Deep Learning (DL) methods to Learn the Parameters of physics-based models (PBM),
which is called "differentiable parameter learning" (dPL).
"""
from typing import Optional, Union
import torch
from torch import nn
from torch import Tensor

from hydromodel.models.model_config import MODEL_PARAM_DICT
from hydromodel.models.dpl_basic import (
    KernelConv,
    SimpleAnn,
    SimpleLSTM,
    ann_pbm,
    lstm_pbm,
)


PRECISION = 1e-5


def calculate_evap(
    lm, c, wu0, wl0, prcp, pet
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Three-layers evaporation model from "Watershed Hydrologic Simulation" written by Prof. RenJun Zhao.

    The book is Chinese, and its real name is 《流域水文模拟》;
    The three-layers evaporation model is descibed in Page 76;
    The method is same with that in Page 22-23 in "Hydrologic Forecasting (5-th version)" written by Prof. Weimin Bao.
    This book's Chinese name is 《水文预报》

    Parameters
    ----------
    lm
        average soil moisture storage capacity of lower layer (mm)
    c
        coefficient of deep layer
    wu0
        initial soil moisture of upper layer; update in each time step (mm)
    wl0
        initial soil moisture of lower layer; update in each time step (mm)
    prcp
        basin mean precipitation (mm/day)
    pet
        potential evapotranspiration (mm/day)

    Returns
    -------
    torch.Tensor
        eu/el/ed are evaporation from upper/lower/deeper layer, respectively
    """
    tensor_min = torch.full(wu0.size(), 0.0).to(prcp.device)
    # when using torch.where, please see here: https://github.com/pytorch/pytorch/issues/9190
    # it's element-wise operation, no problem here. For example:
    # In: torch.where(torch.Tensor([2,1])>torch.Tensor([1,1]),torch.Tensor([1,2]),torch.Tensor([3,4]))
    # Out: tensor([1., 4.])
    eu = torch.where(wu0 + prcp >= pet, pet, wu0 + prcp)
    ed = torch.where(
        (wl0 < c * lm) & (wl0 < c * (pet - eu)), c * (pet - eu) - wl0, tensor_min
    )
    el = torch.where(
        wu0 + prcp >= pet,
        tensor_min,
        torch.where(
            wl0 >= c * lm,
            (pet - eu) * wl0 / lm,
            torch.where(wl0 >= c * (pet - eu), c * (pet - eu), wl0),
        ),
    )
    return eu, el, ed


def calculate_prcp_runoff(b, im, wm, w0, pe):
    """
    Calculates the amount of runoff generated from rainfall after entering the underlying surface

    Same in "Watershed Hydrologic Simulation" and "Hydrologic Forecasting (5-th version)"

    Parameters
    ----------
    b
        B exponent coefficient
    im
        IMP imperiousness coefficient
    wm
        average soil moisture storage capacity
    w0
        initial soil moisture
    pe
        net precipitation

    Returns
    -------
    torch.Tensor
        r -- runoff; r_im -- runoff of impervious part
    """
    wmm = wm * (1 + b)
    a = wmm * (1 - (1 - w0 / wm) ** (1 / (1 + b)))
    if any(torch.isnan(a)):
        raise ValueError(
            "Error: NaN values detected. Try set clamp function or check your data!!!"
        )
    r_cal = torch.where(
        pe > 0.0,
        torch.where(
            pe + a < wmm,
            # torch.clamp is used for gradient not to be NaN, see more in xaj_sources function
            pe - (wm - w0) + wm * (1 - torch.clamp(a + pe, max=wmm) / wmm) ** (1 + b),
            pe - (wm - w0),
        ),
        torch.full(pe.size(), 0.0).to(pe.device),
    )
    if any(torch.isnan(r_cal)):
        raise ValueError(
            "Error: NaN values detected. Try set clamp function or check your data!!!"
        )
    r = torch.clamp(r_cal, min=0.0)
    r_im_cal = pe * im
    r_im = torch.clamp(r_im_cal, min=0.0)
    return r, r_im


def calculate_w_storage(um, lm, dm, wu0, wl0, wd0, eu, el, ed, pe, r):
    """
    Update the soil moisture values of the three layers.

    According to the runoff-generation equation 2.60 in the book "SHUIWENYUBAO", dW = dPE - dR

    Parameters
    ----------
    um
        average soil moisture storage capacity of the upper layer (mm)
    lm
        average soil moisture storage capacity of the lower layer (mm)
    dm
        average soil moisture storage capacity of the deep layer (mm)
    wu0
        initial values of soil moisture in upper layer
    wl0
        initial values of soil moisture in lower layer
    wd0
        initial values of soil moisture in deep layer
    eu
        evaporation of the upper layer; it isn't used in this function
    el
        evaporation of the lower layer
    ed
        evaporation of the deep layer
    pe
        net precipitation; it is able to be negative value in this function
    r
        runoff

    Returns
    -------
    torch.Tensor
        wu,wl,wd -- soil moisture in upper, lower and deep layer
    """
    xaj_device = pe.device
    tensor_zeros = torch.full(wu0.size(), 0.0).to(xaj_device)
    # pe>0: the upper soil moisture was added firstly, then lower layer, and the final is deep layer
    # pe<=0: no additional water, just remove evapotranspiration,
    # but note the case: e >= p > 0
    # (1) if wu0 + p > e, then e = eu (2) else, wu must be zero
    wu = torch.where(
        pe > 0.0,
        torch.where(wu0 + pe - r < um, wu0 + pe - r, um),
        torch.where(wu0 + pe > 0.0, wu0 + pe, tensor_zeros),
    )
    # calculate wd before wl because it is easier to cal using where statement
    wd = torch.where(
        pe > 0.0,
        torch.where(
            wu0 + wl0 + pe - r > um + lm, wu0 + wl0 + wd0 + pe - r - um - lm, wd0
        ),
        wd0 - ed,
    )
    # water balance (equation 2.2 in Page 13, also shown in Page 23)
    # if wu0 + p > e, then e = eu; else p must be used in upper layer,
    # so no matter what the case is, el didn't include p, neither ed
    wl = torch.where(pe > 0.0, wu0 + wl0 + wd0 + pe - r - wu - wd, wl0 - el)
    # the water storage should be in reasonable range
    tensor_mins = torch.full(um.size(), 0.0).to(xaj_device)
    wu_ = torch.clamp(wu, min=tensor_mins, max=um)
    wl_ = torch.clamp(wl, min=tensor_mins, max=lm)
    wd_ = torch.clamp(wd, min=tensor_mins, max=dm)
    return wu_, wl_, wd_


def xaj_generation(
    p_and_e: Tensor,
    k,
    b,
    im,
    um,
    lm,
    dm,
    c,
    wu0: Tensor = None,
    wl0: Tensor = None,
    wd0: Tensor = None,
) -> tuple:
    """
    Single-step runoff generation in XAJ.

    Parameters
    ----------
    p_and_e
        precipitation and potential evapotranspiration (mm/day)
    k
        ratio of potential evapotranspiration to reference crop evaporation
    b
        exponent parameter
    um
        average soil moisture storage capacity of the upper layer (mm)
    lm
        average soil moisture storage capacity of the lower layer (mm)
    dm
        average soil moisture storage capacity of the deep layer (mm)
    im
        impermeability coefficient
    c
        coefficient of deep layer
    wu0
        initial values of soil moisture in upper layer (mm)
    wl0
        initial values of soil moisture in lower layer (mm)
    wd0
        initial values of soil moisture in deep layer (mm)

    Returns
    -------
    tuple[torch.Tensor]
        (r, rim, e, pe), (wu, wl, wd)
    """
    # make sure physical variables' value ranges are correct
    prcp = torch.clamp(p_and_e[:, 0], min=0.0)
    pet = torch.clamp(p_and_e[:, 1] * k, min=0.0)
    # wm
    wm = um + lm + dm
    if wu0 is None:
        # use detach func to make wu0 no_grad as it is an initial value
        wu0 = 0.6 * (um.detach())
    if wl0 is None:
        wl0 = 0.6 * (lm.detach())
    if wd0 is None:
        wd0 = 0.6 * (dm.detach())
    w0_ = wu0 + wl0 + wd0
    # w0 need locate in correct range so that following calculation could be right
    # To make sure the gradient is also not NaN (see case in xaj_sources),
    # we'd better minus a precision (1e-5), although we've not met this situation (grad is NaN)
    w0 = torch.clamp(w0_, max=wm - 1e-5)

    # Calculate the amount of evaporation from storage
    eu, el, ed = calculate_evap(lm, c, wu0, wl0, prcp, pet)
    e = eu + el + ed

    # Calculate the runoff generated by net precipitation
    prcp_difference = prcp - e
    pe = torch.clamp(prcp_difference, min=0.0)
    r, rim = calculate_prcp_runoff(b, im, wm, w0, pe)
    # Update wu, wl, wd;
    # we use prcp_difference rather than pe, as when pe<0 but prcp>0, prcp should be considered
    wu, wl, wd = calculate_w_storage(
        um, lm, dm, wu0, wl0, wd0, eu, el, ed, prcp_difference, r
    )

    return (r, rim, e, pe), (wu, wl, wd)


def xaj_sources(
    pe,
    r,
    sm,
    ex,
    ki,
    kg,
    s0: Optional[Tensor] = None,
    fr0: Optional[Tensor] = None,
    book="HF",
) -> tuple:
    """
    Divide the runoff to different sources

    We use the initial version from the paper of the inventor of the XAJ model -- Prof. Renjun Zhao:
    "Analysis of parameters of the XinAnJiang model". Its Chinese name is <<新安江模型参数的分析>>,
    which could be found by searching in "Baidu Xueshu".
    The module's code can also be found in "Watershed Hydrologic Simulation" (WHS) Page 174.
    It is nearly same with that in "Hydrologic Forecasting" (HF) Page 148-149
    We use the period average runoff as input and the unit period is day so we don't need to difference it as books show


    Parameters
    ------------
    pe
        net precipitation (mm/day)
    r
        runoff from xaj_generation (mm/day)
    sm
        areal mean free water capacity of the surface layer (mm)
    ex
        exponent of the free water capacity curve
    ki
        outflow coefficients of the free water storage to interflow relationships
    kg
        outflow coefficients of the free water storage to groundwater relationships
    s0
        initial free water capacity (mm)
    fr0
        runoff area of last period

    Return
    ------------
    torch.Tensor
        rs -- surface runoff; ri-- interflow runoff; rg -- groundwater runoff

    """
    xaj_device = pe.device
    # maximum free water storage capacity in a basin
    ms = sm * (1 + ex)
    if fr0 is None:
        fr0 = torch.full(sm.shape[0], 0.1).to(xaj_device)
    if s0 is None:
        s0 = 0.5 * (sm.clone().detach())
    # For free water storage, because s is related to fr and s0 and fr0 are both values of last period,
    # we have to trans the initial value of s from last period to this one.
    # both WHS（流域水文模拟）'s sample code and HF（水文预报） use s = fr0 * s0 / fr.
    # I think they both think free water reservoir as a cubic tank. Its height is s and area of bottom rectangle is fr
    # we will have a cubic tank with varying bottom and height,
    # and fixed boundary (in HF sm is fixed) or none-fixed boundary (in EH smmf is not fixed)
    # but notice r's list like" [1,0] which 1 is the 1st period's runoff and 0 is the 2nd period's runoff
    # after 1st period, the s1 could not be zero, but in the 2nd period, fr=0, then we cannot set s=0, because some water still in the tank
    # fr's formula could be found in Eq. 9 in "Analysis of parameters of the XinAnJiang model",
    # Here our r doesn't include rim, so there is no need to remove rim from r; this is also the method in HF
    # Moreover, to make sure ss is not larger than sm, otherwise au will be nan value.
    # It is worth to note that we have to use a precision here -- 1e-5, otherwise the gradient will be NaN;
    # I guess maybe when calculating gradient -- Δy/Δx, Δ brings some precision problem when we need exponent function.

    # NOTE: when r is 0, fr should be 0, however, s1 may not be zero and it still hold some water,
    # then fr can not be 0, otherwise when fr is used as denominator it lead to error,
    # so we have to deal with this case later, for example, when r=0, we cannot use pe * fr to replace r
    # because fr get the value of last period, and it is not 0

    # cannot use torch.where, because it will cause some error when calculating gradient
    # fr = torch.where(r > 0.0, r / pe, fr0)
    # fr just use fr0, and it can be included in the computation graph, so we don't detach it
    fr = torch.clone(fr0)
    fr_mask = r > 0.0
    fr[fr_mask] = r[fr_mask] / pe[fr_mask]
    if any(torch.isnan(fr)):
        raise ValueError(
            "Error: NaN values detected. Try set clamp function or check your data!!!"
        )
    if any(fr == 0.0):
        raise ArithmeticError(
            "Please check fr's value, fr==0.0 will cause error in the next step!"
        )
    ss = torch.clone(s0)
    s = torch.clone(s0)

    ss[fr_mask] = fr0[fr_mask] * s0[fr_mask] / fr[fr_mask]

    if book == "HF":
        ss = torch.clamp(ss, max=sm - PRECISION)
        au = ms * (1.0 - (1.0 - ss / sm) ** (1.0 / (1.0 + ex)))
        if any(torch.isnan(au)):
            raise ValueError(
                "Error: NaN values detected. Try set clamp function or check your data!!!"
            )

        rs = torch.full_like(r, 0.0, device=xaj_device)
        rs[fr_mask] = torch.where(
            pe[fr_mask] + au[fr_mask] < ms[fr_mask],
            # equation 2-85 in HF
            # it's weird here, but we have to clamp so that the gradient could be not NaN;
            # otherwise, even the forward calculation is correct, the gradient is still NaN;
            # maybe when calculating gradient -- Δy/Δx, Δ brings some precision problem
            # if we need exponent function.
            fr[fr_mask]
            * (
                pe[fr_mask]
                - sm[fr_mask]
                + ss[fr_mask]
                + sm[fr_mask]
                * (
                    (
                        1
                        - torch.clamp(pe[fr_mask] + au[fr_mask], max=ms[fr_mask])
                        / ms[fr_mask]
                    )
                    ** (1 + ex[fr_mask])
                )
            ),
            # equation 2-86 in HF
            fr[fr_mask] * (pe[fr_mask] + ss[fr_mask] - sm[fr_mask]),
        )
        rs = torch.clamp(rs, max=r)
        # ri's mask is not same as rs's, because last period's s may not be 0
        # and in this time, ri and rg could be larger than 0
        # we need firstly calculate the updated s, s's mask is same as fr_mask,
        # when r==0, then s will be equal to last period's
        # equation 2-87 in HF, some free water leave or save, so we update free water storage
        s[fr_mask] = ss[fr_mask] + (r[fr_mask] - rs[fr_mask]) / fr[fr_mask]
        s = torch.clamp(s, max=sm)
    elif book == "EH":
        smmf = ms * (1 - (1 - fr) ** (1 / ex))
        smf = smmf / (1 + ex)
        ss = torch.clamp(ss, max=smf - PRECISION)
        au = smmf * (1 - (1 - ss / smf) ** (1 / (1 + ex)))
        if torch.isnan(au).any():
            raise ArithmeticError(
                "Error: NaN values detected. Try set clip function or check your data!!!"
            )
        rs = torch.full_like(r, 0.0, device=xaj_device)
        rs[fr_mask] = torch.where(
            pe[fr_mask] + au[fr_mask] < smmf[fr_mask],
            (
                pe[fr_mask]
                - smf[fr_mask]
                + ss[fr_mask]
                + smf[fr_mask]
                * (
                    1
                    - torch.clamp(
                        pe[fr_mask] + au[fr_mask],
                        max=smmf[fr_mask],
                    )
                    / smmf[fr_mask]
                )
                ** (ex[fr_mask] + 1)
            )
            * fr[fr_mask],
            (pe[fr_mask] + ss[fr_mask] - smf[fr_mask]) * fr[fr_mask],
        )
        rs = torch.clamp(rs, max=r)
        s[fr_mask] = ss[fr_mask] + (r[fr_mask] - rs[fr_mask]) / fr[fr_mask]
        s[fr_mask] = torch.clamp(s[fr_mask], max=smf[fr_mask])
        s = torch.clamp(s, max=smf)
    else:
        raise ValueError("Please set book as 'HF' or 'EH'!")
    # equation 2-88 in HF, next interflow and ground water will be released from the updated free water storage
    # We use the period average runoff as input and the unit period is day.
    # Hence, we directly use ki and kg rather than ki_{Δt} in books.
    ri = ki * s * fr
    rg = kg * s * fr
    # equation 2-89 in HF; although it looks different with that in WHS, they are actually same
    # Finally, calculate the final free water storage
    s1 = s * (1 - ki - kg)
    return (rs, ri, rg), (s1, fr)


def xaj_sources5mm(
    pe,
    runoff,
    sm,
    ex,
    ki,
    kg,
    s0=None,
    fr0=None,
    book="HF",
):
    """
    Divide the runoff to different sources according to books -- 《水文预报》HF 5th edition and 《工程水文学》EH 3rd edition

    Parameters
    ----------
    pe
        net precipitation
    runoff
        runoff from xaj_generation
    sm
        areal mean free water capacity of the surface layer
    ex
        exponent of the free water capacity curve
    ki
        outflow coefficients of the free water storage to interflow relationships
    kg
        outflow coefficients of the free water storage to groundwater relationships
    s0
        initial free water capacity
    fr0
        initial area of generation
    time_interval_hours
        由于Ki、Kg、Ci、Cg都是以24小时为时段长定义的,需根据时段长转换
    book
        the methods in 《水文预报》HF 5th edition and 《工程水文学》EH 3rd edition are different,
        hence, both are provided, and the default is the former -- "ShuiWenYuBao";
        the other one is "GongChengShuiWenXue"

    Returns
    -------
    tuple[tuple, tuple]
        rs_s -- surface runoff; rss_s-- interflow runoff; rg_s -- groundwater runoff;
        (fr_ds[-1], s_ds[-1]): state variables' final value;
        all variables are numpy array
    """
    xaj_device = pe.device
    # 流域最大点自由水蓄水容量深
    smm = sm * (1 + ex)
    if fr0 is None:
        fr0 = torch.full_like(sm, 0.1, device=xaj_device)
    if s0 is None:
        s0 = 0.5 * (sm.clone().detach())
    fr = torch.clone(fr0)
    fr_mask = runoff > 0.0
    fr[fr_mask] = runoff[fr_mask] / pe[fr_mask]
    if torch.all(runoff < 5):
        n = 1
    else:
        r_max = torch.max(runoff).detach().cpu().numpy()
        residue_temp = r_max % 5
        if residue_temp != 0:
            residue_temp = 1
        n = int(r_max / 5) + residue_temp
    rn = runoff / n
    pen = pe / n
    kss_d = (1 - (1 - (ki + kg)) ** (1 / n)) / (1 + kg / ki)
    kg_d = kss_d * kg / ki
    if torch.isnan(kss_d).any() or torch.isnan(kg_d).any():
        raise ValueError("Error: NaN values detected. Check your parameters setting!!!")
    # kss_d = ki
    # kg_d = kg

    rs = torch.full_like(runoff, 0.0, device=xaj_device)
    rss = torch.full_like(runoff, 0.0, device=xaj_device)
    rg = torch.full_like(runoff, 0.0, device=xaj_device)

    s_ds = []
    fr_ds = []
    s_ds.append(s0)
    fr_ds.append(fr0)
    for j in range(n):
        fr0_d = fr_ds[j]
        s0_d = s_ds[j]
        # equation 5-32 in HF, but strange, cause each period, rn/pen is same
        # fr_d = torch.full_like(fr0_d, PRECISION, device=xaj_device)
        # fr_d_mask = fr > PRECISION
        # fr_d[fr_d_mask] = 1 - (1 - fr[fr_d_mask]) ** (1 / n)
        fr_d = fr

        ss_d = torch.clone(s0_d)
        s_d = torch.clone(s0_d)

        ss_d[fr_mask] = fr0_d[fr_mask] * s0_d[fr_mask] / fr_d[fr_mask]

        if book == "HF":
            # ms = smm
            ss_d = torch.clamp(ss_d, max=sm - PRECISION)
            au = smm * (1.0 - (1.0 - ss_d / sm) ** (1.0 / (1.0 + ex)))
            if torch.isnan(au).any():
                raise ValueError(
                    "Error: NaN values detected. Try set clip function or check your data!!!"
                )
            rs_j = torch.full_like(rn, 0.0, device=xaj_device)
            rs_j[fr_mask] = torch.where(
                pen[fr_mask] + au[fr_mask] < smm[fr_mask],
                # equation 5-26 in HF
                fr_d[fr_mask]
                * (
                    pen[fr_mask]
                    - sm[fr_mask]
                    + ss_d[fr_mask]
                    + sm[fr_mask]
                    * (
                        (
                            1
                            - torch.clamp(pen[fr_mask] + au[fr_mask], max=smm[fr_mask])
                            / smm[fr_mask]
                        )
                        ** (1 + ex[fr_mask])
                    )
                ),
                # equation 5-27 in HF
                fr_d[fr_mask] * (pen[fr_mask] + ss_d[fr_mask] - sm[fr_mask]),
            )
            rs_j = torch.clamp(rs_j, max=rn)
            s_d[fr_mask] = ss_d[fr_mask] + (rn[fr_mask] - rs_j[fr_mask]) / fr_d[fr_mask]
            s_d = torch.clamp(s_d, max=sm)

        elif book == "EH":
            smmf = smm * (1 - (1 - fr_d) ** (1 / ex))
            smf = smmf / (1 + ex)
            ss_d = torch.clamp(ss_d, max=smf - PRECISION)
            au = smmf * (1 - (1 - ss_d / smf) ** (1 / (1 + ex)))
            if torch.isnan(au).any():
                raise ValueError(
                    "Error: NaN values detected. Try set clip function or check your data!!!"
                )
            rs_j = torch.full(rn.size(), 0.0).to(xaj_device)
            rs_j[fr_mask] = torch.where(
                pen[fr_mask] + au[fr_mask] < smmf[fr_mask],
                (
                    pen[fr_mask]
                    - smf[fr_mask]
                    + ss_d[fr_mask]
                    + smf[fr_mask]
                    * (
                        1
                        - torch.clamp(
                            pen[fr_mask] + au[fr_mask],
                            max=smmf[fr_mask],
                        )
                        / smmf[fr_mask]
                    )
                    ** (ex[fr_mask] + 1)
                )
                * fr_d[fr_mask],
                (pen[fr_mask] + ss_d[fr_mask] - smf[fr_mask]) * fr_d[fr_mask],
            )
            rs_j = torch.clamp(rs_j, max=rn)
            s_d[fr_mask] = ss_d[fr_mask] + (rn[fr_mask] - rs_j[fr_mask]) / fr_d[fr_mask]
            s_d = torch.clamp(s_d, max=smf)
        else:
            raise NotImplementedError(
                "We don't have this implementation! Please chose 'HF' or 'EH'!!"
            )

        rss_j = s_d * kss_d * fr_d
        rg_j = s_d * kg_d * fr_d
        s1_d = s_d * (1 - kss_d - kg_d)

        rs = rs + rs_j
        rss = rss + rss_j
        rg = rg + rg_j
        # 赋值s_d和fr_d到数组中，以给下一段做初值
        s_ds.append(s1_d)
        fr_ds.append(fr_d)

    return (rs, rss, rg), (s_ds[-1], fr_ds[-1])


def linear_reservoir(x, weight, last_y: Optional[Tensor] = None):
    """
    Linear reservoir's release function

    Parameters
    ----------
    x
        the input to the linear reservoir
    weight
        the coefficient of linear reservoir
    last_y
        the output of last period

    Returns
    -------
    torch.Tensor
        one-step forward result
    """
    weight1 = 1 - weight
    if last_y is None:
        last_y = torch.full(weight.size(), 0.001).to(x.device)
    y = weight * last_y + weight1 * x
    return y


class Xaj4Dpl(nn.Module):
    """
    XAJ model for Differential Parameter learning
    """

    def __init__(
        self,
        kernel_size: int,
        warmup_length: int,
        source_book="HF",
        source_type="sources",
    ):
        """
        Parameters
        ----------
        kernel_size
            the time length of unit hydrograph
        warmup_length
            the length of warmup periods;
            XAJ needs a warmup period to generate reasonable initial state values
        """
        super(Xaj4Dpl, self).__init__()
        self.params_names = MODEL_PARAM_DICT["xaj_mz"]["param_name"]
        param_range = MODEL_PARAM_DICT["xaj_mz"]["param_range"]
        self.k_scale = param_range["K"]
        self.b_scale = param_range["B"]
        self.im_sacle = param_range["IM"]
        self.um_scale = param_range["UM"]
        self.lm_scale = param_range["LM"]
        self.dm_scale = param_range["DM"]
        self.c_scale = param_range["C"]
        self.sm_scale = param_range["SM"]
        self.ex_scale = param_range["EX"]
        self.ki_scale = param_range["KI"]
        self.kg_scale = param_range["KG"]
        self.a_scale = param_range["A"]
        self.theta_scale = param_range["THETA"]
        self.ci_scale = param_range["CI"]
        self.cg_scale = param_range["CG"]
        self.kernel_size = kernel_size
        self.warmup_length = warmup_length
        # there are 2 input variables in XAJ: P and PET
        self.feature_size = 2
        self.source_book = source_book
        self.source_type = source_type

    def forward(self, p_and_e, parameters, return_state=False):
        """
        run XAJ model

        Parameters
        ----------
        p_and_e
            precipitation and potential evapotranspiration
        parameters
            parameters of XAJ model
        return_state
            if True, return state values, mainly for warmup periods

        Returns
        -------
        torch.Tensor
            streamflow got by XAJ
        """
        xaj_device = p_and_e.device
        # denormalize the parameters to general range
        k = self.k_scale[0] + parameters[:, 0] * (self.k_scale[1] - self.k_scale[0])
        b = self.b_scale[0] + parameters[:, 1] * (self.b_scale[1] - self.b_scale[0])
        im = self.im_sacle[0] + parameters[:, 2] * (self.im_sacle[1] - self.im_sacle[0])
        um = self.um_scale[0] + parameters[:, 3] * (self.um_scale[1] - self.um_scale[0])
        lm = self.lm_scale[0] + parameters[:, 4] * (self.lm_scale[1] - self.lm_scale[0])
        dm = self.dm_scale[0] + parameters[:, 5] * (self.dm_scale[1] - self.dm_scale[0])
        c = self.c_scale[0] + parameters[:, 6] * (self.c_scale[1] - self.c_scale[0])
        sm = self.sm_scale[0] + parameters[:, 7] * (self.sm_scale[1] - self.sm_scale[0])
        ex = self.ex_scale[0] + parameters[:, 8] * (self.ex_scale[1] - self.ex_scale[0])
        ki_ = self.ki_scale[0] + parameters[:, 9] * (
            self.ki_scale[1] - self.ki_scale[0]
        )
        kg_ = self.kg_scale[0] + parameters[:, 10] * (
            self.kg_scale[1] - self.kg_scale[0]
        )
        # ki+kg should be smaller than 1; if not, we scale them, but note float only contain 4 digits, so we need 0.999
        ki = torch.where(
            ki_ + kg_ < 1.0,
            ki_,
            (1 - PRECISION) / (ki_ + kg_) * ki_,
        )
        kg = torch.where(
            ki_ + kg_ < 1.0,
            kg_,
            (1 - PRECISION) / (ki_ + kg_) * kg_,
        )
        a = self.a_scale[0] + parameters[:, 11] * (self.a_scale[1] - self.a_scale[0])
        theta = self.theta_scale[0] + parameters[:, 12] * (
            self.theta_scale[1] - self.theta_scale[0]
        )
        ci = self.ci_scale[0] + parameters[:, 13] * (
            self.ci_scale[1] - self.ci_scale[0]
        )
        cg = self.cg_scale[0] + parameters[:, 14] * (
            self.cg_scale[1] - self.cg_scale[0]
        )

        # initialize state values
        warmup_length = self.warmup_length
        if warmup_length > 0:
            # set no_grad for warmup periods
            with torch.no_grad():
                p_and_e_warmup = p_and_e[0:warmup_length, :, :]
                cal_init_xaj4dpl = Xaj4Dpl(
                    self.kernel_size, 0, self.source_book, self.source_type
                )
                if cal_init_xaj4dpl.warmup_length > 0:
                    raise RuntimeError("Please set init model's warmup length to 0!!!")
                _, _, *w0, s0, fr0, qi0, qg0 = cal_init_xaj4dpl(
                    p_and_e_warmup, parameters, return_state=True
                )
        else:
            # use detach func to make wu0 no_grad as it is an initial value
            w0 = (0.5 * (um.detach()), 0.5 * (lm.detach()), 0.5 * (dm.detach()))
            s0 = 0.5 * (sm.detach())
            fr0 = torch.full(ci.size(), 0.1).to(xaj_device)
            qi0 = torch.full(ci.size(), 0.1).to(xaj_device)
            qg0 = torch.full(cg.size(), 0.1).to(xaj_device)

        inputs = p_and_e[warmup_length:, :, :]
        runoff_ims_ = torch.full(inputs.shape[:2], 0.0).to(xaj_device)
        rss_ = torch.full(inputs.shape[:2], 0.0).to(xaj_device)
        ris_ = torch.full(inputs.shape[:2], 0.0).to(xaj_device)
        rgs_ = torch.full(inputs.shape[:2], 0.0).to(xaj_device)
        es_ = torch.full(inputs.shape[:2], 0.0).to(xaj_device)
        for i in range(inputs.shape[0]):
            if i == 0:
                (r, rim, e, pe), w = xaj_generation(
                    inputs[i, :, :], k, b, im, um, lm, dm, c, *w0
                )
                if self.source_type == "sources":
                    (rs, ri, rg), (s, fr) = xaj_sources(
                        pe, r, sm, ex, ki, kg, s0, fr0, book=self.source_book
                    )
                elif self.source_type == "sources5mm":
                    (rs, ri, rg), (s, fr) = xaj_sources5mm(
                        pe, r, sm, ex, ki, kg, s0, fr0, book=self.source_book
                    )
                else:
                    raise NotImplementedError("No such divide-sources method")
            else:
                (r, rim, e, pe), w = xaj_generation(
                    inputs[i, :, :], k, b, im, um, lm, dm, c, *w
                )
                if self.source_type == "sources":
                    (rs, ri, rg), (s, fr) = xaj_sources(
                        pe, r, sm, ex, ki, kg, s, fr, book=self.source_book
                    )
                elif self.source_type == "sources5mm":
                    (rs, ri, rg), (s, fr) = xaj_sources5mm(
                        pe, r, sm, ex, ki, kg, s, fr, book=self.source_book
                    )
                else:
                    raise NotImplementedError("No such divide-sources method")
            # impevious part is pe * im
            runoff_ims_[i, :] = rim
            # so for non-imprvious part, the result should be corrected
            rss_[i, :] = rs * (1 - im)
            ris_[i, :] = ri * (1 - im)
            rgs_[i, :] = rg * (1 - im)
            es_[i, :] = e
            # rss_[i, :] = 0.7 * r
            # ris_[i, :] = 0.2 * r
            # rgs_[i, :] = 0.1 * r
        # seq, batch, feature
        runoff_im = torch.unsqueeze(runoff_ims_, dim=2)
        rss = torch.unsqueeze(rss_, dim=2)
        es = torch.unsqueeze(es_, dim=2)

        conv_uh = KernelConv(a, theta, self.kernel_size)
        qs_ = conv_uh(runoff_im + rss)

        qs = torch.full(inputs.shape[:2], 0.0).to(xaj_device)
        for i in range(inputs.shape[0]):
            if i == 0:
                qi = linear_reservoir(ris_[i], ci, qi0)
                qg = linear_reservoir(rgs_[i], cg, qg0)
            else:
                qi = linear_reservoir(ris_[i], ci, qi)
                qg = linear_reservoir(rgs_[i], cg, qg)
            qs[i, :] = qs_[i, :, 0] + qi + qg
        # seq, batch, feature
        q_sim = torch.unsqueeze(qs, dim=2)
        if return_state:
            return q_sim, es, *w, s, fr, qi, qg
        return q_sim, es


class DplLstmXaj(nn.Module):
    def __init__(
        self,
        n_input_features,
        n_output_features,
        n_hidden_states,
        kernel_size,
        warmup_length,
        param_limit_func="sigmoid",
        param_test_way="final",
        source_book="HF",
        source_type="sources",
    ):
        """
        Differential Parameter learning model: LSTM -> Param -> XAJ

        The principle can be seen here: https://doi.org/10.1038/s41467-021-26107-z

        Parameters
        ----------
        n_input_features
            the number of input features of LSTM
        n_output_features
            the number of output features of LSTM, and it should be equal to the number of learning parameters in XAJ
        n_hidden_states
            the number of hidden features of LSTM
        kernel_size
            the time length of unit hydrograph
        warmup_length
            the length of warmup periods;
            hydrologic models need a warmup period to generate reasonable initial state values
        param_limit_func
            function used to limit the range of params; now it is sigmoid or clamp function
        param_test_way
            how we use parameters from dl model when testing;
            now we have three ways:
            1. "final" -- use the final period's parameter for each period
            2. "mean_time" -- Mean values of all periods' parameters is used
            3. "mean_basin" -- Mean values of all basins' final periods' parameters is used
        """
        super(DplLstmXaj, self).__init__()
        self.dl_model = SimpleLSTM(n_input_features, n_output_features, n_hidden_states)
        self.pb_model = Xaj4Dpl(
            kernel_size, warmup_length, source_book=source_book, source_type=source_type
        )
        self.param_func = param_limit_func
        self.param_test_way = param_test_way

    def forward(self, x, z):
        """
        Differential parameter learning

        z (normalized input) -> lstm -> param -> + x (not normalized) -> xaj -> q
        Parameters will be denormalized in xaj model

        Parameters
        ----------
        x
            not normalized data used for physical model; a sequence-first 3-dim tensor. [sequence, batch, feature]
        z
            normalized data used for DL model; a sequence-first 3-dim tensor. [sequence, batch, feature]

        Returns
        -------
        torch.Tensor
            one time forward result
        """
        q, e = lstm_pbm(self.dl_model, self.pb_model, self.param_func, x, z)
        return q


class DplAnnXaj(nn.Module):
    def __init__(
        self,
        n_input_features: int,
        n_output_features: int,
        n_hidden_states: Union[int, tuple, list],
        dr: Union[int, tuple, list],
        kernel_size: int,
        warmup_length: int,
        param_limit_func="sigmoid",
        param_test_way="final",
        source_book="HF",
        source_type="sources",
    ):
        """
        Differential Parameter learning model only with attributes as DL model's input: ANN -> Param -> Gr4j

        The principle can be seen here: https://doi.org/10.1038/s41467-021-26107-z

        Parameters
        ----------
        n_input_features
            the number of input features of ANN
        n_output_features
            the number of output features of ANN, and it should be equal to the number of learning parameters in XAJ
        n_hidden_states
            the number of hidden features of ANN; it could be Union[int, tuple, list]
        kernel_size
            the time length of unit hydrograph
        warmup_length
            the length of warmup periods;
            hydrologic models need a warmup period to generate reasonable initial state values
        param_limit_func
            function used to limit the range of params; now it is sigmoid or clamp function
        param_test_way
            how we use parameters from dl model when testing;
            now we have three ways:
            1. "final" -- use the final period's parameter for each period
            2. "mean_time" -- Mean values of all periods' parameters is used
            3. "mean_basin" -- Mean values of all basins' final periods' parameters is used
        """
        super(DplAnnXaj, self).__init__()
        self.dl_model = SimpleAnn(
            n_input_features, n_output_features, n_hidden_states, dr
        )
        self.pb_model = Xaj4Dpl(
            kernel_size, warmup_length, source_book=source_book, source_type=source_type
        )
        self.param_func = param_limit_func
        self.param_test_way = param_test_way

    def forward(self, x, z):
        """
        Differential parameter learning

        z (normalized input) -> ANN -> param -> + x (not normalized) -> gr4j -> q
        Parameters will be denormalized in gr4j model

        Parameters
        ----------
        x
            not normalized data used for physical model; a sequence-first 3-dim tensor. [sequence, batch, feature]
        z
            normalized data used for DL model; a 2-dim tensor. [batch, feature]

        Returns
        -------
        torch.Tensor
            one time forward result
        """
        q, e = ann_pbm(self.dl_model, self.pb_model, self.param_func, x, z)
        return q
