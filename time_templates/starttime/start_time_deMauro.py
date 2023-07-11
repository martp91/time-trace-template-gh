"""
From de Mauro Thesis 2020

Just some code to go from Rc to DXmax and vice versa

Mart Pothast 2022
"""
import numpy as np
from time_templates.utilities import atmosphere
from time_templates.utilities.constants import *  # BAD
from numba import njit


# From peppe de Mauro 2020 thesis:
# Table 5.2
# A = 416.03
# B = 1.16
# C = -134.66
# AERR = 4.84
# BERR = 0.03
# CERR = 31.33

# Newly calibrated by MP
# new on hybrid data icrc2019 theta < 60 deg, SdlgE > 19
# on all events, so no cut on p value
# Only Sdr_new_min > 500
A = 460
B = 1.06
C = -147.6
AERR = 6.1
BERR = 0.038
CERR = 23.1

MEAN_DXRC = 855.4  # corrected, typo in thesis
MEAN_LGE = 1.15  # lg(E/EeV)
SIGMA_C = 42
# UB offlinev3r99p2a EPOS LHC proton iron average lgE 19-20 flat, theta < 53 deg
# no cuts on p value or something.
# updated with cut on min(r) > 500
# Some discrepancy between XmaxInterpolated and XmaxGH
# This is GH
AMC = 529.4  # pm 1.5
BMC = 1.24  # pm 0.00
CMC = -135  # pm 3.4 ?


ATM = atmosphere.Atmosphere(model=21)


def DXmaxRc_callibration(
    DXRc, lgE, a=A, b=B, c=C, mean_DXRc=MEAN_DXRC, mean_lgE=MEAN_LGE
):
    "DXmaxFD from DXRc"
    return a + b * (DXRc - mean_DXRc + c * ((lgE - 18) - mean_lgE))


def inv_DXmaxRc_callibration(
    DXmax, lgE, a=A, b=B, c=C, mean_DXRc=MEAN_DXRC, mean_lgE=MEAN_LGE
):
    "DXRc from DXmaxFD"
    return (DXmax - a) / b + mean_DXRc - c * ((lgE - 18) - mean_lgE)


# var(DXRc) = -var(Rc) (dXRc/dRc /cos(theta))**2
def DX_at_Rc(Rc, theta, atm=ATM, hground=1400):
    # sure about this?
    Xg = atm.slant_depth_at_height(hground, 0)
    XRc = atm.slant_depth_at_height(hground + Rc * np.cos(theta), 0)
    return (Xg - XRc) / np.cos(theta)


def Rc_at_DXRc(DXRc, theta, atm=ATM, hground=1400):
    Xg = atm.slant_depth_at_height(hground, 0)
    XRc = Xg - DXRc * np.cos(theta)
    hRC = atm.height_at_slant_depth(XRc, 0) - hground
    return hRC / np.cos(theta)


def DXmax_at_Rc(Rc, theta, lgE, atm=ATM, hground=1400, is_data=True):
    # DXmax FD
    DXRc = DX_at_Rc(Rc, theta, atm, hground)
    if is_data:
        DXmax = DXmaxRc_callibration(DXRc, lgE)
    else:
        # average proton/iron EPOS_LHC lgE 19-20, theta < 53 deg
        DXmax = DXmaxRc_callibration(DXRc, lgE, a=AMC, b=BMC, c=CMC)
    return DXmax


def Rc_at_DXmax(DXmax, theta, lgE, atm=ATM, hground=1400, is_data=True):
    if is_data:
        DXRc = inv_DXmaxRc_callibration(DXmax, lgE)
    else:
        DXRc = inv_DXmaxRc_callibration(DXmax, lgE, a=AMC, b=BMC, c=CMC)
    return Rc_at_DXRc(DXRc, theta, atm, hground)


# DXRc = DXRc - meanDXRc
# lgE = (lgE-18) - meanlgE
# var(DXmax) = var(a) + var(b)*(DXRc + c*lgE) + var(c) * b * lgE + var(DXRc)*b + var(lgE) * b * c


def var_DXRc(varRc, Rc, theta, atm=ATM, hground=1400):
    return varRc * (atm.dXdh(hground + Rc * np.cos(theta)) / np.cos(theta)) ** 2


def var_DXmaxRc(
    varRc,
    Rc,
    theta,
    lgE,
    var_lgE=0,
    atm=ATM,
    hground=1400,
    var_a=AERR ** 2,
    var_b=BERR ** 2,
    var_c=CERR ** 2,
    sigma_c=SIGMA_C,
):
    # TODO make this nice and modular
    varDXRc = var_DXRc(varRc, Rc, theta, atm, hground)
    DXRc = DX_at_Rc(Rc, theta, atm, hground) - MEAN_DXRC
    lgE_ = (lgE - 18) - MEAN_LGE
    return (
        var_a
        + var_b * (DXRc + C * lgE_) ** 2
        + var_c * (B * lgE_) ** 2
        + varDXRc * B ** 2
        + var_lgE * (B * C) ** 2
        + sigma_c ** 2  # to make chi2/ndf=1
    )


def catenary(x, a):
    """catenary.

    Parameters
    ----------
    x : distance to shwoer core in m
        x
    a :
        curvature parameter in meter
    """
    return a * np.cosh(x / a)


def start_time_plane_front_catenary_Rc(r, Rc):
    """ I hope I understand his thesis correctly
    and this is indeed start time wrt plane front time
    and x = r distance in shower plane
    """
    return (catenary(r, Rc) - Rc) / C0


def start_time_plane_front_DXmax(r, DXmax, theta, lgE, atm=ATM, hground=1400):
    """start_time_plane_front_DXmax.

    Parameters
    ----------
    r : float
        distanace to core in shower plane in meter
    DXmax :
        DXmax distance to xmax g/cm2
    theta :
        theta zenith in rad
    lgE :
        lgE log10(E/eV)
    hground :
        hground in meter
    model :
        model atmosphere model from corsika (what if data?)
    """
    Rc = Rc_at_DXmax(DXmax, theta, lgE, atm, hground)
    return start_time_plane_front_catenary_Rc(r, Rc)


@njit
def start_time_variance(r, theta, S, RiseTime=None):
    """Giuseppe de Mauro thesis

    Does not work vectorized... too bad! #TODO
    """
    a = 1.12
    b = 10.07
    c = 2.98e-4
    R = 1.8
    H = 1.2
    RTL = 1 / (np.cos(theta) + (2 * H / (np.pi * R) * np.sin(theta)))
    n = S / RTL
    if n < 10 or RiseTime is None:
        RiseTime = 0.9253 * r ** 0.8471 * np.exp(-0.01152 * theta)
        n = 12

    T = RiseTime / 0.59
    return a ** 2 * (T ** 2 / n ** 2 + c * r ** 2 * np.cos(theta) ** 2) + b ** 2


@njit
def v_start_time_variance(r, theta, S, RiseTime, nstations):

    out = np.zeros(nstations)
    for i in range(nstations):
        out[i] = start_time_variance(r[i], theta, S[i], RiseTime[i])

    return out
