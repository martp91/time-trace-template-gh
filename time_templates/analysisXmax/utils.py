import os
import numpy as np
from time_templates.misc.Xmax import Fd_Xmax_resolution_at_lgE
from time_templates import package_path


#################
# Load data
#################


##############
# Resolution
##############

# as fit on hybrid data

HYBRID_XMAX_TTT_OFFSET = 28.3
HYBRID_XMAX_TTT_OFFSET_ERR = 2.8

file_P_HYBRID_XMAX_RESOLUTION = os.path.join(
    package_path, "data", "resolution_hybrid_SD_Xmax_p.txt"
)
file_COV_HYBRID_XMAX_RESOLUTION = os.path.join(
    package_path, "data", "resolution_hybrid_SD_Xmax_cov.txt"
)

P_HYBRID_XMAX_RESOLUTION = np.loadtxt(file_P_HYBRID_XMAX_RESOLUTION)
COV_HYBRID_XMAX_RESOLUTION = np.loadtxt(file_COV_HYBRID_XMAX_RESOLUTION)


def resolution_func(lgE, a, b):
    "sqrt(FD**2 + SD**2)"
    return a + b * 10 ** (19 - lgE)


def SD_Xmax_resolution(
    lgE, a=P_HYBRID_XMAX_RESOLUTION[0], b=P_HYBRID_XMAX_RESOLUTION[1]
):
    total = resolution_func(lgE, a, b)
    FD = Fd_Xmax_resolution_at_lgE(lgE)
    return np.sqrt(total**2 - FD**2)


#################
# DXmax bias MC
#################
# as determined on EPOS proton/iron showers lgE: 19-20
DX0 = 600
LGE_0 = 19.5

file_P_DXMAX_BIAS_CORR = os.path.join(package_path, "data", "DXmax_bias_corr_p.txt")
file_COV_DXMAX_BIAS_CORR = os.path.join(package_path, "data", "DXmax_bias_corr_cov.txt")

P_DXMAX_BIAS_CORR = np.loadtxt(file_P_DXMAX_BIAS_CORR)

COV_DXMAX_BIAS_CORR = np.loadtxt(file_COV_DXMAX_BIAS_CORR)
# print(P_DXMAX_BIAS_CORR)
# print(COV_DXMAX_BIAS_CORR)


def DXmax_bias(DXmax, lgE=19.5, p=P_DXMAX_BIAS_CORR, DXbreak=DX0):
    """From fit to DXmax(TTT)-DXmax(MC) vs DXmax(MC) on EPOS proton/iron showers
    lgE/eV: 19-20
    Cuts as in Table 6.1

    Parameters
    ----------
    DXmax : float or array of float
        DXmax defined as distance to Xmax from shower core
        DXmax = atm.slant_depth_at_height(hcore, zenith) - Xmax

    Returns
    ---------
    polyonimal in x=DXmax/DX0 -1
    """
    lgE_ = lgE - LGE_0
    a = p[1] * lgE_
    b = p[2] + p[3] * lgE_
    c = p[4]
    d = p[5]
    x = DXmax / DXbreak - 1
    out = a + b * x + c * x**2 + d * x**3
    return p[0] + np.where(DXmax > DXbreak, 0, out)


def unbias_DXmax(DXmax, lgE=19.5, DXbreak=DX0, lgE0=LGE_0, p=P_DXMAX_BIAS_CORR):
    """unbias DXmax from a measured DXmax(TTT), by solving 3rd degree poly

    Parameters
    ----------
    DXmax : float (no arrays!)
        DXmax see above
    lgE : float
        lgE/eV
    DXbreak : float
        Flat above this, not necesearly DX0
    lgE0 :
        lgE0
    p : list of fit values
        p
    """
    lgE_ = lgE - lgE0
    a = p[0] + p[1] * lgE_
    b = p[2] + p[3] * lgE_
    c = p[4]
    d = p[5]
    if DXmax > DXbreak:
        return DXmax - p[0]
    return (
        np.real((np.roots([d, c, b + DXbreak, a - DXmax + DXbreak])[-1] + 1)) * DXbreak
    )
