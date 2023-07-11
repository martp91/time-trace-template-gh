import os
import json
import numpy as np
from numba import njit
from time_templates.templates.universality.names import (
    eMUON,
    eEM_PURE,
    eEM_MU,
    eEM_HAD,
)
from time_templates import package_path

data_path = os.path.join(package_path, "data")

DXREF = 400  # g/cm2

# Ave et al 2017 APP. With probablly typo for eEM_MU that should be -250
DX0_COMP = {eMUON: -250, eEM_PURE: -500, eEM_MU: -250, eEM_HAD: -800}


FILE_GH_COMP = os.path.join(data_path, "shower_size_GH_fit_params.json")
FILE_SEC_THETA_CORR_COMP = os.path.join(data_path, "shower_size_sec_theta_corr.json")
FILE_RMU_CORR_COMP = os.path.join(data_path, "shower_size_Rmu_corr.json")

try:
    with open(FILE_GH_COMP, "r") as infile:
        GH_COMP = json.load(infile)
except FileNotFoundError:
    print(f"Warning could not find {FILE_GH_COMP}, using hard coded vals")
    GH_COMP = {
        eMUON: {
            "lgSref": [1.339458010819829, 0.9347142869098212],
            "DXmax": [340.6073377476682, 0],
            "lam": [423.1153254551724, 0],
        },
        eEM_PURE: {
            "lgSref": [0.8362878253043337, 0.9853569469017787],
            "DXmax": [168.56905439057792, 0],
            "lam": [82.50395482932659, 8.677096776417851],
        },
        eEM_MU: {
            "lgSref": [0.5939719993677353, 0.9264912285083613],
            "DXmax": [283.2723689405857, 0],
            "lam": [361.4090701354997, 0],
        },
        eEM_HAD: {
            "lgSref": [0.7362206200083165, 0.9224575359090643],
            "DXmax": [39.89438156013308, 0],
            "lam": [88.01495041722774, 8.742619156500457],
        },
    }
try:
    with open(FILE_SEC_THETA_CORR_COMP, "r") as infile:
        SEC_THETA_CORR_COMP = json.load(infile)
except FileNotFoundError:
    print(f"Warning could not find {FILE_SEC_THETA_CORR_COMP}, using hard coded vals")
    SEC_THETA_CORR_COMP = {
        eMUON: [
            0.9894533136604929,
            -0.11051466895944895,
            0.1360878719429421,
            0.32324450662370524,
        ],
        eEM_PURE: [
            0.9809880408072699,
            0.024263986929437764,
            0.3599783473452425,
            -0.5591773282491483,
        ],
        eEM_MU: [
            0.9816404150195188,
            -0.12296445255760591,
            0.29710393979048505,
            0.2392304959898658,
        ],
        eEM_HAD: [
            0.968327102030941,
            -0.12309687004194796,
            0.5676943892506003,
            0.2580832357411827,
        ],
    }
try:
    with open(FILE_RMU_CORR_COMP, "r") as infile:
        RMU_CORR_COMP = json.load(infile)
except FileNotFoundError:
    print(f"Warning could not find {FILE_RMU_CORR_COMP}, using hard coded vals")
    RMU_CORR_COMP = {
        eMUON: 1,
        eEM_PURE: -0.024638935731785994,
        eEM_MU: 0.9643188342098775,
        eEM_HAD: 1.3692733098004606,
    }


# @njit(cache=True, nogil=True, error_model="numpy")
def modified_Gaisser_Hillas(DX, DXmax, DX0, lam):
    GH = ((DX - DX0) / (DXREF - DX0)) ** ((DXmax - DX0) / lam) * np.exp(
        (DXREF - DX) / lam
    )
    return np.where(DX - DX0 > 0, GH, 0)


def S1000_func(DX, Sref, DXmax, DX0, lam):
    return Sref * modified_Gaisser_Hillas(DX, DXmax, DX0, lam)


def Rmu_corr(Rmu, b):
    return 1 + b * (Rmu - 1)


# Lin function for lgSref, DXmax, lambda
def lin_lgE(lgE, a, b):
    return a + b * (lgE - 19)


def sec_theta_func(sec_theta, a, b, c, d):
    x = sec_theta - 1.33
    return a + b * x + c * x**2 + d * x**3


def Rmu_func(Rmu, alpha):
    return 1 + alpha * (Rmu - 1)


def S1000_comp_model(
    lgE,
    DX,
    theta,
    Rmu=1,
    comp=eMUON,
    gh_comp=GH_COMP,
    sec_theta_corr_comp=SEC_THETA_CORR_COMP,
    rmu_corr_comp=RMU_CORR_COMP,
):
    Rmu_corr_ = Rmu_func(Rmu, rmu_corr_comp[comp])
    if sec_theta_corr_comp is not None:
        sct_corr = sec_theta_func(1 / np.cos(theta), *sec_theta_corr_comp[comp])
    else:
        sct_corr = 1
    gh_params = gh_comp[comp]
    return (
        Rmu_corr_
        * sct_corr
        * S1000_func(
            DX,
            10 ** lin_lgE(lgE, *gh_params["lgSref"]),
            lin_lgE(lgE, *gh_params["DXmax"]),
            DX0_COMP[comp],
            lin_lgE(lgE, *gh_params["lam"]),
        )
    )


def set_Rmu_df(df):
    Smu1000_pred = S1000_comp_model(
        df["MClgE"].values, df["MCDX_1000"].values, df["MCTheta"].values, 1, eMUON
    )
    df["Rmu"] = df["MC_mu1000"] / Smu1000_pred
