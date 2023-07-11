import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from scipy.optimize import least_squares, minimize
from numba import njit

from time_templates import package_path, data_path
from time_templates.signalmodel import signal_model
from time_templates.utilities.nb_funcs import interp
from time_templates.utilities.fitting import plot_fit_curve
from time_templates.templates.universality.names import (
    DICT_COMP_LABELS,
    DICT_COMP_SIGNALKEY,
    eMUON,
    eEM_PURE,
    eEM_MU,
    eEM_HAD,
)

SQRT2PI = np.sqrt(2 * np.pi)

# filedir = os.path.dirname(os.path.abspath(__file__))


MS_PARAMS_FILE = os.path.join(
    package_path, "data", "lognormal_m_s_interpolation_table.json"
)
NPARAMETERS = 4

try:
    with open(MS_PARAMS_FILE, "r") as input_file:
        DICT_MS_DX_FIT = json.load(input_file)
    RDENSE = np.array(sorted(DICT_MS_DX_FIT["rdense"]))
    for comp in DICT_COMP_SIGNALKEY.keys():
        for key in ["m", "s"]:
            # convert to array because json has only lists
            DICT_MS_DX_FIT[comp][key] = np.array(DICT_MS_DX_FIT[comp][key])
            DICT_MS_DX_FIT[comp][key + "err"] = np.array(
                DICT_MS_DX_FIT[comp][key + "err"]
            )
except FileNotFoundError as e:
    print("First run this script too fill table")

# njit makes about 4 times faster
@njit(fastmath=True, cache=True, nogil=True, error_model="numpy")
def lognormal_pdf(t, m, s, t0=0):
    prefac = 1 / (s * SQRT2PI)
    s2_05_inv = 0.5 / s**2
    # slightly faster
    nt = len(t)
    out = np.zeros(nt)
    for i, _t in enumerate(t):
        if _t > t0:
            _t = _t - t0
            out[i] = prefac * np.exp(-s2_05_inv * (np.log(_t) - m) ** 2) / _t

    # t_ = t - t0
    # out = np.where(
    #    t_ > 0, prefac * np.exp(-0.5 * (np.log(t_) - m) ** 2 / s ** 2) / t_, 0,
    # )

    return out


signal_responses = {
    "UUB_WCD": signal_model.UUB_WCD_t_response,
    "UUB_SSD": signal_model.UUB_SSD_t_response,
    "UB_WCD": signal_model.UB_WCD_t_response,
    "UB_SSD": signal_model.UB_SSD_t_response,
}


def lognormal_signal(t, m, s, t0=0, det="UUB_WCD"):
    response = signal_responses[det]
    out = signal_model.evaluate_convolution(
        t, lambda t: response(lognormal_pdf(t, m, s, t0))
    )
    outsum = out.sum()
    if outsum > 0:
        return out / (outsum * (t[1] - t[0]))
    return out


def ms_parameters_func(DX, a, b, c=0, d=0):
    DXref = DX / 400 - 1
    return a + b * DXref + c * DXref**2 + d * DXref**3


def get_interpolated_r_ms_parameters(r, comp, key, kind="linear"):
    x = RDENSE
    nx = NPARAMETERS  # len(dict_ms_x0[key][comp])
    out = np.zeros(nx)
    ys = DICT_MS_DX_FIT[comp][key]
    yerrs = DICT_MS_DX_FIT[comp][key + "err"]
    lgr = np.log10(r)
    lgx = np.log10(x)
    for ip in range(nx):
        y = ys[:, ip]
        yerr = yerrs[:, ip]
        if kind == "linear":  # about 10x faster
            out[ip] = interp(lgr, lgx, y)
        else:
            spl = splrep(lgx, y, w=1 / yerr, k=3, s=len(y))
            out[ip] = splev(lgr, spl)
    return out


def get_m_s_lognormal_comp(DX, r, comp, interp_kind="linear"):
    mparams = get_interpolated_r_ms_parameters(r, comp, "m", interp_kind)
    sparams = get_interpolated_r_ms_parameters(r, comp, "s", interp_kind)
    m = ms_parameters_func(DX, *mparams)
    s = ms_parameters_func(DX, *sparams)
    # HACK
    m = min(max(m, 1), 10)
    s = min(max(s, 0.1), 3)
    return m, s


def fit_func_bootstrap(
    x, m, merr, fitfunc=ms_parameters_func, p0=[1, 1, 1, 1], ax=None
):
    _, (pfit, perr, chi2, ndof) = plot_fit_curve(
        x, m, yerr=merr, p0=p0, func=fitfunc, ax=None
    )
    mpred = fitfunc(x, *pfit)
    residual = np.abs((m - mpred) / merr)
    mask = residual < 5
    if len(x[mask]) < 4:
        return pfit, perr, chi2, ndof
    else:
        ax, (pfit, perr, chi2, ndof) = plot_fit_curve(
            x[mask],
            m[mask],
            yerr=merr[mask],
            p0=pfit,
            func=fitfunc,
            ax=ax,
            smoother_x=True,
        )
    return pfit, perr, chi2, ndof


def parmaterize_DX(df):
    print("running...")
    f, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    f2, axes2 = plt.subplots(2, 2, figsize=(12, 12))
    axes2 = axes2.flatten()
    d_DX_fit = {}
    rs = np.sort(df.index.get_level_values(level=0).unique())
    nr = len(rs)
    empty_array = np.empty((nr, NPARAMETERS))
    d_DX_fit["rdense"] = rs

    for comp, ax1, ax2 in zip(DICT_COMP_LABELS, axes, axes2):
        d_DX_fit[comp] = {}
        empty_array[:] = np.nan
        d_DX_fit[comp]["m"] = empty_array.copy()  # .tolist()
        d_DX_fit[comp]["s"] = empty_array.copy()  # .tolist()
        d_DX_fit[comp]["merr"] = empty_array.copy()  # .tolist()
        d_DX_fit[comp]["serr"] = empty_array.copy()  # .tolist()
        for i, r in enumerate(rs):
            df_ = df.loc[r]
            dx = df_["MCDXstation"]
            red_chi2 = df_[f"wcd_{comp}_trace_redchi2"]
            m = df_[f"wcd_{comp}_trace_mfit"]
            merr = df_[f"wcd_{comp}_trace_merr"]  # * np.sqrt(red_chi2)
            s = df_[f"wcd_{comp}_trace_sfit"]  # * np.sqrt(red_chi2)
            serr = df_[f"wcd_{comp}_trace_serr"]
            success = df_[f"wcd_{comp}_trace_success"]
            t0 = df_[f"wcd_{comp}_trace_t0"]
            mask = np.isfinite(dx * m * merr * s * serr) & (
                success
            )  # & (serr/s < 1) & (merr/m < 1) & (red_chi2 < 100) & (t0 < 500)
            if len(dx[mask]) < 4:
                continue
            #             d_DX_fit[comp][r] = {}
            pfit, perr, chi2, ndof = fit_func_bootstrap(
                dx[mask], m[mask], merr[mask], ax=ax1, p0=[1, 1, 0, 0]
            )
            #             print(pfit)
            d_DX_fit[comp]["m"][i, :] = pfit
            d_DX_fit[comp]["merr"][i, :] = perr
            #             d_DX_fit[comp][r]['m'] = pfit
            #             d_DX_fit[comp][r]['merr'] = perr
            pfit, perr, chi2, ndof = fit_func_bootstrap(
                dx[mask], s[mask], serr[mask], ax=ax2, p0=[1, 1]
            )
            d_DX_fit[comp]["s"][i, :2] = pfit
            d_DX_fit[comp]["serr"][i, :2] = perr
            d_DX_fit[comp]["s"][i, 2:] = 0
            d_DX_fit[comp]["serr"][i, 2:] = 0
    #             d_DX_fit[comp][r]['s'] = pfit
    #             d_DX_fit[comp][r]['serr'] = perr
    print("...done")
    return d_DX_fit


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-file",
        default=os.path.join(data_path, "lognormal_fit", "df_fitted_lognormals.pl"),
    )

    args = parser.parse_args()
    print(f"Reading {args.file}")
    df = pd.read_pickle(args.file)

    d_DX_fit = parmaterize_DX(df)

    dict_ms_fit = {}

    for comp in DICT_COMP_LABELS:
        dict_ms_fit[comp] = {}
        dict_ms_fit[comp]["m"] = d_DX_fit[comp]["m"].tolist()
        dict_ms_fit[comp]["s"] = d_DX_fit[comp]["s"].tolist()
        dict_ms_fit[comp]["merr"] = d_DX_fit[comp]["merr"].tolist()
        dict_ms_fit[comp]["serr"] = d_DX_fit[comp]["serr"].tolist()
    dict_ms_fit["rdense"] = d_DX_fit["rdense"].tolist()
    print(dict_ms_fit["muon"])
    print(f"Saving at {MS_PARAMS_FILE}")
    with open(MS_PARAMS_FILE, "w") as output:
        json.dump(dict_ms_fit, output, indent=4)

    print("...done")
