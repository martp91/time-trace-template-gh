#!/usr/bin/env python
from collections import defaultdict
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from iminuit import Minuit
from iminuit.cost import LeastSquares, NormalConstraint

from time_templates.templates.universality.lognormal_templates import (
    lognormal_signal,
    lognormal_pdf,
)
from time_templates import data_path
from time_templates.utilities.nb_funcs import interp
from time_templates.starttime.start_time_deMauro import start_time_variance


def fit_lognormal(
    t, y, yerr, det="UUB_WCD", t0=0, fit_t0=False, cdf_min=0.0, cdf_max=0.95, t0_err=100
):

    #     # make sure _y is still in ds/dt 1/s, model returns the same

    cs = np.cumsum(y)
    cs /= cs[-1]
    t01 = interp(cdf_min, cs, t)
    t99 = interp(cdf_max, cs, t)
    tmed = interp(0.5, cs, t)

    dt = t[1] - t[0]

    if cdf_min == 0:
        imin = 0
    else:
        imin = int(t01 / dt)
    imax = int(t99 / dt)

    _t = t[imin:imax]
    _y = y[imin:imax]
    _yerr = yerr[imin:imax]

    mask = _yerr > 0
    _t = _t[mask]
    _y = _y[mask]
    _yerr = _yerr[mask]
    # _yerr = np.maximum(_yerr, _y / 1000)

    N = _y.sum() * dt
    if N <= 0:
        raise RuntimeError
    _y /= N
    _yerr /= N

    p0 = [np.log(tmed), 0.5, t0]
    bounds = [(None, None), (0.0001, None), (-1000, 1000)]
    boost_func = lambda x, m, s, t0: lognormal_signal(x, m, s, t0=t0, det=det)
    # boost_func = lambda x, m, s, t0: lognormal_pdf(x, m, s, t0=t0)

    lq = LeastSquares(_t, _y, _yerr, boost_func)
    ncs = NormalConstraint("t0", value=0, error=t0_err)
    minfunc = lq + ncs
    m = Minuit(minfunc, *p0)
    m.fixed["t0"] = not fit_t0
    m.limits = bounds
    # m.simplex()
    m.migrad()
    # if not (
    #    m.valid
    #    and m.accurate
    #    and not m.fmin.has_parameters_at_limit
    #    and m.fmin.has_posdef_covar
    # ):
    #    _yerr = np.maximum(_yerr, _y / 100)
    #    lq = LeastSquares(_t, _y, _yerr, boost_func)
    #    m = Minuit(lq, *p0)
    #    m.limits = bounds
    #    m.fixed["t0"] = True
    #    m.fixed["m"] = True
    #    m.migrad()
    #    m.fixed["m"] = False
    #    m.simplex()
    #    m.migrad()

    return m, imin, imax


REBIN = True


def get_trace_fit(
    df_row, key="wcd_em_trace", ax=None, fit_t0=False, cdf_min=0.01, cdf_max=0.95
):
    y = df_row[key + "_mean"]
    yerr = np.sqrt(df_row[key + "_var"] / df_row["nstations"])
    if "ssd" in key:
        det = "UUB_SSD"
    else:
        det = "UUB_WCD"

    dt = df_row["dt"]
    if REBIN:
        y = y.reshape((200, 3)).mean(axis=-1)  # WARNING hardcoded
        yerr = np.sqrt(np.sum(yerr.reshape((200, 3)) ** 2, axis=-1) / 3)
        dt = 25
        if "ssd" in key:
            det = "UB_SSD"
        else:
            det = "UB_WCD"

    t = np.linspace(0, (len(y) - 1) * dt, len(y))
    if np.all(y == 0) or y.sum() <= 0:
        print(
            f'All zero for {df_row[["MCr", "MCCosTheta", "MCDXstation", "MCcospsi"]]}'
        )
        raise RuntimeError

    t0_var = (
        start_time_variance(df_row["MCr"], df_row["MCTheta"], df_row["WCDTotalSignal"])
        # / df_row["nstations"]
    )

    m, imin, imax = fit_lognormal(
        t,
        y,
        yerr,
        det=det,
        t0=0,
        fit_t0=fit_t0,
        cdf_min=cdf_min,
        cdf_max=cdf_max,
        t0_err=np.sqrt(t0_var),
    )
    p = m.values
    if ax is not None:
        ax.set_title(f"{df_row['SdCosTheta']} {df_row['Sdcospsi']}")
        ax.errorbar(t, y / (y.sum() * dt), yerr / (y.sum() * dt), ls="", errorevery=1)
        ax.plot(t, lognormal_signal(t, *p, det))
        ax.plot(t[imin], 2e-5, marker="v", color="r", ls="", ms=10, lw=3)
        ax.plot(t[imax], 2e-5, marker="v", color="r", ls="", ms=10)
        ax.set_xlim([0, 2000])
        plt.show()
    return m


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-file", default=data_path + "/mean_df/df_means_merged.pl")

    args = parser.parse_args()

    # traces are binned wrt t0 from de Mauro
    df = pd.read_pickle(args.file)
    df.index.rename([name + "_idx" for name in df.index.names], inplace=True)

    trace_keys = [key[:-5] for key in df.keys() if "trace_mean" in key]
    # trace_keys = ["wcd_em_trace"]

    min_stations = 10
    ax = None

    dd = defaultdict(list)
    for (ct2_bin, DX_bin, r, cp), row in df.iterrows():
        dofit = True
        print()
        print(ct2_bin, DX_bin, r, cp)

        if row["nstations"] < min_stations:
            dofit = False

        # for col, val in df_bin.items():
        #    dd[col].append(val)

        d_fit = {}

        for key in trace_keys:
            success = True
            if dofit:
                try:
                    m = get_trace_fit(row, key, cdf_min=0.0, cdf_max=0.95, ax=ax)
                    if not (
                        m.valid
                        and m.accurate
                        and not m.fmin.has_parameters_at_limit
                        and m.fmin.has_posdef_covar
                    ):
                        m = get_trace_fit(row, key, cdf_min=0.05, cdf_max=0.95, ax=ax)

                    if not (
                        m.valid
                        and m.accurate
                        and not m.fmin.has_parameters_at_limit
                        and m.fmin.has_posdef_covar
                    ):
                        pfit = [np.nan, np.nan]
                        perr = [np.nan, np.nan]
                        red_chi2 = np.nan
                        success = False
                    else:
                        success = True
                        pfit = m.values
                        red_chi2 = m.fmin.reduced_chi2
                        perr = m.errors * np.sqrt(red_chi2)
                except RuntimeError:
                    pfit = [np.nan, np.nan]
                    perr = [np.nan, np.nan]
                    red_chi2 = np.nan
                    success = False
            else:
                success = False
                pfit = [np.nan, np.nan]
                perr = [np.nan, np.nan]
                red_chi2 = np.nan

            if dofit and not success and r < 2000 and "wcd" in key:
                print()
                print(f"FAILED at r < 1600 m for {key}", ct2_bin, DX_bin, r, cp)
                print()
                print(m)
                f, ax = plt.subplots(1)
                m = get_trace_fit(row, key, cdf_min=0.05, cdf_max=0.95, ax=ax)
                # plt.show()

            if success:
                print("Success", ct2_bin, DX_bin, r, cp, key)

            d_fit[key + "_mfit"] = pfit[0]
            d_fit[key + "_sfit"] = pfit[1]
            d_fit[key + "_merr"] = perr[0]
            d_fit[key + "_serr"] = perr[1]
            d_fit[key + "_redchi2"] = red_chi2
            d_fit[key + "_success"] = success

        for key in d_fit:
            dd[key].append(d_fit[key])

    df_fit = pd.DataFrame(dd)

    idx = df.index.names

    df_join = df.reset_index().join(df_fit)
    df_join = df_join.set_index(idx)
    outputfile = os.path.join(data_path, args.file[:-3] + "_fitted_lognormal.pl")
    print("Saved at", outputfile)
    df_join.to_pickle(outputfile)
    print("...done")
