import os
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from iminuit import Minuit
from iminuit.cost import LeastSquares, NormalConstraint

from time_templates.utilities.fitting import plot_fit_curve
from time_templates.templates.universality.lognormal_templates import (
    lognormal_signal,
    lognormal_pdf,
)
from time_templates.templates.universality.names import (
    DICT_COMP_COLORS,
    eMUON,
    eEM_HAD,
    eEM_MU,
    eEM_PURE,
    DICT_COMP_LABELS,
)
from time_templates.utilities.nb_funcs import interp

from time_templates.starttime.start_time_deMauro import start_time_variance

from time_templates import data_path


def make_mean(df):
    df_gb = df.groupby(["MCDXstation_bin", "MCr_round"])
    df_mean = df_gb.mean()
    for key in df.keys():
        if "trace" in key:
            sum_trace = df_gb[key].sum()
            sum_sq_trace = df_gb[key].apply(lambda x: np.sum(x ** 2, axis=0))
            n = df_gb[key].count()
            df_mean[key + "_mean"] = np.where(
                n > 1, sum_trace / np.maximum(n, 1), np.nan
            )
            df_mean[key + "_var"] = np.where(
                n > 1,
                sum_sq_trace / np.maximum(n, 1)
                - sum_trace ** 2 / np.maximum(n, 1) ** 2,
                np.nan,
            )
            df_mean["nstations"] = n
    return df_mean.dropna(subset="wcd_muon_trace_var")


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


def is_valid(m):
    if (
        m.valid
        and m.accurate
        and not m.fmin.has_parameters_at_limit
        and m.fmin.has_posdef_covar
    ):
        return True
    else:
        return False


PNAN = [np.nan, np.nan, np.nan, np.nan]


def fit_traces(df):
    print("running")
    dd = defaultdict(list)
    trace_keys = [key[:-5] for key in df.keys() if "trace_mean" in key]
    min_stations = 10
    for (DX_bin, r), row in df.iterrows():
        dofit = True
        if row["nstations"] < min_stations:
            dofit = False
        d_fit = {}
        for key in trace_keys:
            success = True
            if dofit:
                try:
                    m = get_trace_fit(row, key, cdf_min=0.0, cdf_max=0.95, fit_t0=True)
                    if not is_valid(m):
                        m.simplex()
                        m.migrad()
                    if not is_valid(m):
                        m = get_trace_fit(
                            row, key, cdf_min=0.0, cdf_max=0.95, fit_t0=False
                        )
                        m.simplex()
                        m.migrad()

                    if not is_valid(m):
                        pfit = PNAN
                        perr = PNAN
                        red_chi2 = np.nan
                        success = False
                    else:
                        success = True
                        pfit = m.values
                        red_chi2 = m.fmin.reduced_chi2
                        perr = m.errors
                except RuntimeError as e:
                    pfit = PNAN
                    perr = PNAN
                    red_chi2 = np.nan
                    success = False
                    print(e)
            else:
                success = False
                pfit = PNAN
                perr = PNAN
                red_chi2 = np.nan

            d_fit[key + "_mfit"] = pfit[0]
            d_fit[key + "_sfit"] = pfit[1]
            d_fit[key + "_merr"] = perr[0]
            d_fit[key + "_serr"] = perr[1]
            d_fit[key + "_t0"] = pfit[2]
            d_fit[key + "_t0err"] = perr[2]
            d_fit[key + "_redchi2"] = red_chi2
            d_fit[key + "_success"] = success

        for key in d_fit:
            dd[key].append(d_fit[key])

    df_fit = pd.DataFrame(dd)
    idx = df.index.names
    df_join = df.reset_index().join(df_fit)
    print("...done")
    return df_join.set_index(idx).swaplevel(0, 1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files", nargs="+", help="dataframe with traces in phase wrt t0"
    )
    parser.add_argument(
        "-outfile", help="outputfile", default="df_fitted_lognormals.pl"
    )

    args = parser.parse_args()
    print(args)

    dfs = []
    for fl in args.files:
        dfs.append(pd.read_pickle(fl))

    df = pd.concat(dfs)
    print("making mean traces")
    df = make_mean(df)
    print("Fitting")
    df = fit_traces(df)
    print(f"Saving at {args.outfile}")
    df.to_pickle(args.outfile)
