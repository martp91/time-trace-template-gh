"""
Fit MPD histogram for every r (integrated over Ekin)
And then get Xmumax(r=1000)
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd
import hist
import uproot
from scipy.optimize import curve_fit
from numba import njit

from time_templates.utilities.fitting import plot_fit_curve


@njit
def cubic_interp_func(x, a, b, c, d):
    return a + c * (x - b) ** 2 + d * (x - b) ** 3


def fit_Xmumax(hX, refit=True, ax=None):
    """
    Fit Xmumax on histogram, interpolating function same as MuonProfileBuilder
    refit is recommended
    """
    X = hX.axes[0].centers
    y = hX.values()
    #     Xmax = X[np.argmax(y)]
    hX_rebinned = hX[::5j]
    Xmax = hX_rebinned.axes[0].centers[np.argmax(hX_rebinned.values())]

    mask = (X > Xmax - 300) & (X < Xmax + 300) & (y > 5) & np.isfinite(y)
    bounds = [(0, 50, -np.inf, -1), (np.inf, 1700, np.inf, 1)]
    try:
        popt, pcov = curve_fit(
            cubic_interp_func,
            X[mask],
            y[mask],
            p0=[y[mask].max(), Xmax, -1, 0],
            sigma=np.sqrt(y[mask]),
            bounds=bounds,
        )
    except (RuntimeError, ValueError):
        try:
            popt, pcov = curve_fit(
                cubic_interp_func, X[mask], y[mask], sigma=np.sqrt(y[mask])
            )
        except (RuntimeError, ValueError):
            print("FIT FAILED")
            return 0, 0

    Xmumax = popt[1]
    Xmumax_err = np.sqrt(pcov[1, 1])
    if refit:
        if Xmumax_err < 100:
            mask = (X > Xmumax - 200) & (X < Xmumax + 200) & (y > 5)
            try:
                popt, pcov = curve_fit(
                    cubic_interp_func,
                    X[mask],
                    y[mask],
                    p0=popt,
                    sigma=np.sqrt(y[mask]),
                    bounds=bounds,
                )
                Xmumax = popt[1]
                Xmumax_err = np.sqrt(pcov[1, 1])
            except (ValueError, RuntimeError):
                print("Refit FAILED")
                pass

    if ax is not None:
        hX.plot(ax=ax, yerr=False)
        ax.plot(
            X[mask],
            cubic_interp_func(X[mask], *popt),
            "r--",
            label=f"$X^\\mu_{{\\rm max}} = {Xmumax:.0f} \\pm {Xmumax_err:.0f} \ \\rm g/cm^2$",
        )
        #         ax.set_xlim([500, 800])
        #         ax.set_ylim([2e5, None])
        ax.legend()
        ax.set_xlabel("X [g/cm2]")
        ax.set_ylabel("#")
    return Xmumax, Xmumax_err


def Xmumax_1000_from_hist(h3d, plot=False):
    if plot:
        f, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
    h2d_rebinned = h3d[::2j, ::2j, sum]
    r_bins = h2d_rebinned.axes[1]
    Xmumaxs = np.zeros(r_bins.size)
    Xmumax_errs = np.zeros(r_bins.size)
    for i in range(r_bins.size):
        try:
            ax = axes[i]
        except:
            ax = None
        rmin, rmax = r_bins[i]
        hX = h2d_rebinned[:, i]
        try:
            Xmumaxs[i], Xmumax_errs[i] = fit_Xmumax(hX, ax=ax)
        except (RuntimeError, ValueError):
            Xmumaxs[i], Xmumax_errs[i] = (0, 0)
        if plot:
            ax.set_xlim([100, 1500])
            ax.set_title(f"${rmin:.2f} < \\lg{{r/ \\rm m}} < {rmax:.2f}$")
    if plot:
        plt.tight_layout()
        f, ax = plt.subplots(1)
    else:
        ax = None

    mask = (Xmumaxs > 0) & (Xmumax_errs > 0)
    success = True
    try:
        ax, (fitp, fitp_err, chi2, ndof) = plot_fit_curve(
            r_bins.centers[mask],
            Xmumaxs[mask],
            lambda x, a, b: a + b * (x - 3),
            yerr=Xmumax_errs[mask],
            ax=ax,
        )
    except:
        print("FAILED")
        fitp = [0, 0]
        fitp_err = np.array([0, 0])
        chi2 = 0
        ndof = 0
        success = False
    fitp_err *= np.sqrt(chi2 / ndof)
    if plot:
        ax.set_xlabel("lgr")
        ax.set_ylabel("Xmumax [g/cm2]")
        ax.legend()

    Xmumax1000, dXmumax_dr = fitp
    Xmumax1000_err, dXmumax_dr_err = fitp_err
    return Xmumax1000, dXmumax_dr, Xmumax1000_err, dXmumax_dr_err, success


def Xmumax_from_hist_from_rootfile(rootfilename, plot=False):
    ids = []
    Xmumax_1000_ = []
    Xmumax_1000_err_ = []
    dXmumax_dr_ = []
    dXmumax_dr_err_ = []

    simshower_ids = []
    Xmumax_ = []
    fit_success_ = []
    Xmumax_1700_ = []
    Xmumax_1700_err_ = []
    try:
        rootfl = uproot.open(rootfilename)
    except:
        print(f"Could not read {rootfilename}")
        return (
            simshower_ids,
            Xmumax_1000_,
            Xmumax_1000_err_,
            dXmumax_dr_,
            dXmumax_dr_err_,
            Xmumax_,
            fit_success_,
        )

    for key in rootfl.keys():
        print(key)
        simshower_ids.append(int(key.split("_")[2]))

        h3d = rootfl[key].to_hist()
        Xmumax, Xmumax_err = fit_Xmumax(h3d[:, sum, sum])
        Xmumax_.append(Xmumax)
        (
            Xmumax_1000,
            dXmumax_dr,
            Xmumax_1000_err,
            dXmumax_dr_err,
            fit_success,
        ) = Xmumax_1000_from_hist(h3d, plot=plot)

        idx1700 = h3d.axes[1].index(np.log10(1700))
        idx4000 = h3d.axes[1].index(np.log10(4000))
        Xmumax1700, Xmumax1700_err = fit_Xmumax(h3d[:, idx1700:idx4000:sum, sum])
        Xmumax_1700_.append(Xmumax1700)
        Xmumax_1700_err_.append(Xmumax1700_err)
        Xmumax_1000_.append(Xmumax_1000)
        Xmumax_1000_err_.append(Xmumax_1000_err)
        dXmumax_dr_.append(dXmumax_dr)
        dXmumax_dr_err_.append(dXmumax_dr_err)
        fit_success_.append(fit_success)

    return (
        simshower_ids,
        Xmumax_1700_,
        Xmumax_1700_err_,
        Xmumax_1000_,
        Xmumax_1000_err_,
        dXmumax_dr_,
        dXmumax_dr_err_,
        Xmumax_,
        fit_success_,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("files", nargs="+")
    parser.add_argument("-output_file", "-o", default="df_MPD.pl")

    args = parser.parse_args()

    dd = defaultdict(list)

    i = 0
    n = len(args.files)

    for fl in args.files:
        print(f"{i}/{n}")
        i += 1
        print(fl)
        results = Xmumax_from_hist_from_rootfile(fl)
        for (
            shower_id,
            Xmumax_1700,
            Xmumax_1700_err,
            Xmumax_1000,
            Xmumax_1000_err,
            dXmumax_dr,
            dXmumax_dr_err,
            Xmumax,
            fit_success,
        ) in zip(*results):
            dd["shower_id"].append(shower_id)
            dd["Xmumax_1700"].append(Xmumax_1700)
            dd["Xmumax_1700_err"].append(Xmumax_1700_err)
            dd["Xmumax_1000"].append(Xmumax_1000)
            dd["Xmumax_1000_err"].append(Xmumax_1000_err)
            dd["dXmumax_dr"].append(dXmumax_dr)
            dd["dXmumax_dr_err"].append(dXmumax_dr_err)
            dd["Xmumax"].append(Xmumax)
            dd["Xmumax_fit_success"].append(fit_success)

    df = pd.DataFrame(dd)

    print(f"Saving at {args.output_file}")
    df.to_pickle(args.output_file)
