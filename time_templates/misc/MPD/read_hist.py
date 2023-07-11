import os
import argparse
from collections import defaultdict
import re
import numpy as np
import uproot
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.style.use("science")


def GH(X, Nmax, Xmax, _lambda, X0=-45, ground=2000):
    x = (X - X0) / _lambda
    m = (Xmax - X0) / _lambda
    out = Nmax * (x / m) ** m * np.exp(m - x)
    out[X > ground] = 0
    out[~np.isfinite(out)] = 0
    return out


def func_cubic(x, a, b, c, d):
    return a + c * (x - b) ** 2 + d * (x - b) ** 3


def USP(X, Nmax, Xmax, R, L, ground=2000):
    Xp = X - Xmax
    out = Nmax * (1 + R * Xp / L) ** (R**-2) * np.exp(-Xp / (L * R))
    out[X > ground] = 0
    out[~np.isfinite(out)] = 0
    return out


def fit_Xmax(x, y, yerr=None, ground=2000):
    """
    Fit MPD histogram, copied from offline essentially
    x = depth
    y = n particles
    """

    if yerr is None:
        yerr = np.ones_like(y)

    mask = np.isfinite(x) & np.isfinite(y) & (yerr > 0) & np.isfinite(yerr)

    Xmax = x[np.argmax(y)]

    p0 = [y.max(), Xmax, 0.42, 265.7]
    bounds = [(0, 200, 0.01, 100), (np.inf, 1200, 0.8, 400)]
    popt, pcov = curve_fit(
        lambda x, *args: USP(x, *args, ground),
        x[mask],
        y[mask],
        sigma=yerr[mask],
        absolute_sigma=True,
        p0=p0,
        bounds=bounds,
    )

    #    popt_GH, pcov_GH = curve_fit(lambda x, *args: GH(x, *args, ground),
    #                                 x[mask], y[mask],
    #                                 p0=[y.max(), Xmax, 100],
    #                                 bounds=[(0, 200, 10), (np.inf, 1300, 500)],
    #                                 sigma=yerr[mask], absolute_sigma=True)
    #    ndof = len(x[mask])
    #    chi2 = np.sum((y[mask] - GH(x[mask], *popt_GH, ground))**2/yerr[mask]**2)

    return popt, pcov


def split_key(key):
    "Get info from the tree name for file reading root, regex magic"
    lgE, ct, Xmax = re.findall(r"[-+]?\d*\.\d+|\d+", key)
    return float(lgE), float(ct), float(Xmax)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("file")

    args = parser.parse_args()

    uproot_fl = uproot.open(args.file)

    baseflname = os.path.splitext(os.path.basename(args.file))[0]

    dd = defaultdict(list)

    for histname in uproot_fl:
        histname = histname.split(";")[0]  # split ;1 or whatever
        if histname.endswith("_w2"):
            continue
        lgE, cosTheta, Xmax = split_key(histname)

        hist = uproot_fl[histname]
        # get separate weighted^2 histogram
        hist_w2 = uproot_fl[histname + "_w2"]
        MPD_hist, MPD_bins, lgr_bins, lgE_bins = hist.to_numpy()
        MPD_hist_w2 = hist_w2.values()
        MPD_hist = MPD_hist.sum(axis=-1)  # ignore lgE for now
        MPD_errs = np.sqrt(MPD_hist_w2.sum(axis=-1))
        n, m = MPD_hist.shape

        for i in range(m):
            dd["lgE"].append(lgE)
            dd["cosTheta"].append(cosTheta)
            dd["Xmax"].append(Xmax)
            dd["lgrmin"].append(lgr_bins[i])
            dd["lgrmax"].append(lgr_bins[i + 1])
            dd["hist"].append(MPD_hist[:, i])
            dd["hist_err"].append(MPD_errs[:, i])

    df = pd.DataFrame(dd)

    df_gb = df.groupby(["cosTheta", "lgrmin", "lgE"])
    df_gb_mean = df_gb.mean().reset_index()
    df_gb_mean["hist"] = [x for x in df_gb["hist"].apply(lambda x: np.sum(x))]
    df_gb_mean["hist_err"] = [
        x for x in df_gb["hist_err"].apply(lambda x: np.sqrt(np.sum(x**2)))
    ]

    df = df_gb_mean
    df["MPD_bins"] = [MPD_bins for _ in range(df.shape[0])]

    cosThetas = np.array(sorted(df["cosTheta"].unique()))
    lgrmins = np.array(sorted(df["lgrmin"].unique()))
    lgrmaxs = np.array(sorted(df["lgrmax"].unique()))
    x = (MPD_bins[1:] + MPD_bins[:-1]) / 2
    dx = MPD_bins[1] - MPD_bins[0]
    Xmumax_arr = np.zeros((len(cosThetas), len(lgrmins)))
    Xmumax_err_arr = np.zeros((len(cosThetas), len(lgrmins)))

    # f, axes = plt.subplots(len(cosThetas), 1, figsize=(10, 15), sharex=True)
    #
    #
    #
    # for i, (cosTheta, ax) in enumerate(zip(cosThetas, axes)):
    #
    #    ground = 880/cosTheta
    #
    #    for j, (lgrmin, lgrmax) in enumerate(zip(lgrmins, lgrmaxs)):
    #        df_ = df.query(f'lgrmin == {lgrmin} & lgE == 19 & cosTheta == {cosTheta}')
    #        hist = df_['hist'].iloc[0]
    #        yerr = df_['hist_err'].iloc[0]
    #        y = hist/(hist.sum()*dx)
    #        yerr = yerr/(hist.sum()*dx)
    #        popt, pcov = fit_Xmax(x, y, yerr, ground)
    #
    #        Xmumax_arr[i, j] = popt[1]
    #        Xmumax_err_arr[i, j] = np.sqrt(np.diag(pcov))[1]
    #
    #        if not j % 4:
    #            pl = ax.errorbar(x, y, yerr, errorevery=1, marker='.', ls='',
    #                    label=f'${lgrmin:.1f} < \\lg r/\\mathrm{{m}} < {lgrmax:.1f}$')
    #
    #            ax.plot(x, USP(x, *popt, ground), color=pl[0].get_color())
    ##            ax.plot(Xmax, 0, color=pl[0].get_color(), marker='o')
    #
    ##    ax.set_xlim([0, 800])
    #    ax.legend(fontsize=10)
    #    ax.set_xlim([0, 1800])
    #    ax.set_xlabel('$X\, [\\mathrm{g/cm^2}]$')
    #    ax.set_ylabel('$\\frac{dN}{dX}\, [\\mathrm{cm^2/g}]$')
    #
    # plt.tight_layout()
    # plt.savefig(baseflname+'_dNdX_cos_theta_lgr.pdf', bbox_inches='tight')
    #
    # lgr = (lgrmins + lgrmaxs)/2
    #
    # pfits = []
    # pfits_cov = []
    # f, ax = plt.subplots(1)
    #
    # for i, cosTheta in enumerate(cosThetas):
    #
    #    pl = ax.errorbar(lgr, Xmumax_arr[i], Xmumax_err_arr[i], marker='o', ls='',
    #                label=f'$\\theta = {np.rad2deg(np.arccos(cosTheta)):.0f}^\\circ$')
    #
    #    pfit, pcov = np.polyfit(lgr-3, Xmumax_arr[i], w=1/Xmumax_err_arr[i], deg=1, cov=True)
    #    ax.plot(lgr, np.poly1d(pfit)(lgr-3), color=pl[0].get_color())
    #    pfits.append(pfit)
    #    pfits_cov.append(pcov)
    #
    # pfits = np.array(pfits)
    # pfits_cov = np.array(pfits_cov)
    #
    # ax.legend()
    # ax.set_xlabel('lgr')
    # ax.set_ylabel('Xmumax')
    # plt.tight_layout()
    # plt.savefig(baseflname+'_Xmumax_vs_lgr.pdf', bbox_inches='tight')
    #
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    #
    # ax1.errorbar(cosThetas, pfits[:, 0], yerr=np.sqrt(pfits_cov[:, 0, 0]), marker='o', ls='')
    # ax2.errorbar(cosThetas, pfits[:, 1], yerr=np.sqrt(pfits_cov[:, 1, 1]), marker='o', ls='')
    #
    # ax1.set_xlabel('cos theta')
    # ax1.set_ylabel('slope')
    # ax2.set_ylabel('offset')
    # ax2.set_xlabel('cos theta')
    # ax1.set_ylim([-350, 0])
    # ax2.set_ylim([500, 700])
    #
    # plt.savefig(baseflname+'_fitp_costheta.pdf', bbox_inches='tight')

    f, ax = plt.subplots(1)

    cosTheta = 1
    i = -1

    ground = 880 / cosTheta

    for j, (lgrmin, lgrmax) in enumerate(zip(lgrmins, lgrmaxs)):
        df_ = df.query(f"lgrmin == {lgrmin} & lgE == 19 & cosTheta == {cosTheta}")
        hist = df_["hist"].iloc[0]
        yerr = df_["hist_err"].iloc[0]
        y = hist / (hist.sum() * dx)
        yerr = yerr / (hist.sum() * dx)
        popt, pcov = fit_Xmax(x, y, yerr, ground)

        Xmumax_arr[i, j] = popt[1]
        Xmumax_err_arr[i, j] = np.sqrt(np.diag(pcov))[1]

        if not j % 4 and j != 0:
            pl = ax.errorbar(
                x,
                y,
                yerr,
                errorevery=1,
                marker=".",
                ls="",
                label=f"${lgrmin:.2f} < \\lg r/\\mathrm{{m}} < {lgrmax:.2f}$",
            )

            ax.plot(x, USP(x, *popt, ground), color=pl[0].get_color())
    #            ax.plot(Xmax, 0, color=pl[0].get_color(), marker='o')

    #    ax.set_xlim([0, 800])
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1000])
    ax.set_xlabel("$X\, [\\mathrm{g/cm^2}]$")
    ax.set_ylabel("$\\frac{1}{N}\\frac{dN}{dX}\, [\\mathrm{cm^2/g}]$")

    plt.tight_layout()
    plt.savefig(baseflname + "_dNdX_vertical_lgr.pdf", bbox_inches="tight")
    plt.show()
