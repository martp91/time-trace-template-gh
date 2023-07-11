import os
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares
from time_templates.templates.muon_dsdt import MuondSdt
from time_templates.utilities import strings
from time_templates.utilities.fitting import fit_xy_xerr_yerr, plot_nice_fit
from time_templates.utilities.nb_funcs import interp

# Functions to to Xmumax, lam fit and parametrize Xmumax
# as from above

# fit Xmumax, rough
P_A = [512.2550407, 264.07819244, -82.87266986]
P_B = [-0.24140282, -0.07757393, -0.04418292]


def f_Xmumax_r_psi_theta(r, psi, theta):
    a0, a1, a2 = P_A
    b0, b1, b2 = P_B
    cos_psi = np.cos(psi)
    sintheta2 = np.sin(theta) ** 2
    a = a0 + sintheta2 * (a1 + a2 * cos_psi)
    b = b0 + sintheta2 * (b1 + b2 * cos_psi)

    return a * (r / 1000) ** b


def muon_func(t, Xmumax, lam, muon_dsdt, det="wcd", fix_lambda=False):
    muon_dsdt.set_Xmumax_lambda(Xmumax)
    if not fix_lambda:
        muon_dsdt.lam = lam
    if det.lower() == "ssd":
        out = muon_dsdt.ds_dt_ssd(t)
    else:
        out = muon_dsdt.ds_dt_wcd(t)

    if not np.all(np.isfinite(out)):
        return np.zeros_like(out)

    return out  # /out.sum() / dt


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


def fit_Xmumax_lam(
    t, y, yerr, muon_dsdt, p0=[550, 50], det="wcd", fix_Xmumax=False, fix_lambda=False
):
    # only take bins with max/100
    # cutoff = y.max() / 50

    # mask = y > cutoff
    ## For the model the t should be in smooth order (with distance always
    ## 25 or 25/3 ns) or the convolution does not work BUG/FEATURE
    ## so that the trace where the cutoff is first and last valid
    # imin = np.argmax(mask)
    # imax = len(mask) - np.argmax(mask[::-1])
    # if imax * 25 / 3 > muon_dsdt.r * 2:
    #    imax = int((muon_dsdt.r * 2) / (25 / 3))
    # dt = t[1] - t[0]
    # _t = t[imin:imax]
    # _y = y[imin:imax]
    # _yerr = yerr[imin:imax]
    cdf_min = 0
    cdf_max = 0.95

    cs = np.cumsum(y)
    cs /= cs[-1]
    t01 = interp(cdf_min, cs, t)
    t99 = interp(cdf_max, cs, t)

    dt = t[1] - t[0]

    if cdf_min == 0:
        imin = 0
    else:
        imin = int(t01 / dt)
    imax = int(t99 / dt)

    _t = t[imin:imax]
    _y = y[imin:imax]
    _yerr = yerr[imin:imax]
    _yerr[_yerr <= 1e-9] = _yerr.max()
    if len(_y) < 10:
        raise RuntimeError
    # make sure _y is still in ds/dt 1/s, model returns the same
    N = _y.sum() * dt
    _y /= N
    _yerr /= N
    lq = LeastSquares(
        _t,
        _y,
        _yerr,
        lambda t, Xmumax, lam: muon_func(t, Xmumax, lam, muon_dsdt, det, fix_lambda),
    )
    m = Minuit(lq, Xmumax=p0[0], lam=p0[1])
    bounds = [(50, 2000), (5, 400)]
    m.limits = bounds

    m.fixed["Xmumax"] = fix_Xmumax
    m.fixed["lam"] = fix_lambda
    m.migrad()
    if not is_valid(m):
        m.simplex()
        m.migrad()

    if not is_valid(m):
        _yerr = np.maximum(_yerr, _y / 100)
        lq = LeastSquares(
            _t,
            _y,
            _yerr,
            lambda t, Xmumax, lam: muon_func(
                t, Xmumax, lam, muon_dsdt, det, fix_lambda=fix_lambda
            ),
        )
        m = Minuit(lq, Xmumax=p0[0], lam=p0[1])
        bounds = [(50, 2000), (5, 400)]
        m.limits = bounds
        m.fixed["lam"] = fix_lambda
        if not fix_lambda:
            m.fixed["lam"] = True
            m.migrad()
            m.fixed["lam"] = False
        m.simplex()
        m.migrad()

    if not is_valid(m):
        print(m)
        plt.errorbar(_t, _y, _yerr, ls="")
        plt.show()

    return m, imin, imax


def fit_muon_traces_Xmumax_lam(
    df, fix_Xmumax=False, fix_lambda=False, det="wcd", PLOT=True, gamma=2.6
):
    df = df.query("mean_total_wcd_signal > 10")
    cos_theta2_mids = df["costheta2"].apply(lambda x: x.mid).unique()
    cos_theta2_bins = df["costheta2"].unique()
    # selected rs
    rs = np.array([500, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500])
    cos_psis = np.sort(df["cos_psi"].apply(lambda x: x.mid).unique())

    nt = df.iloc[0]["nt"]
    dt = df.iloc[0]["dt"]

    # make sure dt=dt and nt=nt
    t = np.arange(0, nt * dt, dt)
    if len(t) < nt:
        t = np.append(t, t[-1] + dt)
    elif len(t) > nt:
        t = t[:-1]

    #    t = np.linspace(0, nt * dt, nt) # this does not give dt correctly

    # trace_keys = df.keys()[df.keys().str.contains('trace_mean')].str[:-5]
    # trace_keys = ['wcd_muon_trace']  # , 'ssd_muon_trace']
    key = "wcd_muon_trace"
    _df = df.set_index(["costheta2", "r", "cos_psi"])
    dd = defaultdict(list)

    param_names = [strings.XMUMAX, "\\lambda"]
    units = [strings.GCM2, strings.GCM2]

    p0 = [550, 50]

    for i, cos_theta2 in enumerate(cos_theta2_mids):
        cos_theta = np.sqrt(cos_theta2)
        theta = np.arccos(cos_theta)
        print("Working on costheta2", cos_theta2)
        if PLOT:
            f, axes = plt.subplots(len(rs), 2, figsize=(16, 24))
        for ir, r in enumerate(rs):
            for cp in cos_psis[::-1]:
                try:
                    df_ct_r_cp = _df.loc[cos_theta2, r, cp]
                except KeyError:
                    print(f"no stations for {cos_theta2}, {r}, {cp}")
                    continue

                n = df_ct_r_cp["n"]
                psi = df_ct_r_cp["psi_rad"]
                psi_deg = df_ct_r_cp["psi"]
                n = df_ct_r_cp[key + "_n"]
                if n < 5:
                    continue

                mean_trace = df_ct_r_cp[key + "_mean"]
                err_trace = df_ct_r_cp[key + "_std"] / np.sqrt(n)
                Xmumax0 = f_Xmumax_r_psi_theta(r, psi, theta)
                muon_dsdt = MuondSdt(r, psi, theta)
                muon_dsdt.set_Xmumax_lambda(Xmumax0)

                p0 = [muon_dsdt.Xmumax, muon_dsdt.lam]

                muon_dsdt.set_gamma(gamma)
                # Kinematic delay pushse Xmumax -> 0 (200-300 g/cm2)
                # muon_dsdt.use_kinematic_delay = 5
                #
                # imin, imax the fit boundaries
                m, imin, imax = fit_Xmumax_lam(
                    t,
                    mean_trace,
                    err_trace,
                    muon_dsdt,
                    det=det,
                    p0=p0,
                    fix_Xmumax=fix_Xmumax,
                    fix_lambda=fix_lambda,
                )

                if not m.accurate or not m.valid or m.fmin.has_parameters_at_limit:
                    p = p0
                    pcov = np.array([[1000, 0], [0, 1000]], dtype=float)
                    red_chi2 = 1e20
                    print("WARNING: not accurate at:")
                    print(cos_theta2, r, cp)
                    print(m)
                #   print(mean_trace)
                #   print(err_trace)
                #   fig, axx = plt.subplots(1)
                #   plot_nice_fit(
                #       t,
                #       mean_trace,
                #       lambda t, Xmumax, lam: muon_func(t, Xmumax, lam, muon_dsdt),
                #       m.values,
                #       m.errors,
                #       yerr=err_trace,
                #       chi2=m.fmin.fval,
                #       ndof=m.ndof,
                #       ax=axx,
                #       param_names=param_names,
                #       units=units,
                #       ebar_kws=dict(ls="", errorevery=3),
                #       smoother_x=False,
                #       custom_label=f"$\\psi = {psi_deg:.0f}^\\circ$",
                #   )
                #   plt.show()
                else:
                    p = np.array(m.values)
                    pcov = np.array(m.covariance)
                    red_chi2 = m.fmin.reduced_chi2

                pcov *= red_chi2
                Xmumax, lam = p
                perr = np.sqrt(np.diag(pcov))

                dd["costheta2"].append(df_ct_r_cp.name[0])
                dd["r"].append(r)
                dd["cp"].append(np.cos(psi))
                dd["trace_key"].append(key)
                dd["Xmumax"].append(Xmumax)
                dd["lam"].append(lam)
                dd["cov"].append(np.array(pcov))
                dd["red_chi2"].append(red_chi2)
                dd[key + "_mean_trace"].append(mean_trace)
                dd[key + "_err_trace"].append(err_trace)

                if PLOT and (cp > 0.9 or cp < -0.9):

                    iax = ir
                    try:
                        ax1 = axes[iax, 0]
                        ax2 = axes[iax, 1]
                    except:
                        continue

                    plot_nice_fit(
                        t,
                        mean_trace,
                        lambda t, Xmumax, lam: muon_func(t, Xmumax, lam, muon_dsdt),
                        p,
                        perr,
                        yerr=err_trace,
                        chi2=m.fmin.fval,
                        ndof=m.ndof,
                        ax=ax1,
                        param_names=param_names,
                        units=units,
                        ebar_kws=dict(ls="", errorevery=3),
                        smoother_x=False,
                        custom_label=f"$\\psi = {psi_deg:.0f}^\\circ$",
                    )
                    plot_nice_fit(
                        t,
                        mean_trace,
                        lambda t, Xmumax, lam: muon_func(t, Xmumax, lam, muon_dsdt),
                        p,
                        perr,
                        yerr=err_trace,
                        chi2=m.fmin.fval,
                        ndof=m.ndof,
                        ax=ax2,
                        param_names=param_names,
                        units=units,
                        ebar_kws=dict(ls="", errorevery=3),
                        smoother_x=False,
                    )
                    ax1.set_xlim([imin * 25 / 3, imax * 25 / 3])
                    ax2.set_xlim([0, 600 * 25 / 3])
                    ax2.set_yscale("log")
                    ax2.set_ylim([1e-7, 0.01])
                    ax1.set_title(f"r = {r}")
                    ax1.legend()

        if PLOT:
            for ax in axes.flatten():
                ax.set_ylabel(
                    "$\\frac{1}{S^\\mu}\\frac{dS^\\mu}{dt} \ [\\mathrm{ns^{-1}}]$"
                )
                ax.set_xlabel("$t - t_{\\mathrm{pf}}\ [\\mathrm{ns}]$")
            ct2min = cos_theta2_bins[i].left
            ct2max = cos_theta2_bins[i].right
            f.suptitle(f"${ct2min:.2f} < \\cos^2{{\\theta}} < {ct2max:.2f}$")
            f.subplots_adjust(hspace=0.5, top=0.95)

    df_fits = pd.DataFrame(dd)

    # Multiply by red chi2 gives OK results, not yet for the r fit
    # but later on OK
    # df_fits['cov'] = df_fits['cov'] * df_fits['red_chi2']

    df_fits["Xmumax_err"] = df_fits["cov"].apply(lambda x: np.sqrt(x[0, 0]))
    df_fits["lam_err"] = df_fits["cov"].apply(lambda x: np.sqrt(x[1, 1]))
    #     print('...done')
    return df_fits


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--pretty_plot", action="store_true")
    parser.add_argument("--fix_Xmumax", action="store_true")
    parser.add_argument("--fix_lambda", action="store_true")

    args = parser.parse_args()
    if args.pretty_plot:
        plt.style.use("science")  # uncomment for nice plots, but less speed

    print("input file:", args.file)
    print("---------------------")
    print(
        "WARNING",
        ": These traces should be the mean over bins of cos^2 theta and r and psi",
    )
    print(
        "WARNING: Make sure the traces are already normalized(as ds/dt 1/s)"
        "otherwise the mean still makes sense but the std will not"
    )
    print("don't worry you always get these 'warnings'")
    print("---------------------")

    df = pd.read_pickle(args.file)
    df["costheta"] = df["costheta2"].apply(lambda x: np.sqrt(x.mid))
    df["psi_rad"] = np.deg2rad(df["psi"])
    df_fits = fit_muon_traces_Xmumax_lam(
        df,
        fix_Xmumax=args.fix_Xmumax,
        fix_lambda=args.fix_lambda,
        det="wcd",
        PLOT=args.plot,
    )

    HIM, primary, lgE_min, lgE_max, basename = strings.get_info_from_file_str(args.file)
    outdir = "../data/"
    outfile = f"Xmumax_lam_fitted_{HIM}_{primary}_lgE_{lgE_min}-{lgE_max}.pl"
    if args.fix_Xmumax:
        outfile = (
            f"Xmumax_lam_fitted_{HIM}_{primary}_lgE_{lgE_min}-{lgE_max}_fixed_Xmumax.pl"
        )
    if args.fix_lambda:
        outfile = (
            f"Xmumax_lam_fitted_{HIM}_{primary}_lgE_{lgE_min}-{lgE_max}_fixed_lambda.pl"
        )

    print(f"saving at {outdir}{outfile}")
    df_fits.to_pickle(os.path.join(outdir, outfile))

    plt.show()
