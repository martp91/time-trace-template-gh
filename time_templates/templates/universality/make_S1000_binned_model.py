#!/usr/bin/env python
# coding: utf-8


from collections import defaultdict
import json
import numpy as np
from time_templates.datareader.get_data import fetch_MC_data_from_tree
from time_templates.utilities.plot import plot_hist, plot_profile_1d, add_identity
from time_templates.utilities.misc import histedges_equalN
from time_templates.utilities.fitting import plot_fit_curve, fit_curve

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from time_templates.templates.universality.names import (
    DICT_COMP_SIGNALKEY,
    eEM_HAD,
    eEM_MU,
    eEM_PURE,
    eMUON,
    DICT_COMP_LABELS,
    DICT_COMP_COLORS,
    DICT_COMP_MARKERS,
)
from time_templates.templates.universality.S1000_model import (
    S1000_func,
    S1000_comp_model,
    DX0_COMP,
    sec_theta_func,
    Rmu_func,
    lin_lgE,
    FILE_GH_COMP,
    FILE_SEC_THETA_CORR_COMP,
    FILE_RMU_CORR_COMP,
)
from time_templates import data_path


FIT_DX0 = False
SAVE = True
FORCE = False
BOOTSTRAPS = 0
if SAVE:
    plt.style.use("science")
    BOOTSTRAPS = 1000

gcm2 = "\\mathrm{g/cm^2}"
label_DX = f"$\\Delta X \, [{gcm2}]$"
label_VEM = "$\\rm Signal\, [\\rm VEM]$"

DICT_FRAC_LABELS = {
    comp: f"$S_{{{label}}}(1000) / \\widehat{{S}}_{{{label}}}(1000)$"
    for comp, label in DICT_COMP_LABELS.items()
}

lgE_bins = np.arange(18.5, 20.01, 0.1)
nE = lgE_bins
DXmax_bins = np.arange(0, 1200, 50)

dfs = []
for energy in ["18.5_19", "19_19.5", "19.5_20"]:
    df = fetch_MC_data_from_tree(
        primary="proton", energy=energy, no_traces=True, dense=True, force=FORCE,
    )
    dfs.append(df)
    df = fetch_MC_data_from_tree(
        primary="iron", energy=energy, no_traces=True, dense=True, force=FORCE,
    )
    dfs.append(df)

df = pd.concat(dfs)
df = df.query("MCr_round == 1000 & MCDXstation > 10 & MCCosTheta >= 0.5")

df["MClgE_bin"] = pd.cut(df["MClgE"], lgE_bins)
# Take DX as as 1000m at psi=+-np.pi/2=+-90 deg
df["MCDX_bin"] = pd.cut(df["MCDX_1000"], DXmax_bins)

df.to_pickle(data_path + "/df_shower_size.pl")

df_p = df.query("primary == 'proton'")
df_Fe = df.query("primary == 'iron'")


# Fit on proton only !
df_gb = df_p.groupby(["MClgE_bin"])


print()
print("First fitting S vs DX")

f, ax = plt.subplots(1)

dd = defaultdict(list)

for lgE_bin, df_ in df_gb:
    #     print(len(df_.groupby('EventId').count()))
    df_gb_ = df_.groupby("MCDX_bin")
    df_mean = df_gb_.mean()
    df_std = df_gb_.std()
    n = df_gb_.count()
    x = df_mean["MCDX_1000"].values

    for comp, signal_key in DICT_COMP_SIGNALKEY.items():
        y = df_mean[signal_key].values

        yerr = np.sqrt(
            df_std[signal_key].values ** 2 / n[signal_key].values
        )  # + 0.001*y)

        mask = np.isfinite(x * y * yerr) & (yerr > 0)
        if FIT_DX0:
            fitfunc = lambda DX, Sref, DXmax, lam, DX0: S1000_func(
                DX, Sref, DXmax, DX0, lam
            )
            p0 = [10 ** (1 + lgE_bin.mid - 19), 300, 200, DX0_COMP[comp]]
        else:
            fitfunc = lambda DX, Sref, DXmax, lam: S1000_func(
                DX, Sref, DXmax, DX0_COMP[comp], lam
            )
            p0 = [10 ** (1 + lgE_bin.mid - 19), 300, 200]

        comp_label = DICT_COMP_LABELS[comp]
        # Plotting an example
        if 19.5 in lgE_bin:
            ax, (fitp, fitp_err, chi2, ndof) = plot_fit_curve(
                x[mask],
                y[mask],
                yerr=yerr[mask],
                func=fitfunc,
                p0=p0,
                ax=ax,
                smoother_x=True,
                ebar_kws=dict(
                    color=DICT_COMP_COLORS[comp], ls="", marker=DICT_COMP_MARKERS[comp]
                ),
                param_names=[
                    f"S_{{{comp_label}}}^{{\\rm ref}}",
                    "\Delta X_{\\rm max}",
                    "\\lambda",
                ],
                units=["\\rm VEM", gcm2, gcm2],
                custom_label=f"${comp_label}$",
            )
        #             print(lgE_bin)
        else:
            fitp, fitp_err, chi2, ndof = fit_curve(
                x[mask], y[mask], yerr=yerr[mask], func=fitfunc, p0=p0
            )
        if FIT_DX0:
            Sref, DXmax, lam, DX0 = fitp
            Sref_err, DXmax_err, lam_err, DX0_err = fitp_err
            dd["DX0"].append(DX0)
            dd["DX0_err"].append(DX0_err)
        else:
            Sref, DXmax, lam = fitp
            Sref_err, DXmax_err, lam_err = fitp_err

        dd["lgE"].append(lgE_bin.mid)
        dd["comp"].append(comp)
        dd["Sref"].append(Sref)
        dd["DXmax"].append(DXmax)
        dd["lam"].append(lam)
        dd["Sref_err"].append(Sref_err)
        dd["DXmax_err"].append(DXmax_err)
        dd["lam_err"].append(lam_err)
        dd["chi2"].append(chi2)
        dd["ndof"].append(ndof)

ax.set_xlabel(label_DX)
ax.set_ylabel(label_VEM)
labels = [f"${lab}$" for lab in DICT_COMP_LABELS.values()]

handles = [
    Line2D([0], [0], color=DICT_COMP_COLORS[key], marker=DICT_COMP_MARKERS[key])
    for key in DICT_COMP_LABELS.keys()
]
ax.legend(handles, labels, loc=3)
ax.set_yscale("log")

if SAVE:
    plt.savefig("DXmax_vs_signal_example.pdf", bbox_inches="tight")

df_fit = pd.DataFrame(dd)

print()
print("Now fitting parameters as a function of lgE")

df_fit["lgSref"] = np.log10(df_fit["Sref"])
df_fit["lgSref_err"] = df_fit["Sref_err"] / df_fit["Sref"] * 1 / np.log(10)

dict_GH_params = {}
dict_param_keys = {
    "lgSref": lambda x: f"$\\log_{{10}}\\left(S_{x}^{{\\rm ref}} \\right)$",
    "DXmax": lambda x: "$\\Delta X_{\\rm max}$",
    "lam": lambda x: "$\\lambda$",
}

f, axes = plt.subplots(4, 3, figsize=(12, 12))

for irow, comp in enumerate(DICT_COMP_SIGNALKEY.keys()):
    df_ = df_fit.query(f'comp == "{comp}"')
    axes_row = axes[irow]
    x = df_["lgE"]
    dict_GH_params[comp] = {}
    for ax, (key, label) in zip(axes_row, dict_param_keys.items()):
        y = df_[key]
        yerr = df_[key + "_err"] * np.sqrt(df_["chi2"] / df_["ndof"])
        if key == "lgSref" or (key == "lam" and (comp == eEM_PURE or comp == eEM_HAD)):
            _, (fitp, fitp_err, chi2_1, ndof_1) = plot_fit_curve(
                x, y, yerr=yerr, func=lambda x, a, b: a + b * (x - 19), ax=ax
            )
            a, b = fitp
        else:
            _, (fitp, fitp_err, chi2_0, ndof_0) = plot_fit_curve(
                x, y, yerr=yerr, func=lambda x, a: a * np.ones_like(x), ax=ax
            )
            a = fitp[0]
            b = 0

        dict_GH_params[comp][key] = [a, b]

        ax.set_ylabel(label(DICT_COMP_LABELS[comp]))
        ax.set_xlabel("$\\log_{10}\\left(E/ \\rm eV \\right)$")

plt.subplots_adjust(wspace=0.4, hspace=0.4)
if SAVE:
    plt.savefig("S1000_params_lin_lgE.pdf", bbox_inches="tight")

print()
print(f"Saving at {FILE_GH_COMP}")

with open(FILE_GH_COMP, "w") as outfile:
    json.dump(dict_GH_params, outfile)


lgE = df["MClgE"].values
DXmax = df["MCDX_1000"].values
for key, val in dict_GH_params.items():
    if FIT_DX0:
        df[key + "_pred"] = S1000_func(
            DXmax,
            10 ** lin_lgE(lgE, *val["lgSref"]),
            lin_lgE(lgE, *val["DXmax"]),
            lin_lgE(lgE, *val["DX0"]),
            lin_lgE(lgE, *val["lam"]),
        )
    else:
        df[key + "_pred"] = S1000_func(
            DXmax,
            10 ** lin_lgE(lgE, *val["lgSref"]),
            lin_lgE(lgE, *val["DXmax"]),
            DX0_COMP[key],
            lin_lgE(lgE, *val["lam"]),
        )
    df[key + "_ratio"] = df[DICT_COMP_SIGNALKEY[key]] / df[key + "_pred"]
    df[key + "_diff"] = (
        2
        * (df[DICT_COMP_SIGNALKEY[key]] - df[key + "_pred"])
        / (df[DICT_COMP_SIGNALKEY[key]] + df[key + "_pred"])
    )

# Take all sims (also iron) and calc mean at 1000m
df_gb = df.groupby("EventId")
df = df_gb.mean()
df["primary"] = df_gb["primary"].min()

df_p = df.query("primary == 'proton'")
df_Fe = df.query("primary == 'iron'")

print()
print("Correcting for theta")

dict_sec_theta_corr = {}

f, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

bins = histedges_equalN(df_p["MCSecTheta"], 15)

for (comp, signal_key), ax in zip(DICT_COMP_SIGNALKEY.items(), axes):
    ax, (x, y, yerr) = plot_profile_1d(
        df_p["MCSecTheta"],
        df_p[comp + "_ratio"],
        ax=ax,
        bins=bins,
        color=DICT_COMP_COLORS[comp],
        stat="mean",
        bootstraps=BOOTSTRAPS,
    )
    ax, (fitp, _, _, _) = plot_fit_curve(
        x,
        y,
        yerr=yerr,
        func=sec_theta_func,
        ax=ax,
        ebar_kws=dict(alpha=0, color=DICT_COMP_COLORS[comp]),
        smoother_x=True,
    )
    dict_sec_theta_corr[comp] = list(fitp)
    #     ax.set_title(DICT_COMP_LABELS[comp])
    ax.set_xlabel("$\\sec{\\theta}$")
    ax.set_ylabel(DICT_FRAC_LABELS[comp])
    ax.legend()
    ax.set_ylim([0.8, 1.2])
    ax.axhline(1, ls="--", color="k")
    ax.axhspan(0.95, 1.05, color="k", alpha=0.1)
    ax.set_xlim([1, 2])
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    new_tick_locations = np.array([1, 1.15, 1.3, 1.55, 2])
    ax_top.set_xticks(new_tick_locations)
    new_tick_labels = np.rad2deg(np.arccos(1 / new_tick_locations))
    ax_top.set_xticklabels([f"${int(round(l, 0))}^\\circ$" for l in new_tick_labels])
    #     ax_top.set_xlabel('$\\theta$ [deg]')
    ax_top.tick_params(
        axis="x",
        which="minor",
        direction="in",
        top=False,
        labeltop=True,
        bottom=False,
        labelbottom=False,
    )
plt.subplots_adjust(hspace=0.35, wspace=0.3)
if SAVE:
    plt.savefig("signal_ratio_sec_theta_dependence.pdf", bbox_inches="tight")


with open(FILE_SEC_THETA_CORR_COMP, "w") as outfile:
    json.dump(dict_sec_theta_corr, outfile)


# Predict on all
lgE = df["MClgE"].values
DX = df["MCDX_1000"].values
theta = df["MCTheta"].values
df["total_pred"] = 0
for comp in DICT_COMP_LABELS:
    # Hack here but do not use Rmu yet anyway
    df[comp + "_pred"] = S1000_comp_model(
        lgE,
        DX,
        theta,
        1,
        comp=comp,
        gh_comp=dict_GH_params,
        sec_theta_corr_comp=dict_sec_theta_corr,
        rmu_corr_comp={eMUON: 1, eEM_MU: 1, eEM_HAD: 1, eEM_PURE: 0},
    )
    df[comp + "_ratio"] = df[DICT_COMP_SIGNALKEY[comp]] / df[comp + "_pred"]
    df["total_pred"] += df[comp + "_pred"]
df["total_ratio"] = df["WCDTotalSignal"] / df["total_pred"]
df["Rmu"] = df[eMUON + "_ratio"]


df_p = df.query("primary == 'proton'")
df_Fe = df.query("primary == 'iron'")


Rmu_mean_Fe = df_Fe["Rmu"].mean()
print("Iron Rmu mean", Rmu_mean_Fe)

print()
print("Correcting for Rmu")

dict_Rmu_corr = {}


# For fitting
bins = np.linspace(0.8, 1.5, 20)

f, axes = plt.subplots(1, 3, figsize=(16, 5))
axes = axes.flatten()

iax = 0
for comp in DICT_COMP_SIGNALKEY.keys():
    ax = axes[iax]
    if comp != eMUON:
        iax += 1

    ax.scatter(df_p["Rmu"], df_p[comp + "_ratio"], color="b", alpha=0.02, marker=".")
    ax.scatter(df_Fe["Rmu"], df_Fe[comp + "_ratio"], color="r", alpha=0.02, marker=".")
    ax, (x, y, yerr) = plot_profile_1d(
        df[eMUON + "_ratio"],
        df[comp + "_ratio"],
        ax=ax,
        bins=bins,
        marker="",
        alpha=0,
        stat="mean",
        bootstraps=BOOTSTRAPS,
    )
    mask = np.isfinite(x * y * yerr) & (yerr > 0)
    ax, (pfit, pfit_err, chi2, ndf) = plot_fit_curve(
        x[mask],
        y[mask],
        yerr=yerr[mask],
        func=Rmu_func,
        ax=ax,
        ebar_kws=dict(color="k", ls="", marker="", alpha=0),
    )
    dict_Rmu_corr[comp] = pfit[0]

    ax, (x, y, yerr) = plot_profile_1d(
        df_p["Rmu"],
        df_p[comp + "_ratio"],
        ax=ax,
        bins=histedges_equalN(df_p[eMUON + "_ratio"], 10),
        color="b",
    )
    ax, (x, y, yerr) = plot_profile_1d(
        df_Fe["Rmu"],
        df_Fe[comp + "_ratio"],
        ax=ax,
        bins=histedges_equalN(df_Fe[eMUON + "_ratio"], 10),
        color="r",
    )

    if comp == eEM_PURE:
        ax.axhline(1, ls=":", color="k")
    else:
        add_identity(ax, color="k", ls=":")
    ax.set_xlabel("$R^\\mu$")
    ax.set_ylabel(DICT_FRAC_LABELS[comp])
    ax.annotate(
        xy=(0.1, 0.95),
        text=f"$\\alpha_{{{DICT_COMP_LABELS[comp]}}} = {pfit[0]:.3f} \\pm {pfit_err[0]*np.sqrt(chi2/ndf):.3f}$",
        fontsize=18,
        xycoords="axes fraction",
        ha="left",
        va="top",
    )
    #     ax.legend()
    ax.set_ylim([0.6, 1.7])
    ax.set_xlim([0.6, 1.7])
    if comp == eMUON:
        ax.clear()

plt.subplots_adjust(wspace=0.4)
if SAVE:
    plt.savefig("signal_ratio_Rmu_dependence.pdf", bbox_inches="tight")


dict_Rmu_corr[eMUON] = 1

print()
print(f"Saving at {FILE_RMU_CORR_COMP}")

with open(FILE_RMU_CORR_COMP, "w") as outfile:
    json.dump(dict_Rmu_corr, outfile)

dict_Rmu_corr


lgE = df["MClgE"].values
DX = df["MCDX_1000"].values
theta = df["MCTheta"].values
df["Rmu"] = 1
df["Rmu"].loc[df["primary"] == "proton"] = 1
df["Rmu"].loc[df["primary"] == "iron"] = Rmu_mean_Fe  # ~1.33
df["total_pred"] = 0
Rmu = df["Rmu"].values
for comp in DICT_COMP_LABELS:
    df[comp + "_pred"] = S1000_comp_model(
        lgE,
        DX,
        theta,
        Rmu,
        comp=comp,
        gh_comp=dict_GH_params,
        sec_theta_corr_comp=dict_sec_theta_corr,
        rmu_corr_comp=dict_Rmu_corr,
    )
    df[comp + "_ratio"] = df[DICT_COMP_SIGNALKEY[comp]] / df[comp + "_pred"]
    df["total_pred"] += df[comp + "_pred"]
df["total_ratio"] = df["WCDTotalSignal"] / df["total_pred"]


df_p = df.query("primary == 'proton'")
df_Fe = df.query("primary == 'iron'")

print("Plotting more checks")


f, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
bins = np.linspace(0.5, 1.5, 50)
for (comp, signal_key), ax in zip(DICT_COMP_SIGNALKEY.items(), axes):
    plot_hist(
        df_p[comp + "_ratio"],
        ax=ax,
        bins=bins,
        histtype="step",
        color="b",
        density=True,
    )
    plot_hist(
        df_Fe[comp + "_ratio"],
        ax=ax,
        bins=bins,
        histtype="step",
        color="r",
        density=True,
    )
    #     ax.set_title(DICT_COMP_LABELS[comp])
    ax.legend(ncol=2)
    ax.axvline(1, ls="--", color="k")
    ax.set_ylabel(DICT_FRAC_LABELS[comp])
    ax.set_ylabel("probability density")

plt.subplots_adjust(hspace=0.3)
if SAVE:
    plt.savefig("signal_ratio_hist_Rmu_theta_corr.pdf", bbox_inches="tight")


bins = np.linspace(0.6, 1.4, 20)
df_p_ = df_p.query("MCCosTheta > 0.5 & MClgE > 19.4 & MClgE < 19.6")
df_Fe_ = df_Fe.query("MCCosTheta > 0.5 & MClgE > 19.4 & MClgE < 19.6")

f, ax = plt.subplots(1)  # , figsize=(8, 6))
plot_hist(
    df_p_["total_ratio"], color="b", bins=bins, histtype="step", fit_norm=False, ax=ax
)
plot_hist(
    df_Fe_["total_ratio"], color="r", bins=bins, histtype="step", fit_norm=False, ax=ax
)
ax.axvline(1, ls="--", color="k")

ax.legend(ncol=1)
ax.axvline(1, ls="--", color="k")
ax.set_xlabel(f"$S_{{\\rm total}}(1000) / \\widehat{{S}}_{{\\rm total}}(1000)$")
ax.set_ylabel("probability density")
# ax.set_yscale('log')
if SAVE:
    plt.savefig("total_signal_ratio_hist.pdf")


f, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
lgE_bins = np.arange(18.5, 20.01, 0.1)
for (comp, signal_key), ax in zip(DICT_COMP_SIGNALKEY.items(), axes):
    ax, (x, y, yerr) = plot_profile_1d(
        df_p["MClgE"],
        df_p[comp + "_ratio"],
        ax=ax,
        bins=lgE_bins,
        color="b",
        bootstraps=BOOTSTRAPS,
    )
    ax, (x, y, yerr) = plot_profile_1d(
        df_Fe["MClgE"],
        df_Fe[comp + "_ratio"],
        ax=ax,
        bins=lgE_bins,
        color="r",
        bootstraps=BOOTSTRAPS,
    )

    ax.set_ylim([0.8, 1.2])
    ax.axhspan(0.95, 1.05, color="k", alpha=0.1)
    ax.axhline(1, ls="--", color="k")
    ax.set_ylabel(DICT_FRAC_LABELS[comp])
    ax.set_xlabel("$\\log_{10}\\left(E/ \\rm eV \\right)$")
plt.subplots_adjust(hspace=0.3, wspace=0.25)
if SAVE:
    plt.savefig("signal_ratio_comps_profile_all_corrected_lgE.pdf", bbox_inches="tight")


f, ax = plt.subplots(1)

plot_profile_1d(
    df_p["MClgE"],
    df_p["total_ratio"],
    bins=lgE_bins,
    ax=ax,
    color="b",
    bootstraps=BOOTSTRAPS,
)
plot_profile_1d(
    df_Fe["MClgE"],
    df_Fe["total_ratio"],
    bins=lgE_bins,
    ax=ax,
    color="r",
    bootstraps=BOOTSTRAPS,
)
ax.set_ylim([0.8, 1.2])
ax.axhspan(0.95, 1.05, color="k", alpha=0.1)
ax.axhline(1, ls="--", color="k")
ax.set_ylabel(f"$S_{{\\rm total}}(1000) / \\widehat{{S}}_{{\\rm total}}(1000)$")
ax.set_xlabel("$\\log_{10}\\left(E/ \\rm eV \\right)$")
if SAVE:
    plt.savefig("total_signal_ratio_profile_all_corrected_lgE.pdf", bbox_inches="tight")


f, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

bins = histedges_equalN(df["MCSecTheta"], 15)
for (comp, signal_key), ax in zip(DICT_COMP_SIGNALKEY.items(), axes):
    ax, (x, y, yerr) = plot_profile_1d(
        df_p["MCSecTheta"],
        df_p[comp + "_ratio"],
        ax=ax,
        bins=bins,
        color="b",
        bootstraps=BOOTSTRAPS,
    )
    ax, (x, y, yerr) = plot_profile_1d(
        df_Fe["MCSecTheta"],
        df_Fe[comp + "_ratio"],
        ax=ax,
        bins=bins,
        color="r",
        bootstraps=BOOTSTRAPS,
    )

    ax.set_ylim([0.8, 1.2])
    ax.axhspan(0.95, 1.05, color="k", alpha=0.1)
    ax.axhline(1, ls="--", color="k")
    ax.set_ylabel(DICT_FRAC_LABELS[comp])
    ax.set_xlabel("$\\sec{\\theta}$")

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    new_tick_locations = np.array([1, 1.15, 1.3, 1.55, 2])
    ax_top.set_xticks(new_tick_locations)
    new_tick_labels = np.rad2deg(np.arccos(1 / new_tick_locations))
    ax_top.set_xticklabels([f"${int(round(l, 0))}^\\circ$" for l in new_tick_labels])
    ax_top.tick_params(
        axis="x",
        which="minor",
        direction="in",
        top=False,
        labeltop=True,
        bottom=False,
        labelbottom=False,
    )
plt.subplots_adjust(hspace=0.4, wspace=0.25)
if SAVE:
    plt.savefig(
        "signal_ratio_comp_profile_all_corrected_sec_theta.pdf", bbox_inches="tight"
    )


f, ax = plt.subplots(1)

plot_profile_1d(
    df_p["MCSecTheta"],
    df_p["total_ratio"],
    bins=bins,
    ax=ax,
    color="b",
    bootstraps=BOOTSTRAPS,
)
plot_profile_1d(
    df_Fe["MCSecTheta"],
    df_Fe["total_ratio"],
    bins=bins,
    ax=ax,
    color="r",
    bootstraps=BOOTSTRAPS,
)
ax.set_ylim([0.8, 1.2])
ax.axhspan(0.95, 1.05, color="k", alpha=0.1)
ax.axhline(1, ls="--", color="k")
ax.set_ylabel("$S_{{\\rm total}}(1000) / \\widehat{{S}}_{{\\rm total}}(1000)$")
ax.set_xlabel("$\\sec{\\theta}$")

ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
new_tick_locations = np.array([1, 1.15, 1.3, 1.55, 2])
ax_top.set_xticks(new_tick_locations)
new_tick_labels = np.rad2deg(np.arccos(1 / new_tick_locations))
ax_top.set_xticklabels([f"${int(round(l, 0))}^\\circ$" for l in new_tick_labels])
ax_top.tick_params(
    axis="x",
    which="minor",
    direction="in",
    top=False,
    labeltop=True,
    bottom=False,
    labelbottom=False,
)
ax.axvspan(1.667, 2, color="k", hatch="//", alpha=0.4)
ax.set_xlim([1, 2])
if SAVE:
    plt.savefig("total_signal_ratio_profile_all_corrected_sec_theta.pdf")

print("...done")
