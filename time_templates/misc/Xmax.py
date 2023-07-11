"""
Copy from auger open data
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from time_templates.utilities.plot import plot_sys_brackets

file_path = os.path.dirname(os.path.realpath(__file__))

# data
# https://arxiv.org/pdf_Xmax_data/1409.4809.pdf_Xmax_data
# This is from ICRC2019

arr = np.genfromtxt(os.path.join(file_path, "AugerICRC2019_Xmax_Moments.txt"))
df_Xmax_data = pd.DataFrame(
    arr,
    columns=[
        "meanlgE",
        "nEvts",
        "mean",
        "meanErr",
        "meanSystUp",
        "meanSystLow",
        "sigma",
        "sigmaErr",
        "sigmaSystUp",
        "sigmaSystLow",
    ],
)


Xmax_data_interp = interp1d(
    df_Xmax_data["meanlgE"].values,
    df_Xmax_data["mean"].values,
    fill_value=(df_Xmax_data["mean"].iloc[0], df_Xmax_data["mean"].iloc[-1]),
    bounds_error=False,
)

# arr = np.genfromtxt(os.path.join(file_path, "AugerICRC2019_lnA_Moments.txt"))
# df_lnA_data = pd.DataFrame(
#    arr,
#    columns=[
#        "meanlgE",
#        "nEvts",
#        "mean",
#        "meanErr",
#        "meanSystLow",
#        "meanSystUp",
#        "var",
#        "varErr",
#        "varSystLow",
#        "varSystUp",
#    ],
# )


def mean_Xmax_data(E):
    """If outside bounds value is linearly extrapolated"""
    lgE = np.log10(E)
    return Xmax_data_interp(lgE)


# Numbers from GAP-2020-058
mean_params = {
    # units in g/cm2. X0, D, xi, delta
    "Sibyll2.3d": [815.87, 57.873, -0.3035, 0.7963],
    "EPOS-LHC": [806.04, 56.295, 0.3463, 1.0442],
    "EPOS_LHC": [806.04, 56.295, 0.3463, 1.0442],
    "EPOSLHC": [806.04, 56.295, 0.3463, 1.0442],
    "QGSJetII-04": [790.15, 54.191, -0.4170, 0.6927],
    "QGSJet-II.04": [790.15, 54.191, -0.4170, 0.6927],
    "QGSJET-II.04": [790.15, 54.191, -0.4170, 0.6927],
    "QGSII04": [790.15, 54.191, -0.4170, 0.6927],
}

sigma_sq_params = {
    "Sibyll2.3d": [3.727e3, -4.838e2, 1.325e2, -4.055e-1, -2.536e-4, 4.749e-2],
    "EPOS-LHC": [3.269e3, -3.053e2, 1.239e2, -4.605e-1, -4.751e-4, 5.838e-2],
    "EPOS_LHC": [3.269e3, -3.053e2, 1.239e2, -4.605e-1, -4.751e-4, 5.838e-2],
    "EPOSLHC": [3.269e3, -3.053e2, 1.239e2, -4.605e-1, -4.751e-4, 5.838e-2],
    "QGSJetII-04": [3.688e3, -4.279e2, 5.096e1, -3.939e-1, 1.314e-3, 4.512e-2],
    "QGSJet-II.04": [3.688e3, -4.279e2, 5.096e1, -3.939e-1, 1.314e-3, 4.512e-2],
    "QGSJET-II.04": [3.688e3, -4.279e2, 5.096e1, -3.939e-1, 1.314e-3, 4.512e-2],
    "QGSII04": [3.688e3, -4.279e2, 5.096e1, -3.939e-1, 1.314e-3, 4.512e-2],
}


# From  https://arxiv.org/abs/1301.6637 (Auger lnA paper)
def mean_Xmax_func(E, A, X0, D, xi, delta, E0=1e19):
    """mean_Xmax_func.

    Parameters
    ----------
    E : float
        primary energy in eV
    A : int
        Atomic mass number, proton=1, iron=52
    X0 : float
        mean of proton at E0
    D : float
        elongation rate
    xi :
        0 for superposition model
    delta :
        0 for superposition model
    """

    return (
        X0
        + D * np.log10(E / (E0 * A))
        + xi * np.log(A)
        + delta * np.log(A) * np.log10(E / E0)
    )


def sigma_sq_Xmax_func(E, A, p0, p1, p2, a0, a1, b, E0=1e19):
    sigma_p_sq = p0 + p1 * np.log10(E / E0) + p2 * np.log10(E / E0) ** 2
    a = a0 + a1 * np.log10(E / E0)
    return sigma_p_sq * (1 + a * np.log(A) + b * np.log(A) ** 2)


convert_primary_A = {"proton": 1, "helium": 4, "nitrogen": 14, "iron": 52}
primary_colors = {"proton": "b", "helium": "c", "nitrogen": "orange", "iron": "r"}
HIM_ls = {"EPOS-LHC": "-", "QGSJetII-04": "--", "Sibyll2.3d": ":"}


def mean_Xmax_MC(E, primary, HIM):
    A = convert_primary_A[primary]
    return mean_Xmax_func(E, A, *mean_params[HIM])


def sigma_sq_Xmax_MC(E, primary, HIM):
    A = convert_primary_A[primary]
    return sigma_sq_Xmax_func(E, A, *sigma_sq_params[HIM])


def sigma_Xmax_MC(E, primary, HIM):
    return np.sqrt(sigma_sq_Xmax_MC(E, primary, HIM))


def lnA_from_Xmax(Xmax, E, HIM):
    Xmax_p = mean_Xmax_MC(E, "proton", HIM)
    Xmax_Fe = mean_Xmax_MC(E, "iron", HIM)
    return (Xmax - Xmax_p) / (Xmax_Fe - Xmax_p) * np.log(55.854)


# from https://www.auger.unam.mx/AugerWiki/Hybrid_Xmax_acceptance_and_resolution
# Eric Mayotte parametrization


nbin, lgEmin, lgEmax, sigma1, sigma1err, sigma2, sigma2Err, fraction = np.genfromtxt(
    os.path.join(file_path, "resolution-v3r99.txt")
).T
sigma_total = np.sqrt(fraction * sigma1**2 + (1 - fraction) * sigma2**2)

lgE = (lgEmin + lgEmax) / 2
Xmax_resolution_interp = interp1d(
    lgE, sigma_total, fill_value=(sigma_total[0], sigma_total[-1])
)


def Fd_Xmax_resolution_at_lgE(lgE):
    return Xmax_resolution_interp(lgE)


def poly(x, p):
    "reverse arguments"
    return np.poly1d(p[::-1])(x)


# Don't understand this parametrization. DO NOT USE
p_mu = [-4.53037, -0.79847, 0.96660]
p_sigma = [11.51004, 15.40049, 1.53474, 0.52767]
p_sigmaRatio = [1.94336, 0.31835, 0]
p_frel = [0.80970, -0.07642, 0]


def Fd_Xmax_bias_at_E(E):
    lgE_18 = np.log10(E) - 18
    p = p_mu
    return p[0] + p[1] * lgE_18 + p[2] * lgE_18**2


p_X1 = [579.83060, 77.62593, -43.80631]
p_L1 = [121.31259, 230.65110, 65.48331]
p_X2 = [893.16899, 16.14445, -4.65494]
p_L2 = [112.82722, 62.08177, 5.16993]


def Fd_Xmax_X1(z18):
    return poly(z18, p_X1)


def Fd_Xmax_X2(z18):
    return poly(z18, p_X2)


def Fd_Xmax_L1(z18):
    return poly(z18, p_L1)


def Fd_Xmax_L2(z18):
    return poly(z18, p_L2)


def Fd_Xmax_acceptance(Xmax, E):
    z18 = np.log10(E) - 18.0
    X1 = poly(z18, p_X1)
    L1 = poly(z18, p_L1)
    X2 = poly(z18, p_X2)
    L2 = poly(z18, p_L2)
    return np.where(
        Xmax <= X1,
        np.exp((Xmax - X1) / L1),
        np.where(Xmax > X2, np.exp(-(Xmax - X2) / L2), 1),
    )


def get_FD_Xmax(lgEmin, lgEmax):
    df_ = df_Xmax_data.query(f"meanlgE > {lgEmin} & meanlgE < {lgEmax}")
    Xmax = df_["mean"]
    err = df_["meanErr"]
    syst_low = df_["meanSystLow"]
    syst_up = df_["meanSystUp"]
    return Xmax, err, syst_low, syst_up, df_["meanlgE"]


def plot_mean_Xmax_data(
    ax,
    lgEmin=17,
    lgEmax=20.2,
    interpolation=False,
    data_color="k",
    data_marker="s",
    data_kwargs=dict(),
    x_offset=0,
    y_offset=0,
    more_MC_lines=False,
    plot_sys=True,
):
    E = np.logspace(lgEmin, lgEmax)

    if more_MC_lines:
        prims = ["proton", "helium", "nitrogen", "iron"]
    else:
        prims = ["proton", "iron"]

    for HIM in ["EPOS-LHC", "QGSJetII-04", "Sibyll2.3d"]:
        for prim in prims:
            ax.plot(
                E,
                mean_Xmax_MC(E, prim, HIM),
                label=f"{HIM}, {prim}",
                color=primary_colors[prim],
                ls=HIM_ls[HIM],
            )

    ax.set_ylabel("$X_{\\rm max}\, [\\rm g/cm^2]$")
    ax.set_xlabel("$E\, [\\rm eV]$")

    ax.set_xscale("log")
    df_ = df_Xmax_data.query(f"meanlgE > {lgEmin} & meanlgE < {lgEmax}")

    ax.errorbar(
        10 ** (df_["meanlgE"] + x_offset),
        df_["mean"] + y_offset,
        yerr=df_["meanErr"],
        marker=data_marker,
        color=data_color,
        ls="",
        **data_kwargs,
    )
    if plot_sys:
        plot_sys_brackets(
            10 ** df_["meanlgE"].values + x_offset,
            df_["mean"].values + y_offset,
            df_["meanSystLow"].values + y_offset,
            df_["meanSystUp"].values + y_offset,
            ax=ax,
            color=data_color,
            size=14,
            **data_kwargs,
        )
    if interpolation:
        ax.plot(E, mean_Xmax_data(E), color=data_color, ls="-", label="auger data")


def get_FD_Xmax_sys_uncertainty(lgE):
    low = np.interp(lgE, df_Xmax_data["meanlgE"], df_Xmax_data["meanSystLow"])
    up = np.interp(lgE, df_Xmax_data["meanlgE"], df_Xmax_data["meanSystUp"])
    return low, up


def plot_sigma_Xmax_data(
    ax,
    lgEmin=17,
    lgEmax=20.2,
    interpolation=False,
    data_color="k",
    data_marker="s",
    data_kwargs=dict(),
    x_offset=0,
    y_offset=0,
    more_MC_lines=False,
    plot_sys=True,
):
    E = np.logspace(lgEmin, lgEmax)

    if more_MC_lines:
        prims = ["proton", "helium", "nitrogen", "iron"]
    else:
        prims = ["proton", "iron"]

    for HIM in ["EPOS-LHC", "QGSJetII-04", "Sibyll2.3d"]:
        for prim in prims:
            ax.plot(
                E,
                sigma_Xmax_MC(E, prim, HIM),
                label=f"{HIM}, {prim}",
                color=primary_colors[prim],
                ls=HIM_ls[HIM],
            )

    ax.set_xscale("log")
    df_ = df_Xmax_data.query(f"meanlgE > {lgEmin} & meanlgE < {lgEmax}")

    ax.errorbar(
        10 ** df_["meanlgE"] + x_offset,
        df_["sigma"] + y_offset,
        yerr=df_["sigmaErr"],
        marker=data_marker,
        color=data_color,
        ls="",
        **data_kwargs,
    )
    if plot_sys:
        plot_sys_brackets(
            10 ** df_["meanlgE"].values + x_offset,
            df_["sigma"].values + y_offset,
            df_["sigmaSystLow"].values + y_offset,
            df_["sigmaSystUp"].values + y_offset,
            ax=ax,
            color=data_color,
            size=14,
            **data_kwargs,
        )
    if interpolation:
        ax.plot(E, mean_Xmax_data(E), "k-", label="auger data")


def plot_lnA(Xmax, axes=None):
    if axes is None:
        f, axes = plt.subplots(1, 3, figsize=(6, 2.5), sharey=True)

    for HIM, ax in zip(["EPOS-LHC", "QGSJet-II.04", "Sibyll2.3d"], axes):

        ax.axhline(0, ls=":", color="k")
        ax.axhline(np.log(4), ls=":", color="k")
        ax.axhline(np.log(14), ls=":", color="k")
        ax.axhline(np.log(55.8), ls=":", color="k")
        ax.set_ylim([-0.3, 4.5])
        ax.set_xscale("log")
        ax.set_title(HIM)
        ax.set_xlabel("$E$ [eV]")

    axes[0].set_ylabel("$\\langle \\ln{A} \\rangle$")
    x = 1.7e20
    axes[2].annotate("Fe", xy=(x, np.log(55.8)), annotation_clip=False)
    axes[2].annotate("N", xy=(x, np.log(14)), annotation_clip=False)
    axes[2].annotate("He", xy=(x, np.log(4)), annotation_clip=False)
    axes[2].annotate("p", xy=(x, 0.0), annotation_clip=False)
    f.subplots_adjust(wspace=0.0)


if __name__ == "__main__":

    f, ax = plt.subplots(1)

    plot_mean_Xmax_data(ax)

    f, ax = plt.subplots(1)

    E = np.logspace(18, 20)

    ax.plot(E, Fd_Xmax_resolution_at_lgE(np.log10(E)))
    ax.set_ylabel("sigma Fd Xmax")
    ax.set_xlabel("E")
    ax.set_xscale("log")

    f, ax = plt.subplots(1)

    Xmax = np.linspace(400, 1400)

    for E in [1e19, 10**19.5, 1e20]:
        ax.plot(Xmax, Fd_Xmax_acceptance(Xmax, E), label=np.log10(E))

    ax.legend()

    plt.show()
