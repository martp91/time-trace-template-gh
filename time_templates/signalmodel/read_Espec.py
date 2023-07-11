import matplotlib.pyplot as plt
import numpy as np
from time_templates.utilities.plot import plot_profile_1d
import uproot
import pandas as pd
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy import integrate as scint
from re import findall

plt.rcParams.update({"text.usetex": True, "font.size": 20, "font.family": "sans"})
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
markers = ["s", "o", "^", "v", "x", "*"]
lss = ["-", "--", ":", "-."]


def split_key(key):
    "Get info from the tree name for file reading root, regex magic"

    try:
        lgE, ct, Xmax, _ = findall(r"[-+]?\d*\.\d+|\d+", key)
    except ValueError:
        print(key)
    if "electron" in key:
        ptype = "electron"
    elif "photon" in key:
        ptype = "photon"
    else:
        ptype = "muon"
    return float(lgE), float(ct), float(Xmax), ptype


def integrate_spec(f, low=4, high=14, step=None, args=()):
    if step is None:
        step = (high - low) / 10000
    x = np.arange(low, high, step)
    return f(x, *args).sum() * step


# return scint.quad(f, low, high, args=args)[0]


def load_spec(fl, dd=None, Ech=40e6, Evem=400e6):

    if Ech == 0:
        lgEch = -1
    else:
        lgEch = np.log10(Ech)
    lgEvem = np.log10(Evem)
    if dd is None:
        dd = defaultdict(list)

    try:
        hist = uproot.open(fl)
    except OSError:
        return

    for key in hist.iterkeys(
        filter_name="muon*"
    ):  # just getting the key name (no loop actually)
        lgE, cosTheta, Xmax, ptype = split_key(key)
        muon_spec, lgr_bins, psi_bins, lgE_pkin_bins = hist[key].to_numpy(flow=True)
        lgr_bins[0] = 2
        lgr_bins[-1] = 4
        lgE_pkin_bins[0] = np.log10(0.05e6)  # from ECUTS corsika Aab
        lgE_pkin_bins[-1] = 14
        lgEmean = (lgE_pkin_bins[1:] + lgE_pkin_bins[:-1]) / 2
        Emean = 10 ** lgEmean
        lgrmean = (lgr_bins[1:] + lgr_bins[:-1]) / 2

        dlgE = lgE_pkin_bins[1:] - lgE_pkin_bins[:-1]
        electron_key = "electron_" + "_".join(key.split("_")[1:])
        electron_spec = hist[electron_key].values(flow=True)
        photon_key = "photon_" + "_".join(key.split("_")[1:])
        photon_spec = hist[photon_key].values(flow=True)

        # ignoring psi dependence
        mu_spec = muon_spec.sum(axis=1)  # /dlgE
        el_spec = electron_spec.sum(axis=1)  # /dlgE
        ph_spec = photon_spec.sum(axis=1)  # /dlgE

        for i in range(1, len(lgrmean) - 1):
            lgr = lgrmean[i]
            lgrl = lgr_bins[i]
            lgrh = lgr_bins[i + 1]
            mu_spec_r = interp1d(
                lgEmean, mu_spec[i] / dlgE, bounds_error=False, fill_value=0
            )
            ry = ph_spec[i].sum() / el_spec[i].sum()
            aveEe = (el_spec[i] * Emean).sum() / (el_spec[i]).sum()
            aveEph = (ph_spec[i] * Emean).sum() / (ph_spec[i]).sum()

            mu_spec_E2 = lambda x: mu_spec_r(x) * (10 ** x / Evem) ** 2
            Nmu = mu_spec[i].sum()

            Emu1_sq = integrate_spec(mu_spec_E2, lgEch, lgEvem) / Nmu

            fmu2 = integrate_spec(mu_spec_r, lgEvem, 14) / Nmu
            dd["lgE"].append(lgE)
            dd["cosTheta"].append(cosTheta)
            dd["Xmax"].append(Xmax)
            dd["lgr"].append(lgr)
            dd["lgrl"].append(lgrl)
            dd["lgrh"].append(lgrh)
            dd["filename"].append(fl)

            dd["ry"].append(ry)
            dd["Ee"].append(aveEe)
            dd["Ey"].append(aveEph)
            dd["Emu1sq"].append(Emu1_sq)
            dd["fmu2"].append(fmu2)
            theta = np.round(np.arccos(cosTheta), 2)
            dd["theta"].append(theta)
