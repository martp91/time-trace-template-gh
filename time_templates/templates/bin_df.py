import argparse
import numpy as np
import pandas as pd

from time_templates.preprocessing.phase_traces_df import phase_traces_df
from time_templates.datareader.get_data import fetch_MC_data_from_tree
from time_templates import data_path


def bin_df_ct2_DX_r_cp(df, ct2_bins=6, dDX=50, Scut=5, Xmumax=False, phase_wrt_t0=True):
    "dDX should not be too large, 25 would be better"
    print("making some cuts")
    df = df.query("MCCosTheta > 0.6")
    mc_axis = df[["MCAxis.fX", "MCAxis.fY", "MCAxis.fZ"]].values
    sd_axis = df[["SdAxis.fX", "SdAxis.fY", "SdAxis.fZ"]].values
    df["mc_sd_axis_dot"] = np.einsum("ij,ij->i", mc_axis, sd_axis)
    mc_core = df[["MCCore.fX", "MCCore.fY"]].values
    sd_core = df[["SdCore.fX", "SdCore.fY"]].values

    df["mc_sd_core_xydiff"] = np.sqrt(np.sum((mc_core - sd_core) ** 2, axis=1))

    ptr_diff = np.abs(df["MCPlaneTimeRes"] - df["PlaneTimeRes"])
    df["ptr_diff"] = ptr_diff - np.mean(ptr_diff)
    # some quality cuts
    # cut on difference from mean of MC plane time and sd rec plane time
    # this removes some weird traces timing, and I checked that the cos theta, lgE, r, distr does not change
    # 50 ns seems reasonable. This cuts slightly removes more vertical showers (costheta > 0.9)
    df = df.query("mc_sd_axis_dot > 0.999 & mc_sd_core_xydiff < 150 & ptr_diff < 100")
    # for trigger prob
    print("cut for trigger prob")
    df = df.query(f"WCDTotalSignal_pred > {Scut}")

    if isinstance(ct2_bins, int):
        df["MCCosTheta_sq_bin"] = pd.cut(
            df["MCCosTheta"] ** 2, np.linspace(0.36, 1, ct2_bins)
        )
    elif isinstance(ct2_bins, (list, np.ndarray, np.generic)):
        df["MCCosTheta_sq_bin"] = pd.cut(df["MCCosTheta"] ** 2, ct2_bins)
    else:
        raise TypeError(
            "ct2_bins type not understood should be one of [int, array-like]"
        )

    # TODO: for muon this should be Xmumax
    if Xmumax:
        df["Xmumax_50_bin"] = pd.cut(df["Xmumax_50"], np.arange(450, 850, 25))
        df["MCCosTheta_sq_bin"] = pd.cut(
            df["MCCosTheta"] ** 2, np.linspace(0.36, 1, 12)
        )

        df = df.set_index(
            ["MCCosTheta_sq_bin", "Xmumax_50_bin", "MCr_round", "MCcospsi_round"]
        )
    else:
        df["MCDXstation_bin"] = pd.cut(df["MCDXstation"], np.arange(0, 1200, dDX))

        df = df.set_index(
            ["MCCosTheta_sq_bin", "MCDXstation_bin", "MCr_round", "MCcospsi_round"]
        )
    # sort index for speed
    df = df.reset_index().set_index(df.index.names).sort_index()

    # Take mean values in bin to reduce spread of traces due to time uncertainty
    # Because stupid MC plane does not match sd rec
    # The fix is to shift by the mean in each bin seperatlty

    # Not sure if still use this?
    # df["MCPlaneTimeRes_"] = df["MCPlaneTimeRes"]
    # df["MCPlaneTimeRes_shift"] = df.groupby(df.index.names).apply(
    #    lambda x: (x["MCPlaneTimeRes"] - x["PlaneTimeRes"]).mean()
    # )
    # df["MCPlaneTimeRes"] = df["MCPlaneTimeRes"] - df["MCPlaneTimeRes_shift"]
    # drop some columns to save memory
    df = df.drop(
        [
            "primary",
            "MCCore.fX",
            "MCCore.fY",
            "MCCore.fZ",
            "SdCore.fX",
            "SdCore.fY",
            "SdCore.fZ",
            "StationPos.fX",
            "StationPos.fY",
            "StationPos.fZ",
            "Is6T5",
            "YYMMDD",
            "HHMMSS",
            "GPSSecond",
            "GPSNanoSecond",
            "PMT1Charge",
            "PMT2Charge",
            "PMT3Charge",
            "PMT5Charge",
            "PMT1Peak",
            "PMT2Peak",
            "PMT3Peak",
            "PMT5Peak",
            "PMT1DAratio",
            "PMT2DAratio",
            "PMT3DAratio",
            "PMT5DAratio",
            "IsDense",
            "TriggerName",
        ],
    )
    # uses average ptr, t0 in bin, as set above
    # This can now be on MC plane time res because of shift above, else be careful
    # there is height difference btween MC core and SdCore and other effects (??) that # make the timing different
    if phase_wrt_t0:
        t0_key = "t0_wrt_pft_Rc_fit"
    else:
        t0_key = None
    df = phase_traces_df(
        df, univ_comp=True, ptf_key="PlaneTimeRes_new", t0_key=t0_key, normalize=True,
    )
    print("...done")
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-HIM", default="EPOS_LHC")
    parser.add_argument("-primary", default="proton")
    parser.add_argument(
        "-energy",
        default="19_19.5",
        help="should be any of [18.5_19, 19_19.5, 19.5_20]",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-dDX", default=50, type=int)
    parser.add_argument("--Xmumax", action="store_true")
    parser.add_argument("--phase_wrt_ptf", action="store_true")
    # TODO: set nDX nct in bin func
    args = parser.parse_args()
    HIM = args.HIM
    primary = args.primary
    energy = args.energy
    print("args", args)

    if args.Xmumax:
        pickle_filename = (
            data_path
            + f"/binned_df/Xmumax/binned_df_{HIM}_{primary}_{energy}_phased_traces_Xmumax.pl"
        )
    else:
        pickle_filename = (
            data_path
            + f"/binned_df/binned_df_{HIM}_{primary}_{energy}_phased_traces.pl"
        )

    df = fetch_MC_data_from_tree(
        key="new_UUB_SSD_rcola",
        energy=energy,
        primary=primary,
        HIM=HIM,
        force=args.force,
        dense=True,
        do_Rc_fit=True,
        univ_comp=True,
        save=True,
    )
    df = df.reset_index()
    print("binning")

    # Default cut above ct > 0.6
    df = df.query("MCCosTheta > 0.6")
    ct2_bins = np.linspace(0.36, 1, 6)

    df = bin_df_ct2_DX_r_cp(
        df,
        ct2_bins=ct2_bins,
        dDX=args.dDX,
        Xmumax=args.Xmumax,
        phase_wrt_t0=not args.phase_wrt_ptf,
    )
    print("Saving for later", pickle_filename)
    df.to_pickle(pickle_filename)

    print("making means, var in bins")
    print(
        "Making dataframes for different cos theta and binning these in DX, for dense stations r/psi"
    )
    print("Grouping")
    df_gb = df.groupby(df.index.names)

    print("Taking mean")

    df_gb_mean = df_gb.mean()
    df_gb_var = df_gb.var()
    n = df_gb["MClgE"].count()
    df_gb_mean["nstations"] = n

    for key in df.keys():
        if not "trace" in key:
            continue
        print(f"at {key}")
        mean_trace = df_gb[key].mean()

        x2_trace = df_gb[key].apply(lambda x: np.mean(x ** 2))
        var_trace = x2_trace - mean_trace ** 2
        # var_trace = np.var(traces, dtype="float64", axis=0)
        # median does not work whatever
        # median_trace = df_gb[key].aggregate(np.median)
        df_gb_mean[key + "_mean"] = mean_trace
        # df_gb_mean[key + "_median"] = median_trace
        df_gb_mean[key + "_var"] = var_trace

    for key in df_gb_var.keys():
        df_gb_mean[key + "_var"] = df_gb_var[key]
    # df_gb_mean.dropna(inplace=True)
    if args.Xmumax:
        mean_pickle_filename = (
            data_path + f"/mean_df/Xmumax/mean_df_{HIM}_{primary}_{energy}_Xmumax.pl"
        )
    else:
        mean_pickle_filename = (
            data_path + f"/mean_df/mean_df_{HIM}_{primary}_{energy}.pl"
        )
    print()
    print("Number of rows", len(df_gb_mean))
    print("Saving mean df at", mean_pickle_filename)

    df_gb_mean.to_pickle(mean_pickle_filename)
