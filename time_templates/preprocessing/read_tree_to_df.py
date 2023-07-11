#! /usr/bin/env python
"""
This reads the root tree (made from adst with adst-to-tree)
and does some preprocessing and saves it to a pandas df
"""

import os
import argparse
import numpy as np
import uproot
import pandas as pd
from time_templates import package_path, data_path
from time_templates.utilities import atmosphere
from time_templates.utilities.reduce_df_mem import reduce_mem_usage
from time_templates.starttime.fit_start_time import fit_start_times_df
from time_templates.templates.universality.rho_model import set_total_signal_pred_df
from time_templates.templates.universality.S1000_model import set_Rmu_df
from time_templates.MPD.Xmumax_50 import Xmumax_att_factor

from time_templates.preprocessing.apply_cuts_df import apply_cuts_df
import traceback

pd.options.mode.chained_assignment = None  # ignore SettingWithCopyWarning
atm = atmosphere.Atmosphere()


def get_year_month_day(YYMMDD):
    YYMMDD = str(YYMMDD)
    if len(YYMMDD) == 6:
        return pd.Series(
            [2000 + int(YYMMDD[:2]), int(YYMMDD[2:4]), int(YYMMDD[4:])],
            index=["year", "month", "day"],
        )
    elif len(YYMMDD) == 5:
        return pd.Series(
            [2000 + int(YYMMDD[0]), int(YYMMDD[1:3]), int(YYMMDD[3:])],
            index=["year", "month", "day"],
        )
    else:
        raise ValueError


def get_hour_minute_sec(HHMMSS):
    HHMMSS = str(HHMMSS)
    #     print(HHMMSS)
    if len(HHMMSS) == 1:
        return pd.Series([0, 0, int(HHMMSS)], index=["hour", "minute", "second"])
    elif len(HHMMSS) == 2:
        return pd.Series([0, 0, int(HHMMSS)], index=["hour", "minute", "second"])
    elif len(HHMMSS) == 3:
        return pd.Series(
            [0, int(HHMMSS[0]), int(HHMMSS[1:])], index=["hour", "minute", "second"]
        )
    elif len(HHMMSS) == 4:
        return pd.Series(
            [0, int(HHMMSS[0:2]), int(HHMMSS[2:])], index=["hour", "minute", "second"]
        )
    elif len(HHMMSS) == 5:
        return pd.Series(
            [int(HHMMSS[0]), int(HHMMSS[1:3]), int(HHMMSS[3:])],
            index=["hour", "minute", "second"],
        )
    elif len(HHMMSS) == 6:
        return pd.Series(
            [int(HHMMSS[0:2]), int(HHMMSS[2:4]), int(HHMMSS[4:])],
            index=["hour", "minute", "second"],
        )
    else:
        raise ValueError


# ususal 10 dense stations
cos_psis = np.unique(np.round(np.cos(np.arange(0, 2 * np.pi, 2 * np.pi / 10)), 6))

psis = np.linspace(0, 180, 6)
cp_to_psi = {}
for cp in cos_psis:
    psi = np.rad2deg(np.arccos(cp))
    cp_to_psi[cp] = psis[np.argmin(np.abs(psis - psi))]


def read_tree(
    input_file,
    output_file=None,
    dense=False,
    no_traces=False,
    save=True,
    is_data=False,
    ssd_traces=False,
    univ_comp=True,
    do_Rc_fit=False,
    reduce_mem=False,
    savetype="pickle",
    get_dates=False,
    cuts=None,
    **kwargs,
):
    """
    Read a root tree file that was made with adst-to-tree
    Returns pandas dataframe
    """

    print("loading tree to dataframe")
    if is_data:
        get_dates = True

    # First read all the non traces

    def check_names(x):
        """ Ignore branch names that have this"""
        if "trace" in x:
            return False
        elif "TObject" in x:
            return False
        else:
            return True

    tree = uproot.open(input_file)["tree"]
    df = tree.arrays(library="pd", filter_name=lambda x: check_names(x))
    df["MCTheta"] = np.arccos(df["MCCosTheta"])
    df["SdTheta"] = np.arccos(df["SdCosTheta"])
    df["MCSecTheta"] = 1 / df["MCCosTheta"]
    df["SdSecTheta"] = 1 / df["SdCosTheta"]
    df["MCcospsi"] = np.cos(df["MCpsi"])
    df["Sdcospsi"] = np.cos(df["Sdpsi"])

    if do_Rc_fit:
        df = fit_start_times_df(df)

    if not no_traces:
        print("Getting traces, takes some time...")
        if ssd_traces:
            detectors = ["wcd", "ssd"]
        else:
            detectors = ["wcd"]
        for det in detectors:
            if univ_comp and not is_data:
                muon_traces = tree["%s_muon_trace" % det].array(library="np")
                em_pure_traces = tree["%s_em_pure_trace" % det].array(library="np")
                em_mu_traces = tree["%s_em_mu_trace" % det].array(library="np")
                em_had_traces = tree["%s_em_had_trace" % det].array(library="np")
                df["%s_muon_trace" % det] = list(muon_traces)
                df["%s_em_trace" % det] = list(em_pure_traces)
                df["%s_em_mu_trace" % det] = list(em_mu_traces)
                df["%s_em_had_trace" % det] = list(em_had_traces)
                # least memory consumption like this, altering it changes memory usage?

            else:
                if not is_data:
                    muon_traces = tree["%s_muon_trace" % det].array(library="np")
                    df["%s_muon_trace" % det] = list(muon_traces)
            total_traces = tree["%s_total_trace" % det].array(library="np")
            df["%s_total_trace" % det] = list(total_traces)

    df = df.reset_index().set_index(["EventId", "StationId"])
    df.sort_index(inplace=True)

    # Hack to get mean muon MC signal at 1000 meter
    # Takes some time, not too much?
    # Needs to be after trace getting
    if not is_data:
        print("Getting MC 1000")
        try:
            df["MCr_round"] = df["MCr"].round(0)
            df_dense_1000 = df.query("MCr_round == 1000 & IsDense == 1")
            df_dense_1000_gb = df_dense_1000.groupby(["EventId", "SdlgE"])
            MC_mu1000 = df_dense_1000_gb["WCDMuonSignal"].mean()
            MC_S1000 = df_dense_1000_gb["WCDTotalSignal"].mean()
            out = []
            out_S1000 = []
            for eventid, new_df in df.groupby(["EventId", "SdlgE"]):
                new_df["MC_mu1000"] = MC_mu1000.loc[eventid]
                new_df["MC_S1000"] = MC_S1000.loc[eventid]
                out.append(new_df["MC_mu1000"].values)
                out_S1000.append(new_df["MC_S1000"].values)
            df["MC_mu1000"] = np.hstack(out)
            df["MC_S1000"] = np.hstack(out_S1000)
        except:
            print("Failed, is this data?")
            df["MC_mu1000"] = np.nan
            df["MC_S1000"] = np.nan

    if dense:
        print("Only saving dense station")
        df = df.query("IsDense == 1")
        df["MCr"] = df["MCr"].round(0)
    else:
        print("not saving dense stations")
        df = df.query("IsDense == 0 & IsCandidate == 1")

    SD_core = df[["SdCore.fX", "SdCore.fY", "SdCore.fZ"]].values
    StationPos = df[["StationPos.fX", "StationPos.fY", "StationPos.fZ"]].values

    df["Sdr_ground"] = np.sqrt(np.sum((SD_core - StationPos) ** 2, axis=1))

    print("Getting rmin")
    df = df.reset_index()
    df.set_index("EventId", inplace=True)
    df["Sdr_min"] = df.groupby("EventId")["Sdr"].min()
    df["Sdr_ground_min"] = df.groupby("EventId")["Sdr_ground"].min()

    if do_Rc_fit:
        rmin = df.groupby("EventId")["Sdr_new"].min()
        df["Sdr_new_min"] = rmin

    df = df.reset_index().set_index(["EventId", "StationId"])
    df.sort_index(inplace=True)

    print("Some more preprocessing")
    if not univ_comp:
        df["WCDEMSignal"] = df["WCDTotalSignal"] - df["WCDMuonSignal"]
        df["SSDEMSignal"] = df["SSDTotalSignal"] - df["SSDMuonSignal"]

    # WARNING this could be wrong
    tpf = df["MCStationPlaneFrontTimeS"] / 1e9 + df["MCStationPlaneFrontTimeNS"]
    t0 = (df["TimeS"] / 1e9 + df["TimeNS"]) - tpf
    df["MCPlaneTimeRes"] = -t0

    df["WCDCharge"] = (df["PMT1Charge"] + df["PMT2Charge"] + df["PMT3Charge"]) / df[
        "nWorkingPmts"
    ]

    df["WCDPeak"] = (df["PMT1Peak"] + df["PMT2Peak"] + df["PMT3Peak"]) / df[
        "nWorkingPmts"
    ]

    # Remove any outliers and set to mean
    # TODO: if 0 set to other pmts?

    def pmt_pulse_shape(x):
        return np.where(x < 30, x.mean(), np.where(x > 80, x.mean(), x))

    for pmtid in [1, 2, 3]:
        df[f"PMT{pmtid}PulseShape"] = pmt_pulse_shape(df[f"PMT{pmtid}PulseShape"])

    df["WCDPulseShape"] = (
        df["PMT1PulseShape"] + df["PMT2PulseShape"] + df["PMT3PulseShape"]
    ) / 3
    df["WCDPulseShapeRMS"] = np.std(
        [df["PMT1PulseShape"], df["PMT2PulseShape"], df["PMT3PulseShape"]], axis=0
    )

    if not is_data:
        df["primary"] = df["primary"].map({2212: "proton", 1000026056: "iron"})
        df["MCDXstation"] = atm.DX_at_station(
            df["MCr"].values,
            df["MCpsi"].values,
            df["MCTheta"].values,
            df["MCXmax"].values,
            1400,
        )
        df["MCDXmax"] = (
            atm.slant_depth_at_height(1400, df["MCTheta"].values) - df["MCXmax"].values
        )

        df["MCDX_1000"] = atm.DX_at_station(
            1000, np.pi / 2, df["MCTheta"].values, df["MCXmax"].values, 1400
        )
        df["MCcospsi_round"] = df["MCcospsi"].apply(
            lambda x: cos_psis[np.argmin(np.abs(cos_psis - x))]
        )
        df["MCSinThetaCosPsi"] = np.sin(df["MCTheta"]) * df["MCcospsi"]
        df["MClgr"] = np.log(df["MCr"])

        try:
            set_Rmu_df(df)
        except:
            print(traceback.format_exc())
            print("not setting Rmu")
            pass

        try:
            set_total_signal_pred_df(df, Rmu=df["Rmu"].values)  # only for MC of course
        except:
            print(traceback.format_exc())
            print("total signal pred failed")
            pass

        try:
            print("Getting Xmumax 50")
            df["shower_id"] = df["adstFilename"].str[-11:-5].astype("int")
            df_mpd = pd.read_pickle(data_path + "/MPD/df_MPD.pl")
            df = (
                df.reset_index()
                .merge(df_mpd, on="shower_id")
                .set_index(["EventId", "StationId"])
            )
            # from fit to sec theta depenence on xmumax1700. average epos proton/iron
            df["Xmumax_50"] = df["Xmumax_1700"] - Xmumax_att_factor(df["MCSecTheta"])

            df["DXmumax_50"] = (
                atm.slant_depth_at_height(1400, df["MCTheta"].values)
                - df["Xmumax_50"].values
            )
            df["DXmustation_50"] = atm.DX_at_station(
                df["MCr"].values,
                df["MCpsi"].values,
                df["MCTheta"].values,
                df["Xmumax_50"],
                1400,
            )

            df["LXmumax_50"] = (
                atm.height_at_slant_depth(df["Xmumax_50"].values, df["MCTheta"].values)
                - 1400
            ) / np.cos(df["MCTheta"].values)
            # to station
            df["LXmustation_50"] = (
                atm.height_at_slant_depth(df["Xmumax_50"].values, df["MCTheta"].values)
                - (
                    1400
                    + df["MCr"].values
                    * df["MCcospsi"].values
                    * np.sin(df["MCTheta"].values)
                )
            ) / np.cos(df["MCTheta"].values)
            df["LXmustation_1700"] = (
                atm.height_at_slant_depth(
                    df["Xmumax_1700"].values, df["MCTheta"].values
                )
                - (
                    1400
                    + df["MCr"].values
                    * df["MCcospsi"].values
                    * np.sin(df["MCTheta"].values)
                )
            ) / np.cos(df["MCTheta"].values)
        except Exception as e:
            print("WARNING: Getting Xmumax failed")
            print(e)

    if get_dates:
        print("Setting year, month day")
        # This is slow
        df[["year", "month", "day"]] = df["YYMMDD"].apply(get_year_month_day)
        df[["hour", "minute", "second"]] = df["HHMMSS"].apply(get_hour_minute_sec)

    # Sort by total signal default
    df.sort_values(["EventId", "WCDTotalSignal"], ascending=False, inplace=True)

    # Do I want this?
    df = df[~df.index.duplicated()]
    df = apply_cuts_df(df, cuts)

    if is_data:
        df = df.loc[:, ~df.columns.str.contains("MC")]
    if reduce_mem:
        try:
            df = reduce_mem_usage(df)
        except:
            print("Reducing mem failed")

    if save:
        if output_file is None:
            output_name = "df_" + os.path.splitext(os.path.basename(input_file))[0]
            if dense:
                output_name += "_dense"

            if no_traces:
                output_name += "_no_traces"

            if savetype == "pickle":
                ext = ".pl"
            elif savetype == "parquet":
                ext = ".parquet"

            output_file = os.path.join(data_path, output_name + ext)
        print("Saving at %s" % output_file)
        if savetype == "pickle":
            df.to_pickle(output_file)
        elif savetype == "parquet":
            df.to_parquet(output_file)
    else:
        print("not saving")

    print("...done")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="input root tree")
    parser.add_argument("-output_file", help="specify pandas out file")
    parser.add_argument(
        "--dense",
        help="if specified, assume dense station \
                        and only take these",
        action="store_true",
    )
    parser.add_argument(
        "--no_traces",
        help="if specified average over \
                        dense rings (no traces saved!)",
        action="store_true",
    )
    parser.add_argument("--is_data", action="store_true")
    parser.add_argument("--do_Rc_fit", action="store_true")

    args = parser.parse_args()
    read_tree(**vars(args))
