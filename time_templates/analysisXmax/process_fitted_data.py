import os
import argparse
import numpy as np
import pandas as pd
import json

from time_templates.datareader.get_data import fetch_MC_data_from_tree, fetch_data
from time_templates.misc import energy as energy_tools
from time_templates.analysisXmax.utils import SD_Xmax_resolution
from time_templates import data_path
from time_templates.preprocessing.apply_cuts_df import apply_cuts_df


csv_data_path = os.path.join(data_path, "fitted_csv")

with open("thesis_cuts.json") as infile:
    CUTS = json.load(infile)

FORCE = True


def load_fitted_df(
    is_data=False,
    primary="proton",
    energy="19.0_19.5",
    HIM="EPOSLHC",
    key_lib="MCTask_Offline_v3r99p2a",
    key="Xmax_SD",
):
    if is_data:
        df = fetch_data(key=key, do_Rc_fit=True, force=FORCE)
        df_fit = pd.read_csv(os.path.join(csv_data_path, f"events_fitted_{key}.csv"))
    else:
        df_fit = pd.read_csv(
            os.path.join(
                csv_data_path,
                f"events_fitted_{key_lib}_{HIM}_{primary}_{energy}_{key}.csv",
            )
        )
        df = fetch_MC_data_from_tree(
            primary=primary,
            energy=energy,
            HIM=HIM,
            key=key_lib,
            do_Rc_fit=True,
            force=FORCE,
        )

    df = apply_cuts_df(df, CUTS)
    df_fit.set_index("EventId", inplace=True)
    df = df.groupby("EventId").mean()
    df = df_fit.join(df, on="EventId")
    if not is_data:
        df["HIM"] = HIM
        df["primary"] = primary
        df["MCDXmax"] = df["Xg"] - df["MCXmaxGH"]

    df["fit_success"] = df["fit_success"].astype(bool)
    df["Xmax_fit"] = df["Xmax_fit"].astype(float)
    # df['Xmax_fit_err'] = np.sqrt(df['cov_Xmax'])
    df["Rmu_fit"] = df["Rmu_fit"].astype(float)
    df["DXmax_fit"] = df["Xg"] - df["Xmax_fit"]
    df["red_deviance_fit"] = df["fit_deviance"] / df["fit_ndof"]

    df["SdlgE_err"] = energy_tools.SdlgE_resolution(df["SdlgE"])
    df["Xmax_fit_err"] = SD_Xmax_resolution(df["SdlgE"])
    cols_drop = []
    for col in df.keys():
        if "rho" in col:
            cols_drop.append(col)
        if "Signal_pred" in col:
            cols_drop.append(col)

        if "PMT" in col:
            cols_drop.append(col)

        if is_data:
            if "MC" in col:
                cols_drop.append(col)

    df.drop(cols_drop, axis=1, inplace=True)

    df = apply_cuts_df(df, CUTS)

    if is_data:
        output_file = os.path.join(
            data_path, f"fitted_merged/events_ttt_fitted_{key}.csv"
        )
    else:
        output_file = os.path.join(
            data_path,
            f"fitted_merged/events_ttt_fitted_{key_lib}_{HIM}_{primary}_{energy}_{key}.csv",
        )
    print(f"Saving at {output_file}")
    df.to_csv(output_file)


if __name__ == "__main__":

    HIM = "EPOSLHC"
    key_lib = ("MCTask_Offline_v3r99p2a",)
    key = ("Xmax_SD",)
    is_data = False
    # for primary in ["proton", "iron"]:
    #    for energy in ["19.0_19.5", "19.5_20.0"]:
    #        load_fitted_df(is_data=False, HIM=HIM, primary=primary, energy=energy)

    # load_fitted_df(is_data=True, key="observer_icrc19_SD_SdlgE19_theta53")
    load_fitted_df(is_data=True, key="observer_icrc19_Golden_SdlgE18.8")
