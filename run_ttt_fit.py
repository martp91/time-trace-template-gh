#!/usr/bin/env python
"""
Run time template fit on data files (adst), or root trees or pandas dataframes (the later two are created
from adst files)

Mart Pothast Nov 2022
mart.pothast@gmail.com
"""

import argparse
import os
import json
import sys
import subprocess
import numpy as np
import pandas as pd


NO_FUTURE = False  # just set to true if you don't want to know the amount of power used
if not NO_FUTURE:
    try:
        from codecarbon import EmissionsTracker
    except:
        print("get codecarbon: pip install codecarbon to track power usage")
        NO_FUTURE = True

# import pandas as pd

from time_templates import data_path, package_path
from time_templates.datareader.get_data import fetch_dataframe

from time_templates.fittemplate.fit_events import fit_events


FILEDIR = os.path.dirname(__file__)

try:
    AUGEROFFLINEROOT = os.environ["AUGEROFFLINEROOT"]
except KeyError:
    try:
        AUGEROFFLINEROOT = os.environ["ADSTROOT"]
    except KeyError as e:
        print("You might need Oflfine or atleast ADST reading capabilities")
        print("Or you might have the processed files already")
        AUGEROFFLINEROOT = ""
        # TODO: maybe if you have processed files you don't?

print(f"Offline (or ADST) install at: {AUGEROFFLINEROOT}")
print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-adst_files", "-a", nargs="+", help="Adst file(s), can be multiple or just one"
    )
    parser.add_argument(
        "-tree_file",
        "-t",
        help="One root tree file, processed adst files with adst_to_tree/read_adst",
    )
    parser.add_argument(
        "-df_file",
        "-df",
        help="One pandas dataframe tree file, processed adst files with adst_to_tree/read_adst -> preprocessing",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run on some test files and check Xmax is same as in thesis",
    )
    parser.add_argument(
        "-selectADST_cfg", "-s", default=None, help="if None, no selection"
    )
    parser.add_argument(
        "-save_key", default="", help="Add this str to saved output file"
    )
    parser.add_argument("--isMC", action="store_true", help="if MC use this")
    parser.add_argument(
        "--force",
        action="store_true",
        help="force reading adst files, preprocessing etc",
    )
    parser.add_argument(
        "--calc_co2",
        action="store_true",
        help="for fun calc co2 with emission package (needs install",
    )
    parser.add_argument(
        "-output_file",
        "-o",
        help="output file for fitted events, default will be determined from input",
    )
    parser.add_argument(
        "-fit_event_options_file",
        "-options",
        default="fit_event_options/sd_Xmax_fit.json",
    )
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()

    if (
        args.adst_files is None
        and args.tree_file is None
        and args.df_file is None
        and not args.test
    ):
        parser.error("have to specify adst_files or --test")

    adst_files = args.adst_files
    tree_file = args.tree_file
    df_file = args.df_file

    if (
        adst_files is not None
        and tree_file is None
        and df_file is None
        and not args.test
    ):

        print("Reading ADST files")

        selectADST_cfg = args.selectADST_cfg

        first_file = adst_files[0]
        adst_file_dir, first_filename = os.path.split(first_file)
        first_filename, first_file_ext = os.path.splitext(first_filename)
        adst_selected_filename_noext = first_filename + "_selected"
        adst_selected_filename = adst_selected_filename_noext + ".root"

        nfiles = len(adst_files)
        if nfiles == 1 and selectADST_cfg is None:
            print("No selection necessary")
            adst_selected_output = adst_files[0]
        else:
            if selectADST_cfg is None:
                selectADST_cfg = "selectADST/TTT_ICRC2019/empty.cfg"

            adst_selected_output = os.path.join(
                data_path, "adst", adst_selected_filename
            )

            subprocess.run(
                [
                    "selectADSTEvents",
                    "-c",
                    selectADST_cfg,
                    "-o",
                    adst_selected_output,
                    " ".join(adst_files),
                ],
                check=True,
            )

        tree_filename = "tree_" + adst_selected_filename
        tree_file = os.path.join(data_path, "trees", tree_filename)
        print()

        output_file_str = adst_selected_filename_noext

        if (
            (not os.path.isfile(tree_file))
            or args.force
            or (args.selectADST_cfg is not None)
        ):
            print(
                "Reading adst to tree, this takes some time, but only has to be done once"
            )
            print(f"Saving tree at {tree_file}")
            subprocess.run(
                [
                    "adst_to_tree/read_adst",
                    "-i",
                    adst_selected_output,
                    "-o",
                    tree_file,
                ],
                check=True,
            )

    if args.test:
        print("reading test file")
        df = pd.read_pickle(
            os.path.join(package_path, "data", "df_observer_Golden_icrc19_test.pl")
        )
    else:
        if df_file is not None:
            if df_file.endswith(".csv"):
                df = pd.read_csv(df_file)
            elif df_file.endswith(".pl"):
                df = pd.read_pickle(df_file)
            else:
                raise NameError(
                    f"df file {df_file} is not readable, should be .pl or .csv"
                )
            _, output_file_str = os.path.split(df_file)
            output_file_str, _ = os.path.splitext(
                output_file_str.replace("df_tree_", "")
            )
        else:
            _, output_file_str = os.path.split(tree_file)
            output_file_str, _ = os.path.splitext(output_file_str.replace("tree_", ""))
            df = fetch_dataframe(tree_file, is_data=not args.isMC, force=args.force)

    print("succes reading data to dataframe")
    print()
    print("Fitting time trace templates to all events")

    print(f"loading options from {args.fit_event_options_file}")
    with open(args.fit_event_options_file) as infile:
        fit_event_options = json.load(infile)

    print()
    print("Options used:", fit_event_options)

    if args.test:
        df = df.loc[[2984180, 21535440]]
        Xmax_fits_thesis = [758.45, 774.91]

    if not NO_FUTURE and args.calc_co2:
        tracker = EmissionsTracker()
        tracker.start()

    df_fit = fit_events(df, plot=args.plot, **fit_event_options)

    if not NO_FUTURE and args.calc_co2:
        tracker.stop()

    if args.test:
        for Xmax_fit, Xmax_fit_thesis in zip(
            df_fit["Xmax_fit"].values, Xmax_fits_thesis
        ):
            s = f"Xmax fit not correct is {Xmax_fit}, should be {Xmax_fit_thesis}"
            assert np.isclose(Xmax_fit, Xmax_fit_thesis), s
        print()
        print("===============")
        print("All tests OK!")
        print("===============")
        sys.exit()

    fit_output_filename = "df_events_fitted_" + output_file_str + args.save_key + ".csv"
    print()

    fit_output_file = os.path.join(data_path, "fitted", fit_output_filename)
    df_fit.set_index("EventId", inplace=True)
    df = df.groupby("EventId").mean()
    df = df_fit.join(df, on="EventId")
    if args.output_file is not None:
        fit_output_file = args.output_file

    i = 2
    filename, ext = os.path.splitext(fit_output_file)
    while os.path.isfile(fit_output_file):
        new_filename = filename + str(i)
        fit_output_file = new_filename + ext
        i += 1

    if i > 0:
        print("File already existed")

    print(f"saving at {fit_output_file}")
    df.to_csv(fit_output_file)

    print("... done")
