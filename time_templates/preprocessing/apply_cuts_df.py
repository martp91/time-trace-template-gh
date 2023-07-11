"""
Simple module to make cuts on pandas dataframe
as created by preprocessing
Default gives some verbose messages on the number of events/stations
that survive
"""
import os
import argparse
import json
import pandas as pd

from time_templates.utilities.misc import cut_df
from time_templates import package_path

def apply_cuts_df(df, cuts=None, verbose=1):
    if cuts is None:
        return df
    return cut_df(df, cuts, verbose=verbose)


def apply_cuts(
    df_pickle_file, cuts=None, output=None, cuts_file=None, verbose=1,
):

    if verbose:
        print(f"Loading {df_pickle_file}")

    df = pd.read_pickle(df_pickle_file)
    df = df.reset_index()
    df = df.set_index("EventId")

    if cuts is None:
        if cuts_file is not None:
            cuts = json.load(open(cuts_file, "r"))

    df = apply_cuts_df(df, cuts, verbose)

    if output is not None:
        if verbose:
            print(f"Saving at {output}")
        df.to_pickle(output)

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("df_pickle_file")
    parser.add_argument("-output", default=None)
    parser.add_argument(
        "-cuts_file",
        default=os.path.join(package_path, "preprocessing/default_cuts.json"),
    )
    parser.add_argument("-verbose", type=int, default=1)

    args = parser.parse_args()

    apply_cuts(*args)
