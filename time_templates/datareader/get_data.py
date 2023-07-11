"""
Some convenience functions to get desired data
"""

import os
import glob
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda x: x
from time_templates.preprocessing.read_tree_to_df import read_tree
from time_templates.preprocessing.apply_cuts_df import apply_cuts_df
from time_templates.datareader.event import Event
from time_templates import package_path, data_path

# datapath = os.path.join(package_path, "data/")
# TODO: set data path correctly
# datapath = "/home/mart/auger/data/time_templates/"


def fetch_dataframe(tree_file, force=False, cuts=None, **read_tree_kws):
    df_file_name = os.path.join(
        data_path, "df_" + os.path.splitext(os.path.basename(tree_file))[0]
    )
    if "dense" in read_tree_kws:
        if read_tree_kws["dense"]:
            df_file_name += "_dense"
    if "no_traces" in read_tree_kws:
        if read_tree_kws["no_traces"]:
            df_file_name += "_no_traces"
    df_file_name += ".pl"
    try:
        if force:
            raise FileNotFoundError
        print(f"Trying to read {df_file_name}")
        df = pd.read_pickle(df_file_name)
    except FileNotFoundError:
        print(f"Reading {tree_file}")
        df = read_tree(tree_file, **read_tree_kws)

    return apply_cuts_df(df, cuts)


def fetch_data(
    fl=None, key="Golden", hybrid=True, cuts=None, force=False, **read_tree_kws
):
    if fl is None:
        if hybrid:
            one_file = 0
            for fl in glob.glob(data_path + f"/trees/*{key}*.root"):
                one_file += 1

            if one_file > 1:
                print(f"WARNING: multiple files found, only getting {fl}")
        else:
            raise NotImplementedError

    if fl is None:
        raise FileNotFoundError("No file was found")

    try:
        dffl = data_path + "df_" + os.path.splitext(os.path.basename(fl))[0] + ".pl"
        if not force:
            print(f"reading {dffl}")
            df = pd.read_pickle(dffl)
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print(f"{dffl} not found. Reading tree")

        df = read_tree(fl, is_data=True, **read_tree_kws)

    return apply_cuts_df(df, cuts)


def fetch_MC_data_from_tree(
    fl=None,
    key="new_UUB_SSD_rcola",
    HIM="EPOS_LHC",
    primary="proton",
    energy="18.5_19",
    test_events=False,
    cuts=None,
    force=False,
    **read_tree_kws,
):
    "Get MC data from pickled dataframe"
    print("Fetching")
    if test_events:
        return pd.read_pickle(data_path + "test_events.pl")

    if fl is None:
        one_file = 0
        treepath = data_path + f"/trees/*{key}*{HIM}*{primary}*{energy}.root"
        for fl in glob.glob(treepath):
            one_file += 1

        if one_file > 1:
            print(f"WARNING: multiple files found, only getting {fl}")
        if fl is None:
            raise FileNotFoundError(f"No file was found at {treepath}")

    df_file_name = os.path.join(
        data_path, "df_" + os.path.splitext(os.path.basename(fl))[0]
    )
    if "dense" in read_tree_kws:
        if read_tree_kws["dense"]:
            df_file_name += "_dense"
    if "no_traces" in read_tree_kws:
        if read_tree_kws["no_traces"]:
            df_file_name += "_no_traces"
    df_file_name += ".pl"
    try:
        if force:
            raise FileNotFoundError
        print(f"Trying to read {df_file_name}")
        df = pd.read_pickle(df_file_name)
    except FileNotFoundError:
        print(f"Reading {fl}")
        df = read_tree(fl, **read_tree_kws)

    return apply_cuts_df(df, cuts)


# TODO, read adst directly, tree directly


## TODO more, and non MC
def print_event_info(event, eventid):
    print("------------------------------")
    print(f"event id: {eventid}")
    print("------------------------------")
    try:
        print("MC info:")
        print(f"lgE/eV: {event['MClgE'].iloc[0]:.2f}")
        print(f"theta: {np.rad2deg(event['MCTheta'].iloc[0]):.0f} deg")
        print(f"Xmax : {event['MCXmax'].iloc[0]:.0f} g/cm2")
        print(f"MC mu1000: {event['MC_mu1000'].iloc[0]:.0f} VEM")
    except KeyError:
        pass
    try:
        print("FD info:")
        print(f"lgE/eV: {event['FdlgE'].iloc[0]:.2f}")
        print(f"Xmax : {event['FdXmax'].iloc[0]:.0f} g/cm2")
    except KeyError:
        pass
    try:
        print("SD info:")
        print(f"lgE/eV: {event['SdlgE'].iloc[0]:.2f}")
        print(f"theta: {np.rad2deg(event['SdTheta'].iloc[0]):.0f} deg")
    except KeyError:
        pass

    print("")


def print_station_info(station, stationid):
    print("------------------------------")
    print(f"station id {stationid}")
    print("------------------------------")
    print(f"r: {station['Sdr']:.0f} m")
    print(f"psi : {np.rad2deg(station['Sdpsi']):.0f} deg")
    print(f"WCD total signal : {station['WCDTotalSignal']:.1f} VEM charge")
    print(f"WCD Muon signal : {station['WCDMuonSignal']:.1f} VEM Charge")
    print(f"WCD EM signal : {station['WCDEMSignal']:.1f} VEM Charge")
    print(f"SSD total signal : {station['SSDTotalSignal']:.1f} MIP charge")
    print("")


def get_event_from_df(df, eventid=None, verbose=False, **kwargs):
    """get_event_from_df.

    Parameters
    ----------
    df : pandas dataframe
        this dataframe should have as index eventid, stationid
    eventid : int
        eventid, sd event id from adst
    verbose : bool
        verbose
    """
    if eventid is not None:
        try:
            event = df.loc[eventid]
        except KeyError as e:
            print(f"ERROR event id {eventid} was given but could not find it")
            raise e
    else:
        eventids = df.index.get_level_values(level=0).unique()
        eventid = np.random.choice(eventids)
        event = df.loc[eventid]
    if verbose:
        print_event_info(event, eventid)
    return Event(event, eventid, **kwargs)

    # return event


def event_provider(df, nmax=None, get_random=False, verbose=False, **kwargs):
    """event_provider.

    Parameters
    ----------
    df : pandas dataframe
        this dataframe should have as index eventid, stationid

    Yield
    ----------
    Custom Dataclass Event object
    """
    eventids = df.index.get_level_values(level=0).unique()
    if nmax is None:
        nmax = len(eventids)
    if get_random:
        eventids = np.random.choice(eventids, nmax, replace=False)
    else:
        eventids = eventids[:nmax]

    print("Number of events = ", nmax)
    for eventid in tqdm(eventids):
        try:
            yield get_event_from_df(df, eventid, verbose, **kwargs)
        except IndexError:
            continue
