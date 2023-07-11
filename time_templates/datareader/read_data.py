""" 
From any file type to custom Event class
"""
import numpy as np
from event import Event


# TODO, read adst directly, tree directly


## TODO more, and non MC
def print_event_info(event, eventid):
    print("------------------------------")
    print(f"event id: {eventid}")
    print("------------------------------")
    print("MC info:")
    print(f"lgE/eV: {event['MClgE'].iloc[0]:.2f}")
    print(f"theta: {np.rad2deg(event['MCTheta'].iloc[0]):.0f} deg")
    print(f"Xmax : {event['MCXmax'].iloc[0]:.0f} g/cm2")
    print(f"MC mu1000: {event['MC_mu1000'].iloc[0]:.0f} VEM")
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


def get_event_from_df(df, eventid=None, verbose=True):
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
    return event, eventid


def event_provider(df, verbose=False, MC=False):
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
    nevents = len(eventids)
    print("Number of events = ", nevents)
    for eventid in eventids:
        event = Event(*get_event_from_df(df, eventid, verbose=verbose), MC=MC)
        yield event
