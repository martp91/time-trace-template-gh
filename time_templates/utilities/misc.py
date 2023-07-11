from collections import defaultdict
import numpy as np
import pandas as pd
from numba import njit, prange


def cos_chi(r, theta, psi, zmax):
    Delta = r * np.tan(theta) * np.cos(psi)
    return zmax * np.cos(theta) / np.sqrt(r ** 2 + (zmax - Delta) ** 2)


def f_chi(r, theta, psi, zmax=40000):
    return np.arccos(cos_chi(r, theta, psi, zmax))


def get_event_info_str(event, eid=None, sep="\n", MC=True, FD=False):
    if MC:
        sub1 = "MC"
        sub2 = "MC"
        sub3 = "MC"
    else:
        sub1 = "Sd"
        if FD:
            sub2 = "Fd"
        else:
            sub2 = "Sd"
        sub3 = "Fd"

    #    event = event.iloc[0]
    if eid is not None:
        eventid = eid
    else:
        try:
            eventid = int(event["event_id"])
        except:
            eventid = 0

    textstr = sep.join(
        (
            "EventId= %s" % (eventid,),
            "$\\log{E/\\mathrm{eV}}=%.1f$" % (event[sub2 + "lgE"],),
            "$\\theta=%.0f^\\circ$"
            % (np.rad2deg(np.arccos(event[sub1 + "CosTheta"])),),
            "$X_{\\mathrm{max}}=%.0f \\ \\mathrm{g\\,cm^{-2}}$"
            % (event[sub3 + "Xmax"],),
        )
    )
    return textstr


def get_station_info_str(station, sep="\n", MC=True):
    if MC:
        sub = "MC"
    else:
        sub = "sd_"
    textstr = sep.join(
        (
            "$r=%.0f$ m" % (station[sub + "r"]),
            "$\\psi=%.0f^\\circ$" % (np.rad2deg(np.arccos(station[sub + "psi"]))),
        )
    )
    return textstr


@njit(fastmath=True)
def find_integral_cutoff(mu, cutoff=1):
    """By summing of the mu (without detector response),
    you auto get the number of integrated particles.
    find the bins where the integral before and after
    is lower than @cutoff (for example 1 particle)
    """
    _sum = 0
    ifirst = 0

    # forward integral
    for i, _mu in enumerate(mu):
        _sum += _mu
        if _sum >= cutoff:
            ifirst = i
            break

    _sum = 0
    ilast = len(mu) - 1
    for i, _mu in enumerate(mu[::-1]):
        _sum += _mu
        if _sum > cutoff:
            ilast = len(mu) - i - 1
            break

    if ifirst >= ilast:
        ifirst = 0
        ilast = len(mu) - 1
    return ifirst, ilast


@njit(fastmath=True)
def convolve(tr, resp):
    n = len(tr)
    m = len(resp)
    out = np.zeros(n)

    for i in range(n):
        jmin = i - m
        if jmin < 0:
            jmin = 0
        jmax = n
        for j in range(jmin, jmax):
            if i - j >= 0:
                out[i] += tr[j] * resp[i - j]

    return out


class DictToDF(object):
    def __init__(self, savefile):
        self.savefile = savefile
        self.dd = defaultdict(list)

    def append(self, val, key):
        self.dd[key].append(val)

    def save(self):
        df = pd.DataFrame(self.dd)
        print("Save df to %s" % self.savefile)
        df.save_pickle(self.savefile)


def histedges_equalN(x, nbin):
    """ from https://stackoverflow.com/questions/39418380/histogram-with-equal-number-of-points-in-each-bin
    """
    npt = len(x)
    bins = np.interp(np.linspace(0, npt, nbin + 1), np.arange(npt), np.sort(x))
    return np.unique(bins)  # if bins are equal get error


def get_mean_delta_bins(bins):
    mb = (bins[1:] + bins[:-1]) / 2
    db = bins[1:] - bins[:-1]
    return mb, db


def mid(x):
    try:
        return 0.5 * (x[1:] + x[:-1])
    except:
        x = np.array(x)
        return 0.5 * (x[1:] + x[:-1])


def cut_df(df, cuts, verbose=False) -> pd.DataFrame:
    total_before = len(df)
    #    try:
    total_before_events = len(df.index.get_level_values(level=0).unique())
    #    except:
    #        total_before_events = 0

    if verbose:
        print()
        print("x: [low, high] means low < x < high. None means +-np.inf")
        print("-" * 50)
    if isinstance(cuts, dict):
        cuts = list(cuts.items())
    for cutkey, cut in cuts:
        before = len(df)
        #        try:
        before_events = len(df.index.get_level_values(level=0).unique())
        #        except:
        #            before_events = 0
        if verbose:
            print(
                f"Applying cut on {cutkey}: {cut}. Nevents BEFORE: {before_events}, Nstations BEFORE: {before}"
            )
        #        try:
        #            df = df.loc[np.isfinite(df[cutkey])]
        #        except TypeError:
        #            pass
        try:
            if isinstance(cut, (list, tuple, np.ndarray)):
                low, high = cut
                if high is None:
                    high = np.inf
                if cutkey == "NStations":
                    counts = df.index.get_level_values(level=0).value_counts()
                    eids = counts.index[(counts > low) & (counts < high)]
                    df = df.loc[eids]
                else:
                    df = df.loc[(df[cutkey] > low) & (df[cutkey] < high)]
            elif isinstance(cut, (float, int, bool)):
                df = df.loc[df[cutkey] == cut]
            elif callable(cut):
                if cutkey in df.keys():
                    df = df.loc[cut(df[cutkey])]
                else:
                    df = df.loc[cut(df)]
            elif isinstance(cut, str):
                df = df.loc[df[cutkey] == cut]
            else:
                raise NotImplementedError
            after = len(df)
            #        try:
            after_events = len(df.index.get_level_values(level=0).unique())
            #        except:
            #            after_events = 0
            try:
                efficiency = after / float(before) * 100
            except ZeroDivisionError:
                efficiency = 0

            try:
                efficiency_events = after_events / float(before_events) * 100
            except ZeroDivisionError:
                efficiency_events = 0

            if verbose:
                print(
                    f"Applying cut on {cutkey}: {cut}. Nevents AFTER : {after_events}, Nstations AFTER : {after}"
                )
                print("Efficiency events = {:.2f} %".format(efficiency_events))
                print("Efficiency stations = {:.2f} %".format(efficiency))
        except KeyError:
            print(f"WARNING: Key {cutkey} not found NOT applying cut")

    if verbose:
        print("-" * 50)
        print(f"Total events left = {after_events}")
        print(
            "Total efficiency events = {:.2f} %".format(
                after_events / float(total_before_events) * 100
            )
        )
        print(f"Total stations left = {after}")
        print(
            "Total efficiency stations = {:.2f} %".format(
                after / float(total_before) * 100
            )
        )
        print("-" * 50)
    return df


def merge_drop(left, right, **kwargs):
    """
    merge dataframes with duplicate columns
    and keep only one side of columns

    This is how you merge on eventid, stationid:
    merge = left.merge(
        right, on=["EventId", "StationId"], how="left", suffixes=["", "_drop"]
    )
    """
    merge = left.merge(right, suffixes=["", "_drop"], **kwargs)
    return merge.drop([key for key in merge.keys() if key.endswith("drop")], axis=1)
