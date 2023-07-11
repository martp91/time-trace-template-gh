"""
Simple fitting algorithm for start time from de Mauro thesis
Takes about 2 per event on laptop
Slowest is getting data atmosphere from database

Mart Pothast 2022
"""
import numpy as np
from iminuit import Minuit
from numba import njit
import pandas as pd
import scipy.stats as scistats
from time_templates.starttime.start_time_deMauro import (
    v_start_time_variance,
    start_time_variance,
    C0,
    Rc_at_DXmax,
    start_time_plane_front_catenary_Rc,
)
from time_templates.misc.Xmax import mean_Xmax_data, mean_Xmax_MC
from time_templates.utilities import atmosphere, geometry


@njit(cache=True, nogil=True, fastmath=True)
def plane_time_residual(Cti, xi, Ctc, axis):
    """
    In units of meter
    """
    Cptf = np.sum(xi * axis, axis=1)  # dot prodcut but faster?
    return Cptf - (Ctc - Cti)


# This is from Offline and holds for spherical shower front so catenary does not work here
# stationToCenter = -xi + Rc * axis
# rStation = np.linalg.norm(stationToCenter, axis=1)
# time_res = (Cti - rStation) - (Ctc - Rc)
# chi2 = np.sum(time_res ** 2 / Ct_var)
@njit(cache=True, nogil=True, fastmath=True)
def curve_time_residual(Cti, xi, Rc, Ctc, axis):
    Cptf = np.sum(xi * axis, axis=-1)  # dot prodcut but faster?
    r = np.sqrt(np.sum(xi * xi, axis=-1) - Cptf ** 2)  # this is correct
    Ct_Rc = Rc * np.cosh(r / Rc) - Rc
    return Cptf - Ct_Rc - (Ctc - Cti)


@njit(cache=True, nogil=True, fastmath=True)
def chi2_start_times(Rc, Ctc, u, v, Cti, xi, Ct_var):
    """This function minimes the start time residuals for Rc, tc and axis

    xi, rs, ti, t_var can(should) be arrays

    Parameters
    ----------
    Rc :
        Radius of curvature (or whatever goes into catenary)
    tc : float
        core time in ns
    axis : vector(3) array
        axis
    ti : array like
        start times in ns
    xi : vector(3, n) array
        stationpos-core 
    rs : (n) array
        distance to core in shower plane
    t_var : (n) array
        time variance from model

    Returns : 
        p: array(4)
            fitted parameters [Rc, tc, u, v]
        red_chi2: float
    """
    w = np.sqrt(np.maximum(1 - u ** 2 - v ** 2, 0))
    axis = np.array([u, v, w])  # , dtype=float)  # time drain
    Ctres = curve_time_residual(Cti, xi, Rc, Ctc, axis)
    return np.sum(Ctres ** 2 / Ct_var)


def fit_start_times(
    xi, axis, Cti, Ct_var, Ctc_0, Rc_0=1e4, fix_Rc=False, fix_axis=False, fix_tc=False,
):

    m = Minuit(
        lambda Rc, Ctc, u, v: chi2_start_times(Rc, Ctc, u, v, Cti, xi, Ct_var),
        Rc=Rc_0,
        Ctc=Ctc_0,
        u=axis[0],
        v=axis[1],
    )
    m.limits = [(1e3, 1e5), (-1e5, 1e5), (-1, 1), (-1, 1)]
    m.errors = (10, 10, 0.01, 0.01)  # set initial step size similar to offline
    m.errordef = 1

    m.fixed = False
    m.fixed["Rc"] = fix_Rc
    m.fixed["Ctc"] = fix_tc
    m.fixed["u"] = fix_axis
    m.fixed["v"] = fix_axis

    # Doesnt matter if you fix Rc first and then fit again
    #    m.fixed["Rc"] = True
    #    m.migrad()
    #    m.fixed = False
    m.migrad()

    return m


class EventStartTime:
    def __init__(self, event):
        self.event = event
        core = event.core
        self.axis = event.axis
        theta = event.theta
        rs = event["r"]
        tiS = event["TimeS"]
        tiNS = event["TimeNS"]
        tcS = event["SdCoreTimeS"]
        tcNS = event["SdCoreTimeNS"]

        tiS_mean = np.mean(tiS)
        tiNS_mean = np.mean(tiNS)
        tiS = tiS - tiS_mean
        tiNS = tiNS - tiNS_mean
        tcS = tcS - tiS_mean
        tcNS = tcNS - tiNS_mean
        ti = tiS * 1e9 + tiNS
        tc_0 = tcS * 1e9 + tcNS

        self.RT = event["WCDRiseTime"]
        self.S = event["WCDTotalSignal"]
        station_pos = event["station_pos"]

        #    RC_0 = event.Xmax #TODO set RC start value
        self.Rc_0 = Rc_at_DXmax(
            event.get_distance_Xmax(mean_Xmax_data(10 ** event.lgE)),
            event.theta,
            event.lgE,
            event.atm,
            event.core_height,
        )

        # This can change in the fit because r changes
        t_var = v_start_time_variance(rs, theta, self.S, self.RT, self.event.nstations)
        # t_var = np.array(
        #    [start_time_variance(rs[i], theta, S[i], RT[i]) for i in range(nstations)]
        # )
        self.xi = station_pos - core

        self.Cti = C0 * ti
        self.Ctc_0 = C0 * tc_0
        self.Ct_var = C0 ** 2 * t_var

    def chi2_func(self, Rc, Ctc, u, v):
        return chi2_start_times(Rc, Ctc, u, v, self.Cti, self.xi, self.Ct_var)

    def fit(self, fix_Rc=False, fix_axis=False, fix_tc=False, ax=None, plt_kws=dict()):
        m = fit_start_times(
            self.xi,
            self.axis,
            self.Cti,
            self.Ct_var,
            self.Ctc_0,
            self.Rc_0,
            fix_Rc,
            fix_axis,
            fix_tc,
        )
        # calc with fit values and renew start time var
        Rc, Ctc, u, v = m.values
        axis = np.array([u, v, np.sqrt(1 - u ** 2 - v ** 2)])
        theta = np.arccos(axis[-1])
        xi = self.xi
        Cti = self.Cti
        # axis /= np.linalg.norm(axis)
        Cptf = np.sum(xi * axis, axis=1)  # dot prodcut but faster
        r = np.sqrt(np.sum(xi * xi, axis=1) - Cptf ** 2)  # this is correct
        t_var = v_start_time_variance(r, theta, self.S, self.RT, self.event.nstations)
        Ct_var = C0 ** 2 * t_var
        Ct_pf_res = plane_time_residual(Cti, xi, Ctc, axis)
        Ct_curv_res = curve_time_residual(Cti, xi, Rc, Ctc, axis)

        chi2_new = np.sum(Ct_curv_res ** 2 / Ct_var)
        ndof = self.event.nstations - m.nfit

        # Checks:
        if abs(np.dot(axis, self.event.axis) - 1) > 0.01:
            print("Warning axis changed significantly", axis, self.event.axis)
            print("using old axis")
        else:
            self.event.set_geometry(self.event.core, axis)
        if abs(self.Ctc_0 - Ctc) > 100 * C0:
            print("Warning core time changes significantly", Ctc, self.Ctc_0)
            print("Using old core time")
        else:
            for i, station in enumerate(self.event.stations):
                # no need to correct for barytime, only difference matters
                station.set_plane_front_time(
                    Cti[i] / C0, Ctc / C0, axis, self.event.core
                )

        if ax is not None:
            pl = ax.errorbar(r, Ct_pf_res / C0, yerr=np.sqrt(t_var), ls="", **plt_kws)
            rspace = np.linspace(np.minimum(r[0], 300), np.maximum(r[-1], 2000))
            ax.plot(
                rspace,
                start_time_plane_front_catenary_Rc(rspace, Rc),
                "-",
                color=pl[0].get_color(),
            )
            ax.plot(
                self.event["Sdr"],
                -self.event["PlaneTimeRes"],
                color=pl[0].get_color(),
                marker="X",
                ls="",
            )
            ax.set_xlabel("r [m]")
            ax.set_ylabel("t-tpf [ns]")

        return m, chi2_new, ndof


# DEPRECATED?
def fit_start_times_event(
    event, fix_Rc=False, fix_axis=False, fix_tc=False, ax=None, plt_kws=dict()
):
    """fit_start_times.

    Parameters
    ----------
    event : class
        event class from datareader
    """
    return EventStartTime(event).fit()


def fit_start_times_df_event(df_event):
    """fit_start_times.
    Assumes NC atmosphere

    Checked: gives same fit as function above

    Parameters
    ----------
    df_event : pandas dataframe
        df_event for 1 event
    """
    df_event = df_event.loc[df_event["IsCandidate"].astype(bool)]
    core = df_event.iloc[0][["SdCore.fX", "SdCore.fY", "SdCore.fZ"]].values.astype(
        float
    )
    station_pos = df_event[
        ["StationPos.fX", "StationPos.fY", "StationPos.fZ"]
    ].values.astype(float)
    axis = df_event.iloc[0][["SdAxis.fX", "SdAxis.fY", "SdAxis.fZ"]].values.astype(
        float
    )
    rs = df_event["Sdr"].values
    theta = df_event.iloc[0]["SdTheta"]
    tiS = df_event["TimeS"].values
    tiNS = df_event["TimeNS"].values
    tcS = df_event.iloc[0]["SdCoreTimeS"]
    tcNS = df_event.iloc[0]["SdCoreTimeNS"]

    tiS_mean = np.mean(tiS)
    tiNS_mean = np.mean(tiNS)
    tiS = tiS - tiS_mean
    tiNS = tiNS - tiNS_mean
    tcS = tcS - tiS_mean
    tcNS = tcNS - tiNS_mean
    ti = tiS * 1e9 + tiNS
    tc_0 = tcS * 1e9 + tcNS

    nstations = len(rs)
    S = df_event["WCDTotalSignal"].values
    RT = df_event["WCDRiseTime"].values
    t_var = v_start_time_variance(rs, theta, S, RT, nstations)
    xi = (station_pos - core).astype(float)

    lgE = df_event.iloc[0]["SdlgE"]
    DXmax = atmosphere.slant_depth_isothermal(1400, theta, 21) - mean_Xmax_MC(
        10 ** lgE, "proton", "EPOS-LHC",
    )
    Rc_0 = Rc_at_DXmax(DXmax, theta, lgE, hground=1400)

    Cti = C0 * ti
    Ctc_0 = C0 * tc_0
    Ct_var = C0 ** 2 * t_var

    m = fit_start_times(xi, axis, Cti, Ct_var, Ctc_0, Rc_0)

    ndof = nstations - m.nfit

    Rc, Ctc, u, v = m.values
    Rc_err = m.errors["Rc"]
    axis = np.array([u, v, np.sqrt(1 - u ** 2 - v ** 2)])
    theta = np.arccos(axis[-1])
    Cptf = np.sum(xi * axis, axis=1)
    r = np.sqrt(np.sum(xi * xi, axis=1) - Cptf ** 2)
    t_var = np.array(
        [start_time_variance(r[i], theta, S[i], RT[i]) for i in range(nstations)]
    )
    Ct_var = C0 ** 2 * t_var
    Ct_pf_res = plane_time_residual(Cti, xi, Ctc, axis)
    Ct_curv_res = curve_time_residual(Cti, xi, Rc, Ctc, axis)
    chi2_new = np.sum(Ct_curv_res ** 2 / Ct_var)
    tpf_res = Ct_pf_res / C0

    pval = scistats.chi2.sf(chi2_new, ndof)
    success = (
        m.valid
        and not m.fmin.has_parameters_at_limit
        and not m.fmin.has_reached_call_limit
        and not m.fmin.is_above_max_edm
    )

    # psi = [
    #    geometry.calc_shower_plane_angle(station_pos[i], core, axis)
    #    for i in range(nstations)
    # ]

    core_time = Ctc / C0
    core_time_NS = core_time + tiNS_mean
    core_timeS = tiS_mean  # + core_time / 1e9

    return (
        Rc,
        Rc_err,
        core_timeS,
        core_time_NS,
        axis[0],
        axis[1],
        axis[2],
        pval,
        success,
    )


names = [
    "Rc_fit",
    "Rc_fit_err",
    "Rc_fit_core_time_S",
    "Rc_fit_core_time_NS",
    "Rc_fit_axis.fX",
    "Rc_fit_axis.fY",
    "Rc_fit_axis.fZ",
    "Rc_fit_pval",
    "Rc_fit_success",
]


def fit_start_times_groupby(df_event, verbose=True):
    """Use this on df.groupby('EventId').apply(fit_start_times_groupby)
    """
    if verbose:
        print("\rAt", df_event.index[0][0], end="")
    return pd.Series(fit_start_times_df_event(df_event), index=names)


def fit_start_times_df(df, verbose=True):
    print("Fit start times. Takes some time...")
    df = df.reset_index().set_index(["EventId", "StationId"])
    print("Number of events", df.index.get_level_values(level=0).nunique())
    print("Doing rc fit")
    df_Rc = df.groupby("EventId").apply(fit_start_times_groupby, verbose)
    print("joining Rc fit with rest")
    df = df.join(df_Rc)
    axis = df[["Rc_fit_axis.fX", "Rc_fit_axis.fY", "Rc_fit_axis.fZ"]].values.astype(
        float
    )
    core = df[["SdCore.fX", "SdCore.fY", "SdCore.fZ"]].values.astype(float)
    spos = df[["StationPos.fX", "StationPos.fY", "StationPos.fZ"]].values.astype(float)
    x = spos - core
    Delta = np.einsum("ij,ij->i", x, axis)
    df["Sdr_new"] = np.sqrt(np.sum(x ** 2, axis=-1) - Delta ** 2)
    df["SdCosTheta_new"] = df["Rc_fit_axis.fZ"]
    df["SdSecTheta_new"] = 1 / df["SdCosTheta_new"]

    barytime_S = df["TimeS"].groupby("EventId").mean()
    barytime_NS = df["TimeNS"].groupby("EventId").mean()
    ti = (df["TimeS"] - barytime_S) * 1e9 + (df["TimeNS"] - barytime_NS)
    tc = (df["Rc_fit_core_time_S"] - barytime_S) * 1e9 + (
        df["Rc_fit_core_time_NS"] - barytime_NS
    )
    ptf_res = Delta / C0 - (tc - ti)
    df["PlaneTimeRes_new"] = -ptf_res
    df["t0_wrt_pft_Rc_fit"] = start_time_plane_front_catenary_Rc(
        df["Sdr_new"], df["Rc_fit"]
    )

    df["Sdpsi_new"] = [
        geometry.calc_shower_plane_angle(spos[i], core[i], axis[i])
        for i in range(len(df))
    ]
    df["Sdcospsi_new"] = np.cos(df["Sdpsi_new"])
    print()
    return df
