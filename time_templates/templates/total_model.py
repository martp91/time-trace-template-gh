from time_templates.templates.universality.S1000_model import S1000_comp_model
from time_templates.templates.universality.rho_model import (
    make_X_rho,
    get_rho_comp,
)
from time_templates.templates.trace_templates import TimeTraceTemplate

import numpy as np
from time_templates.utilities.atmosphere import Atmosphere
from time_templates.templates.universality.names import (
    DICT_COMP_LABELS,
    eMUON,
    eEM_PURE,
    eEM_MU,
    eEM_HAD,
    DICT_COMP_COLORS,
)
from time_templates.starttime import start_time_deMauro

ATM = Atmosphere()


def total_signal_model(
    r,
    psi,
    theta,
    lgE,
    Xmax,
    Rmu,
    atm=None,
    station_height=1400,
    core_height=1400,
    comps=False,
):
    """Return total signal as predicted by universality model

    Parameters
    ----------
    r : float
        distance to core in shower plane in m
    psi : float
        azimuth in shower plane in rad
    theta : float
        zenith angle in rad
    lgE : float
        log of primary energy log(E/eV)
    Xmax : float
        distance to shower max in g/cm^2
    Rmu : float
        relative number of muons Nmu/Nmu(EPOS, proton, E, theta, Xmax)
    atm : Atmosphere class
        Atmosphere class as from time_templates/utilities/atmosphere.py/Atmopshere, default=None will switch
        to Malargue March
    station_height : float
        station_height in meter [default=1400m]
    core_height : float
        core_height can be different than station height [default=1400m]
    comps : bool
        if True return split into 4 comps [default=False]
    """
    if r < 500 or r > 2000:
        print(
            "WARNING, not tested/made for r < 500 or r > 2000, although some +- is probably OK"
        )
    if lgE < 19:
        print("Warning not tested for lgE < 19")
    if theta > np.deg2rad(53):
        print("Warning not tested for theta > 53 deg")
    if atm is None:
        print("WARNING no atmosphere given, switching to default Malargue March")
        atm = ATM
    Stots = {}
    DXmax = atm.slant_depth_at_height(core_height, theta) - Xmax
    Stot = 0

    DX = atm.DX_at_station(r, psi, theta, Xmax, station_height)
    Xrho = make_X_rho(theta, DX, r, psi, 1)

    for comp in DICT_COMP_LABELS:
        S1000 = S1000_comp_model(lgE, DXmax, theta, Rmu, comp)
        rho = get_rho_comp(Xrho, comp)[0]
        S = S1000 * rho
        Stots[comp] = S
        Stot += S
    if comps:
        return Stots
    else:
        return Stot


def total_trace_model(
    t,
    r,
    psi,
    theta,
    lgE,
    Xmax,
    Rmu,
    atm=ATM,
    station_height=1400,
    core_height=1400,
    comps=False,
    UUB=False,
    is_data=True,
    delta_t0=0,
    return_var=False,
):
    """Return total time-trace signal as predicted by universality model

    Parameters
    ----------
    t : numpy array of float
        time with respect to plane front in ns, make sure the binning is equal. dt should be 25ns for UB or
        25/3 ns for UUB
    r : float
        distance to core in shower plane in m
    psi : float
        azimuth in shower plane in rad
    theta : float
        zenith angle in rad
    lgE : float
        log of primary energy log(E/eV)
    Xmax : float
        distance to shower max in g/cm^2
    Rmu : float
        relative number of muons Nmu/Nmu(EPOS, proton, E, theta, Xmax)
    atm : Atmosphere class
        Atmosphere class as from time_templates/utilities/atmosphere.py/Atmopshere, default=None will switch
        to Malargue March
    station_height : float
        station_height in meter [default=1400m]
    core_height : float
        core_height can be different than station height [default=1400m]
    comps : bool
        if True return split into 4 comps [default=False]
    UUB : bool
        if True use UUB, dt=25/3, important for detector response convolution [default=False]
    is_data : bool
        if is data the parmaetrization for t0(DXmax) is different [default=True]
    delta_t0 : float
        move the start time by so many ns [default=0]
    """
    if r < 500 or r > 2000:
        print(
            "Warning, not tested/made for r < 500 or r > 2000, although some +- is probably OK"
        )
    if lgE < 19:
        print("Warning not tested for lgE < 19")
    if theta > np.deg2rad(53):
        print("Warning not tested for theta > 53 deg")
    if atm is None:
        print("WARNING no atmosphere given, switching to default Malargue March")
        atm = ATM

    dt = t[1] - t[0]
    if UUB and not np.isclose(dt, 25 / 3):
        raise ValueError("UUB = True and dt != 25/3, but it should be")
    if not UUB and not np.isclose(dt, 25):
        raise ValueError("UUB = False (so UB is assumed) and dt != 25 but it should be")
    ttt = TimeTraceTemplate(
        r,
        psi,
        theta,
        Xmax,
        atm=atm,
        station_height=station_height,
        core_height=core_height,
        UUB=UUB,
    )
    DXmax = atm.slant_depth_at_height(core_height, theta) - Xmax
    Rc = start_time_deMauro.Rc_at_DXmax(
        DXmax, theta, lgE, atm=atm, hground=core_height, is_data=is_data
    )
    t0 = start_time_deMauro.start_time_plane_front_catenary_Rc(r, Rc)
    ttt.set_t0(t0 + delta_t0)
    Scomps = total_signal_model(
        r, psi, theta, lgE, Xmax, Rmu, atm, station_height, core_height, comps=True
    )
    traces = {}
    dt = t[1] - t[0]
    if comps:
        for comp in Scomps:
            traces[comp] = Scomps[comp] * dt * ttt.get_wcd_comp_trace_pdf(t, comp)

        return traces
    else:
        sig = ttt.get_wcd_total_trace(
            t, Scomps[eMUON], Scomps[eEM_PURE], Scomps[eEM_MU], Scomps[eEM_HAD]
        )
        if return_var:
            var = ttt.get_variance_wcd_total(
                t, Scomps[eMUON], Scomps[eEM_PURE], Scomps[eEM_MU], Scomps[eEM_HAD]
            )

            return sig, var
        else:
            return sig
