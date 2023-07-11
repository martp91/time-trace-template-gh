import numpy as np
from iminuit import Minuit, _core
from collections import defaultdict
from math import ceil
import matplotlib.pyplot as plt

from time_templates.utilities import poisson, atmosphere
from time_templates.misc.energy import SdlgE_resolution
from time_templates.misc.sd_signal import poisson_factor, wcd_total_signal_variance
from time_templates.starttime import (
    fit_start_time,
    start_time_deMauro,
)
from time_templates.signalmodel import signal_model
from time_templates.templates.trace_templates import (
    TimeTraceTemplate,
    calc_chi2_trace_inv_cov,
)
from time_templates.MPD.Xmumax_50 import Xmumax_50_from_Xmax
from time_templates.templates.universality.S1000_model import S1000_comp_model
from time_templates.templates.universality.names import (
    eMUON,
    eEM_PURE,
    eEM_MU,
    eEM_HAD,
    DICT_COMP_COLORS,
)
from time_templates.templates.universality.rho_model import (
    make_X_rho,
    get_rho_comp,
)

from time_templates.misc.Xmax import mean_Xmax_MC, mean_Xmax_data


ATM = atmosphere.Atmosphere()

# Pad in front to make sure t0 is fitted correctly
# And make sure you do not cut these empty bins
# check mask setting
NPAD_FRONT = 100
PAD_IN_FRONT = np.zeros(NPAD_FRONT)

T0_REG_FUDGE = 1


def get_start_time_from_Xmax(r, Xmax, theta, lgE, atm=ATM, hground=1400, is_data=True):
    DXmax = atm.slant_depth_at_height(hground, theta) - Xmax
    Rc = start_time_deMauro.Rc_at_DXmax(
        DXmax, theta, lgE, atm=atm, hground=hground, is_data=is_data
    )
    # t0 in ns wrt plane time front
    t0 = start_time_deMauro.start_time_plane_front_catenary_Rc(r, Rc)
    return t0


# average proton/iron epos-lhc lgE 19-20. when fitting only Xmax with lognormals(?)


def calc_rho_all_comps(theta, DX, r, psi):
    X = make_X_rho(theta, DX, r, psi)
    return (
        get_rho_comp(X, eMUON),
        get_rho_comp(X, eEM_PURE),
        get_rho_comp(X, eEM_MU),
        get_rho_comp(X, eEM_HAD),
    )


v_calc_rho_all_comps = np.vectorize(calc_rho_all_comps)


COMPS = [eMUON, eEM_MU, eEM_HAD, eEM_PURE]  # to get order same for below 2 funcs


def get_total_signals(
    rs,
    psis,
    Rmu,
    lgE,
    Xmax,
    theta,
    atm=ATM,
    Xgs=None,
    Xg_1000=None,
    core_height=1400,
):
    "convenience func when you want to plot smooth templates or something"
    if Xg_1000 is None:
        Xg_1000 = atm.Xg_at_station(
            1000, np.pi / 2, theta, observation_level=core_height
        )
    DX_1000 = Xg_1000 - Xmax

    # Rmu scaling = in S1000_comp_model
    S1000_mu = S1000_comp_model(lgE, DX_1000, theta, Rmu, eMUON)
    S1000_empure = S1000_comp_model(lgE, DX_1000, theta, Rmu, eEM_PURE)
    S1000_emmu = S1000_comp_model(lgE, DX_1000, theta, Rmu, eEM_MU)
    S1000_emhad = S1000_comp_model(lgE, DX_1000, theta, Rmu, eEM_HAD)

    # proj distance to Xmax alongs shower axis for stations
    if Xgs is None:
        # Not exact because no station heights
        Xgs = atm.Xg_at_station(rs, psis, theta, observation_level=core_height)
    DX = Xgs - Xmax
    n = len(rs)
    if n != len(psis):
        raise ValueError
    X = make_X_rho(np.ones(n) * theta, DX, rs, psis, n)

    # Faster than using v_calc_total_signal_comp
    rho_mu = get_rho_comp(X, eMUON)
    rho_empure = get_rho_comp(X, eEM_PURE)
    rho_emmu = get_rho_comp(X, eEM_MU)
    rho_emhad = get_rho_comp(X, eEM_HAD)
    # These are nuissance parameters that multiply the total signal

    Smu_LDF = S1000_mu * rho_mu
    Sem_pure_LDF = S1000_empure * rho_empure
    Sem_mu_LDF = S1000_emmu * rho_emmu
    Sem_had_LDF = S1000_emhad * rho_emhad
    return Smu_LDF, Sem_mu_LDF, Sem_had_LDF, Sem_pure_LDF


def get_ttt(rs, psis, Xmax, theta=0, atm=None, t=None):
    "convenience func when you want to plot smooth templates or something"
    n = len(rs)
    if n != len(psis):
        raise ValueError
    if t is None:
        t = np.arange(0, 6000, 25 / 3)
    out = np.zeros((n, len(t), 4))
    for i in range(n):
        ttt = TimeTraceTemplate(rs[i], psis[i], theta, Xmax, atm=atm, UUB=True)
        for k, comp in enumerate(COMPS):
            out[i, :, k] = ttt.get_wcd_comp_trace_pdf(t, comp)

    return out


class TemplateFit(object):
    """class that get inits with Event dataclass and sets up the time templates
    and allows to fit it with poisson deviance
    """

    def __init__(
        self,
        event,
        station_cuts=None,
        verbose=False,
        do_start_time_fit=True,
        use_Xmumax=False,
        use_data_pulse_shape=True,
    ):
        """__init__.

        Parameters
        ----------
        event : class
            see datareader/event.py
        station_cuts : dict (optional)
            dictionary to cut away stations of the form: {"WCDTotalSignal" : (20, 20000)}
        """

        self.event = event
        self.start_time_event_class = fit_start_time.EventStartTime(event)
        self.MC = self.event.MC
        self.station_cuts = station_cuts
        self.lgE = event.lgE
        if self.lgE < 19:
            print("WARNING: not tested for lgE < 19")
        if self.event.is_data:
            self.Xmax = mean_Xmax_data(10**self.lgE)
        else:
            self.Xmax = (
                mean_Xmax_MC(10**self.lgE, "proton", "EPOS-LHC")
                + mean_Xmax_MC(10**self.lgE, "iron", "EPOS-LHC")
            ) / 2
        self.theta = event.theta
        self.Rmu = 1
        self.Xg = self.event.atm.slant_depth_at_height(
            self.event.core_height, self.theta
        )
        self.DXmax = self.Xg - self.Xmax
        self.Xmumax_50 = Xmumax_50_from_Xmax(self.Xmax, self.theta)
        self.use_Xmumax = use_Xmumax
        self.Xmumax_50_0 = self.Xmumax_50
        self.DeltaXmumaxXmax = 0

        self.verbose = verbose
        self.use_data_pulse_shape = use_data_pulse_shape

        self.minuit = None
        self.fix_templates = False
        self.ndof = 0

        self.reg_t0 = T0_REG_FUDGE
        self.Deltat0s = None  # set to none and fill later
        self.setup_all(1, self.lgE, self.Xmax, 0, do_start_time_fit)
        self.nstations = 0
        self.stations = []
        self.ts = []  # times for wcd trace
        self.t0s = []
        self.t0_errs = []
        self.data = []
        self.rs = []
        self.psis = []
        self.total_signals = []
        self.total_signals_var = []
        self.TTTs = []  # trace time templates
        self.Xgs = []  # station vertical depth projected along shower axis
        self.D = 0
        self.reg = 0

    def __del__(self):
        "del some stuff manually because there are/were some memory leaks"
        print("Deleting ET")
        del self.event
        del self.start_time_event_class
        del self.stations
        del self.ts
        del self.t0s
        del self.t0_errs
        del self.data
        del self.rs
        del self.psis
        del self.total_signals
        del self.total_signals_var
        del self.TTTs
        del self.Xgs

    def setup_all(
        self,
        Rmu,
        lgE,
        Xmax,
        DeltaXmumaxXmax=0,
        do_start_time_fit=False,
        tq_cut=None,
        trace_cut=0,
        neff_cut=0,
    ):
        "Setup some stuff, selects stations etc."
        # Order is important here
        self.setup_stations()
        self.set_fit_values(Rmu, lgE, Xmax, DeltaXmumaxXmax)
        if do_start_time_fit:
            self.fit_start_times()
        self.setup_scale_mask(tq_cut, trace_cut, neff_cut)

    def __repr__(self):
        return repr(self.event)

    def __str__(self):
        # TODO: this should be different and show fit results and stuff
        return str(self.event)

    def print(self, *args):
        if self.verbose:
            print(*args)

    def setup_stations(self):
        """convenience func to setup the stations and ttt"""
        # just make sure to reset these
        self.nstations = 0
        self.stations = []
        self.ts = []  # times for wcd trace
        self.t0s = []
        self.t0_errs = []
        self.data = []
        self.rs = []
        self.psis = []
        self.total_signals = []
        self.total_signals_var = []
        self.TTTs = []  # trace time templates
        self.Xgs = []  # station vertical depth projected along shower axis

        # Some sort of hack
        for station in self.event.stations:
            try:
                del station.TT
            except AttributeError:
                continue
        #            print(station.TT)

        for station in self.event.iter_stations(self.station_cuts):
            ttt = TimeTraceTemplate(
                station.r,
                station.psi,
                self.theta,
                Xmax=self.Xmax,
                UUB=station.UUB,
                station_height=station.station_height,  # OK, 1400m all for sim
                core_height=self.event.core_height,
                use_Xmumax=self.use_Xmumax,
            )
            if self.event.is_data and self.use_data_pulse_shape:
                ttt.WCD_response = signal_model.DetectorTimeResponse(
                    UUB=False, tau=station["WCDPulseShape"]
                )
            t = station.wcd_trace.t  # TODO: only WCD here
            dt = station.wcd_trace.dt
            total_wcd_signal = station.wcd_trace.get_total_signal()

            self.t0_errs.append(
                np.sqrt(
                    start_time_deMauro.start_time_variance(
                        station.r, self.theta, total_wcd_signal, station["WCDRiseTime"]
                    )
                )
            )

            trace = station.wcd_trace.get_total_trace()
            trace[trace < 0] = 0
            if station.UUB:
                NPAD_END = 2048 - len(trace)
            else:
                NPAD_END = 768 - len(trace)
            # pad in front and back to have same length trace always
            # but cut on t < t95 removes end anyway...
            # NPAD_END = 0
            trace = np.insert(trace, 0, PAD_IN_FRONT)
            trace = np.append(trace, np.zeros(NPAD_END))
            t = np.linspace(
                t[0] - dt * NPAD_FRONT,
                t[-1] + NPAD_END * dt,
                len(t) + NPAD_FRONT + NPAD_END,
            )

            self.stations.append(station)
            self.t0s.append(station.tpf)

            self.TTTs.append(ttt)
            self.ts.append(t)
            self.rs.append(station.r)
            self.psis.append(station.psi)
            self.Xgs.append(station.Xg)
            self.data.append(trace)
            self.total_signals.append(total_wcd_signal)
            self.total_signals_var.append(
                wcd_total_signal_variance(total_wcd_signal, self.theta)
            )
            self.nstations += 1

        # Put these in awkward arrays for faster computation in PoissonDeviance
        self.rs = np.array(self.rs)
        self.psis = np.array(self.psis)
        self.Xgs = np.array(self.Xgs)
        # self.data_ak = ak.Array(self.data)  # time sink
        self.t0_errs = np.array(self.t0_errs)
        self.total_signals = np.array(self.total_signals)
        self.total_signals_var = np.array(self.total_signals_var)
        if self.Deltat0s is None:
            self.Deltat0s = np.zeros(self.nstations)

    def setup_scale_mask(self, tq_cut=None, trace_cut=0, neff_cut=0):
        "Set the poisson factor scale from the current fit values"
        self.ndata = 0
        self.scales = []  # scales for neff particles
        self.masks = []
        for i in range(self.nstations):
            Smu = self.Smu_LDF[i]
            Sem = self.Sem_pure_LDF[i]
            Semmu = self.Sem_mu_LDF[i]
            Semhad = self.Sem_had_LDF[i]
            ttt = self.TTTs[i]
            t = self.ts[i]
            trace = self.data[i]

            scale = ttt.get_wcd_scale(t, Smu, Sem, Semmu, Semhad)
            scale[~np.isfinite(scale)] = 0
            # scale = np.maximum(scale, 1e-3)
            trace_pred = ttt.get_wcd_total_trace(t, Smu, Sem, Semmu, Semhad)
            neff = trace_pred * scale

            mask = (trace >= trace_cut) & (neff >= neff_cut)
            if tq_cut is not None:
                cs = np.cumsum(trace_pred)
                cs /= cs[-1]
                tq = np.interp(tq_cut, cs, t)
                mask = mask & (t < tq)

            self.ndata += len(trace[mask])
            self.scales.append(scale)
            self.masks.append(mask)

        # self.scales_ak = ak.Array(self.scales)  # time sink
        # self.masks_ak = ak.Array(self.masks)  # time sink
        # self.data_ak_masked = self.data_ak[self.masks]
        # self.scales_masked = self.scales_ak[self.masks]

    def set_total_signals_comp(self, Rmu, lgE, Xmax):
        """Set S1000 and rho for each component for each station in event
        for given params. LDF signal is stored in self.
        """
        # TODO Could also depend on Xmumax
        # TODO This should be better documented at some point
        DX_1000 = self.Xg - Xmax

        # Rmu scaling = in S1000_comp_model
        self.S1000_mu = S1000_comp_model(lgE, DX_1000, self.theta, Rmu, eMUON)
        self.S1000_empure = S1000_comp_model(lgE, DX_1000, self.theta, Rmu, eEM_PURE)
        self.S1000_emmu = S1000_comp_model(lgE, DX_1000, self.theta, Rmu, eEM_MU)
        self.S1000_emhad = S1000_comp_model(lgE, DX_1000, self.theta, Rmu, eEM_HAD)

        # proj distance to Xmax alongs shower axis for stations
        DX = self.Xgs - Xmax
        if self.nstations > 1:
            X = make_X_rho(
                np.ones(self.nstations) * self.theta,
                DX,
                self.rs,
                self.psis,
                n=self.nstations,
            )
        else:
            X = make_X_rho(self.theta, DX, self.rs, self.psis, n=self.nstations)

        # COuld create Xpoly here just once and gain some speed
        # Faster than using v_calc_total_signal_comp
        rho_mu = get_rho_comp(X, eMUON)
        rho_empure = get_rho_comp(X, eEM_PURE)
        rho_emmu = get_rho_comp(X, eEM_MU)
        rho_emhad = get_rho_comp(X, eEM_HAD)

        self.Smu_LDF = self.S1000_mu * rho_mu
        self.Sem_pure_LDF = self.S1000_empure * rho_empure
        self.Sem_mu_LDF = self.S1000_emmu * rho_emmu
        self.Sem_had_LDF = self.S1000_emhad * rho_emhad

    def yield_expected_trace(
        self,
        Rmu=1,
        lgE=None,
        Xmax=None,
        Xmumax_50=None,
        Deltat0s=None,
    ):
        "yield predicted trace for RMu, lgE, XMax. Deltat0s is length of stations and is the offset from t0"
        if self.nstations <= 0:
            raise RuntimeError("No stations satisfy cutoff")

        if not self.TTTs:
            raise RuntimeError("TimeTemplates not set. Why?")

        if Xmax is None:
            Xmax = self.Xmax
        if Xmumax_50 is None:
            Xmumax_50 = self.Xmumax_50

        if Xmumax_50 < 0:
            print(Xmumax_50, Xmax)
            raise ValueError("Xmumax was below 0, something goes wrong")

        if lgE is None:
            lgE = self.lgE

        if Deltat0s is None:
            Deltat0s = np.zeros(self.nstations)

        if not self.fix_templates:  # if fix templates skip this for some speedup
            self.set_total_signals_comp(Rmu, lgE, Xmax)

        DXmax = self.Xg - Xmax
        Rc = start_time_deMauro.Rc_at_DXmax(
            DXmax,
            self.theta,
            lgE,
            atm=self.event.atm,
            hground=self.event.core_height,
            is_data=self.event.is_data,
        )
        self.expected_traces = []
        for i in range(self.nstations):
            ttt = self.TTTs[i]
            ttt.set_DXmax(Xmax)
            t0 = start_time_deMauro.start_time_plane_front_catenary_Rc(ttt.r, Rc)
            ttt.set_t0(t0 + Deltat0s[i])
            ttt.muon_template.set_Xmumax_from_Xmumax_50(Xmumax_50)
            ttt.set_t_muon_offset(Deltat0s[i])  # correct muon pft
            t = self.ts[i]
            mu = np.maximum(
                ttt.get_wcd_total_trace(
                    t,
                    self.Smu_LDF[i],
                    self.Sem_pure_LDF[i],
                    self.Sem_mu_LDF[i],
                    self.Sem_had_LDF[i],
                ),
                1e-20,
            )
            self.expected_traces.append(mu)
            yield mu

    def get_wcd_event_template(self, *args):
        "Get templates for all stations in list"
        expected_traces = []
        for i, mu in enumerate(self.yield_expected_trace(*args)):
            expected_traces.append(mu)

        self.expected_traces = expected_traces

        return self.expected_traces

    def calc_event_poisson_deviance(self, *args, noscale=False):
        """Calculate deviance for args: fit parameters.
        noscale=True will not scale the signal to effectice particles"""

        total = 0

        for i, mu in enumerate(self.yield_expected_trace(*args)):
            mask = self.masks[i]
            data = self.data[i][mask]
            scale = self.scales[i][mask]
            nt = len(data)

            if nt < 2:
                continue
            if noscale:
                total += poisson.PoissonDeviance(data, mu[mask], nt)
            else:
                total += poisson.PoissonDeviance(data, mu[mask], nt, scale)

        return total

    def calc_event_poisson_deviance_noscale(
        self,
        *args,
    ):
        return self.calc_event_poisson_deviance(*args, noscale=True)

    def set_fit_values(
        self,
        Rmu_fit,
        lgE_fit,
        Xmax_fit,
        DeltaXmumaxXmax_fit,
        Deltat0s=None,
    ):
        "Set parameters in class"

        self.Rmu = Rmu_fit
        self.lgE = lgE_fit

        self.Xmax = Xmax_fit
        self.DeltaXmumaxXmax = DeltaXmumaxXmax_fit
        self.Xmumax_50 = Xmumax_50_from_Xmax(Xmax_fit, self.theta) + DeltaXmumaxXmax_fit

        if Deltat0s is not None:
            self.Deltat0s = Deltat0s

        # This sets S_LDF
        self.set_total_signals_comp(Rmu_fit, lgE_fit, Xmax_fit)
        # This is so fucking inconsistent FIXME
        DXmax = self.Xg - self.Xmax
        self.Rc_ttt_fit = start_time_deMauro.Rc_at_DXmax(
            DXmax,
            self.theta,
            self.lgE,
            atm=self.event.atm,
            hground=self.event.core_height,
            is_data=self.event.is_data,
        )

        # There is still a lot of double code here TODO FIXME

        for i in range(self.nstations):
            ttt = self.TTTs[i]
            ttt.set_DXmax(self.Xmax)
            t0 = start_time_deMauro.start_time_plane_front_catenary_Rc(
                ttt.r, self.Rc_ttt_fit
            )
            ttt.set_t0(t0 + self.Deltat0s[i])
            ttt.muon_template.set_Xmumax_from_Xmumax_50(self.Xmumax_50)
            ttt.set_t_muon_offset(self.Deltat0s[i])  # correct muon pft
            station = self.stations[i]
            station.add_TimeTemplate(
                ttt,
                self.Smu_LDF[i],
                self.Sem_pure_LDF[i],
                self.Sem_mu_LDF[i],
                self.Sem_had_LDF[i],
            )

    def plot_traces(self, plotMC=False):
        axes = self.event.plot_traces(plotMC=plotMC, plotTT=True)
        return axes

    def reset_fit(self):
        self.minuit = None

    def convert_fit_parameters_array(self, x):
        Rmu, lgE, Xmax, DeltaXmumaxXmax = x[:4]
        Deltat0s = np.array(x[4 : 4 + self.nstations])  # slower but oh well
        return Rmu, lgE, Xmax, DeltaXmumaxXmax, Deltat0s

    def calc_regularization(self, x, x0, regs):
        # normal constraint
        n = len(x)
        out = 0
        for i in range(n):
            out += (x[i] - x0[i]) ** 2 * regs[i]

        return out

    def fit(
        self,
        Rmu_0=1,
        lgE_0=None,
        Xmax_0=None,
        DeltaXmumaxXmax_0=0,
        Deltat0s_0=None,
        fix_Rmu=False,
        fix_lgE=False,
        fix_Xmax=False,
        fix_Xmumax=True,
        fix_t0s=True,
        reg_Rmu=0,
        reg_lgE=None,
        reg_Xmax=0,
        reg_DeltaXmumaxXmax=0,
        reg_t0s=None,
        no_scale=True,
        simplex=False,
    ):
        "Call fit routine for time-trace-template fit"
        if no_scale:
            deviance = lambda *x: self.calc_event_poisson_deviance_noscale(*x)
        else:
            deviance = lambda *x: self.calc_event_poisson_deviance(*x)

        if reg_lgE is None:
            reg_lgE = 1 / SdlgE_resolution(lgE_0) ** 2
        if reg_t0s is None:
            reg_t0s = self.reg_t0 / self.t0_errs**2

        regs = [
            reg_Rmu,
            reg_lgE,
            reg_Xmax,
            reg_DeltaXmumaxXmax,
            *reg_t0s,
        ]

        # This should be average Xmax from energy
        if Xmax_0 is None:
            Xmax_0 = self.Xmax

        if lgE_0 is None:
            lgE_0 = self.lgE

        # Set minuit fit parameters
        p0 = [Rmu_0, lgE_0, Xmax_0, DeltaXmumaxXmax_0]
        bounds = [(0.05, 4), (18, 21), (500, 1500), (-400, 400)]
        p_names = ["Rmu", "lgE", "Xmax", "DeltaXmumaxXmax"]
        if Deltat0s_0 is None:
            Deltat0s_0 = self.Deltat0s
        for i in range(self.nstations):
            p0.append(Deltat0s_0[i])
            bounds.append((None, None))
            p_names.append(f"Deltat{i}")

        # Set delta t0 to regularize it to 0 instead of to p0
        p0_reg = p0.copy()
        p0_reg[4:] = self.nstations * [0]

        def minfunc(x):
            "boosted function"
            (
                Rmu,
                lgE,
                Xmax,
                DeltaXmumaxXmax,
                Deltat0s,
            ) = self.convert_fit_parameters_array(x)
            Xmumax_50 = Xmumax_50_from_Xmax(Xmax, self.theta) + DeltaXmumaxXmax
            self.D = deviance(Rmu, lgE, Xmax, Xmumax_50, Deltat0s)
            self.reg = self.calc_regularization(x, p0_reg, regs)

            return self.D + self.reg

        m = Minuit(minfunc, p0, name=p_names)
        self.minuit = m

        # If redo fit with previous result and maybe want to use scale now:
        # have to reset FCN to minfunc(x) with hack
        # makes sure delta t0s are taken from previous fit
        # m._fcn = _core.FCN(minfunc, None, True, 0)
        # m.values["Rmu"] = Rmu_0
        # m.values["lgE"] = lgE_0
        # m.values["Xmax"] = Xmax_0
        # m.values["DeltaXmumaxXmax"] = DeltaXmumaxXmax_0

        m.errordef = 1  # Deviance is 2x likelihood
        try:
            m.limits = bounds
        except ValueError as e:
            print(len(bounds), len(m.values))
            print(bounds)
            print(m)
            raise e
        m.fixed["Rmu"] = fix_Rmu
        m.fixed["lgE"] = fix_lgE

        m.fixed["Xmax"] = fix_Xmax
        m.fixed["DeltaXmumaxXmax"] = fix_Xmumax

        self.fix_templates = False
        if fix_Rmu and fix_lgE and fix_Xmax:
            self.fix_templates = True

        for name in p_names:
            if "Deltat" in name:
                m.fixed[name] = fix_t0s

        if self.nstations < 1:
            return m
        try:
            if simplex:
                m.simplex()
            m.migrad()
        except Exception as e:
            print("FAILED")
            print(self)
            print(m)
            raise e

        if not m.valid or not m.accurate:  # or m.fmin.has_parameters_at_limit:
            print("Not converged trying simplex")
            m.simplex()
            m.migrad()

        if not m.valid or not m.accurate:  # or m.fmin.has_parameters_at_limit:
            print("WARNING still fit did not converge")
            print(self)
            print(m)

        (
            Rmu,
            lgE,
            Xmax,
            DeltaXmumaxXmax,
            Deltat0s,
        ) = self.convert_fit_parameters_array(m.values)

        self.ndof_tt = self.ndata - m.nfit

        self.set_fit_values(
            Rmu,
            lgE,
            Xmax,
            DeltaXmumaxXmax,
            Deltat0s,
        )

        return m

    def calc_goodness_of_fit(self, x, neff_cut=1):
        "From fit, calc deviance(*x) and ndof with neff_cut >= 0"
        # not for fitting
        self.setup_scale_mask(neff_cut=neff_cut)
        (
            Rmu,
            lgE,
            Xmax,
            DeltaXmumaxXmax,
            Deltat0s,
        ) = self.convert_fit_parameters_array(x)
        Xmumax_50 = Xmumax_50_from_Xmax(Xmax, self.theta) + DeltaXmumaxXmax
        self.deviance = self.calc_event_poisson_deviance(
            Rmu,
            lgE,
            Xmax,
            Xmumax_50,
            Deltat0s,
        )
        self.ndof = self.ndata - len(Deltat0s) - 2  # usual but hardcode warning #FIXME
        return self.deviance, self.ndof

    def fit_start_times(self, ax=None):
        "Fit start times all stations"

        # This also sets tpf and r from new axis and core time
        m, chi2, ndof = self.start_time_event_class.fit(
            ax=ax, plt_kws=dict(marker="o", color="k")
        )
        self.theta = self.event.theta
        # m, chi2, ndof = fit_start_time.fit_start_times_event(
        #    self.event, ax=ax, plt_kws=dict(marker="o", color="k")
        # )
        Rc, Ctc, u, v = m.values

        self.Rc_fit = Rc
        self.DXRc_fit = start_time_deMauro.DX_at_Rc(
            Rc, self.theta, self.event.atm, self.event.core_height
        )
        self.var_DXRc_fit = start_time_deMauro.var_DXRc(
            m.errors["Rc"] ** 2, Rc, self.theta, self.event.atm, self.event.core_height
        )

        self.DXmax_Rc_fit = start_time_deMauro.DXmax_at_Rc(
            Rc,
            self.theta,
            self.lgE,
            self.event.atm,
            self.event.core_height,
            is_data=self.event.is_data,
        )
        # On sims uncertainty is not correct
        self.var_DXmax_Rc_fit = start_time_deMauro.var_DXmaxRc(
            m.errors["Rc"] ** 2,
            Rc,
            self.theta,
            self.lgE,
            var_lgE=SdlgE_resolution(self.lgE) ** 2,
            atm=self.event.atm,
            hground=self.event.core_height,
        )

        self.chi2_Rc_fit = chi2
        self.ndof_Rc_fit = ndof

        # set t0 in tracetemplate from model
        # Maybe take tstart if t0 > tstart?
        # Or fit nuissance Deltat0
        self.t0s = []  # make sure empty list
        self.ts = []
        for station in self.stations:
            ttt = station.TT
            t0 = start_time_deMauro.start_time_plane_front_catenary_Rc(station.r, Rc)
            t0 = min(t0, station.tpf)
            self.t0s.append(t0)
            ttt.set_t0(t0)
            # also t for traces has now changed so reset
            t = station.wcd_trace.t  # TODO: only WCD here
            self.ts.append(t)

        self.Xmax_Rc_fit = self.event.Xg - self.DXmax_Rc_fit
        return self.Xmax_Rc_fit

    def fit_total_signals(
        self, lgE_0=None, reg_lgE=None, reg_Rmu=0, fix_lgE=True, ax=None, Xmax=None
    ):
        "Total signal fit. simple LDF fit with scale as set by poisson factor"

        mask = (self.rs > 490) & (self.rs < 2100)

        scale = 1 / poisson_factor(self.theta) ** 2
        neff_data = self.total_signals * scale

        def minfunc(Rmu, lgE, Xmax):
            self.set_total_signals_comp(Rmu, lgE, Xmax)
            Spred = np.maximum(
                self.Smu_LDF + self.Sem_pure_LDF + self.Sem_mu_LDF + self.Sem_had_LDF,
                1e-20,
            )
            # var_pred = wcd_total_signal_variance(Spred, self.theta)
            # scale = Spred / var_pred
            reg = reg_lgE * (lgE - lgE_0) ** 2 + reg_Rmu * (Rmu - 1) ** 2
            return (
                poisson.PoissonDeviance(
                    neff_data[mask], Spred[mask] * scale, len(Spred[mask])
                )
                + reg
            )

        if lgE_0 is None:
            lgE_0 = self.lgE

        if reg_lgE is None:
            reg_lgE = 1 / SdlgE_resolution(lgE_0) ** 2
        if Xmax is None:
            Xmax = self.Xmax
        m = Minuit(minfunc, Rmu=1, lgE=lgE_0, Xmax=Xmax)

        m.limits = [(0.05, 4), (18, 21), (500, 1400)]
        m.fixed["Xmax"] = True
        m.fixed["lgE"] = fix_lgE

        m.migrad()

        Rmu, lgE, Xmax = m.values
        self.set_fit_values(Rmu, lgE, Xmax, self.DeltaXmumaxXmax)
        self.ndof_LDF = self.nstations - m.nfit
        self.Rmu_cov = m.covariance[0, 0]
        if ax is not None:
            rs = [station.r for station in self.event.stations]
            Stots = np.array(
                [station.WCDTotalSignal for station in self.event.stations]
            )
            Svar = wcd_total_signal_variance(Stots, self.theta)

            ax.errorbar(
                rs,
                Stots,
                yerr=np.sqrt(Svar),
                marker="o",
                color="k",
                ls="",
                label="wcd",
            )
            r = np.linspace(500, 2000)
            d = {}
            d[0] = {}
            d[np.pi / 2] = {}
            d[np.pi] = {}

            dd = defaultdict(list)
            d = {}
            for comp in DICT_COMP_COLORS.keys():
                d[comp] = []

            d["total"] = []

            for i, psi in enumerate([0, np.pi / 2, np.pi]):
                Smu, Sem_mu, Sem_had, Sem_pure = get_total_signals(
                    r,
                    np.ones_like(r) * psi,
                    Rmu,
                    lgE,
                    Xmax,
                    self.theta,
                    self.event.atm,
                    core_height=self.event.core_height,
                )
                d[eMUON].append(Smu)
                d[eEM_PURE].append(Sem_pure)
                d[eEM_MU].append(Sem_mu)
                d[eEM_HAD].append(Sem_had)

                d["total"].append(Smu + Sem_pure + Sem_mu + Sem_had)

            for comp in ["total", eMUON, eEM_PURE, eEM_MU, eEM_HAD]:
                ax.plot(
                    r, np.mean(d[comp], axis=0), color=DICT_COMP_COLORS[comp], ls="--"
                )
                ax.fill_between(
                    r,
                    np.min(d[comp], axis=0),
                    np.max(d[comp], axis=0),
                    alpha=0.2,
                    color=DICT_COMP_COLORS[comp],
                    lw=0,
                )

            ax.set_yscale("log")
            # ax.set_xscale("log")
            ax.set_xlabel("$r$ [m]")
            ax.set_ylabel("$S$ [VEM]")
            ax.set_ylim([1e-1, None])
        return m

    # FIXME: below does not work
    # def sample_event(
    #    self,
    #    Rmu=1,
    #    lgE=None,
    #    Xmax=None,
    #    Xmumax_50=None,
    #    Deltat0s=None,
    # ):

    #    if self.nstations <= 0:
    #        raise RuntimeError("No stations satisfy cutoff")

    #    if not self.TTTs:
    #        raise RuntimeError("TimeTemplates not set. Why?")

    #    if Xmax is None:
    #        Xmax = self.Xmax
    #    if Xmumax_50 is None:
    #        Xmumax_50 = self.Xmumax_50

    #    if lgE is None:
    #        lgE = self.lgE

    #    if Deltat0s is None:
    #        Deltat0s = np.zeros(self.nstations)

    #    sampled_traces = []

    #    # Set LDFs
    #    if not self.fix_templates:  # if fix templates skip this for some speedup
    #        self.set_total_signals_comp(Rmu, lgE, Xmax, factorSmu, factorSem)

    #    for i in range(self.nstations):
    #        ttt = self.TTTs[i]
    #        ttt.set_DXmax(Xmax)
    #        ttt.set_t0(self.t0s[i] + Deltat0s[i])
    #        ttt.muon_template.set_Xmumax_from_Xmumax_50(self.Xmumax_50)
    #        ttt.set_t_muon_offset(Deltat0s[i])  # correct muon pft
    #        t = self.ts[i]

    #        sample = ttt.sample_wcd_signal(
    #            t,
    #            self.Smu_LDF[i],
    #            self.Sem_pure_LDF[i],
    #            self.Sem_mu_LDF[i],
    #            self.Sem_had_LDF[i],
    #        ) #        sampled_traces.append(sample)
    #    # sampled_traces = ak.Array(sampled_traces)

    #    return sampled_traces

    # FIXME
    # TODO: fix this with new poisson deviance

    # def calc_poisson_deviance_sample(self, *args, force=False):

    #    # Do not recalculate mu
    #    try:
    #        if force:  # force to reset Rmu, lgE, Xmax
    #            raise AttributeError
    #        mu = self.expected_traces
    #    except AttributeError:
    #        self.setup_scale_mask()
    #        mu = self.get_wcd_event_template(*args)

    #    sample = self.sample_event(*args)

    #    return (
    #        poisson.PoissonDeviance_2D(
    #            sample[self.masks_ak], mu[self.masks_ak], self.scales_masked
    #        ),
    #        sample,
    #    )

    # def calc_chi2_cov(self, *args):
    #    chi2 = 0
    #    ys = self.data_ak
    #    mus = self.get_wcd_event_template(*args)
    #    masks = self.masks_ak

    #    for i, tt in enumerate(self.TTTs):
    #        cov = tt.cov  # should be there
    #        inv_cov = np.linalg.pinv(cov)  # OK?
    #        y = ak.to_numpy(ys[i])
    #        mu = ak.to_numpy(mus[i])
    #        mask = ak.to_numpy(masks[i])

    #        chi2 += calc_chi2_trace_inv_cov(y, mu, inv_cov, mask)

    #    return chi2
