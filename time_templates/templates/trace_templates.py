"""
This module contains a class that can be initialized with r, psi, theta, Xmax
and has some methods to get a prediction for the traces for every component
and some more

Mart Pothast 2022
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags

from numba import njit


from time_templates.signalmodel import signal_model
from time_templates.templates.muon.muon_dsdt import MuondSdt
from time_templates.utilities.atmosphere import Atmosphere
from time_templates.utilities import poisson
from time_templates.utilities.nb_funcs import interp, normalize, make_cdf


from time_templates.templates.universality.lognormal_templates import (
    lognormal_pdf,
    get_interpolated_r_ms_parameters,
    ms_parameters_func,
)
from time_templates.templates.universality.names import (
    DICT_COMP_SIGNALKEY,
    DICT_COMP_COLORS,
    DICT_COMP_LABELS,
    eMUON,
    eEM_PURE,
    eEM_MU,
    eEM_HAD,
)
from time_templates.misc.sd_signal import poisson_factor
from time_templates.MPD.Xmumax_50 import Xmumax_50_from_Xmax
from time_templates.starttime import start_time_deMauro


warnings.filterwarnings("ignore")


# this should probably be moved somewhere else
def calc_chi2_trace_inv_cov(trace, model, inv_cov, mask):
    """ calculated chi2 with inv cov"""
    res = trace - model
    return res[mask] @ (inv_cov @ res)[mask]


# For sampling, deprecated for now
# @njit
# def sample_t(cdf, t, size=1):
#    "Inverse transform sampling"
#    # faster than scipy.interp1d
#    u = np.random.random(int(round(size)))
#    return interp(u, cdf, t)
#
#
# @njit
# def bin_signal(ts, bins):
#    return np.histogram(ts, bins)[0]


class TimeTraceTemplate:
    """TimeTraceTemplate class to predict traces for wcd traces (#TODO ssd)
    """

    def __init__(
        self,
        r,
        psi,
        theta,
        Xmax,
        Xmumax=None,
        atm=None,
        station_height=1400,
        core_height=1400,
        UUB=False,
        use_Xmumax=False,
    ):
        if atm is None:
            self.atm = Atmosphere(model=21, isothermal=True)
        else:
            self.atm = atm  # atmosphere class /utitlities/atmosphere.py

        self.muon_template = MuondSdt(
            r, psi, theta, Xmumax, z_station=station_height, UUB=UUB, atm=self.atm
        )
        if Xmumax is None:
            Xmumax_50 = Xmumax_50_from_Xmax(Xmax)
            self.muon_template.set_Xmumax_from_Xmumax_50(Xmumax_50)

        self.set_geometry(theta, r, psi, station_height, core_height)
        self.UUB = UUB
        self.t0 = 0
        # average wcd signal per muon and em
        # TODO set these somehow, and clarify why these numbers
        self.alpha = 0.04  # em pure
        self.beta = 1 * self.Lwcd
        self.gamma = 0.04  # emmu
        self.delta = 0.04  # emhad
        self.sigma_bl = 0.0001  # bl? should not be zero or you get overflows in scale
        self.TL_68 = signal_model.calc_q68_track_length(self.theta, signal=True)
        self.use_TL = True
        self.f_Stot = poisson_factor(self.theta)
        # TODO set if appropiate
        self.correct_total_signal_uncertainty = True
        if self.UUB:
            self.WCD_response = signal_model.UUB_WCD_t_response
            self.SSD_response = signal_model.UUB_SSD_t_response
            self.dt = 25 / 3
        else:
            self.WCD_response = signal_model.UB_WCD_t_response
            self.SSD_response = signal_model.UB_SSD_t_response
            self.dt = 25
        self.tstart = 0
        self.t0 = 0
        self.use_Xmumax = use_Xmumax
        self.t_muon_offset = 0
        self.interpolated_ms_params = {}
        self.ln_ms_params = {}
        self.set_lognormal_params()
        self.set_DXmax(Xmax)  # also sets DX by default

    def __del__(self):
        del self.WCD_response
        del self.SSD_response
        del self.interpolated_ms_params
        del self.ln_ms_params
        del self.atm

    def set_lognormal_params(self):
        for comp in DICT_COMP_SIGNALKEY.keys():
            self.ln_ms_params[comp] = {}
            self.interpolated_ms_params[comp] = {}
            mparams = get_interpolated_r_ms_parameters(self.r, comp, "m", kind="linear")
            sparams = get_interpolated_r_ms_parameters(self.r, comp, "s", kind="linear")
            self.interpolated_ms_params[comp]["m"] = mparams
            self.interpolated_ms_params[comp]["s"] = sparams

    def get_ms_lognormal_params_comp_DX(self, DX, comp):
        m = ms_parameters_func(DX, *self.interpolated_ms_params[comp]["m"],)
        s = ms_parameters_func(DX, *self.interpolated_ms_params[comp]["s"],)
        # Hack to not get too large, small
        m = min(max(m, 1), 10)
        s = min(max(s, 0.1), 3)
        return m, s

    def set_ms_lognormal_params_DX(self, DX):
        for comp in DICT_COMP_SIGNALKEY.keys():
            m, s = self.get_ms_lognormal_params_comp_DX(DX, comp)
            self.ln_ms_params[comp]["m"] = m
            self.ln_ms_params[comp]["s"] = s

    def __str__(self):
        s = (
            f"TimeTraceTemplate \n r = {self.r} \n psi = {self.psi} \n"
            f"theta = {self.theta} \n Xmax = {self.Xmax} \n"
            f"Xmumax = {self.muon_template.Xmumax}"
        )
        return s

    def set_geometry(self, theta, r, psi, station_height=0, core_height=0):
        self.theta = theta
        if self.theta > np.deg2rad(60):
            print("WARNING: this works for vertical showers and theta > 60")
        self.r = r
        self.psi = psi
        self.chi = self.theta  # for now, chi is the zenith of incoming particles
        self.Lwcd = signal_model.f_Lwcd(self.chi)  # average!
        self.station_height = station_height
        self.core_height = core_height
        self.muon_template.set_geometry(
            theta, r, psi, core_height, station_height
        )  # Also set z_core??
        self.Xg_station = self.atm.Xg_at_station(r, psi, theta, station_height)
        self.Xg_core = self.atm.slant_depth_at_height(core_height, theta)

    def set_t0(self, t0):
        """start time of trace
        wrt plane time front
        For lognormal(t, m, s, t0)

        """
        self.t0 = t0  # this can be changed by model but t0 < tstart

    def set_t0_from_Rc(self, Rc):
        self.set_t0(start_time_deMauro.start_time_plane_front_catenary_Rc(self.r, Rc))

    def set_t0_from_Xmax_Rc(self, Xmax, is_data=False):
        DXmax = self.Xg_core - Xmax
        Rc = start_time_deMauro.Rc_at_DXmax(
            DXmax,
            self.theta,
            19.2,  # TODO: not impelment lgE dependence
            atm=self.atm,
            hground=self.core_height,
            is_data=is_data,
        )
        self.set_t0_from_Rc(Rc)

    def set_t_muon_offset(self, t_muon_offset):
        # This offset is for muon_template
        # Because here there is no t0, so need to shift time if needed
        # By Deltati for each station in EVentTemplate fit
        self.t_muon_offset = t_muon_offset

    def set_DXmax(self, Xmax):
        self.Xmax = Xmax
        # to the core
        self.DXmax = self.Xg_core - self.Xmax
        # to the station along shower axis
        self.DX = self.Xg_station - self.Xmax

        # maybe todo: set t0 here (but de Mauro model is very bad for simulation)

        # lognormal time parameters are set
        self.set_ms_lognormal_params_DX(self.DX)

        return self.DXmax

    def get_ms_lognormal_params_comp(self, comp):
        return self.ln_ms_params[comp]["m"], self.ln_ms_params[comp]["s"]

    # energy_flow means wcd signal deposited without time convolution

    def get_wcd_comp_energy_flow(self, t, comp):
        return normalize(
            lognormal_pdf(t, *self.get_ms_lognormal_params_comp(comp), self.t0),
            self.dt,
        )

    def get_wcd_comp_trace_pdf(self, t, comp):
        # Comment on evaluate convolution:
        # make sure that time is well (20 bins or so by default)
        # before trace start or you get an up going part due to the
        # convolution which is fake
        return signal_model.evaluate_convolution(
            t, lambda t: self.WCD_response(self.get_wcd_comp_energy_flow(t, comp))
        )

    def get_wcd_muon_energy_flow_pdf(self, t):
        if self.use_Xmumax:
            return self.muon_template.ds_dt(t - self.t_muon_offset)
        else:
            return self.get_wcd_comp_energy_flow(t, eMUON)

    def get_wcd_em_energy_flow_pdf(self, t):
        return self.get_wcd_comp_energy_flow(t, eEM_PURE)

    def get_wcd_em_mu_energy_flow_pdf(self, t):
        return self.get_wcd_comp_energy_flow(t, eEM_MU)

    def get_wcd_em_had_energy_flow_pdf(self, t):
        return self.get_wcd_comp_energy_flow(t, eEM_HAD)

    def get_wcd_muon_trace_pdf(self, t):
        return signal_model.evaluate_convolution(
            t, lambda t: self.WCD_response(self.get_wcd_muon_energy_flow_pdf(t)),
        )

    def get_wcd_em_trace_pdf(self, t):
        return self.get_wcd_comp_trace_pdf(t, eEM_PURE)

    def get_wcd_em_mu_trace_pdf(self, t):
        return self.get_wcd_comp_trace_pdf(t, eEM_MU)

    def get_wcd_em_had_trace_pdf(self, t):
        return self.get_wcd_comp_trace_pdf(t, eEM_HAD)

    def get_wcd_total_energy_flow(self, t, Smu, Sem, Semmu, Semhad):
        """
        4 comp universality
        Sums to total signal but without time convolution
        """
        muon = Smu * self.get_wcd_muon_energy_flow_pdf(t)
        em = Sem * self.get_wcd_em_energy_flow_pdf(t)
        emmu = Semmu * self.get_wcd_em_mu_energy_flow_pdf(t)
        emhad = Semhad * self.get_wcd_em_had_energy_flow_pdf(t)
        # Stot = Smu + Sem + Semmu + Semhad
        total = muon + em + emmu + emhad

        return total * self.dt

    # Some small differences between doing convolution once or for every comp
    # when using small t range. Then the convolution is cutoff
    # And because the total signal is fixed this can give small discrepancies
    # But you save ~100 microsec wrt to get_wcd_total_trace_2
    def get_wcd_total_trace(self, t, Smu, Sem, Semmu, Semhad):
        return signal_model.evaluate_convolution(
            t,
            lambda t: self.WCD_response(
                self.get_wcd_total_energy_flow(t, Smu, Sem, Semmu, Semhad)
            ),
        )

    # def get_wcd_total_trace_2(self, t, Smu, Sem, Semmu, Semhad):
    #    dt = t[1] - t[0]
    #    muon = (
    #        Smu * dt * self.get_wcd_muon_trace_pdf(t)
    #    )  # self.muon_template.ds_dt(t)  # no detector
    #    em = Sem * dt * self.get_wcd_em_trace_pdf(t)
    #    emmu = Semmu * dt * self.get_wcd_em_mu_trace_pdf(t)
    #    emhad = Semhad * dt * self.get_wcd_em_had_trace_pdf(t)

    #    return muon + em + emmu + emhad

    # TODO: fix variance calc
    def get_variance_wcd_muon_no_det(self, t, Smu):
        "Smu is total muon signal in VEM charge"
        # S_expect = n_expect * beta
        # var_S = n_expect * beta**2 = S_expect * beta
        # I think the directly below only works with muon_template not lognormal, so comment that out
        beta = self.beta  # * self.muon_template.wcd_signal_per_muon(t)
        # beta = np.maximum(beta, 0.01)
        n = Smu * self.dt * self.get_wcd_muon_energy_flow_pdf(t) / beta
        # TODO: IS this correct? should be: sigL^2 * S^2/L^2 + S*L

        return self.TL_68 ** 2 * (beta / self.Lwcd) ** 2 * n ** 2 + beta ** 2 * n

    # = sig_L^2 * Smu/L^2 + L * Smu
    # return beta ** 2 * n

    def get_variance_wcd_em_no_det(self, t, Sem):
        "Sem is total EM signal in VEM charge"
        # Assume no time dependence of energy per particle (this is wrong)
        # S_expect = n_expect * alpha
        # var_S = n_expect * alpha**2 = S_expect * alpha
        return self.alpha * Sem * self.dt * self.get_wcd_em_energy_flow_pdf(t)

    def get_variance_wcd_em_mu_no_det(self, t, Semmu):
        "Sem is total EM signal in VEM charge"
        # Assume no time dependence of energy per particle (this is wrong)
        # S_expect = n_expect * alpha
        # var_S = n_expect * alpha**2 = S_expect * alpha
        return self.gamma * Semmu * self.dt * self.get_wcd_em_mu_energy_flow_pdf(t)

    def get_variance_wcd_em_had_no_det(self, t, Semhad):
        "Sem is total EM signal in VEM charge"
        # Assume no time dependence of energy per particle (this is wrong)
        # S_expect = n_expect * alpha
        # var_S = n_expect * alpha**2 = S_expect * alpha
        return self.delta * Semhad * self.dt * self.get_wcd_em_had_energy_flow_pdf(t)

    def get_total_variance_wcd_no_det(self, t, Smu, Sem, Semmu, Semhad):
        return (
            self.get_variance_wcd_muon_no_det(t, Smu)
            + self.get_variance_wcd_em_no_det(t, Sem)
            + self.get_variance_wcd_em_mu_no_det(t, Semmu)
            + self.get_variance_wcd_em_had_no_det(t, Semhad)
            # + self.get_variance_S_from_t_uncertainty_prop(t, Smu, Sem, Semmu, Semhad)
            + self.sigma_bl ** 2
        )

    def get_covariance_wcd_var(self, var):
        """Uncertainty propagation
        see wikipedia
        """
        # A = self.WCD_response.make_conv_matrix(len(var))
        nt = len(var)
        A = self.WCD_response.A[:nt, :nt]
        cov = diags(var)
        return A.multiply(cov).multiply(A.T).toarray()

    def get_covariance_wcd(
        self, t, Smu, Sem, Semmu, Semhad,
    ):
        total_var_no_det = self.get_total_variance_wcd_no_det(
            t, Smu, Sem, Semmu, Semhad
        )
        cov = self.get_covariance_wcd_var(total_var_no_det)
        # Normalize to total signal uncertainty
        if self.correct_total_signal_uncertainty:
            var_Stot = self.f_Stot ** 2 * (Smu + Sem + Semmu + Semhad)
            sum_cov = cov.sum()
            cov *= var_Stot / sum_cov

        # self.cov = cov  # Save for later
        return cov

    def get_variance_correct_wout_cov(self, t, Smu, Sem, Semmu, Semhad):
        "Fast and correct way to get only variance wout having to calculate huge matrix"
        total_var_no_det = self.get_total_variance_wcd_no_det(
            t, Smu, Sem, Semmu, Semhad
        )

        nt = len(t)
        A = self.WCD_response.make_conv_matrix(nt)
        A2 = A @ A
        sum_cov = np.sum(A2 * total_var_no_det)
        var_Stot = self.f_Stot ** 2 * (Smu + Sem + Semmu + Semhad)
        del A, A2
        return total_var_no_det * var_Stot / sum_cov

    def get_variance_wcd_total(
        self, t, Smu, Sem, Semmu, Semhad,
    ):
        if self.correct_total_signal_uncertainty:
            return signal_model.evaluate_convolution(
                t,
                lambda t: self.WCD_response(
                    self.get_variance_correct_wout_cov(t, Smu, Sem, Semmu, Semhad),
                    square=True,
                ),
            )
            # this is much slower than above
            # return np.diag(self.get_covariance_wcd(t, Smu, Sem, Semmu, Semhad))
        # more than twice faster when not have to calc covariance
        return signal_model.evaluate_convolution(
            t,
            lambda t: self.WCD_response(
                self.get_total_variance_wcd_no_det(t, Smu, Sem, Semmu, Semhad),
                square=True,
            ),
        )

    def get_wcd_scale(self, t, Smu, Sem, Semmu, Semhad):
        """
        Average signal per particles such that:
            n = S*scale
        """
        return self.get_wcd_total_trace(
            t, Smu, Sem, Semmu, Semhad
        ) / self.get_variance_wcd_total(t, Smu, Sem, Semmu, Semhad)

    def get_wcd_pdf(self, t, Smu, Sem, Semmu, Semhad):
        Stot = Smu + Sem + Semmu + Semhad
        return self.get_wcd_total_trace(t, Smu, Sem, Semmu, Semhad) / (Stot * self.dt)

    def get_wcd_cdf(self, t, Smu, Sem, Semmu, Semhad):
        return make_cdf(self.get_wcd_pdf(t, Smu, Sem, Semmu, Semhad), self.dt)

    def get_pdf_start_time(self, t1, Ntot, only_muons=True, signal=True):
        """
        Extreme value transformation
        See thesis G. de Mauro or A. Schulz
        """
        tend = max(2 * self.r * np.cos(self.theta), 1000)
        t = np.arange(0, tend, self.dt)
        t = t - self.dt / 2
        if signal:
            dsdt = self.get_wcd_pdf(t, 1, 0, 0, 0)
        else:
            # This works best somehow, best correlation with Xmumax
            # without time convolution
            dsdt = self.get_wcd_muon_energy_flow_pdf(t)

        dsdt = np.maximum(dsdt, 1e-20)

        dsdt = dsdt / (dsdt.sum() * self.dt)
        cdf = make_cdf(dsdt, self.dt)
        cdf_t1 = interp(t1, t, cdf)
        dsdt_t1 = interp(t1, t, dsdt)
        # make sure not <= 0 1-cdf
        one_min_cdf_t1 = 1 - cdf_t1
        one_min_cdf_t1 = np.maximum(one_min_cdf_t1, 1e-40)
        pdf = Ntot * one_min_cdf_t1 ** (Ntot - 1) * dsdt_t1
        return pdf

    # DEPRECATED, but do not lose code
    def get_start_time_mean_std(self, Ntot, dt=10):
        tend = max(2 * self.r * np.cos(self.theta), 1000)
        t = np.arange(0, tend, dt)
        pdf = self.get_pdf_start_time(t, Ntot)
        pdf /= np.sum(pdf) * dt
        mean = np.sum(t * pdf) * dt
        var = np.sum((t - mean) ** 2 * pdf) * dt
        std = np.sqrt(var)
        return mean, std

    def calc_poisson_deviance_wcd_trace(
        self, t, Smu, Sem, Semmu, Semhad, trace, cutoff=0,
    ):
        model = np.maximum(self.get_wcd_total_trace(t, Smu, Sem, Semmu, Semhad), 1e-20)
        scale = self.get_wcd_scale(t, Smu, Sem, Semmu, Semhad)
        scale[~np.isfinite(scale)] = 0

        neff = model * scale
        mask = neff >= cutoff
        ndof = len(t[mask])
        if ndof <= 0:
            return 0, ndof

        PD = poisson.PoissonDeviance(trace[mask], model, ndof, scale)
        if not np.isfinite(PD):
            print("poisson deviance gave nan")
            print(self)
            print(t, trace, Smu, Sem, scale)
            raise ValueError("nan found")

        return PD, ndof

    def calc_chi2_trace(self, t, Smu, Sem, Semmu, Semhad, trace, cutoff=0):
        # Deprecated?
        """
        See GAP-Note trace uncertainty 2022
        """

        model = self.get_wcd_total_trace(t, Smu, Sem, Semmu, Semhad)
        neff = model * self.get_wcd_scale(
            t, Smu, Sem, Semmu, Semhad
        )  # = signal**2/sigma**2
        mask = neff > cutoff
        cov = self.get_covariance_wcd(t, Smu, Sem, Semmu, Semhad)
        inv_cov = np.linalg.pinv(cov)
        ndof = len(model[mask])
        return calc_chi2_trace_inv_cov(trace, model, inv_cov, mask), ndof

    def sample_wcd_comp_signal(self, t, Scomp, comp, alpha=1):
        """
        alpha determines the signal [in VEM] per particle
        """
        if Scomp <= 0:
            return np.zeros_like(t)
        n_expect = Scomp * self.dt * self.get_wcd_comp_energy_flow(t, comp) / alpha
        mask = (n_expect < 0) & np.isfinite(n_expect)
        n_expect[mask] = 0
        n = np.random.poisson(n_expect)
        if comp == eMUON and self.use_TL:
            # Preserves signal, but only if alpha = 1*Lwcd(theta)
            # Crazy slow ofcourse
            s = np.zeros(len(n))
            for i, _n in enumerate(n):
                if _n == 0:
                    continue
                s[i] = np.sum(
                    signal_model.sample_track_length_v(
                        self.theta, _n, signal=True, ninterp=100
                    )
                )
        else:
            s = n * alpha
        # No need for eval_convolution here
        return self.WCD_response(s)

    def sample_wcd_em_signal(self, t, Sem):
        return self.sample_wcd_comp_signal(t, Sem, eEM_PURE, self.alpha)

    def sample_wcd_muon_signal(self, t, Smu):
        return self.sample_wcd_comp_signal(t, Smu, eMUON, self.beta)

    def sample_wcd_em_mu_signal(self, t, Semmu):
        return self.sample_wcd_comp_signal(t, Semmu, eEM_MU, self.gamma)

    def sample_wcd_em_had_signal(self, t, Semhad):
        return self.sample_wcd_comp_signal(t, Semhad, eEM_HAD, self.delta)

    def sample_wcd_signal(self, t, Smu, Sem, Semmu, Semhad):
        """Poisson = True is slightly faster, do not see any
       large differences"""

        # TODO
        # Comment: when sampling the sigma_t from the time variance model is overestimated
        # because this also incorporates the inherent fluctuations of particle arrivals
        # but this fluctuation is also in the sampling
        t_rand = t  # + np.random.normal(0, self.sigma_t)
        return (
            self.sample_wcd_muon_signal(t_rand, Smu)
            + self.sample_wcd_em_signal(t_rand, Sem)
            + self.sample_wcd_em_mu_signal(t_rand, Semmu)
            + self.sample_wcd_em_had_signal(t_rand, Semhad)
            + np.random.normal(0, self.sigma_bl, len(t))
        )

    def plot_wcd(self, t, Smu, Sem, Semmu, Semhad, ax=None):
        if ax is None:
            _, ax = plt.subplots(1)

        Smu_t = Smu * self.dt * self.get_wcd_muon_trace_pdf(t)
        Sem_t = Sem * self.dt * self.get_wcd_em_trace_pdf(t)
        Semmu_t = Semmu * self.dt * self.get_wcd_em_mu_trace_pdf(t)
        Semhad_t = Semhad * self.dt * self.get_wcd_em_had_trace_pdf(t)

        y = Smu_t + Sem_t + Semmu_t + Semhad_t
        ax.plot(t, y, "k--", label="WCD model")
        var = self.get_variance_wcd_total(t, Smu, Sem, Semmu, Semhad)
        yerr = np.sqrt(var)
        yerr = np.where(yerr > y, y, yerr)  # clip this for very small y
        y16, y84 = poisson.poisson_68CI(y, yerr)
        ax.fill_between(t, y16, y84, color="k", alpha=0.2, lw=0)

        ax.plot(t, Smu_t, "--", color=DICT_COMP_COLORS[eMUON])
        ax.plot(t, Semmu_t, "--", color=DICT_COMP_COLORS[eEM_MU])
        ax.plot(t, Semhad_t, "--", color=DICT_COMP_COLORS[eEM_HAD])
        ax.plot(t, Sem_t, "--", color=DICT_COMP_COLORS[eEM_PURE])
        # ax.plot(self.t0, 0, "mX", label="t0")
        ax.set_xlabel("$t-t_{\\rm pf}$ [ns]")
        ax.set_ylabel("Signal [VEM]")

        return ax
