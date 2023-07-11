"""
Only vectorized works
too bad
"""
import os
import warnings
import numpy as np
import warnings

# from scipy.integrate import quad, quad_vec
from numba import njit

from time_templates.utilities.interpolate3d import interp3d
from time_templates import package_path
from time_templates.signalmodel import signal_model
from time_templates.utilities import atmosphere
from time_templates.utilities.nb_funcs import normalize
from time_templates.utilities.constants import *

data_path = package_path + "/data/"

ATM = atmosphere.Atmosphere(model=21, isothermal=True)

warnings.filterwarnings("ignore")
# only interpolated with default model params

# integral done in templates/integral for muon signal.ipynb
# Evem = 317 + 105 MeV (from fit to detector response)
model_params = {
    "gamma": 2.6,
    "kappa": 0.8,
    "m": 0.105,
    "pa": 0.0002,
    "Q": 0.17,
    "C0": C0,
    "Evem": 0.422,
}

lookup_Lmu = np.load(data_path + "LXmu_model_lookup_table.npz")
thetas = np.arccos(np.sqrt(lookup_Lmu["ct2"]))
THETA_MIN = thetas.min()
THETA_MAX = thetas.max()

interp3d_p0 = interp3d(
    lookup_Lmu["logr"],
    lookup_Lmu["ct2"],
    lookup_Lmu["cp"],
    lookup_Lmu["p0"],
    vectorized=False,
    fill_value=np.nan,
)
interp3d_p1 = interp3d(
    lookup_Lmu["logr"],
    lookup_Lmu["ct2"],
    lookup_Lmu["cp"],
    lookup_Lmu["p1"],
    vectorized=False,
    fill_value=np.nan,
)


def get_Lmu_for_Xmumax_50(Xmumax_50, theta, r, psi, atm=ATM, hground=1400):
    return (
        atm.height_at_slant_depth(Xmumax_50, theta)
        - (hground + r * np.sin(theta) * np.cos(psi))
    ) / np.cos(theta)


def get_Lmu_p0_p1(theta, r, psi):
    # Means: get Xmumax that goes into muon_dsdt model for Lmu(distance to Xmumax_50)
    ct2 = np.cos(theta) ** 2
    cp = np.cos(psi)
    logr = np.log10(r)
    p0 = interp3d_p0(logr, ct2, cp)
    p1 = interp3d_p1(logr, ct2, cp)
    return p0, p1


def get_Xmumax_for_Lmu_p0_p1(Lmu, p0, p1):
    return p0 * np.exp(p1 * Lmu / 7000)


def get_Xmumax_for_Xmumax_50(Xmumax_50, theta, r, psi, atm=ATM, hground=1400):
    Lmu = get_Lmu_for_Xmumax_50(Xmumax_50, theta, r, psi, atm, hground)
    p0, p1 = get_Lmu_p0_p1(theta, r, psi)
    return get_Xmumax_for_Lmu_p0_p1(Lmu, p0, p1)


muon_energy_integral_filename = (
    f'I_integral_vs_l_r_Evem_gamma_{model_params["gamma"]:.1f}.npz'
)
LOOKUP_MUON_ENERGY_INTEGRAL_WCD = np.load(
    data_path + "/muon_E_integrals/" + muon_energy_integral_filename
)

# be aware the interpolation is only valid for
# kappa = 0.8, m=0.105, pa=0.0002, Q=0.17
MUON_ENERGY_INTEGRAL_WCD = interp3d(
    LOOKUP_MUON_ENERGY_INTEGRAL_WCD["lgl"],
    LOOKUP_MUON_ENERGY_INTEGRAL_WCD["lgr"],
    LOOKUP_MUON_ENERGY_INTEGRAL_WCD["Evem"],
    LOOKUP_MUON_ENERGY_INTEGRAL_WCD["array"],
)


def f_lambda_Xmumax_theta(Xmumax, theta):
    # proton lgE19.5-20 0.6 < costheta < 0.8
    p = [72.5, -43.2, -0.0351, -0.133]
    p00, p01, p10, p11 = p
    cos_theta_sq = np.cos(theta) ** 2
    p0 = p00 + p01 * cos_theta_sq
    p1 = p10 + p11 * cos_theta_sq
    lam = p0 + p1 * (Xmumax - 600)
    lam = min(80, max(lam, 5))
    return lam


@njit(fastmath=True, nogil=True, cache=True, error_model="numpy")
def integral_over_E(x, x0, gamma, kappa):
    return x ** (2 - gamma) * (1 - x0 / x) ** kappa * np.exp(-x)


@njit(fastmath=True, nogil=True, cache=True, error_model="numpy")
def GaisserHillas(X, Xmax, lam, X0):
    "Gaisser Hillas function"
    x = (X - X0) / lam
    m = (Xmax - X0) / lam
    return np.where(x > 0, np.exp(m * np.log(x) - x), 0)


@njit(fastmath=True, nogil=True, cache=True, error_model="numpy")
def epsilon_r_z(z, r):
    """ kinematical delay approx, does not really work for vertical showers"""
    lgz = np.log10(z)
    lgz = np.maximum(lgz, 3.2)
    lgz = np.minimum(lgz, 6)
    p0 = -0.6085 + lgz * (1.955 + lgz * (-0.3299 + lgz * 0.0186))
    return 10 ** p0 * r ** -1.1758


@njit(fastmath=True, nogil=True, cache=True, error_model="numpy")
def z_t(t, r, Delta, c0=0.2998, use_kinematic_delay=0):
    r2 = r * r
    ct = c0 * t
    zmdelta = 0.5 * (r2 / ct - ct)
    zmdelta[zmdelta < 0] = 0
    # Does not work because derivative is now not correct??
    for _ in range(use_kinematic_delay):
        eps = epsilon_r_z(zmdelta, r)
        dcte = ct - eps * 0.5 * r2 / np.sqrt(r2 + zmdelta ** 2)
        zmdelta = 0.5 * (r2 / dcte - dcte)
        zmdelta[zmdelta < 0] = 0
    return zmdelta + Delta


class MuondSdt:
    """
    Muon dS dt
    for WCD and SSD
    integrating over the energy spectrum that is
    given by MPD
    """

    # X0= -45 from https://arxiv.org/pdf/1407.5919.pdf, average between
    # proton iron
    # just use X0=0 for simplicity, hardly matters
    def __init__(
        self,
        r,
        psi,
        theta,
        Xmumax=None,
        lam=None,
        X0=0,
        z_core=1400,
        z_station=1400,
        atm=ATM,
        model_params=model_params,
        use_interpolated_integrals=True,
        UUB=True,
        Xmax=None,
    ):
        self.atm = atm
        self.UUB = UUB
        self.use_interpolated_integrals = use_interpolated_integrals
        self.use_kinematic_delay = 0

        self.set_model_params(model_params)
        self.set_geometry(theta, r, psi, z_core, z_station)
        self.Xmumax = Xmumax
        if Xmumax is None:
            self.set_Xmumax_from_Xmumax_50(600)
        self.lam = lam
        if lam is None:
            self.lam = f_lambda_Xmumax_theta(self.Xmumax, self.theta)
        self.X0 = X0

    def __repr__(self):
        s = ""
        s += f"Xmumax = {self.Xmumax:.2f} \n"
        s += f"lambda = {self.lam:.2f} \n"
        return s

    def __str__(self):
        return str(repr(self))

    def set_Xmumax_lambda(self, Xmumax):
        self.Xmumax = Xmumax
        self.lam = f_lambda_Xmumax_theta(Xmumax, self.theta)

    def set_Xmumax_from_Lmu_50(self, Lmu):
        self.set_Xmumax_lambda(get_Xmumax_for_Lmu_p0_p1(Lmu, self.p0_Lmu, self.p1_Lmu))

    def set_Xmumax_from_Xmumax_50(self, Xmumax_50):
        self.Xmumax_50 = Xmumax_50
        self.Lmu = get_Lmu_for_Xmumax_50(
            self.Xmumax_50, self.theta, self.r, self.psi, self.atm, self.z_station
        )
        self.set_Xmumax_from_Lmu_50(self.Lmu)

    def set_gamma(self, gamma):
        """set gamma custom new inteporlation, slow
        do not change, because changes Xmumax, lambda param
        """
        muon_energy_integral_filename = f"I_integral_vs_l_r_Evem_gamma_{gamma:.1f}.npz"
        LOOKUP_MUON_ENERGY_INTEGRAL_WCD = np.load(
            data_path + "/muon_E_integrals/" + muon_energy_integral_filename
        )

        # Bad coding sorry. Setting global variable
        MUON_ENERGY_INTEGRAL_WCD = interp3d(
            LOOKUP_MUON_ENERGY_INTEGRAL_WCD["lgl"],
            LOOKUP_MUON_ENERGY_INTEGRAL_WCD["lgr"],
            LOOKUP_MUON_ENERGY_INTEGRAL_WCD["Evem"],
            LOOKUP_MUON_ENERGY_INTEGRAL_WCD["array"],
        )
        self.gamma = gamma

    def set_model_params(self, params_dict):
        self.gamma = params_dict["gamma"]
        self.kappa = params_dict["kappa"]
        self.m = params_dict["m"]
        self.pa = params_dict["pa"]
        self.Q = params_dict["Q"]
        self.C0 = params_dict["C0"]
        self.Evem = params_dict["Evem"]
        if self.Evem == 0:
            self.Evem = self.m
        self.m2 = self.m ** 2

    def set_geometry(self, theta, r, psi, z_core=0, z_station=0):
        self.theta = theta
        # if self.theta < THETA_MIN or self.theta > THETA_MAX:
        #    print(f"theta = {self.theta}")
        #    raise ValueError(f"Theta should be between {THETA_MIN} - {THETA_MAX}")
        self.cos_theta = np.cos(theta)
        self.r = r
        # if self.r < 500 or self.r > 2000:
        #    print(f"r = {self.r}")
        #    raise ValueError(f"r should be between 500 - 2000 m")
        self.r2 = r * r
        self.psi = psi
        self.z_core = z_core
        self.z_station = z_station
        # Almost always within a few percent. Actually should do
        # Delta = (station-core).dot(axis) with vectors xyz TODO
        self.Delta = (
            r * np.tan(theta) * np.cos(psi) + (z_station - z_core) / self.cos_theta
        )
        self.lgr = np.log10(r)
        self.p0_Lmu, self.p1_Lmu = get_Lmu_p0_p1(
            self.theta, self.r, self.psi
        )  # this is fixed

    def X_h(self, h):
        "Simple isothermal atmosphere. h above height Malargue"
        return self.atm.slant_depth_at_height(h, self.theta)

        # return atmosphere.slant_depth_isothermal(
        #    h - self.z_core, self.theta, self.atm_model
        # )

    def dNdX(self, X):
        "Gaisser Hillas function"
        return GaisserHillas(X, self.Xmumax, self.lam, self.X0)

    def z_t(self, t):
        return z_t(t, self.r, self.Delta, self.C0, self.use_kinematic_delay)

    def l_z(self, z):
        return np.sqrt((z - self.Delta) ** 2 + self.r2)

    def muon_energy_integral(self, lgl, Evem):
        if self.use_interpolated_integrals:
            I = MUON_ENERGY_INTEGRAL_WCD(lgl, self.lgr, Evem)
        else:
            # TODO: this does not work because limit is vector
            raise NotImplementedError
        # This is the result of integrating over the energy spectrum on the ground
        # using the detector response of the wcd as min((E/Evem)^2, TL)
        # with TL the signal due to track length TL is not implemented here
        return I

    def wcd_signal_per_muon(self, t, Evem=None):
        if Evem is None:
            Evem = self.Evem
        z = self.z_t(t)
        l = self.l_z(z)
        lgl = np.log10(l)

        I0 = self.muon_energy_integral(lgl, self.m)  # signal per muon = 1, SSD
        I1 = self.muon_energy_integral(lgl, Evem)

        return np.where(I0 > 0, I1 / I0, 0)

    def ds_dt(self, t, dndt=False):
        """
        This is geometrical time delay
        if dndt=True this is like SSD
        """

        ct = self.C0 * t
        ct2 = ct * ct
        z = self.z_t(t)
        l = self.l_z(z)
        lgl = np.log10(l)
        sina = self.r / l

        X = self.X_h(z * self.cos_theta + self.z_station)
        dndX = self.dNdX(X)

        # Jacobian J = dX/dt dsina/dr - dsina/dt dX/dr
        # = dX/dz (dz/dt dsina/dr - dsina/dt dz/dr)
        # CHECKED: looks okay, extra 100 microsec or so for data
        dXdz = -self.atm.dXdh(self.z_station + z * self.cos_theta, self.theta)

        J = (self.r2 - ct2) ** 2 / (self.r2 + ct2) ** 2 / t * dXdz

        # Default is ds/dt for wcd so that low energy muons create part of vem
        # which alters the signal time distribution
        if not dndt:
            # WCD, integral over (E/Evem)^2 for E < Evem
            I = self.muon_energy_integral(lgl, self.Evem)

        else:
            # just use same integral but set Evem=m, so that all integral
            # is above Evem and nowhere is the E^2
            I = self.muon_energy_integral(lgl, self.m)

        # Total
        out = dndX * sina ** (self.gamma - 2) * J * I

        # mask below ground
        mask = (z - self.Delta) <= 0 | (t < 0) | (X < 0)
        out[mask] = 0
        out[~np.isfinite(out)] = 0

        # pdf
        return normalize(out, t[1] - t[0])

    def ds_dt_wcd(self, t):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # TODO: need eval convoluteion from singal_model
            if self.UUB:
                return signal_model.evaluate_convolution(
                    t, lambda t: signal_model.UUB_WCD_t_response(self.ds_dt(t))
                )
            else:
                return signal_model.evaluate_convolution(
                    t, lambda t: signal_model.UB_WCD_t_response(self.ds_dt(t))
                )

    def ds_dt_ssd(self, t):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.UUB:
                return signal_model.evaluate_convolution(
                    t, lambda t: signal_model.UUB_SSD_t_response(self.ds_dt(t))
                )
            else:
                return signal_model.evaluate_convolution(
                    t, lambda t: signal_model.UB_SSD_t_response(self.ds_dt(t))
                )
