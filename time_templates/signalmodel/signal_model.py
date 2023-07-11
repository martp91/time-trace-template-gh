import os
import json
import numpy as np
from numba import njit
from math import erf, exp, log, sqrt
from scipy.linalg import convolution_matrix
from scipy import sparse

from time_templates import package_path, data_path

data_path = os.path.join(package_path, "data")  # store small files here to load fast


def exponential_response(t, tau, td):
    if td == 0:
        return np.exp(-t / tau)
    out = np.where(
        t <= td, 1 - np.exp(-t / tau), np.exp(-(t - td) / tau) - np.exp(-t / tau)
    )
    return out / (1 - np.exp(-td / tau))


# These values are from the rec adst when calling station.GetCharge(pmtid)
PMT1Charge_to_VEM = 1606.17
PMT2Charge_to_VEM = 1606.42
PMT3Charge_to_VEM = 1606.81
WCDCharge_to_VEM = (PMT1Charge_to_VEM + PMT2Charge_to_VEM + PMT3Charge_to_VEM) / 3
PMT5Charge_to_VEM = 152.37
PMT1Peak_to_VEM = 215.34
PMT2Peak_to_VEM = 215.74
PMT3Peak_to_VEM = 216.71
PMT5Peak_to_VEM = 51.8

# fit to EPOS LHC proton 10^19 eV
parameters = {
    "a": lambda sct: (0.51 + 0.25 * sct, -0.46 + 0.19 * sct, -0.05),
    "b": lambda sct: (0.066, 0.035, 0.0),
    "c": lambda sct: (0.017, 0.018, 0.032),
    "ry": lambda sct: (25, -28, -16),
    "eps_y_ssd": 0.03,  # photon conversion ssd
    "ssd_factor": 1.13,  # mean to mode
    "ssd_mu_factor": 1.25,  # fudge factor to match reconcstructed sims
}

EVEM = 400e6
ECH = 54e6  # Check this


def poly_lgr(lgr, p0, p1, p2=0):
    return p0 + p1 * lgr + p2 * lgr**2


def get_abcdefry(theta, r, include_TL=False, include_fudge=False):
    """
    Get average signal parameters as a function of
    theta (in rad)
    r = distance to shower core in shower plane in m
    """
    sct = 1 / np.cos(theta)
    pa = parameters["a"](sct)  # average muon signal wcd
    pb = parameters["b"](sct)  # average el signal wcd
    pc = parameters["c"](sct)  # average ph signal wcd
    pd = parameters["ry"](sct)  # photons/electrons
    if include_fudge:
        ssd_factor = parameters["ssd_factor"]
        ssd_mu_factor = parameters["ssd_mu_factor"]
    else:
        ssd_factor = 1
        ssd_mu_factor = 1
    if include_TL:
        # average
        Lwcd = f_Lwcd(theta)
        Lssd = f_Lssd(theta)
    else:
        Lwcd = 1
        Lssd = 1

    lgr = np.log10(r / 1000)

    a = poly_lgr(lgr, *pa) * Lwcd  # average signal per muon wcd
    b = poly_lgr(lgr, *pb)  # average signal per electron wcd
    c = poly_lgr(lgr, *pc)  # average signal per photon wcd
    d = Lssd * ssd_factor * ssd_mu_factor  # average signal per muon ssd
    e = Lssd * ssd_factor  # average signal electron ssd
    f = parameters["eps_y_ssd"] * ssd_factor  # average signal photon ssd
    ry = poly_lgr(lgr, *pd)  # number of photons/electrons

    return a, b, c, d, e, f, ry


@njit
def f_Awcd(chi, R=1.8, H=1.2):
    "Surface WCD under angle wrt zenith chi"
    return np.pi * R**2 * np.abs(np.cos(chi)) + 2 * R * H * np.sin(chi)


@njit
def f_Assd(chi):
    "Surface SSD under angle wrt zenith chi"
    return 3.84 * np.abs(np.cos(chi))


@njit(fastmath=True, nogil=True, cache=True, error_model="numpy")
def f_Lwcd(theta, R=1.8, H=1.2):
    return 1 / (np.abs(np.cos(theta)) + 2 * H / (np.pi * R) * np.sin(theta))


def f_Lssd(chi):
    "Average signal/MIP due to average tracklength SSD at zenith angle chi"
    return 1 / np.cos(chi)


def signal(Nmu, Ne, r, theta):
    """Signal WCD SSD
    Forward folding
    Nmu = Number of muons hitting WCD
    Ne = number of electrons/positrons hitting SSD
    """

    Awcd = f_Awcd(theta)
    Assd = f_Awcd(theta)

    a, b, c, d, e, f, ry = get_abcdefry(r, theta)

    Swcd = Nmu * a + Ne * (b + c * ry)
    Sssd = Assd / Awcd * (Nmu * d + Ne * (e + f * ry))

    return Swcd, Sssd


def matrix_inv(Swcd, Sssd, r, theta):
    """
    Inverting matrix:
    ( Swcd ) = ( a,  (b+cry)              ) ( Nmu )
    ( Sssd ) = ( As/Aw d, As/Aw (e + fry) ) ( Ne  )

    abcdef are determined by r, theta
    Can be array or float
    """

    Awcd = f_Awcd(theta)
    Assd = f_Awcd(theta)

    a, b, c, d, e, f, ry = get_abcdefry(r, theta)

    G = Awcd / Assd

    det = a * (e + f * ry) - b * d - c * d * ry

    Nmu = ((e + f * ry) * Swcd - G * (b + c * ry) * Sssd) / det
    Ne = (G * a * Sssd - Swcd * d) / det

    # Checking if iterable and set nans accordingly in array or not
    iterable = True
    try:
        iter(Nmu)
    except TypeError:
        iterable = False

    if not iterable:
        if Nmu < 0:
            Nmu = np.nan
        if Ne < 0:
            Ne = np.nan
    else:
        Nmu[Nmu < 0] = np.nan
        Ne[Ne < 0] = np.nan
    return Nmu, Ne


def covariance_matrix_signal(Swcd, Sssd, r, theta):
    """
    Covariance matrix from fluctuations of number of particles
    hitting the detector. An additional detector uncertainty can
    (should!) be added.

    ( Swcd ) = ( a,  (b+cry)              ) ( Nmu )
    ( Sssd ) = ( As/Aw d, As/Aw (e + fry) ) ( Ne  )

    abcdef are determined by r, theta
    Can be array or float
    """

    Awcd = f_Awcd(theta)
    Assd = f_Awcd(theta)

    a, b, c, d, e, f, ry = get_abcdefry(r, theta)

    Nmu, Ne = matrix_inv(Swcd, Sssd, r, theta)

    sigma_w2 = Nmu * a**2 + Ne * (b**2 + ry * c**2)
    sigma_s2 = (Assd / Awcd) * (d * Nmu + Ne * (e + ry * f**2))

    sigma_ws = Assd / Awcd * (a * d * Nmu + (b + c * ry) * (e + f * ry) * Ne)

    return np.array([[sigma_w2, sigma_ws], [sigma_ws, sigma_s2]])


@njit(fastmath=True, nogil=True, cache=True, error_model="numpy")
def track_length_cdf(L, theta, R=1.8, H=1.2):
    """
    Track length CDF from GAP-2009-043
    vectorized L
    invert this to sample
     pL(L|theta) = 1/2R P(L/2R, theta)
     int_0^l p(l'| theta) = (arcsin(v) + 2u +
                      (3v-2u)*(sqrt(1-v^2)))/(2u + pi/2)
                              for v <= min(1, u) else 1

    Parameters
    ----------
    L : array like of floats
        track-length in meters
    theta : float
        zenith angle of incoming particle
    R : float, optional
        radius of tank in m default=1.8
    H : float, optional
        height of tank in m default=1.2

    Returns
    --------
    array like floats
        probability

    """

    if theta == 0:  # always vertical through tank
        return np.ones_like(L)

    h = H / (2 * R)
    l = L / (2 * R)
    u = h * np.abs(np.tan(theta))
    v = l * np.sin(theta)

    mask = v <= np.minimum(1, u)
    p = np.ones_like(L)
    p[mask] = (
        np.arcsin(v[mask]) + 2 * u + (3 * v[mask] - 2 * u) * np.sqrt(1 - v[mask] ** 2)
    ) / (2 * u + np.pi / 2)
    return p


@njit(fastmath=True, nogil=True, cache=True, error_model="numpy")
def sample_track_length(theta, R=1.8, H=1.2, signal=False, ninterp=100):
    """sample_track_length from interpolated inverse CDF
    From GAP GAP-2009-043

    Parameters
    ----------
    theta : float
        zenith angle of incoming particles in radians
    R : float, optional
        radius of tank in m default=1.8
    H : float, optional
        height of tank in m default=1.2
    signal : bool, optional
        if True return TL/H is signal in VEM default=False
    ninterp : int, optional
        number of points to use for interpolation default=100
        1000 reproduces mean correctly, 100 is 1% difference

    Returns
    --------
    float
        random track-length
    """

    if theta == 0:  # always
        if signal:
            return 1
        return H
    lmax = 2 * R  # TODO: this can be optimized
    x = np.linspace(0, lmax, ninterp)  # 100 seems fine?
    cdf = track_length_cdf(x, theta, R, H)
    uniform_samples = np.random.random_sample()
    # faster than scipy.interpolate.interp1d
    out = np.interp(uniform_samples, cdf, x)
    if signal:
        return out / H
    return out


@njit(fastmath=True, nogil=True, cache=True, error_model="numpy")
def sample_track_length_v(theta, N, R=1.8, H=1.2, signal=True, ninterp=100):
    out = np.zeros(N)
    for i in range(N):
        out[i] = sample_track_length(theta, R, H, signal, ninterp)

    return out


@njit(fastmath=True, nogil=True, cache=True, error_model="numpy")
def calc_q68_track_length(theta, R=1.8, H=1.2, signal=False, ninterp=100):
    if theta == 0:  # always
        return 0
    lmax = 2 * R  # TODO: this can be optimized
    x = np.linspace(0, lmax, ninterp)  # 100 seems fine?
    cdf = track_length_cdf(x, theta, R, H)

    # Very approximate
    if signal:
        return (np.interp(0.84, cdf, x) - np.interp(0.16, cdf, x)) / (2 * H)

    return (np.interp(0.84, cdf, x) - np.interp(0.16, cdf, x)) / 2


@njit(fastmath=True, nogil=True, cache=True, error_model="numpy")
def calc_median_track_length(theta, R=1.8, H=1.2, signal=False, ninterp=100):
    if theta == 0:  # always
        return 1
    lmax = 2 * R  # TODO: this can be optimized
    x = np.linspace(0, lmax, ninterp)  # 100 seems fine?
    cdf = track_length_cdf(x, theta, R, H)
    if signal:
        return np.interp(0.5, cdf, x) / H
    else:
        return np.interp(0.5, cdf, x)


@njit(fastmath=True, nogil=True, cache=True, error_model="numpy")
def sample_signal_moyal(loc, scale):
    """Random sample from
    moyal distr. copy from scipy
    to get working with njit

    Parameters
    ----------
    loc : float
        loc
    scale : float
        scale
    """
    u = np.random.gamma(0.5, 2)
    sample = -np.log(u) * scale + loc
    if sample > 0:  # is this ok?
        return sample
    else:
        return 0


@njit(fastmath=True, nogil=True, cache=True, error_model="numpy")
def sample_signal_moyal_k(k, loc, scale, norm_cutoff=20):
    """sample_signal_moyal_k.
    Sample signal from moyal for k particles.
    if k > norm_cutoff, the normal approximation can be used

    Parameters
    ----------
    k : int
        k
    loc : float
        loc
    scale : float
        scale
    norm_cutoff : int
        norm_cutoff

    Returns
    --------
    float
    """
    if k > norm_cutoff:
        mean = loc + scale * (np.log(2) + np.euler_gamma)
        std = np.pi * scale / np.sqrt(2)
        return np.random.normal(k * mean, np.sqrt(k) * std)
    y = 0
    for _ in range(int(k)):
        y += sample_signal_moyal(loc, scale)

    return y


# This is now faster than the previous implementation when sampling from normal
# Still do some checks if this is ok


@njit(fastmath=True, nogil=True, cache=True, error_model="numpy")
def sample_signal_moyal_v(k, *args):
    """sample_signal_moyal_v.
    k is number of particles (array)
    loc, scale -> moyal dist
    norm_cutoff=50 is very save, could be lower~20, but speed-up is negligible

    Not sure if below is still true

    This OVERESTIMATES the actual signal,
    because of the landau tail
    and because the data is not corrected for that
    So correct the data with the mean of the moyal

    Parameters
    ----------
    k : array like
        number of particles
    *args: tuple
        additional arguments for sample_signal_moyal
        can be loc, scale, norm_cutoff

    Returns
    --------
    array of floats
    """
    size = len(k)
    y = np.zeros(size)
    for i in range(size):
        _k = k[i]
        y[i] = sample_signal_moyal(_k, *args)
    return y


@njit(fastmath=True, nogil=True, cache=True, error_model="numpy")
def sample_signal_normal_v(k, mean, std):
    """Sample normal for an array k
    with each _k hits. Scale the mean and std appropriately
    """
    size = len(k)
    y = np.zeros(size)
    for i in range(size):
        _k = k[i]
        sample = np.random.normal(_k * mean, np.sqrt(_k) * std)
        if sample > 0:
            y[i] = sample
    return y


# Also faster
@njit(fastmath=True, nogil=True, cache=True, error_model="numpy")
def sample_poisson_v(mu, size):
    k = np.zeros(size)

    for i in range(size):
        k[i] = np.random.poisson(mu[i])

    return k


# Below is all sampling stuff


@njit(fastmath=True, nogil=True, cache=True, error_model="numpy")
def convert_lognorm_mean(mean, std):
    """convert_lognorm_mean.
    returns lognormal mu, s parameters
    pdf = 1/(x*s*sqrt(2*pi)) * exp(-(logx-mu)/(2s^2))

    Parameters
    ----------
    mean : float
        mean
    std : float
        std

    Returns
    --------
    mu: float
    s: float
    """
    mean2 = mean * mean
    std2 = std * std
    mu = log(mean2 / sqrt(std2 + mean2))
    s = sqrt(log(1 + std2 / mean2))
    return mu, s


SQRT2 = np.sqrt(2)
LN2 = np.log(2)


@njit(fastmath=True, nogil=True, cache=True, error_model="numpy")
def convert_moyal_loc_scale(loc, scale):
    "return means, sig"
    mean = loc + scale * (LN2 + np.euler_gamma)
    std = np.pi * scale / SQRT2
    return mean, std


@njit(fastmath=True, nogil=True, cache=True, error_model="numpy")
def convert_moyal_mean_std(mean, std):
    "return loc scale"
    scale = std * SQRT2 / np.pi
    loc = mean - scale * (LN2 * np.euler_gamma)
    return loc, scale


@njit(fastmath=True, nogil=True, cache=True, error_model="numpy")
def x2_logN(mu, s, Evem=EVEM, Ech=ECH):
    """Emu12 = 1/N int_Ech^Evem (E/Evem)^2 dn/de de
    1/N dn/de ~ lognormal(mu, s)
    """
    s2 = s * s
    pref = -0.5 * exp(2 * (mu + s2))
    mu2s2 = mu + 2 * s2
    sqrt2_s = SQRT2 * s
    x1 = (mu2s2 - log(Evem)) / sqrt2_s
    x2 = (mu2s2 - log(Ech)) / sqrt2_s
    f1 = erf(x1)
    f2 = erf(x2)
    return pref * (f1 - f2) / Evem**2


@njit(fastmath=True, nogil=True, cache=True, error_model="numpy")
def f2_logN(mu, s, Evem=EVEM):
    """fmu2 = 1/N int_Evem^inf dn/de de
    1/N dn/de ~ lognormal(mu, s)
    This part should be multiplied by track-length
    """
    x = (log(Evem) - mu) / (SQRT2 * s)
    return 0.5 * (1 - erf(x))


@njit(fastmath=True, nogil=True, cache=True, error_model="numpy")
def f_average_muon_signal_exact(Emu_mean, Emu_std, L=1, Evem=EVEM, Ech=ECH):
    """
    Assuming the muon energy spectrum can be approximated by a lognormal
    Then the average muon signal is the contribution of muons below Evem,
    as E^2/Evem^2 and the fraction of muons above Evem
    The VEM muons need to be multiplied by the track-length normalized signal
    """
    if Emu_mean <= 0:
        return 0
    if Emu_std <= 1e-6 * Emu_mean:
        Emu_std = Emu_mean / 2
    mu, sig = convert_lognorm_mean(Emu_mean, Emu_std)
    a = x2_logN(mu, sig, Evem, Ech) + L * f2_logN(mu, sig, Evem)
    return a


@njit(fastmath=True, nogil=True, cache=True, error_model="numpy")
def f_average_muon_signal_exact_v(Emu_mean, Emu_std, L=1, Evem=EVEM, Ech=ECH):
    """
    This is a large time drain
    Assuming the muon energy spectrum can be approximated by a lognormal
    Then the average muon signal is the contribution of muons below Evem,
    as E^2/Evem^2 and the fraction of muons above Evem
    The VEM muons need to be multiplied by the track-length normalized signal
    """
    size = len(Emu_mean)
    out = np.zeros(size)
    for i in range(size):
        out[i] = f_average_muon_signal_exact(
            Emu_mean[i], Emu_std[i], L=L, Evem=Evem, Ech=Ech
        )

    return out


# tau and start of peak

try:
    expon_detector_time_fit = np.loadtxt(
        os.path.join(
            data_path,
            "detector_response",
            "UB_WCD_double_exponential_params.txt",
        )
    )
    TAU, TPEAK = expon_detector_time_fit
    TPEAK = 0
except FileNotFoundError:
    print("Warnig could not find exponential time fit file, setting hardcoded")
    print("Tau = 61.4, Tpeak = 0")  # 49.5")
    TAU = 61.4  # ns
    TPEAK = 0  # 49.5  # ns


class DetectorTimeResponse:
    # TODO: for data use AOP -> s -> tau.
    # See Orazio Zapparrata talks at Auger OCM November 2021 and March 2022
    """
    Time detector response g(t) such that:
    S(t) = int g(tau) n(t-tau) dtau

    On offline tank simulations with 2 GeV muons (average over many)
    injected at 2 m above ground (such that there is
    a delay of ~ 2.7 ns to WCD and ~1 ns to SSD.
    the 0th entry correspond to the time the particle was injected

    BEWARE: the binning should be the same when doing the convolution
    """

    def __init__(
        self, fl=None, UUB=True, SSD=False, use_expon=True, tau=TAU, tpeak=TPEAK
    ):
        """__init__.

        Parameters
        ----------
        fl : str (default = None)
            filepath to .npy file that contains particle response
            If not specified load fl based on UUB and SSD at default
            data path
        UUB : bool (default=True)
            If True use UUB (25/3 ns timing) else UB (25ns). Not much different
            except binning
        SSD : bool (default=False)
            If True use SSD response, else WCD (default)
        """
        if use_expon and not SSD:
            if UUB:
                t = np.arange(0, 1000, 25 / 3)
            else:
                t = np.arange(0, 1000, 25)
            self.t_response = exponential_response(t, tau, tpeak)
        else:
            # Is this deprecated? because use_expon=True is default now
            if fl is None:
                if UUB:
                    if SSD:
                        self.t_response = np.load(
                            os.path.join(
                                data_path,
                                "detector_response",
                                "UUB_SSD_time_response.npy",
                            )
                        )
                    else:
                        self.t_response = np.load(
                            os.path.join(
                                data_path,
                                "detector_response",
                                "UUB_WCD_time_response.npy",
                            )
                        )
                else:
                    if SSD:
                        self.t_response = np.load(
                            os.path.join(
                                data_path,
                                "detector_response",
                                "UB_SSD_time_response.npy",
                            )
                        )
                    else:
                        self.t_response = np.load(
                            os.path.join(
                                data_path,
                                "detector_response",
                                "UB_WCD_time_response.npy",
                            )
                        )
            else:
                self.t_response = np.load(fl)

        self.t_response /= self.t_response.max()  # normalize on peak
        self.t_response_sum = self.t_response.sum()
        self.AOP = 1 / self.t_response_sum  # Area over peak
        self.t_response_normed = self.t_response * self.AOP
        self.t_response_sq = self.t_response**2
        self.t_response_normed_sq = self.t_response_normed**2
        if UUB:
            if SSD:
                self.t_response = np.load(
                    os.path.join(
                        data_path,
                        "detector_response",
                        "UUB_SSD_time_response.npy",
                    )
                )
            else:
                self.t_response = np.load(
                    os.path.join(
                        data_path,
                        "detector_response",
                        "UUB_WCD_time_response.npy",
                    )
                )
        else:
            if SSD:
                self.t_response = np.load(
                    os.path.join(
                        data_path,
                        "detector_response",
                        "UB_SSD_time_response.npy",
                    )
                )
            else:
                self.t_response = np.load(
                    os.path.join(
                        data_path,
                        "detector_response",
                        "UB_WCD_time_response.npy",
                    )
                )
        # self.try_load_A(UUB, tau, tpeak)

    def try_load_A(self, UUB, tau, tpeak):
        """
        load large convolution matrix from file
        DO NOT USE: MEMORY LEAKS
        """
        if UUB:
            det = "UUB"
        else:
            det = "UB"
        # TODO: SSD?
        tau = round(tau)
        tpeak = round(tpeak)
        try:
            self.A = sparse.load_npz(
                data_path + det + f"_conv_matrix_{tau}_{tpeak}.npz"
            )
            self.A2 = sparse.load_npz(
                data_path + det + f"_conv_matrix_sq_{tau}_{tpeak}.npz"
            )
        except:
            self.A = self.make_conv_matrix(4000)
            self.A2 = self.A @ self.A
            sparse.save_npz(data_path + det + f"_conv_matrix_{tau}_{tpeak}.npz", self.A)
            sparse.save_npz(
                data_path + det + f"_conv_matrix_sq_{tau}_{tpeak}.npz", self.A2
            )

    def __del__(self):
        del self.t_response
        del self.t_response_normed
        del self.t_response_sq
        del self.t_response_normed_sq
        # del self.A
        # del self.A2

    def make_conv_matrix(self, nt, normalize=True):
        # so that y = A * x
        if normalize:
            h = self.t_response_normed
        else:
            h = self.t_response
        A = sparse.csr_matrix(
            convolution_matrix(h, nt, mode="full")[:nt].astype("float32")
        )
        return A

    def convolve_t(self, y, normalize=True, square=False):
        """convolve_t.

        Parameters
        ----------
        y : array float
            the trace to be convolved with the detector response over time
        normalize : bool (default = True)
            If True make sure that the sum(y) = sum(return)

        Returns
        ---------
        array like floats
            the convolution product
        """
        if normalize:
            if square:
                h = self.t_response_normed_sq
            else:
                h = self.t_response_normed
        else:
            if square:
                h = self.t_response_sq
            else:
                h = self.t_response
            # this returns an array with the same length
            # this does mean that the binning of y has to be the same as for t_resopnse
        # numba implement is slower due to BLAS or something
        # So no use in optimizing this
        return np.convolve(y, h, mode="full")[: len(y)]

    def __call__(self, y, normalize=True, square=False):
        return self.convolve_t(y, normalize, square=square)


UB_WCD_t_response = DetectorTimeResponse(UUB=False, SSD=False)
UUB_WCD_t_response = DetectorTimeResponse(UUB=True, SSD=False)
# TODO
UUB_SSD_t_response = DetectorTimeResponse(UUB=True, SSD=True)
UB_SSD_t_response = DetectorTimeResponse(UUB=False, SSD=True)


# Might move this into DetectorTimeResponse
def evaluate_convolution(t, func, nleft=20, nright=0):
    """
    Because the convolution can screw up the distribution if t0 is close to t0
    Take some more bins and calculate and then restore to the original t
    nleft=20 is probably enough, but be careful
    nright=0 because this is probably not needed
    """
    dt = t[1] - t[0]
    start = t[0] - nleft * dt
    stop = start + (len(t) - 1 + nleft + nright) * dt
    new_size = len(t) + nleft + nright
    tnew = np.linspace(start, stop, new_size)
    if nright == 0:
        out = func(tnew)[nleft:]
    else:
        out = func(tnew)[nleft:-nright]
    # try:
    #    assert len(out) == len(t)
    # except AssertionError as e:
    #    print(len(out), len(t))
    #    raise e

    # try:
    #    assert np.isclose(tnew[1] - tnew[0], dt)
    # except AssertionError as e:
    #    print(tnew[1] - tnew[0], dt)
    #    raise e
    return out
