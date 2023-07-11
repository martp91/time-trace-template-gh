from numba import njit, vectorize, optional
from math import exp, log, ceil, erf
from scipy import stats as scistats
from scipy import special
import numpy as np


@njit(fastmath=True)
def gammaln(z):
    """Numerical Recipes 6.1"""
    # Don't use global variables.. (They only can be changed if you recompile the function)
    coefs = np.array(
        [
            57.1562356658629235,
            -59.5979603554754912,
            14.1360979747417471,
            -0.491913816097620199,
            0.339946499848118887e-4,
            0.465236289270485756e-4,
            -0.983744753048795646e-4,
            0.158088703224912494e-3,
            -0.210264441724104883e-3,
            0.217439618115212643e-3,
            -0.164318106536763890e-3,
            0.844182239838527433e-4,
            -0.261908384015814087e-4,
            0.368991826595316234e-5,
        ]
    )

    out = 0
    y = z
    tmp = z + 5.24218750000000000
    tmp = (z + 0.5) * np.log(tmp) - tmp
    ser = 0.999999999999997092

    n = coefs.shape[0]
    for j in range(n):
        y = y + 1.0
        ser = ser + coefs[j] / y

        out = tmp + log(2.5066282746310005 * ser / z)

    return out


@njit
def xlogy(x, y):
    if x == 0:
        return 0
    else:
        return x * np.log(y)


@njit(fastmath=True)
def _poisson_pdf(y, mu, scale=1):
    """
    If scale != 1 then mu is the expectation value of y=scale*k, with k ~ Poisson(y/scale)
    So the poisson pdf is modified by this scale factor
    """
    if scale <= 0:
        raise ValueError("scale should be larger than 0")
    if mu <= 0:
        if y == 0:
            return 1.0
        elif y > 0:
            return 0.0
        else:
            return np.nan
    else:
        return (
            np.exp(xlogy(y / scale, mu / scale) - mu / scale - gammaln(y / scale + 1))
            / scale
        )


@vectorize
def poisson_pdf(k, mu, scale=1):
    return _poisson_pdf(k, mu, scale)


@njit(fastmath=True)
def PoissonLike(signal, Nmu, Nem, Tmu, Tem):
    "From GAP 2004-057"
    size = len(signal)
    EMvem = 25  # 1VEM is 25 EM particles, 1 VEM=1 muon
    lnL = 0
    for i, k in enumerate(signal):
        Ps = 0
        k = ceil(k)
        for m in range(int(k)):  # m is number of muons
            Ps += poisson_pdf(m, Nmu * Tmu[i]) * poisson_pdf(
                Nem * Tem[i], (k - m) * EMvem
            )
        lnL += np.log(Ps + 1e-20)

    return -lnL


# vectorized but slow if use awkward array
# def poisson_deviance(k, mu):
#    return 2 * np.sum(special.xlogy(k, k) - special.xlogy(k, mu) - k + mu)


@njit(cache=True, nogil=True)
def PoissonDeviance(k, mu, size, scale=None):
    """same as scipy.stats.poisson.logpmf(k, mu).sum() -
    scipy.stats.poisson.logpmf(k, k).sum()

    AKA: c-statistic (XSPEC)
    """
    _sum = 0
    for i in range(size):
        _mu = mu[i]
        if _mu < 0:
            return 1e200 #hack
        _k = k[i]
        if _k < 0:
            continue

        # if k == 0 the below is _mu
        # add some eps to _mu to avoid 0
        # this is fater than logxy - logxy
        lnPi = xlogy(_k, _k / (_mu + 1e-20)) - _k + _mu

        if scale is None:
            _sum += lnPi
        else:
            _sum += scale[i] * lnPi

    # ths is hack
    if _sum == 0:
        _sum = 1e200

    return 2 * _sum  # deviance definition


@njit(cache=True, nogil=True)
def PoissonDeviance_2D(k, mu, scale=None):
    """ Can use awkard array for total event"""
    total = 0
    size = len(k)
    for i in range(size):
        size_k = len(k[i])
        if scale is None:
            s = None
        else:
            s = scale[i]
        if size_k > 0:
            total += PoissonDeviance(k[i], mu[i], size_k, s)
    return total


# Just use scipy.stats.poisson.ppf(q, mu)
@njit
def _poisson_quantile(q, mu, scale=1, size=100):
    """
    Up to ~3% differences
    """
    if q == 1:
        return np.inf
    if q == 0:
        return -1
    if mu < 1:
        y = np.linspace(0, mu * 3, size)
    else:
        y = np.linspace(np.maximum(0, mu - 3 * np.sqrt(mu)), mu + 3 * np.sqrt(mu), size)

    cdf = np.cumsum(poisson_pdf(y, mu, scale))
    cdf /= cdf[-1]
    x = np.interp(q, cdf, y)
    return x


@vectorize
def poisson_quantile(q, mu, scale=1):
    return _poisson_quantile(q, mu, scale)


def poisson_CI(k, alpha, scale=1):
    # Scale divides number of counts here
    # Same as would in scipy stats
    # But opposite of what is now in trace_templates.py
    q0 = 0.5 * scistats.chi2.ppf((1 - alpha) / 2, 2 * k / scale) * scale
    q1 = 0.5 * scistats.chi2.ppf((alpha + 1) / 2, 2 * k / scale) * scale
    # if k==0 q0=nan, so fill nan with 0
    try:
        q0[~np.isfinite(q0)] = 0
        q1[~np.isfinite(q1)] = 0
    except TypeError:
        pass
    return q0, q1


def poisson_68CI(y, yerr):
    """
    Return 16 and 84% interval~1 sigma for poisson
    For low counts one should use this instead of
    1 sigma for errorbars, because that can go to negative values
    and sometimes you dont want that

    By default this returns y +- (yerr_min , yerr_plus)
    """
    scale = yerr ** 2 / y
    return poisson_CI(y, 0.68, scale)


def poisson_1sigma(y, yerr):
    return np.abs(y - np.array(poisson_68CI(y, yerr)))
