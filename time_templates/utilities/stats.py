import numpy as np
from scipy.optimize import root_scalar
from scipy.stats import binned_statistic_2d, gaussian_kde
from numba import njit


def bootstrap_mean_sigma_kde(x, nboot=1000, method="scott"):
    kde = gaussian_kde(x, bw_method=method)
    mean_samples = np.zeros(nboot)
    sigma_samples = np.zeros(nboot)
    n = len(x)
    for i in range(nboot):
        sample = kde.resample(n)
        mean_samples[i] = np.mean(sample)
        sigma_samples[i] = np.std(sample)

    return (
        np.mean(mean_samples),
        np.std(mean_samples),
        np.mean(sigma_samples),
        np.std(sigma_samples),
    )


def profile_1d(
    x,
    y,
    bins,
    stat="mean",
    weights=None,
    remove_outliers=(-np.inf, np.inf),
    bootstraps=0,
):
    """profile_1d.

    Parameters
    ----------
    x : array like
        array of values on x axis to bin
    y : array like
        array of values to apply stat on every bin
    bins :
        bins
    stat :
        stat
    remove_outliers :
        remove_outliers
    bootstraps :
        bootstraps
    """
    if isinstance(bins, int):
        nb = bins
        bins = np.linspace(x.min(), x.max(), nb)
    nb = len(bins) - 1
    mb = (bins[1:] + bins[:-1]) / 2
    xs = np.zeros(nb)
    ys = np.zeros(nb)
    yerr = np.zeros(nb)
    ns = np.zeros(nb)
    mask = (y > remove_outliers[0]) & (y < remove_outliers[1])
    y = y[mask]
    x = x[mask]
    for i in range(nb):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        _x = x[mask]
        _y = y[mask]
        n = len(_y)
        if n <= 1:
            mu, std = np.nan, np.nan
        else:
            if stat == "mean":
                if bootstraps > 0:
                    if weights is None:
                        mu, std = bootstrap_func_mean_std(_y, stat, bootstraps)
                    else:
                        raise NotImplementedError(
                            "if weighted bootstraps does not work"
                        )
                        # mu, std = bootstrap_func_mean_std(
                        #    _y,
                        #    lambda x: np.average(x, weights=weights[mask]),
                        #    bootstraps,
                        # )
                else:
                    if weights is None:
                        mu = np.mean(_y)
                        std = np.std(_y) / np.sqrt(n)
                    else:
                        mu = np.average(_y, weights=weights[mask])
                        std = np.sqrt(
                            np.average((_y - mu) ** 2, weights=weights[mask]) / n
                        )
            elif stat == "median":
                if bootstraps > 0:
                    mu, std = bootstrap_func_mean_std(
                        _y, lambda x: np.median(x), bootstraps
                    )
                else:
                    mu = np.median(_y)
                    std = np.std(_y) / np.sqrt(n)
            elif stat == "mode":
                if bootstraps > 0:
                    mu, std = bootstrap_func_mean_std(
                        _y, lambda x: get_kde_mode(x), bootstraps
                    )
                else:
                    mu, std = get_kde_mode(_y, ret_68cl=True)
                    std /= np.sqrt(n)
            else:
                if bootstraps > 0:
                    mu, std = bootstrap_func_mean_std(_y, lambda x: stat(x), bootstraps)
                else:
                    mu = stat(_y)
                    std = 0

        xs[i] = np.mean(_x)
        ys[i] = mu
        yerr[i] = std
        ns[i] = n

    return xs, ys, yerr, ns


def bootstrap_func(y, stat=lambda x: np.mean(x), n_boot=100):
    if stat == "mean":
        func = lambda x: np.mean(x)
    elif stat == "median":
        func = lambda x: np.median(x)
    elif stat == "std":
        func = lambda x: np.std(x)
    elif isinstance(stat, (int, float)):
        func = lambda x: np.percentile(x, stat)
    elif callable(stat):
        func = stat
    else:
        print(stat)
        raise NotImplementedError("stat argument was not correct")
    n = len(y)
    boot_stat = []
    for i in range(n_boot):
        randints = np.random.randint(0, n, n)
        boot_stat.append(func(y.take(randints)))
    return boot_stat


@njit
def bootstrap_mean_median_std(x, n_boot=100):
    means = np.zeros(n_boot)
    medians = np.zeros(n_boot)
    sigmas = np.zeros(n_boot)
    n = len(x)

    for i in range(n_boot):
        randints = np.random.randint(0, n, n)
        randoms = x.take(randints)
        means[i] = np.mean(randoms)
        medians[i] = np.median(randoms)
        sigmas[i] = np.std(randoms)

    return (
        np.mean(means),
        np.std(means),
        np.mean(medians),
        np.std(medians),
        np.mean(sigmas),
        np.std(sigmas),
    )


def bootstrap_func_mean_std(y, stat=lambda x: np.mean(x), n_boot=100):
    boots = bootstrap_func(y, stat, n_boot)
    return np.mean(boots), np.std(boots)


def get_kde_mode(x, xmin=None, xmax=None, nspaces=500, ret_68cl=False):
    if len(x) <= 1:
        return np.nan, np.nan

    try:
        gkde = gaussian_kde(x)
    except:
        if ret_68cl:
            return np.nan, np.nan
        else:
            return np.nan
    if xmin is None:
        xmin = np.quantile(x, 0.1)
    if xmax is None:
        xmax = np.quantile(x, 0.9)
    xspace = np.linspace(xmin, xmax, nspaces)
    pdf = gkde.pdf(xspace)
    mode = xspace[pdf.argmax()]

    if ret_68cl:
        # integrating from -x to x so assuming symmetry
        fx = lambda x: gkde.integrate_box_1d(-x, x) - 0.68
        fp = lambda x: gkde(x) + gkde(-x)
        cl68_sol = root_scalar(fx, fprime=fp, x0=np.std(x), bracket=[0, 1e20])
        return mode, cl68_sol.root
    else:
        return mode


def weighted_cov(x, y, w):
    """Weighted Covariance"""
    return np.average(
        (x - np.average(x, weights=w)) * (y - np.average(y, weights=w)), weights=w
    )


def weighted_corr(x, y, w):
    """Weighted Correlation"""
    return weighted_cov(x, y, w) / np.sqrt(
        weighted_cov(x, x, w) * weighted_cov(y, y, w)
    )


def profile_2d(x, y, z, bins, statistic="mean", bootstraps=0):
    xbins, ybins = bins
    nxb = len(xbins) - 1
    nyb = len(ybins) - 1
    xout = np.zeros((nxb, nyb))
    yout = np.zeros((nxb, nyb))
    zout = np.zeros((nxb, nyb))
    zout_err = np.zeros((nxb, nyb))
    nout = np.zeros((nxb, nyb))

    for i in range(nxb):
        for j in range(nyb):
            mask = (
                (x >= xbins[i])
                & (x < xbins[i + 1])
                & (y >= ybins[j])
                & (y < ybins[j + 1])
            )
            x_ = x[mask]
            y_ = y[mask]
            z_ = z[mask]
            n = len(y_)
            if bootstraps > 0:
                mu, std = bootstrap_func_mean_std(z_, stat=statistic, n_boot=bootstraps)
            else:
                mu, std = np.mean(z_), np.std(z_) / np.sqrt(n)
            zout[i, j] = mu
            zout_err[i, j] = std
            xout[i, j] = np.mean(x_)
            yout[i, j] = np.mean(y_)
            nout[i, j] = n

    return xout, yout, zout, zout_err, nout


def bootstrap_2d_statistic(x, y, z, bins, statistic, nboot=100):
    n, _, _, _ = binned_statistic_2d(x, y, z, bins=bins, statistic="count")
    if nboot <= 0:
        z_mean, xedge, yedge, binnum = binned_statistic_2d(
            x, y, z, bins=bins, statistic=statistic
        )
        z_std, xedge, yedge, binnum = binned_statistic_2d(
            x, y, z, bins=bins, statistic="std",
        )
        z_std /= np.sqrt(n)
    else:
        z_boots = []
        for i in range(nboot):
            ri = np.random.randint(0, len(x), len(x))
            x_ = x[ri]
            y_ = y[ri]
            z_ = z[ri]
            z_boot, xedge, yedge, binnum = binned_statistic_2d(
                x_, y_, z_, bins=bins, statistic=statistic
            )
            z_boots.append(z_boot)

        z_mean = np.mean(z_boots, axis=0)
        z_std = np.std(z_boots, axis=0)
    mbx = (xedge[1:] + xedge[:-1]) / 2.0
    mby = (yedge[1:] + yedge[:-1]) / 2.0
    return z_mean, z_std, mbx, mby, n


def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation
