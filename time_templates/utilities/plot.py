import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from time_templates.utilities.stats import profile_1d, profile_2d
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from time_templates.utilities.poisson import poisson_CI
import uncertainties as un
from time_templates import data_path
from time_templates.utilities.stats import bootstrap_mean_median_std
from matplotlib.lines import Line2D


def plot_trace(
    station,
    detector="wcd",
    tracetype="total",
    ax=None,
    MC=False,
    templates=False,
    **kwargs,
):
    if ax is None:
        f, ax = plt.subplots(1)

    if detector not in ["wcd", "ssd"]:
        raise TypeError("specifcy wcd or ssd for detector")

    if tracetype not in ["total", "em", "muon"]:
        raise TypeError("tracetype needs to be total, em, muon")

    if MC:
        tpf = (
            station["MCStationPlaneFrontTimeS"] / 1e9
            + station["MCStationPlaneFrontTimeNS"]
        )
        t0 = (station["TimeS"] / 1e9 + station["TimeNS"]) - tpf
    else:
        t0 = -station["PlaneTimeRes"]

    try:
        dt = station["dt"]
    except KeyError:
        dt = 25 / 3
    offset = station["TraceOffset"] * dt
    t0 -= offset
    if tracetype == "muon":
        trace = station[detector + "_total_trace"] - station[detector + "_em_trace"]
    else:
        trace = station[detector + "_" + tracetype + "_trace"]

    nt = len(trace)
    t = np.linspace(t0, t0 + nt * dt, nt)

    ax.plot(t, trace, **kwargs)

    if detector == "wcd":
        ax.set_ylabel("Signal [VEM peak]")
    elif detector == "ssd":
        ax.set_ylabel("Signal [MIP peak]")

    ax.set_xlabel("$t - t_{pf}$ [ns]")
    return ax


def plot_profile_1d(
    x,
    y,
    bins,
    stat="mean",
    weights=None,
    remove_outliers=(-np.inf, np.inf),
    bootstraps=0,
    ax=None,
    plot_n_entries=False,
    plot_xerr=False,
    plot_x_mean_bin=False,
    **kwargs,
):

    xs, ys, yerr, ns = profile_1d(
        x, y, bins, stat, weights, remove_outliers, bootstraps
    )

    if ax is None:
        f, ax = plt.subplots(1)

    if "ls" in kwargs.keys():
        ls = kwargs["ls"]
        del kwargs["ls"]
    else:
        ls = ""

    if "marker" in kwargs.keys():
        marker = kwargs["marker"]
        del kwargs["marker"]
    else:
        marker = "o"
    if isinstance(bins, int):
        bins = np.linspace(x.min(), x.max(), bins)
    if plot_xerr:
        xerr = [xs - bins[:-1], bins[1:] - xs]
    else:
        xerr = None

    if plot_x_mean_bin:
        xs = (bins[1:] + bins[:-1]) / 2
    ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, ls=ls, marker=marker, **kwargs)

    if plot_n_entries:
        trans = transforms.blended_transform_factory(
            x_transform=ax.transData, y_transform=ax.transAxes
        )
        for i, n in enumerate(ns):
            ax.annotate(
                int(n),
                xy=(xs[i], 0),
                xycoords=trans,
                fontsize=8,
                ha="center",
                va="bottom",
                xytext=(0, 1),
                textcoords="offset pixels",
            )

    return ax, (xs, ys, yerr)


def add_identity(axes, *line_args, **line_kwargs):
    (identity,) = axes.plot([], [], *line_args, **line_kwargs)

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(axes)
    axes.callbacks.connect("xlim_changed", callback)
    axes.callbacks.connect("ylim_changed", callback)
    return axes


def setAxLinesBW(ax, set_to_black=True):
    """
    https://stackoverflow.com/questions/7358118/matplotlib-black-white-colormap-with-dashes-dots-etc
    Take each Line2D in the axes, ax, and convert the line style to be
    suitable for black and white viewing.
    """
    MARKERSIZE = 3

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    COLORMAP = {
        default_colors[0]: {"marker": None, "dash": (None, None)},
        default_colors[1]: {"marker": None, "dash": [5, 5]},
        default_colors[2]: {"marker": None, "dash": [5, 3, 1, 3]},
        default_colors[3]: {"marker": None, "dash": [1, 3]},
        default_colors[4]: {"marker": None, "dash": [5, 2, 5, 2, 5, 10]},
        default_colors[5]: {"marker": None, "dash": [5, 3, 1, 2, 1, 10]},
        default_colors[6]: {"marker": "o", "dash": (None, None)},  # [1,2,1,10]}
    }

    lines_to_adjust = ax.get_lines()
    try:
        lines_to_adjust += ax.get_legend().get_lines()
    except AttributeError:
        pass

    for line in lines_to_adjust:
        origColor = line.get_color()
        if set_to_black:
            line.set_color("black")
        line.set_dashes(COLORMAP[origColor]["dash"])
        line.set_marker(COLORMAP[origColor]["marker"])
        line.set_markersize(MARKERSIZE)


def setFigLinesBW(fig, set_to_black=True):
    """
    Take each axes in the figure, and for each line in the axes, make the
    line viewable in black and white.
    """
    for ax in fig.get_axes():
        setAxLinesBW(ax, set_to_black)


def plot_hist(
    x, ax=None, labelinfo=True, fit_norm=False, errorbars=False, bootstraps=0, **kwargs
):
    if ax is None:
        f, ax = plt.subplots(1)

    x = np.array(x)
    x = x[np.isfinite(x)]
    if "bins" in kwargs:
        if not isinstance(kwargs["bins"], int):
            x = x[(x > kwargs["bins"][0]) & (x < kwargs["bins"][-1])]

    n = len(x)
    if bootstraps > 0:
        mean, mean_err, median, median_err, std, std_err = bootstrap_mean_median_std(
            x, bootstraps
        )
    else:
        mean = np.mean(x)
        median = np.median(x)
        std = np.std(x)
        mean_err = std / np.sqrt(n)
        median_err = 1.2533 * mean_err  # for normal only
        # approx
        std_err = np.sqrt(
            1 / n * (stats.moment(x, 4) - (n - 3) / (n - 1) * std**4)
        ) / (2 * std)

    textstr = "\n".join(
        (
            r"$\bar{{x}}= {:.uL}$".format(un.ufloat(mean, mean_err)),
            r"$\tilde{{x}}={:.uL}$".format(un.ufloat(median, median_err)),
            r"$\sigma={:.uL}$".format(un.ufloat(std, std_err)),
            r"$n= %i$" % (n,),
        )
    )
    #         r'$\mathrm{RMS}=%.2f$' % (rms)))
    if labelinfo:
        if "label" in kwargs.keys():
            kwargs["label"] = kwargs["label"] + "\n" + textstr
        else:
            kwargs["label"] = textstr

    n, bins, patches = ax.hist(x, **kwargs)
    handles, labels = ax.get_legend_handles_labels()
    yerr = np.abs(n - poisson_CI(n, 0.68))

    if errorbars:
        mb = (bins[1:] + bins[:-1]) / 2
        Nd = 1
        if "density" in kwargs:
            if kwargs["density"]:
                Nd = len(x) * (bins[1:] - bins[:-1])
                n *= Nd

        ax.errorbar(
            mb,
            n / Nd,
            yerr=yerr / Nd,
            ls="",
            marker=".",
            color=patches[0].get_ec(),
            label=labels[0],
        )

    if fit_norm:
        _min = min(bins)
        _max = max(bins)
        _x = x[(x > _min) & (x < _max)]
        N = len(_x)
        mu, sig = stats.norm.fit(_x)
        xspace = np.linspace(_min, _max, 100)
        dx = bins[1] - bins[0]
        try:
            if kwargs["density"]:
                N = 1
                dx = 1
        except KeyError:
            pass
        ax.plot(
            xspace, N * dx * stats.norm.pdf(xspace, mu, sig), color=patches[0].get_ec()
        )

    return ax, (bins, n)


# copy from matplotlib
def confidence_ellipse(x, y, ax, cov=None, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    if cov is None:
        cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def save_data(flname, x, y, xerr=None, yerr=None, **kwargs):
    flpath = data_path + "/plots_data/" + flname
    print(f"saving data at {flname}")
    np.savez(flpath, x=x, y=y, xerr=xerr, yerr=yerr, **kwargs)


def load_data(flname):
    flpath = data_path + "/plots_data/" + flname
    npfl = np.load(flpath)
    return npfl["x"], npfl["y"], npfl["xerr"], npfl["yerr"], npfl


def bin_stat_2d(x, y, z, plot=False, remove_outliers=(-np.inf, np.inf), **kwargs):
    from scipy.stats import binned_statistic_2d

    mask = (z > remove_outliers[0]) & (z < remove_outliers[1])
    x = x[mask]
    y = y[mask]
    z = z[mask]

    z, xedge, yedge, binnum = binned_statistic_2d(x, y, z, **kwargs)
    mbx = (xedge[1:] + xedge[:-1]) / 2.0
    mby = (yedge[1:] + yedge[:-1]) / 2.0
    if plot:
        f, ax = plt.subplots(1)
        im = ax.pcolormesh(mbx, mby, z)
        f.colorbar(im)
        return ax
    else:
        return z.T, xedge, yedge, binnum


def plot_sys_brackets(x, y, ylow, yup, ax, size=14, bracket="[", **kwargs):
    for i, _x in enumerate(x):
        ax.text(
            x[i],
            y[i] + yup[i],
            bracket,
            verticalalignment="center",
            horizontalalignment="center",
            fontsize=size,
            rotation=-90,
            **kwargs,
        )
        ax.text(
            x[i],
            y[i] - ylow[i],
            bracket,
            verticalalignment="center",
            horizontalalignment="center",
            fontsize=size,
            rotation=90,
            **kwargs,
        )
