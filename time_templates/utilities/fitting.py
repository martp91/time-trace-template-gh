import numpy as np
import matplotlib.pyplot as plt
import uncertainties
from scipy.optimize import curve_fit
from scipy import odr


def pretty_print_uncertaity(x, xerr=0, latex=True):
    """pretty_print_uncertaity.

    Parameters
    ----------
    x : float
        x
    xerr : float
        xerr
    latex : bool
        latex
    """
    x = uncertainties.ufloat(x, xerr)
    # TODO: fix when xerr = 0 to display some digitis but not all
    if latex:
        return "{:L}".format(x)
    return "{:u}".format(x)


def calc_chi2(y, ypred, yerr=None):
    if yerr is None:
        yerr = 1
    residuals = (y - ypred) / yerr
    return np.sum(residuals ** 2)


def min_mu12_y12_with_covariances(mu12, y12, inv_covs):
    """
    mu12 and y12 are a (n, 2) array and
    the inv_covs are a (n, 2, 2) array with covariance matrix
    for each n values

    This returns a weighted chi square with these covariances over all n
    """
    # chi 2 with covariance summed over all x
    return np.einsum("ij, ijk, ik", (y12 - mu12), inv_covs, (y12 - mu12))


def fit_poly(x, y, yerr=None, deg=1, return_cov=False):
    x = np.array(x)
    y = np.array(y)
    mask = np.isfinite(x) & np.isfinite(y)
    try:
        with np.errstate(invalid="ignore"):
            mask = mask & np.isfinite(yerr) & (yerr > 0)
    except:
        pass

    x = x[mask]
    y = y[mask]
    try:
        yerr = yerr[mask]
        weights = 1 / yerr
    except TypeError:
        weights = None
    # asume errors are 1/sigma so that cov is not scaled
    fitp, fitp_cov = np.polyfit(x, y, deg=deg, w=weights, cov="unscaled")
    chi2 = calc_chi2(y, np.poly1d(fitp)(x), yerr)
    ndof = len(y) - (deg + 1)
    #    cov *= chi2/ndof
    fitp_err = np.sqrt(np.diag(fitp_cov))

    if return_cov:
        return fitp, fitp_cov, chi2, ndof
    return fitp, fitp_err, chi2, ndof


def fit_curve(
    x, y, func, yerr=None, p0=None, bounds=(-np.inf, np.inf), return_cov=False
):
    x = np.array(x)
    y = np.array(y)
    #    mask = np.isfinite(x) & np.isfinite(y)
    #    try:
    #        with np.errstate(invalid='ignore'):
    #            mask = mask & np.isfinite(yerr) & (yerr > 0)
    #    except:
    #        pass

    #    x = x[mask]
    #    y = y[mask]

    # check if yerr is array
    if yerr is not None:
        sigma = np.array(yerr)
    sigma = yerr
    #    try:
    #        sigma = yerr[mask]
    #        yerr = yerr[mask]
    #    except TypeError:
    #        sigma = None
    fitp, fitp_cov = curve_fit(
        func, x, y, sigma=sigma, p0=p0, bounds=bounds, absolute_sigma=True
    )
    fitp_err = np.sqrt(np.diag(fitp_cov))
    chi2 = calc_chi2(y, func(x, *fitp), yerr)
    ndof = len(x) - len(fitp)
    if return_cov:
        return fitp, fitp_cov, chi2, ndof
    return fitp, fitp_err, chi2, ndof


def plot_nice_fit(
    x,
    y,
    func,
    fitp,
    fitp_err,
    chi2,
    ndof,
    xerr=None,
    yerr=None,
    ax=None,
    ebar_kws=None,
    line_kws=None,
    plot_line=True,
    transform_x=lambda x: x,
    param_names=None,
    units=None,
    smoother_x=False,
    custom_label=None,
    add_label=True,
):
    """plot_nice_fit.

    Parameters
    ----------
    x : array
        x
    y : array
        y
    func : function
        func of the form ypred = func(x, *fitp)
    fitp : array-like with fit parameters that go into func
        fitp
    fitp_err : array-like
        fitp_err
    chi2 :
        chi2
    ndof :
        ndof
    yerr : array-like  or float(?) (optional)
        yerr uncertainty on y for plot
    ax : matplotlib axis
        ax
    ebar_kws :
        ebar_kws
    line_kws :
        line_kws
    plot_line :
        plot_line
    transform_x :
        transform_x
    param_names :
        param_names
    units :
        units
    smoother_x : bool (optional)
        smoother_x if True(default), use more point to plot the line
        than there are for the x points. If false just use the x values
        this is for example necessary when plotting the muon/em trace model
        then the convolution needs the bins to be 25/3 ns or 25 ns wide.
    custom_label: string
        give custom label to add to chi2 ndf etc
    """
    if ax is None:
        f, ax = plt.subplots(1)

    if ebar_kws is None:
        ebar_kws = dict(ls="", marker="o")
    if line_kws is None:
        line_kws = dict()

    if custom_label is not None:
        textlist = [custom_label]
    else:
        textlist = []

    textlist.append("$\\chi^2/\\mathrm{ndf}=%.1f/%.0f$" % (chi2, ndof))

    if param_names is None:
        param_names = [f"p_{i}" for i in range(len(fitp))]

    if units is None:
        units = ["" for i in range(len(fitp))]

    for (p, perr, p_name, unit) in zip(fitp, fitp_err, param_names, units):
        textlist.append(f"${p_name}={pretty_print_uncertaity(p, perr)} \, {unit}$")

    sep = "\n"

    pl = ax.errorbar(x, y, xerr=xerr, yerr=yerr, **ebar_kws)
    if add_label:
        label = sep.join(textlist)
    else:
        label = None
    if plot_line:
        if smoother_x:
            x = np.linspace(x.min(), x.max(), 100)
        ax.plot(
            x,
            func(transform_x(x), *fitp),
            **line_kws,
            label=label,
            color=pl[0].get_color(),
        )

    return ax, pl


#    ax.annotate(sep.join(textlist), xy=(0.2, 0.98), xycoords='axes fraction',
#                    verticalalignment='top')
#    ax.legend()


def plot_fit_curve(
    x,
    y,
    func,
    yerr=None,
    p0=None,
    bounds=(-np.inf, np.inf),
    ax=None,
    ebar_kws=None,
    line_kws=None,
    plot_line=True,
    return_cov=False,
    param_names=None,
    units=None,
    smoother_x=False,
    custom_label=None,
    add_label=True,
):
    if return_cov:
        fitp, fitp_cov, chi2, ndof = fit_curve(
            x, y, func, yerr=yerr, p0=p0, bounds=bounds, return_cov=return_cov
        )
        fitp_err = np.sqrt(np.diag(fitp_cov))
        if yerr is None:
            fitp_cov *= chi2 / ndof
    else:
        fitp, fitp_err, chi2, ndof = fit_curve(
            x, y, func, yerr=yerr, p0=p0, bounds=bounds
        )
        if yerr is None:
            fitp_err *= np.sqrt(chi2 / ndof)

    if ax is not None:

        ax, pl = plot_nice_fit(
            x,
            y,
            func,
            fitp,
            fitp_err,
            chi2,
            ndof,
            yerr=yerr,
            ax=ax,
            ebar_kws=ebar_kws,
            line_kws=line_kws,
            plot_line=plot_line,
            param_names=param_names,
            units=units,
            smoother_x=smoother_x,
            custom_label=custom_label,
            add_label=add_label,
        )
    if return_cov:
        return ax, (fitp, fitp_cov, chi2, ndof)
    return ax, (fitp, fitp_err, chi2, ndof)


def plot_fit_poly(
    x,
    y,
    ax=None,
    yerr=1,
    deg=1,
    ebar_kws=None,
    line_kws=None,
    plot_line=True,
    transform_x=lambda x: x,
    return_cov=False,
    param_names=None,
):

    if return_cov:
        fitp, fitp_cov, chi2, ndof = fit_poly(
            transform_x(x), y, yerr, deg, return_cov=return_cov
        )
        fitp_err = np.sqrt(np.diag(fitp_cov))
    else:
        fitp, fitp_err, chi2, ndof = fit_poly(transform_x(x), y, yerr, deg)

    ax, pl = plot_nice_fit(
        x,
        y,
        lambda x, *p: np.poly1d(p)(x),
        fitp,
        fitp_err,
        chi2,
        ndof,
        yerr=yerr,
        ax=ax,
        ebar_kws=ebar_kws,
        line_kws=line_kws,
        plot_line=plot_line,
        transform_x=transform_x,
        param_names=param_names,
    )
    if return_cov:
        return ax, (fitp, fitp_cov, chi2, ndof)
    return ax, (fitp, fitp_err, chi2, ndof)


from scipy.optimize import minimize


def fit_xy_xerr_yerr(x, y, xerr, yerr, func, p0, odr_fit=True):
    p0, _ = curve_fit(lambda x, *p: func(p, x), x, y, p0=p0)

    if odr_fit:
        model = odr.Model(func)
        data = odr.RealData(x, y, sx=xerr, sy=yerr)
        myodr = odr.ODR(data, model, beta0=p0)
        out = myodr.run()
        pfit = out.beta
        red_chi2 = out.res_var
        ndof = len(x) - 2
        chi2 = red_chi2 * ndof
        # by default cov is multiplied by chi2/ndof, undo that here
        pfit_err = out.sd_beta / np.sqrt(chi2 / ndof)
    else:
        if len(p0) > 2:
            raise RuntimeError(
                "Can only do for 2 parameters (and should be straight line)"
            )

        def minfunc(p):
            a, b = p
            sig_sq = yerr ** 2 + b ** 2 * xerr ** 2
            return np.sum((y - func(p, x)) ** 2 / sig_sq)

        res = minimize(minfunc, x0=p0)
        pfit = res["x"]
        chi2 = res["fun"]
        ndof = len(x) - 2
        pfit_err = np.sqrt(np.diag(res["hess_inv"]))
    return pfit, pfit_err, chi2, ndof


def plot_fit_xy_xerr_yerr(
    x,
    y,
    xerr,
    yerr,
    func,
    p0,
    ax=None,
    ebar_kws=None,
    line_kws=None,
    plot_line=True,
    transform_x=lambda x: x,
    param_names=None,
    units=None,
    smoother_x=True,
    custom_label=None,
    odr_fit=True,
    add_label=True,
):
    """
    Should be like this: func = lambda p, x: p[0] + p[1]*x
    """

    fitp, fitp_err, chi2, ndof = fit_xy_xerr_yerr(
        x, y, xerr, yerr, func, p0=p0, odr_fit=odr_fit
    )

    ax, pl = plot_nice_fit(
        x,
        y,
        lambda x, *p: func(p, x),
        fitp,
        fitp_err,
        chi2,
        ndof,
        xerr=xerr,
        yerr=yerr,
        ax=ax,
        ebar_kws=ebar_kws,
        line_kws=line_kws,
        plot_line=plot_line,
        param_names=param_names,
        units=units,
        smoother_x=smoother_x,
        custom_label=custom_label,
        add_label=add_label,
    )

    return ax, (fitp, fitp_err, chi2, ndof)


def bootstrap_fit(x, y, func, yerr=None, ci=0.68, nboot=200, p0=None):
    out = []
    n = len(x)
    ypreds = []
    xspace = np.linspace(x.min(), x.max(), 100)
    for _ in range(nboot):
        randi = np.random.randint(0, n, n)
        popt, _ = curve_fit(func, x[randi], y[randi], sigma=yerr[randi], p0=p0)
        p0 = popt
        ypreds.append(func(xspace, *popt))
        out.append(popt)
    mean = np.mean(out, axis=0)
    std = np.std(out, axis=0)
    pred_mean = np.mean(ypreds, axis=0)
    pred_low = np.quantile(ypreds, 0.5 - ci / 2, axis=0)
    pred_high = np.quantile(ypreds, 0.5 + ci / 2, axis=0)

    return (
        mean,
        std,
        xspace,
        pred_mean,
        pred_low,
        pred_high,
    )
