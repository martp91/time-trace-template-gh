import argparse
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
import gc

try:
    from tqdm import tqdm
except:
    tqdm = lambda x: x

from matplotlib.gridspec import GridSpec
from time_templates import data_path

from time_templates.misc.energy import SdlgE_resolution
from time_templates.misc.Xmax import Fd_Xmax_resolution_at_lgE
from time_templates.datareader.get_data import (
    fetch_MC_data_from_tree,
    get_event_from_df,
)
from time_templates.fittemplate.template_fit import TemplateFit
from time_templates.preprocessing.apply_cuts_df import apply_cuts_df


from matplotlib.ticker import LogLocator
from time_templates.templates.universality.names import (
    DICT_COMP_COLORS,
    DICT_COMP_LABELS,
)
from matplotlib.lines import Line2D
from matplotlib.container import ErrorbarContainer
from matplotlib.collections import LineCollection
from math import ceil


def event_viewer(TF, plot_trace_station_i=None):
    event = TF.event
    TF.setup_scale_mask(tq_cut=0.95)
    azim = event["SdAzimuth"][0]
    nstations = event.nstations
    nrows = int(ceil(nstations / 2)) + 1
    f, axes = plt.subplots(nrows, 2, figsize=(6, 2 * nrows))
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]

    TF.setup_scale_mask(tq_cut=0.95)

    Deltat0s = TF.Deltat0s
    t0_sigs = TF.t0_errs
    t0s = [TT.t0 for TT in TF.TTTs]

    xcore, ycore = event["SdCore.fX"] / 1000, event["SdCore.fY"] / 1000

    x = [station.station_pos[0] / 1000 - xcore for station in event.stations]
    y = [station.station_pos[1] / 1000 - ycore for station in event.stations]
    s = [station.WCDTotalSignal for station in event.stations]
    t = [station.TimeNS - event.SdCoreTimeNS for station in event.stations]

    im = ax1.scatter(x, y, s=s, c=t, cmap="bwr")
    arrowsize = 50
    ax1.annotate(
        text="",
        xy=(0, 0),
        xytext=(arrowsize * np.cos(azim), arrowsize * np.sin(azim)),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->"),
    )

    for station in event.stations:
        ax1.annotate(
            text=f"{station.stationid}",
            xy=(
                station.station_pos[0] / 1000 - xcore,
                station.station_pos[1] / 1000 - ycore,
            ),
            xytext=(-15, -10),
            textcoords="offset points",
        )

    ax1.set_xlabel("$x - x_{\\rm c} $ [km]")
    ax1.set_ylabel("$y - y_{\\rm c}$ [km]")
    ax1.set_xlim(*np.array(ax1.get_xlim()) * 1.1)
    ax1.set_ylim(*np.array(ax1.get_ylim()) * 1.2)

    TF.fit_total_signals(ax=ax2)
    ax2.set_ylabel("Signal [VEM]")

    ax2.set_xlim([event.stations[0].r - 100, event.stations[-1].r + 100])

    # traces plot
    if plot_trace_station_i is None:
        indexes = list(range(nstations))
    else:
        indexes = plot_trace_station_i

    iax = 2
    for i, station in enumerate(event.stations):
        if i not in indexes:
            continue
        ax = axes.flatten()[iax]
        iax += 1
        if hasattr(station, "TT"):
            station.plot_trace(
                ax=ax, detector="wcd", plotTT=True, plotMC=False, infostr=False
            )
            ymax = ax.get_ylim()[1]
            yarrow = ymax / 25
            ax.annotate(
                text="$\sigma_{t_0}$",
                xy=(t0s[i] - Deltat0s[i], -yarrow * 2.8),
                ha="center",
                va="center",
                fontsize=8,
            )
            ax.set_ylim((-yarrow * 4.4, ymax))
            ax.errorbar(
                t0s[i] - Deltat0s[i],
                -1 * yarrow,
                xerr=t0_sigs[i],
                xuplims=True,
                xlolims=False,
                color="k",
                marker="",
                lw=1,
            )
            ax.errorbar(
                t0s[i] - Deltat0s[i],
                -1 * yarrow,
                xerr=t0_sigs[i],
                xuplims=False,
                xlolims=True,
                color="k",
                marker="",
                lw=1,
            )
            t = TF.ts[i]
            mask = TF.masks[i]
            tmax = np.max(t[mask])
            ax.axvspan(tmax, 5000, hatch="\\\\\\", fc="none", color="grey", lw=0)
            ax.set_xlim([-50, tmax * 1.2])
        else:
            station.plot_trace(
                ax=ax, detector="wcd", plotTT=False, plotMC=False, infostr=False
            )
            #             ax.set_xlim([-50, 3*station.r])
            ax.axvspan(*ax.get_xlim(), hatch="\\\\\\", fc="none", color="grey", lw=0)

        ax.annotate(
            text=f"ID = {station.stationid} \n $r = {station.r:.0f}$ m",
            xy=(0.5, 0.95),
            va="top",
            xycoords="axes fraction",
            fontsize=8,
        )
        #     ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    #     ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    #     axes[1, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # axes[2, 0].set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    #     axes[2, 1].set_yticks([0, 0.1, 0.2, 0.3])
    plt.tight_layout()
    f.align_ylabels(axes[:, 1])
    f.align_ylabels(axes[:, 0])


#     f.subplots_adjust(wspace=0.3, hspace=0.35)

# Labels to save into csv
LABELS = [
    "EventId",
    "Rmu_fit",
    "lgE_fit",
    "Xmax_fit",
    "DeltaXmumaxXmax_fit",
    "Xmumax_50_fit",
    "cov_Rmu",
    "cov_lgE",
    "cov_Xmax",
    "cov_DeltaXmumaxXmax",
    "cov_Rmu_lgE",
    "cov_Rmu_Xmax",
    "cov_Rmu_DeltaXmumaxXmax",
    "cov_lgE_Xmax",
    "cov_lgE_DeltaXmumaxXmax",
    "cov_Xmax_DeltaXmumaxXmax",
    "fit_success",
    "fit_deviance",
    "fit_ndof",
    "DXmax_Rc_fit",
    "DXmax_Rc_err",
    "DXmax_Rc_chi2",
    "DXmax_Rc_ndof",
    "Xmax_Rc_fit",
    "Xg",
    "DXRc_fit",
    "DXRc_fit_err",
    "theta_Rc",
    "nstations_ttt_fit",
    "fit_dev_no_reg",
    "fit_reg",
]


PLOT = False
NO_SCALE = False  # convert S to effective particles

FIX_NUISSANCE_FACTOR = True

# default cuts on stations for ttt fit
CUTS = {
    "rmin": 500,
    "rmax": 2000,
    "Smin": 5,
    "Smax": 4000,
    "tq_cut": 0.95,
    "trace_cut": 0,
    "neff_cut": 0,
}

# Regularization on fit parameters, this is not used by default
REG = {
    "Rmu": 0,  # 1 / 0.5 ** 2,
    "lgE": 10,  # will be multiplied by 1/uncertainty**2 on sd energy
    "Xmax": 0,  # 1 / 100 ** 2,  # from start time fit approx
    "DeltaXmumaxXmax": 1 / 100**2,
}  # TODO: these should reflect signal uncertainty


def fit_event(
    event,
    reg=REG,
    cuts=CUTS,
    plot=PLOT,
    no_scale=False,
    MClgE=False,
    verbose=True,
    fix_lgE=True,
    fix_Rmu_timefit=True,
    fix_Xmax=False,
    useFdlgE=False,
    useFdXmax=False,
    fit_time_templates=True,
    fix_t0s_final=True,  # Warning if false can be unstable
    use_Xmumax=False,
    fix_Xmumax=True,
    MCXmax=False,
    use_data_pulse_shape=True,
    fix_Deltat0s=False,
    lgE_shift=1,
):
    """fit time-trace-templates on event

    Parameters
    ----------
    event : Event class
        see datareader/event.py
    reg : dict
        dictionary with regularization parameters for Rmu, lgE, Xmax (Default is no regularization)
    cuts : dict
       cuts to make on total signal, rmin, rmax and for trace optionally on expected number of particles
       or vem charge in bin. tq_cut cuts at the end of the trace that contains tq (time quantile) of the
       total signal. tq_cut=0.95 is default, because this proved to work best
    plot : bool
        plot if True, beware that the plot will be shown immediately and you have to close it by hand
    no_scale : bool
        do not convert vem into neff particles with poisson factor default=False
    MClgE : bool
        if True, use MC energy and sample from normal distribution with SdlgE resolution to get some fluct.
        default = False
    verbose : bool
        if True print some stuff. default=True
    fix_lgE : bool
        if True fix energy in template fit and also total signal fit, default=True
    fix_Rmu_timefit : bool
        if True fix Rmu in ttt-fit, default=True
    fix_Xmax : bool
        if True fix Xmax in ttt-fit
    useFdlgE : bool
        if True use FdlgE, only works on hybrid, default=False
    useFdXmax : bool
        if True, use FdXmax, only works on hyrbdi (for example if you want to fit Rmu). default=False. see
        also MCXmax
    fit_time_templates : bool
        if True fit ttt (default), but set to false if you only want to fit Rmu from LDF fit for example
    fix_t0s_final : bool
        if True (default), fix t0's in the final fit round (and only fit Xmax). The t0's were allowed to
        fluctuate in an earlier fit, this works best.
    use_Xmumax : bool
        use Xmumax for muon template (experimental!, does not really work yet). default=False
    fix_Xmumax : bool
        if use_Xmumax then still fix_Xmumax, default=True, but has no effect if not use_Xmumax
    MCXmax : bool
        if useFdXmax but it is MC sims, then also set this to True, to sample Xmax from trueMC with resolution
        from FD. default=False
    use_data_pulse_shape : bool
        get MuonPulseShape from average of PMTs and use this as a detector response (tau=40-60ns or
        something). default=True, for MC this has no effect
    fix_Deltat0s : bool
        do not fit nuissance Deltat0s, ttt-fit does not work well with this to True. Default=False
    lgE_shift : bool
        shift the lgE' = lgE_shift * lgE. For example for sys uncertainty lgE_shift=1.14. default=1
    """
    if verbose:
        pprint = lambda *x: print(*x)
    else:
        pprint = lambda *x: None

    pprint(f"EventId: {event.eventid}")

    TF = TemplateFit(
        event,
        verbose=False,
        station_cuts={
            "r": [cuts["rmin"], cuts["rmax"]],
            "LowGainSat": (-0.1, 0.1),
        },
        do_start_time_fit=False,
        use_Xmumax=use_Xmumax,
        use_data_pulse_shape=use_data_pulse_shape,
    )

    pprint("Fitting start times")
    # This is on all stations (checked) no cuts
    TF.fit_start_times(ax=None)  # this is already done at init?
    TF.setup_all(TF.Rmu, TF.lgE, TF.Xmax, TF.DeltaXmumaxXmax)  # do this to set new r

    if useFdlgE:
        if MClgE:
            # sample fd energy sigma=7.5%
            TF.lgE = np.random.normal(event.MClgE, 0.075 / np.log(10))
        else:
            TF.lgE = event.FdlgE
    else:
        if MClgE:
            # Because SdlgE is biased, take energy as sample from MC with resolution from SD
            TF.lgE = np.random.normal(event.MClgE, SdlgE_resolution(event.MClgE))
        else:
            TF.lgE = event.SdlgE

    TF.lgE += np.log10(lgE_shift)

    if useFdXmax:
        if MCXmax:
            TF.Xmax = np.random.normal(event.MCXmax, Fd_Xmax_resolution_at_lgE(TF.lgE))
        else:
            TF.Xmax = event.FdXmax
    else:
        # If uncertainty is too large ignore ?
        if np.sqrt(TF.var_DXmax_Rc_fit) / TF.Xmax_Rc_fit < 0.15:
            TF.Xmax = TF.Xmax_Rc_fit
    pprint(f"Start Xmax (from RC fit) is {TF.Xmax}")
    pprint()
    reg_lgE = reg["lgE"] / SdlgE_resolution(TF.lgE) ** 2
    reg_Rmu = reg["Rmu"]

    pprint("Fitting total signals")
    if fix_Rmu_timefit:
        m = TF.fit_total_signals(
            lgE_0=TF.lgE,
            reg_lgE=reg_lgE,
            reg_Rmu=reg_Rmu,
            fix_lgE=fix_lgE,
            ax=None,
            Xmax=TF.Xmax,
        )
    else:
        m = TF.fit_total_signals(
            lgE_0=TF.lgE,
            reg_lgE=reg_lgE,
            reg_Rmu=reg_Rmu,
            fix_lgE=fix_lgE,
            ax=None,
            Xmax=TF.Xmax,
        )

    if fit_time_templates:
        # Cut on predicted signal from total signal fit
        pprint(
            f"Removing stations S(r) < {cuts['Smin']:.0f} and S(r) > {cuts['Smax']:.0f}"
        )
        pprint(f"Number of stations before: {TF.nstations}")
        TF.station_cuts["Stotal_fit"] = (cuts["Smin"], cuts["Smax"])
        # reset stations with new cut
        TF.setup_all(
            TF.Rmu,
            TF.lgE,
            TF.Xmax,
            0,
            tq_cut=cuts["tq_cut"],
            trace_cut=cuts["trace_cut"],
            neff_cut=cuts["neff_cut"],
        )
        pprint("Number of stations after", TF.nstations)
        pprint(f"after only LDF fit Rmu = {TF.Rmu:.3f}, lgE = {TF.lgE:.3f}")

        TF.reset_fit()

        pprint("Fitting templates for Rmu, lgE, Xmax, with toffset from previous fit")

        m = TF.fit(
            Rmu_0=TF.Rmu,
            lgE_0=TF.lgE,
            Xmax_0=TF.Xmax,
            fix_Rmu=fix_Rmu_timefit,
            fix_lgE=fix_lgE,
            fix_Xmax=fix_Xmax,
            fix_Xmumax=fix_Xmumax,
            fix_t0s=fix_Deltat0s,
            reg_Rmu=reg["Rmu"],
            reg_lgE=reg_lgE,
            reg_Xmax=reg["Xmax"],
            reg_DeltaXmumaxXmax=reg["DeltaXmumaxXmax"],
            no_scale=no_scale,
        )
        TF.Rmu_time_fit = TF.Rmu

        # Refit total signal for Rmu with better Xmax
        if fix_Rmu_timefit:
            pprint("Fitting total signal again for Rmu")
            mtot = TF.fit_total_signals(
                lgE_0=TF.lgE, reg_lgE=reg_lgE, reg_Rmu=reg_Rmu, fix_lgE=fix_lgE, ax=None
            )
            TF.setup_all(
                TF.Rmu,
                TF.lgE,
                TF.Xmax,
                0,
                tq_cut=cuts["tq_cut"],
                trace_cut=cuts["trace_cut"],
                neff_cut=cuts["neff_cut"],
            )

            # Fit again with better Rmu
            pprint(
                f"Fitting ttt again for Xmax with fixed Rmu = {TF.Rmu} and fixed Deltat0s"
            )
            m = TF.fit(
                Rmu_0=TF.Rmu,
                lgE_0=TF.lgE,
                Xmax_0=TF.Xmax,
                fix_Rmu=True,
                fix_lgE=True,
                fix_Xmax=fix_Xmax,
                fix_Xmumax=fix_Xmumax,
                fix_t0s=True,
                reg_Rmu=reg["Rmu"],
                reg_lgE=reg_lgE,
                reg_Xmax=reg["Xmax"],
                reg_DeltaXmumaxXmax=reg["DeltaXmumaxXmax"],
                no_scale=no_scale,
            )

        TF.ndof = TF.ndof_tt
        TF.calc_goodness_of_fit(
            m.values, neff_cut=1
        )  # neff_cut=1 here, this has some influence on mask
    else:
        TF.ndof = TF.ndof_LDF
        TF.deviance = m.fval

    try:
        Rmu_MC = event["Rmu"][0]
        lgE_MC = event["MClgE"]
        Xmax_MC = event["MCXmax"]
        theta_MC_deg = np.rad2deg(event["MCTheta"])
    except AttributeError:
        Rmu_MC = 0
        lgE_MC = 0
        Xmax_MC = 0
        theta_MC_deg = 0
    success = (
        m.valid
        and not m.fmin.has_parameters_at_limit
        and not m.fmin.has_reached_call_limit
        and not m.fmin.is_above_max_edm
    )

    lgE_Sd = event["SdlgE"]
    lgE_Fd = event["FdlgE"]
    Xmax_Fd = event["FdXmax"]
    Xmax_Fd_err = event["FdXmax_err"]
    theta_sd_deg = np.rad2deg(event["SdTheta"])
    Rmu_fit = m.values["Rmu"]
    Xmax_fit = m.values["Xmax"]
    lgE_fit = m.values["lgE"]
    Rmu_fit_err = m.errors["Rmu"]
    Xmax_fit_err = m.errors["Xmax"]
    lgE_fit_err = m.errors["lgE"]
    if fix_Rmu_timefit:
        Rmu_fit_err = np.sqrt(TF.Rmu_cov)
    try:
        MCXmumax = event["Xmumax_50"][0]
    except:
        MCXmumax = 0

    s = (
        f"EventId = {event.eventid} \n"
        "\nMC:\n"
        f"$\\theta = {theta_MC_deg:.0f}^\\circ$ \n"
        f"$R^\mu = {Rmu_MC:.2f}$ \n"
        f"$\\lg(E/\\rm eV) = {lgE_MC:.2f}$\n"
        f"$X_{{\\rm max}} = {Xmax_MC:.0f}\ \\rm g/cm^2$\n"
        f"$X^{{\\mu}}_{{\\rm max}} = {MCXmumax:.0f}$\n"
        "\nSD rec: \n"
        f"$\\theta = {theta_sd_deg:.0f}^\\circ$ \n"
        f"$\\lg(E/\\rm eV) = {lgE_Sd:.2f}$\n"
        "\nTime (curvature) fit: \n"
        f"$R_C = {TF.Rc_fit:.0f}\ \\rm m$\n"
        f"$X_{{\\rm max}}(R_C) = {TF.Xmax_Rc_fit:.0f} \pm {np.sqrt(TF.var_DXmax_Rc_fit):.0f} \ \\rm g/cm^2$\n"
        "\nFd rec: \n"
        f"$\\lg(E/\\rm eV) = {lgE_Fd:.2f}$\n"
        f"$X_{{\\rm max}} = {Xmax_Fd:.0f} \pm {Xmax_Fd_err:.0f} \ \\rm g/cm^2$\n"
        "\nTemplate fit:\n"
        f"$R^\mu = {Rmu_fit:.2f} \pm {Rmu_fit_err:.2f} $ \n"
        f"$\\lg(E/\\rm eV) = {lgE_fit:.2f} \pm {lgE_fit_err:.2f} $\n"
        f"$X_{{\\rm max}} = {Xmax_fit:.0f} \pm {Xmax_fit_err:.0f} \ \\rm g/cm^2$\n"
        f"$X^{{\\mu}}_{{\\rm max}} = {TF.Xmumax_50:.0f}$\n"
        f"Successfull fit = {success}\n"
        f"Deviance/ndf = {TF.deviance:.0f}/{TF.ndof:.0f}"
    )
    # TODO fit information, chi2/ndf etc

    if not success:
        pprint()
        pprint("FIT FAILED for event")
        pprint(event)
        pprint(m)

    if plot:
        event_viewer(TF)
        plt.show()
    return m, TF


def fit_events(
    df,
    is_data=True,
    MC=False,
    reg=REG,
    cuts=CUTS,
    plot=False,
    fix_lgE=True,
    useFdlgE=False,
    fix_Xmax=False,
    useFdXmax=False,
    MCXmax=False,
    MClgE=False,
    use_Xmumax=False,
    fix_Xmumax=False,
    fix_Rmu_timefit=True,
    only_t0_fit=False,
    fix_Deltat0s=False,
    lgE_shift=1,
    reg_Xmax=0,
    no_data_pulse_shape=False,
):
    """fit multiple events in df
    --------------------------
    df: pd.DataFrame
        containts events in pandas df with all the preprocessing things
    yes you need to specify is_data(default=True) and MC(defuault=False).
    If you do is_data=True and MC=True, MC is treated as if it were Sd data
    For other params see: fit_event()
    """

    dd = defaultdict(list)

    eventids = df.index.get_level_values(level=0).unique()
    nevents = len(eventids)
    print()
    print(f"Number of events to fit: {nevents}")
    print()

    reg = REG
    reg["Xmax"] = reg_Xmax

    ndone = 0
    nfailed = 0

    for eid in tqdm(eventids):
        try:
            event = get_event_from_df(df, eid, is_data=is_data, MC=MC)
            res, TF = fit_event(
                event,
                plot=plot,
                fix_Rmu_timefit=fix_Rmu_timefit,
                MClgE=MClgE,
                useFdlgE=useFdlgE,
                fix_lgE=fix_lgE,
                MCXmax=MCXmax,
                useFdXmax=useFdXmax,
                fix_Xmax=fix_Xmax,
                use_Xmumax=use_Xmumax,
                fix_Xmumax=fix_Xmumax,
                fit_time_templates=(not only_t0_fit),
                fix_Deltat0s=fix_Deltat0s,
                lgE_shift=lgE_shift,
                reg=reg,
                use_data_pulse_shape=(not no_data_pulse_shape),
            )
            p = res.values
            Rmu = p["Rmu"]
            lgE = p["lgE"]
            Xmax = p["Xmax"]
            try:
                DeltaXmumaxXmax = p["DeltaXmumaxXmax"]
            except KeyError:
                DeltaXmumaxXmax = 0
            deviance = TF.deviance
            ndof = TF.ndof
            success = (
                res.valid
                and not res.fmin.has_parameters_at_limit
                and not res.fmin.has_reached_call_limit
                and not res.fmin.is_above_max_edm
            )
            cov = res.covariance
            cov_Rmu = cov[0, 0]
            cov_RmulgE = cov[0, 1]
            cov_RmuXmax = cov[0, 2]
            cov_lgE = cov[1, 1]
            cov_lgEXmax = cov[1, 2]
            cov_Xmax = cov[2, 2]
            try:
                cov_DeltaXmumaxXmax = cov[3, 3]
                cov_RmuDeltaXmumaxXmax = cov[0, 3]
                cov_lgEDeltaXmumaxXmax = cov[1, 3]
                cov_XmaxDeltaXmumaxXmax = cov[2, 3]
            except IndexError:
                cov_DeltaXmumaxXmax = 0
                cov_RmuDeltaXmumaxXmax = 0
                cov_lgEDeltaXmumaxXmax = 0
                cov_XmaxDeltaXmumaxXmax = 0
            Xmumax_50_fit = TF.Xmumax_50
            if fix_Rmu_timefit:
                cov_Rmu = TF.Rmu_cov

            res = [
                eid,
                Rmu,
                lgE,
                Xmax,
                DeltaXmumaxXmax,
                Xmumax_50_fit,
                cov_Rmu,
                cov_lgE,
                cov_Xmax,
                cov_DeltaXmumaxXmax,
                cov_RmulgE,
                cov_RmuXmax,
                cov_RmuDeltaXmumaxXmax,
                cov_lgEXmax,
                cov_lgEDeltaXmumaxXmax,
                cov_XmaxDeltaXmumaxXmax,
                int(success),
                deviance,
                ndof,
                TF.DXmax_Rc_fit,
                np.sqrt(TF.var_DXmax_Rc_fit),
                TF.chi2_Rc_fit,
                TF.ndof_Rc_fit,
                TF.Xmax_Rc_fit,
                TF.event.Xg,
                TF.DXRc_fit,
                np.sqrt(TF.var_DXRc_fit),
                TF.theta,
                TF.nstations,
                TF.D,
                TF.reg,
            ]
            del TF
            del event
        except Exception as e:
            # Just catch anything to make sure we move along
            print()
            print(f"FAILED {eid}")
            print()
            print(traceback.format_exc())
            nfailed += 1
            res = len(LABELS) * [0]
            res[0] = eid
        except KeyboardInterrupt:
            print(f"Interrupted at {eid}")
            gc.collect()
            return pd.DataFrame(dd)

        for label, r in zip(LABELS, res):
            dd[label].append(r)
        # write_result_line_to_file(res, opened_file)
        ndone += 1
        print()
        print(f"{ndone} / {nevents} events done")
        print()
        gc.collect()
    print("number of events failed", nfailed)
    return pd.DataFrame(dd)


if __name__ == "__main__":
    print("DEPRECATED")
    quit()

    parser = argparse.ArgumentParser()
    parser.add_argument("-inputfile", "-i", help="should be a pickled dataframe")
    parser.add_argument("-outputfile", "-o")
    parser.add_argument("-primary", default="proton")
    parser.add_argument("-energy", default="19_19.5")
    parser.add_argument("-HIM", default="EPOS_LHC")
    parser.add_argument("-nevents", default=None, type=int)
    parser.add_argument("-key", default="new_UUB_SSD_rcola")
    parser.add_argument("--is_data", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--MClgE", action="store_true")
    parser.add_argument("--fix_lgE", action="store_true")
    parser.add_argument("--useFdlgE", action="store_true")
    parser.add_argument("--fix_Xmax", action="store_true")
    parser.add_argument("--useFdXmax", action="store_true")
    parser.add_argument("--MCXmax", action="store_true")
    parser.add_argument("--use_Xmumax", action="store_true")
    parser.add_argument("--fix_Xmumax", action="store_true")
    parser.add_argument("--MC", action="store_true")  # deprecated?
    parser.add_argument("--fix_Rmu_timefit", action="store_true")
    parser.add_argument("--only_t0_fit", action="store_true")
    parser.add_argument("--fix_Deltat0s", action="store_true")
    parser.add_argument("-lgE_shift", default=1, type=float)
    parser.add_argument("-reg_Xmax", default=0, type=float)
    parser.add_argument("--no_data_pulse_shape", action="store_true")

    args = vars(parser.parse_args())
    print("args", args)
    print("Getting data")

    if args["inputfile"] is None:
        df = fetch_MC_data_from_tree(**args)
    else:
        df = pd.read_pickle(args["inputfile"])

    # nevents_before = df.index.get_level_values(level=0).nunique()
    # Only cut on events here. Standard lgE > 19 costheta > 0.6 ~ 50 deg
    if args["is_data"]:
        df = apply_cuts_df(df, cuts={"SdlgE": (19, 22), "SdCosTheta": (0.6, 1)})
    else:
        df = apply_cuts_df(df, cuts={"MClgE": (19, 22), "MCCosTheta": (0.6, 1)})

    nevents = args["nevents"]

    if nevents is not None:
        print(f"only getting {nevents} events")
        eventids = df.index.get_level_values(level=0).unique()
        if nevents < len(eventids):
            eventids = np.random.choice(eventids, nevents, replace=False)
        else:
            print(f"Warning can only get {nevents}")
        df = df.loc[eventids]

    outputfile = args["outputfile"]
    if outputfile is None:
        if args["fix_lgE"]:
            outputfile = os.path.join(
                data_path,
                "fitted_pl/events_fitted_"
                + f"{args['key']}_{args['HIM']}_{args['primary']}_{args['energy']}_fixlgE.csv",
            )
        else:
            outputfile = os.path.join(
                data_path,
                "fitted_pl/events_fitted_"
                + f"{args['key']}_{args['HIM']}_{args['primary']}_{args['energy']}.csv",
            )

    fit_events(df, outputfile, args)
    print("...done")
