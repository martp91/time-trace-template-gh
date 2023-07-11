import os
from math import ceil
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from time_templates.utilities.poisson import poisson_1sigma
from time_templates.misc.Xmax import mean_Xmax_MC, mean_Xmax_data
from time_templates.datareader.station import Station
from time_templates.utilities.atmosphere import Atmosphere
from time_templates import package_path


# WARNING if adst_to_tree changes have to change this also
event_info_keys = json.load(
    open(os.path.join(package_path, "data", "event_info_keys.json"), "r")
)


class Event:
    def __init__(
        self,
        df_event,
        eventid,
        MC=False,
        allUUB=False,
        is_data=False,
        univ_comp=True,
    ):
        self.eventid = eventid

        # Set event info
        for key in event_info_keys:
            try:
                setattr(self, key, df_event[key].iloc[0])
            except KeyError:
                pass

        self.MC = MC
        self.is_data = is_data

        if MC:
            # ignore linter it is set above
            self.lgE = self.MClgE
            self.theta = self.SdTheta
            self.Xmax = self.MCXmax
        else:
            self.lgE = self.SdlgE
            self.theta = self.SdTheta
            if self.FdXmax > 0:
                self.Xmax = self.FdXmax
            else:
                self.Xmax = 750
                # if self.is_data:
                #    self.Xmax = mean_Xmax_data(10 ** self.lgE)
                # else:
                #    self.Xmax = (
                #        mean_Xmax_MC(10 ** self.lgE, "proton", "EPOS-LHC")
                #        + mean_Xmax_MC(10 ** self.lgE, "iron", "EPOS-LHC")
                #    ) / 2
        self.S1000 = self.SdS1000

        self.core = np.array(
            [self["SdCore.fX"], self["SdCore.fY"], self["SdCore.fZ"]], dtype=float
        )  # ignore linter
        self.axis = np.array(
            [self["SdAxis.fX"], self["SdAxis.fY"], self["SdAxis.fZ"]], dtype=float
        )  # ignore linter

        self.stations = []
        self.station_id_map = {}
        self.nstations = 0
        for i, (sid, station) in enumerate(df_event.iterrows()):
            self.nstations += 1
            self.stations.append(
                Station(
                    station,
                    sid,
                    MC,
                    UUB=allUUB,
                    univ_comp=univ_comp,
                )
            )
            self.station_id_map[sid] = i

        # Sort by distance to core, why not total signal?
        rs = [station.r for station in self.stations]
        self.stations = [station for _, station in sorted(zip(rs, self.stations))]
        self.core_height = self.stations[0].station_height  # hottest station CAVE
        self.Xmax_fit = None
        self.Xmumax_fit = None
        if is_data and not MC:
            try:
                self.atm = Atmosphere(gps_seconds=self.GPSSecond)  # ignore linter
            except IndexError as e:
                print("Could not find GDAS in db for event", self)
                raise e
        else:
            self.atm = Atmosphere(model=21)  # TODO model is probably 21 but check

        self.set_geometry(self.core, self.axis)

    def set_geometry(self, core, axis):
        self.core = core
        self.axis = axis
        self.cos_theta = axis[-1]
        self.theta = np.arccos(self.cos_theta)
        # TODO: set azimuth
        self.Xg = self.atm.slant_depth_at_height(self.core_height, self.theta)
        for station in self.iter_stations():
            station.set_SPD(axis, core)

        self.set_stations_Xg()

    def set_stations_Xg(self):
        for station in self.iter_stations():
            station.set_Xg(
                self.atm.Xg_at_station(
                    station.r, station.psi, self.theta, station.station_height
                )
            )

    def get_station(self, stationid):
        return self.stations[self.station_id_map[stationid]]

    def iter_stations(self, cuts=None, verbose=False):
        for station in self.stations:
            if cuts is None:
                yield station
            else:
                survived = True
                for key, cut in cuts.items():
                    try:
                        if not cut[0] < station[key] < cut[1]:
                            survived = False
                            break
                    except AttributeError:  # HACK, if it does not have this value, skip also
                        survived = False
                        break
                if survived:
                    yield station

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            return np.array([getattr(station, key) for station in self.stations])
        except AttributeError as e:
            raise AttributeError(e)

    def __str__(self):
        s = ""
        s += "------------------------------\n"
        s += f"event id: {self.eventid}\n"
        s += "------------------------------\n"
        if self.MC:
            s += "MC\n"
        s += f"lgE/eV: {self.lgE:.2f}\n"
        s += f"theta: {np.rad2deg(self.theta):.0f} deg\n"
        s += f"Xmax : {self.Xmax:.0f} g/cm2 \n"
        s += f"# stations: {self.nstations} \n"
        s += "\n"
        return s

    def __repr__(self):
        return str(self)

    def __del__(self):
        # print("Deleting event class")
        del self.stations
        del self.station_id_map
        try:
            del self.atm
        except:
            pass

    def get_distance_Xmax(self, Xmax=None, height=None):
        if Xmax is None:
            Xmax = self.Xmax
        if height is None:
            height = self.core_height

        Xg = self.atm.slant_depth_at_height(height, self.theta)
        return Xg - Xmax

    def plot_traces(
        self, detector="wcd", plotMC=False, plotTT=False, infostr=False, nmax=9
    ):

        ny = ceil(self.nstations / 3)

        if detector == "both":
            f = plt.figure(figsize=(20, 7 * ny))
        else:
            f = plt.figure(figsize=(20, 3.5 * ny))

        gs = gridspec.GridSpec(
            ny, 3, wspace=0.25, hspace=0.2
        )  # these are the 9 clusters

        axes = []
        for i in range(min(self.nstations, nmax)):
            if detector == "both":
                nested_gs = gridspec.GridSpecFromSubplotSpec(
                    2, 1, subplot_spec=gs[i], hspace=0.06
                )
                ax_w = plt.Subplot(f, nested_gs[0])
                self.stations[i].plot_trace(
                    ax=ax_w, detector="wcd", plotMC=plotMC, plotTT=plotTT
                )
                ax_s = plt.Subplot(f, nested_gs[1])
                self.stations[i].plot_trace(
                    ax=ax_s, detector="ssd", plotMC=plotMC, plotTT=plotTT, infostr=False
                )
                f.add_subplot(ax_w)
                f.add_subplot(ax_s)
                axes.append(ax_w)
                ax_w.sharex(ax_s)
                ax_w.set_xlabel("")
                ax_w.set_xticklabels("")
            else:
                ax = plt.Subplot(f, gs[i])
                # try:
                #    self.stations[i].TT.set_Xmax(self.Xmax_fit)
                #    self.stations[i].TT.set_Xmumax(self.Xmumax_fit)
                # except:
                #    pass
                self.stations[i].plot_trace(
                    ax=ax, detector=detector, plotMC=plotMC, plotTT=plotTT
                )
                f.add_subplot(ax)
                axes.append(ax)
        if infostr:
            evstr = (
                "$\\lg{E/\\mathrm{eV}} = %.1f$" % self.lgE,
                "$\\theta = %.0f^\\circ$" % np.rad2deg(self.theta),
            )
            axes[0].annotate(
                "\n".join(evstr),
                xy=(0.3, 0.85),
                xycoords="axes fraction",
                va="top",
            )

        #         plt.tight_layout()
        # axes[0].legend(loc=1)
        f.align_ylabels()
        return axes

    def plot_ldf(self, detector="wcd", plotMC=False, ax=None):
        if ax is None:
            f, ax = plt.subplots(1)

        if detector == "wcd":

            ax.errorbar(
                self["r"],
                self["WCDTotalSignal"],
                yerr=poisson_1sigma(self["WCDTotalSignal"], self["WCDTotalSignal_err"]),
                marker="s",
                color="k",
                ls="",
                label="WCD total signal",
            )

            if plotMC:
                ax.plot(
                    self["r"],
                    [station.wcd_trace.get_muon_signal() for station in self.stations],
                    color="b",
                    marker="o",
                    ls="",
                    label="WCD muon signal (MC)",
                )
            ax.set_ylabel("Signal [VEM]")
        else:
            ax.errorbar(
                self["r"],
                self["SSDTotalSignal"],
                yerr=poisson_1sigma(self["SSDTotalSignal"], self["SSDTotalSignal_err"]),
                marker="s",
                color="k",
                ls="",
                label="SSD total signal",
            )
            ax.set_ylabel("Signal [MIP]")

        ax.set_yscale("log")
        ax.set_xlabel("$r$ [m]")
        evstr = (
            "$\\lg{E/\\mathrm{eV}} = %.1f$" % self.lgE,
            "$\\theta = %.0f^\\circ$" % np.rad2deg(self.theta),
        )
        ax.annotate(
            "\n".join(evstr),
            xy=(0.95, 0.6),
            xycoords="axes fraction",
            va="top",
            ha="right",
        )

        return ax
