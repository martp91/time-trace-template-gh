import os
from copy import deepcopy
import json
import numpy as np
from time_templates.misc.read_station_altitude import get_station_height
from time_templates.utilities import atmosphere, geometry

from time_templates.datareader.trace import Trace
from time_templates.utilities.constants import *


from time_templates import package_path


# WARNING if adst_to_tree changes have to change this also
event_info_keys = json.load(
    open(os.path.join(package_path, "data", "event_info_keys.json"), "r")
)


class Station:
    def __init__(self, station, stationid, MC=True, UUB=False, univ_comp=True):
        self.stationid = stationid
        self.TT = None
        for key in station.keys():
            if key not in event_info_keys and "trace" not in key:
                try:
                    setattr(self, key, station[key])
                except KeyError:
                    pass
        try:
            if station.IsUUB:
                self.UUB = True
            else:
                self.UUB = False
        except AttributeError:
            self.UUB = UUB
        #        if MC:
        #            # Discrepancy between MC and Sd, so just use Sd rec
        #            self.tpf = -station["MCPlaneTimeRes"] + 42
        #        else:
        #            self.tpf = -station["PlaneTimeRes"]

        self.tpf = -station["PlaneTimeRes"]
        # TODO WARNING: This needs some work for data
        # total_trace = (pmt1 + pmt2 + pmt3)/3 [vem peak]
        # but now total_trace [vem charge] = (pmt1+pmt2+pmt3)/3 * (pmt1_charge + pmt2_charge +
        # pmt3_charge)/(pmt1_peak + pmt2_peak + pmt3_peak) != (pmt1*pmt1_charge/pmt1_peak + pmt2 *
        # pmt2_charge/pmt2_peak + pmt3*pmt3_charge/pmt3_peak)/3. The later would be correct
        # FIX: use vem charge already in read_adst.cxx
        try:
            self.wcd_trace = Trace(
                station["wcd_total_trace"],
                self.tpf,
                station["WCDCharge"],
                station["WCDPeak"],
                offset=station["TraceOffset"],
                UUB=self.UUB,
            )
            if univ_comp:
                self.wcd_trace.add_MC(
                    station["wcd_muon_trace"],
                    station["wcd_em_trace"],
                    station["wcd_em_mu_trace"],
                    station["wcd_em_had_trace"],
                )
            else:
                self.wcd_trace.add_MC(station["wcd_muon_trace"])
        except KeyError:
            pass
            # print("WARNING no traces")

        # TODO: SSD univ comp
        try:
            self.ssd_trace = Trace(
                station["ssd_total_trace"],
                self.tpf,
                station["PMT5Charge"],
                station["PMT5Peak"],
                offset=station["TraceOffset"],
                UUB=self.UUB,
            )
            if univ_comp:
                self.ssd_trace.add_MC(
                    station["ssd_muon_trace"],
                    station["ssd_em_trace"],
                    station["ssd_em_mu_trace"],
                    station["ssd_em_had_trace"],
                )
            else:
                self.ssd_trace.add_MC(station["ssd_muon_trace"])
            self.has_SSD = True
        except KeyError:
            self.has_SSD = False
            pass
        # coordinates system not the same ??
        self.station_height = get_station_height(self.stationid)  # m wrt sea level
        self.station_pos = np.array(
            [self["StationPos.fX"], self["StationPos.fY"], self["StationPos.fZ"]],
            dtype=float,
        )  # ignore linter

        if MC:
            self.r = self.MCr  # ignore linter it does
            self.psi = self.MCpsi
        else:
            self.r = self.Sdr
            self.psi = self.Sdpsi
        # vertical depth along shower axis for universality.

        self.fitres = None

    def __del__(self):
        # print("Deleting station")
        try:
            if self.TT is not None:
                del self.TT
        except:
            pass
        try:
            del self.wcd_trace
        except:
            pass
        try:
            del self.ssd_trace
        except:
            pass

    def set_Xg(self, Xg):
        "will be set at event level"
        self.Xg = Xg

    def add_TimeTemplate(self, TT, Smu_fit, Sem_fit, Sem_mu_fit, Sem_had_fit):
        # Warning when setting values in TT this can change, no copy
        self.TT = TT  # deepcopy(TT)
        self.Smu_fit = Smu_fit
        self.Sem_fit = Sem_fit
        self.Sem_mu_fit = Sem_mu_fit
        self.Sem_had_fit = Sem_had_fit
        self.Stotal_fit = Smu_fit + Sem_fit + Sem_mu_fit + Sem_had_fit

    def set_plane_front_time(self, time, core_time, axis, core):
        # all nanoseconds or meter
        x = self.station_pos - core
        Cptf = np.dot(x, axis)
        Ctpf_res = Cptf - (core_time - time) * C0
        tpf_res = Ctpf_res / C0
        self.tpf = tpf_res
        self.wcd_trace.set_time(tpf_res)
        try:
            self.TT.set_tstart(tpf_res)
        except AttributeError:
            pass

    def set_SPD(self, axis, core):
        x = self.station_pos - core
        Cptf = np.dot(x, axis)
        self.r = np.sqrt(np.dot(x, x) - Cptf**2)
        self.psi = geometry.calc_shower_plane_angle(self.station_pos, core, axis)

    def __str__(self):
        s = ""
        s += "------------------------------ \n"
        s += f"station id {self.stationid}\n"
        s += "------------------------------\n"
        s += f"r: {self.r:.0f} m\n"
        s += f"tpf: {self.tpf:.1f} ns\n"
        s += f"psi : {np.rad2deg(self.psi):.0f} deg\n"
        s += f"WCD total signal : {self.WCDTotalSignal:.1f} VEM charge\n"  # ignore linter it does
        s += f"SSD total signal : {self.SSDTotalSignal:.1f} MIP charge\n"
        s += "\n"
        return s

    def __repr__(self):
        return str(self)

    #    return "\n".join(str(self.__dict__).split(","))

    def __getitem__(self, key):
        return getattr(self, key)

    def plot_trace(
        self, ax=None, detector="wcd", plotMC=True, plotTT=False, infostr=True
    ):
        if "ssd" in detector:
            ax = self.ssd_trace.plot(
                ax=ax,
                plotMC=plotMC,
                color="darkgrey",
                label="SSD",
            )
            ax.set_ylabel("Signal [MIP]")
            # TODO: the labels here are now set in trace, but only for VEM
        if "wcd" in detector:
            ax = self.wcd_trace.plot(ax=ax, plotMC=plotMC, color="k", label="WCD")
        if isinstance(detector, list):
            ax.set_ylabel("Signal [VEM or MIP]")

        if plotTT:
            if hasattr(self, "TT"):
                self.TT.plot_wcd(
                    self.wcd_trace.t,
                    self.Smu_fit,
                    self.Sem_fit,
                    self.Sem_mu_fit,
                    self.Sem_had_fit,
                    ax=ax,
                )
                t99 = np.interp(
                    0.99,
                    self.TT.get_wcd_cdf(
                        self.wcd_trace.t,
                        self.Smu_fit,
                        self.Sem_fit,
                        self.Sem_mu_fit,
                        self.Sem_had_fit,
                    ),
                    self.wcd_trace.t,
                )
                if not np.isfinite(t99):
                    t99 = None
                ax.set_xlim([0, t99])

        if infostr:
            ststr = (
                # "StationId = %.i" % self.stationid,
                "$r = %.0f$ m" % self.r,
                "$\\psi = %.0f^\\circ$" % np.rad2deg(self.psi),
                "$S = %.0f$ VEM" % self["WCDTotalSignal"],
            )
            ax.annotate(
                "\n".join(ststr),
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                va="top",
                ha="right",
                fontsize=10,
            )
        _, xr = ax.get_xlim()
        #        if xr > self.r * 3:
        #            xr = self.r * 3
        # ax.set_xlim([0, xr])
        return ax
