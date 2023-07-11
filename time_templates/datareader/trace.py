import numpy as np
import matplotlib.pyplot as plt
from time_templates.templates.universality.names import (
    DICT_COMP_COLORS,
    DICT_COMP_LABELS,
    eMUON,
    eEM_HAD,
    eEM_MU,
    eEM_PURE,
)


class Trace:
    def __init__(
        self, trace, plane_time_front_res, charge, peak, UUB=False, offset=20,
    ):
        self.trace = trace.copy()
        self.Stot = self.trace.sum()
        self.nt = len(self.trace)
        self.UUB = UUB
        if UUB:
            self.dt = 25 / 3  # ns
        else:
            self.dt = 25  # ns
        # number of bins before start time recorded trace, set in adst->tree
        self.offset = offset
        self.set_time(plane_time_front_res)
        self.hasMC = False

        # From charge/peak histogram
        if self.Stot > 0:
            self.charge = charge
            self.peak = peak
            self.AOP = charge / peak
            self.POA = peak / charge  # peak over area, faster to do multiply
        else:
            self.charge = 0
            self.peak = 0
            self.AOP = 1
            self.POA = 1

        # Default trace unit is VEMpeak
        # set default units in VEM charge, also for traces
        self.units = "charge"

    def __del__(self):
        # print("Deleting trace")
        del self.trace

    def set_time(self, plane_time_front_res):
        # t that matches trace array, note offset
        # trace has offset number of bins before trigger time
        self.t0 = plane_time_front_res
        # usually trace is saved (from adst_to_tree) with 20 bins before to not miss anything important
        self.tstart = plane_time_front_res - self.offset * self.dt
        self.tend = self.tstart + (self.nt - 1) * self.dt
        # Make sure dt=dt and nt=nt, this is OK
        self.t = np.linspace(self.tstart, self.tend, self.nt)

    def __len__(self):
        return len(self.trace)

    def add_MC(self, muon_trace, empure_trace=None, emmu_trace=None, emhad_trace=None):
        # default units in VEM Charge

        self.muon_trace = muon_trace
        self.empure_trace = empure_trace
        self.emmu_trace = emmu_trace
        self.emhad_trace = emhad_trace

        self.em_trace = self.trace - self.muon_trace

        self.Smu = self.muon_trace.sum()
        self.Sem = self.em_trace.sum()
        self.has_univ_comp = True
        try:
            self.Sempure = self.empure_trace.sum()
            self.Semmu = self.emmu_trace.sum()
            self.Semhad = self.emhad_trace.sum()
        except:
            self.has_univ_comp = False
        self.hasMC = True

    def fix_units(self, x, units=None):
        " Default unit = VEM charge"
        # WARNING: This sucks for data, because it should be done for each PMT
        if units is None:
            units = self.units
        if units == "charge":
            return x
        elif units == "peak":
            return x * self.AOP
        elif units == "fadc":
            return x * self.charge
        else:
            raise ValueError(
                f"units should be one of [charge, peak, fadc], it is {units}"
            )

    def get_total_trace(self, units=None):
        return self.fix_units(self.trace, units)

    def get_total_signal(self, units=None):
        return self.fix_units(self.Stot, units)

    def get_muon_signal(self, units=None):
        return self.fix_units(self.Smu, units)

    def get_muon_trace(self, units=None):
        return self.fix_units(self.muon_trace, units)

    def get_em_trace(self, units=None):
        return self.fix_units(self.em_trace, units)

    def get_em_signal(self, units=None):
        return self.fix_units(self.Sem, units)

    def get_empure_trace(self, units=None):
        return self.fix_units(self.empure_trace, units)

    def get_empure_signal(self, units=None):
        return self.fix_units(self.Sempure, units)

    def get_emmu_trace(self, units=None):
        return self.fix_units(self.emmu_trace, units)

    def get_emmu_signal(self, units=None):
        return self.fix_units(self.Semmu, units)

    def get_emhad_trace(self, units=None):
        return self.fix_units(self.emhad_trace, units)

    def get_emhad_signal(self, units=None):
        return self.fix_units(self.Semhad, units)

    def plot(self, ax=None, plotMC=False, **kwargs):
        if ax is None:
            _, ax = plt.subplots(1)

        ax.plot(self.t, self.get_total_trace(), drawstyle="steps", **kwargs)

        if plotMC:
            if self.hasMC:
                ax.plot(
                    self.t,
                    self.get_muon_trace(),
                    drawstyle="steps",
                    color=DICT_COMP_COLORS[eMUON],
                    label=f"${DICT_COMP_LABELS[eMUON]}$",
                )
                ax.plot(
                    self.t,
                    self.get_emmu_trace(),
                    drawstyle="steps",
                    color=DICT_COMP_COLORS[eEM_MU],
                    label=f"${DICT_COMP_LABELS[eEM_MU]}$",
                )
                ax.plot(
                    self.t,
                    self.get_emhad_trace(),
                    drawstyle="steps",
                    color=DICT_COMP_COLORS[eEM_HAD],
                    label=f"${DICT_COMP_LABELS[eEM_HAD]}$",
                )
                ax.plot(
                    self.t,
                    self.get_empure_trace(),
                    drawstyle="steps",
                    color=DICT_COMP_COLORS[eEM_PURE],
                    label=f"${DICT_COMP_LABELS[eEM_PURE]}$",
                )
            else:
                print("WARNING: plotMC=True, but trace has no MC")

        ylabel = "Signal [VEM]"
        if self.units == "fadc":
            ylabel = "Signal [FADC]"

        xlabel = "$t - t_{\\rm pf}$ [ns]"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return ax

    def __repr__(self):
        return f"t0 (wrt plane front) = {self.t0:.2f} ns, len(trace) = {self.nt:.0f}, dt = {self.dt:.2f} ns, hasMC = {self.hasMC}"
