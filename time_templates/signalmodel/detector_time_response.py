#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from time_templates.signalmodel.signal_model import exponential_response

from load_detector_response import load_detector_data
from time_templates import data_path, package_path

print("Reading data")

UUB_WCD_muon_traces = load_detector_data(
    data_path + "/detector_response/UUB_muon_vert_2GeV_traces_2m.root", hastraces=True
)
UB_WCD_muon_traces = load_detector_data(
    data_path + "/detector_response/UB_muon_vert_2GeV_traces_2m.root", hastraces=True
)

UUB_SSD_muon_traces = load_detector_data(
    data_path + "/detector_response/UUB_SSD_muon_vert_2GeV_traces.root", hastraces=True
)
UB_SSD_muon_traces = load_detector_data(
    data_path + "/detector_response/UB_SSD_muon_vert_2GeV_traces.root", hastraces=True
)


mean_UUB_WCD_trace = np.vstack(UUB_WCD_muon_traces["VEMTrace"]).mean(axis=0)
med_UUB_WCD_trace = np.median(np.vstack(UUB_WCD_muon_traces["VEMTrace"]), axis=0)
mean_UB_WCD_trace = np.vstack(UB_WCD_muon_traces["VEMTrace"]).mean(axis=0)

std_UB_WCD_trace = np.vstack(UB_WCD_muon_traces["VEMTrace"]).std(axis=0)
std_UB_WCD_trace /= mean_UB_WCD_trace.max()

std_UUB_WCD_trace = np.vstack(UUB_WCD_muon_traces["VEMTrace"]).std(axis=0)
std_UUB_WCD_trace /= mean_UUB_WCD_trace.max()
n_UB = len(UB_WCD_muon_traces)
n_UUB = len(UUB_WCD_muon_traces)

mean_UUB_WCD_trace /= mean_UUB_WCD_trace.max()
mean_UB_WCD_trace /= mean_UB_WCD_trace.max()
med_UUB_WCD_trace /= med_UUB_WCD_trace.max()


mean_UUB_SSD_trace = np.vstack(UUB_SSD_muon_traces["PMT5Trace"]).mean(axis=0)
mean_UB_SSD_trace = np.vstack(UB_SSD_muon_traces["PMT5Trace"]).mean(axis=0)

std_UUB_SSD_trace = np.vstack(UUB_SSD_muon_traces["PMT5Trace"]).std(axis=0)
std_UUB_SSD_trace /= mean_UUB_SSD_trace.max()

mean_UUB_SSD_trace /= mean_UUB_SSD_trace.max()
mean_UB_SSD_trace /= mean_UB_SSD_trace.max()

f, ax = plt.subplots(1)

t_UUB = np.linspace(0, len(mean_UUB_WCD_trace) * 25 / 3, len(mean_UUB_WCD_trace))
t_UB = np.linspace(0, len(mean_UB_WCD_trace) * 25, len(mean_UB_WCD_trace))

# ax.plot(t_UUB, mean_UUB_WCD_trace, "k-", label="UUB mean")
# ax.plot(t_UB, mean_UB_WCD_trace, "k--", lw=3, label="UB mean")
#
# UB_WCD_muon_traces_test = np.vstack(UB_WCD_muon_traces["VEMTrace"])  # [:30]
# UB_WCD_muon_traces_test /= UB_WCD_muon_traces_test.max(axis=1)[:, None]
## ax.plot(t_UB, UB_WCD_muon_traces_test.T)
# ax.plot(t_UB, np.median(UB_WCD_muon_traces_test, axis=0), "r-", label="UB median")
# out = np.exp(-t_UB / 67) - np.exp(-t_UB / 9)
# ax.plot(t_UB, out / out.max(), "g:", label="UB expon")

from time_templates.utilities.fitting import plot_fit_curve


from scipy.stats import exponnorm

print("Fitting data")


def exponential_norm(t, A, tau, mu, sig):
    return A * exponnorm.pdf(t, tau, mu, sig)


mask = (t_UUB < 500) & (mean_UUB_WCD_trace > 0.001)

_, (fitp_WCD, fitp_err, chi2, ndof) = plot_fit_curve(
    t_UUB[mask],
    mean_UUB_WCD_trace[mask],
    exponential_response,
    ax=ax,
    p0=[1, 60],
    yerr=np.sqrt((std_UUB_WCD_trace[mask] / np.sqrt(n_UUB)) ** 2),
)
mask = (t_UB < 500) & (mean_UB_WCD_trace > 0.001)

_, (fitp_UB_WCD, fitp_err, chi2, ndof) = plot_fit_curve(
    t_UB[mask],
    mean_UB_WCD_trace[mask],
    exponential_response,
    ax=ax,
    p0=[1, 60],
    yerr=np.sqrt((std_UB_WCD_trace[mask] / np.sqrt(n_UB)) ** 2),
)


mask = (t_UUB < 500) & (mean_UUB_SSD_trace > 0.001)


_, (fitp_SSD, fitp_err, chi2, ndof) = plot_fit_curve(
    t_UUB[mask],
    mean_UUB_SSD_trace[mask],
    exponential_norm,
    ax=ax,
    p0=[2, 40, 40, 10],
    yerr=np.sqrt((std_UUB_SSD_trace[mask] / np.sqrt(n_UUB)) ** 2),
)
ax.set_xlabel("t [ns]")

# ax.set_yscale("log")

ax.legend()


# df_UB_WCD_muon_traces_disk = load_detector_data(
#    "../data/UB_muon_disk_2GeV_traces_2m.root", hastraces=True
# )
#
# f, ax = plt.subplots()
#
# ct_bins = np.linspace(0.5, 1, 6)
#
# for ctmin, ctmax in zip(ct_bins[:-1], ct_bins[1:]):
#
#    df_ = df_UB_WCD_muon_traces_disk.query(f"cosTheta > {ctmin} & cosTheta <= {ctmax}")
#    traces = np.vstack(df_["VEMTrace"])
#    mean = np.median(traces, axis=0)
#    ax.plot(mean / mean.max(), label=f"{ctmin} < cos theta < {ctmax}")
# ax.legend()

# plt.show()

# np.save("../data/UUB_WCD_time_response.npy", mean_UUB_WCD_trace)
# np.save("../data/UB_WCD_time_response.npy", mean_UB_WCD_trace)
# np.save("../data/UUB_SSD_time_response.npy", mean_UUB_SSD_trace)
# np.save("../data/UB_SSD_time_response.npy", mean_UB_SSD_trace)
save_path = package_path + "/data/"

print("Saving")
np.savetxt(
    save_path + "/detector_response/UB_WCD_double_exponential_params.txt", fitp_UB_WCD
)
np.save(
    save_path + "/detector_response/UUB_WCD_time_response.npy",
    exponential_response(t_UUB, *fitp_WCD),
)
np.save(
    save_path + "/detector_response/UB_WCD_time_response.npy",
    exponential_response(t_UB, *fitp_WCD),
)
np.save(
    save_path + "/detector_response/UUB_SSD_time_response.npy",
    exponential_norm(t_UUB, *fitp_SSD),
)
np.save(
    save_path + "/detector_response/UB_SSD_time_response.npy",
    exponential_norm(t_UB, *fitp_SSD),
)
print("...done")
