import os
import argparse
import pandas as pd
import numpy as np

from time_templates.utilities.traces import phase_trace
from time_templates import package_path


def phase_traces(df_file, no_normalize=False, nt=600, dt=25 / 3, univ_comp=False):
    print(f"Loading file {df_file}")
    df = pd.read_pickle(df_file)
    print("By default normalize")
    normalize = True
    if no_normalize:
        normalize = False
        print("Traces will NOT be normalized (non-default)")
    else:
        print("Traces will be normalized to 1/s ds/dt (default)")
    df = phase_traces_df(df, normalize, nt, dt, univ_comp)
    if normalize:
        endstr = "_phased_traces_normalized.pl"
    else:
        endstr = "_phased_traces.pl"
    output_name = os.path.splitext(os.path.basename(df_file))[0] + endstr
    print(f"Saving at {package_path}/data/{output_name}")
    df.to_pickle(os.path.join(package_path + "/data", output_name))


def phase_traces_df(
    df,
    normalize=True,
    nt=600,
    dt=25 / 3,
    univ_comp=False,
    ptf_key="PlaneTimeRes",
    t0_key=None,
):
    nt = int(nt)

    if not univ_comp:

        df["wcd_em_trace"] = df["wcd_total_trace"] - df["wcd_muon_trace"]
        try:
            df["ssd_em_trace"] = df["ssd_total_trace"] - df["ssd_muon_trace"]
        except KeyError:
            pass

    print("Putting traces in phase wrt to plane time front")
    if t0_key is not None:
        print("BEWARE: planetimefront is now not at t=0")

    def apply_phase(row, key, nt=nt, dt=dt, normalize=normalize):
        planetimeres = -row[ptf_key]
        if t0_key is None:
            t0 = 0
        else:
            t0 = row[t0_key]

        # signal can never start before t0
        # TODO this is a (bad) hack
        # t0 = min(t0, planetimeres)

        trace = row[key]
        offset = row["TraceOffset"]
        if np.isnan(planetimeres) or np.isnan(t0) or np.isnan(offset):
            return np.zeros(nt, dtype="float32")
        phased_trace = phase_trace(
            planetimeres, trace, t0=t0, offset=offset, nt=nt, dt=dt, verbose=False
        )

        # Checked: it does not matter if you normalize before or after when taking mean
        # But it does for the variance
        if normalize:
            tracesum = trace.sum()
            # Now can check how much signal is missed by taking this number of bins
            if tracesum > 0:
                phased_trace /= tracesum * dt
        return phased_trace.astype("float32")

    df["wcd_total_trace"] = df.apply(
        lambda x: apply_phase(x, "wcd_total_trace"), axis=1
    )
    df["wcd_muon_trace"] = df.apply(lambda x: apply_phase(x, "wcd_muon_trace"), axis=1)
    df["wcd_em_trace"] = df.apply(lambda x: apply_phase(x, "wcd_em_trace"), axis=1)
    if univ_comp:
        df["wcd_em_mu_trace"] = df.apply(
            lambda x: apply_phase(x, "wcd_em_mu_trace"), axis=1
        )
        df["wcd_em_had_trace"] = df.apply(
            lambda x: apply_phase(x, "wcd_em_had_trace"), axis=1
        )
        try:
            df["ssd_em_mu_trace"] = df.apply(
                lambda x: apply_phase(x, "ssd_em_mu_trace"), axis=1
            )
            df["ssd_em_had_trace"] = df.apply(
                lambda x: apply_phase(x, "ssd_em_had_trace"), axis=1
            )
        except KeyError:
            pass
    df["nt"] = nt
    df["dt"] = dt

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("df_file")
    parser.add_argument("-nt", default=600, type=int)
    parser.add_argument("-dt", default=25 / 3, type=float)
    parser.add_argument("--no_normalize", action="store_true")

    args = parser.parse_args()

    phase_traces(args.df_file, args.no_normalize, args.nt, args.dt)
