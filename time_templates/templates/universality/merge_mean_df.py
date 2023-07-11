import glob
import argparse
from collections import defaultdict
import tqdm
import pandas as pd
import glob
from time_templates import data_path


def merge_means(dfs, verbose=True):
    dd = defaultdict(list)

    ct2_bins = dfs[0].index.get_level_values(level=0).unique()
    DX_bins = dfs[0].index.get_level_values(level=1).unique()
    if verbose:
        pprint = lambda *x: print(*x)
    else:
        pprint = lambda *x: None
    pprint("Getting rs with most entries")  # 18.5 might not have 2000-2500 m
    rs = [0]
    for df in dfs:
        rs_ = df.index.get_level_values(level=2).unique()
        if len(rs_) > len(rs):
            rs = rs_
    cps = df.index.get_level_values(level=3).unique()

    pprint("Bins", ct2_bins, DX_bins, rs, cps)

    for ct2_bin in ct2_bins:
        ct2 = ct2_bin.mid
        pprint("at", ct2)
        for DX_bin in DX_bins:
            DX = DX_bin.mid
            for r in rs:
                for cp in cps:
                    new_row = 0
                    n_total = 0
                    for df in dfs:
                        try:
                            row = df.loc[ct2, DX, r, cp]
                            n = row["nstations"]
                            if n <= 0:
                                continue
                            new_row += row * n
                            n_total += n
                        except KeyError:
                            continue

                    if n_total <= 0:
                        continue

                    new_row /= n_total

                    for key, val in new_row.items():
                        if key == "nstations":
                            continue
                        dd[key].append(val)

                    dd["nstations"].append(n_total)
                    dd["MCCosTheta_sq_bin_idx"].append(ct2_bin)
                    dd["MCDXstation_bin_idx"].append(DX_bin)
                    dd["MCr_round_idx"].append(r)
                    dd["MCcospsi_round_idx"].append(cp)

    return pd.DataFrame(dd).set_index(dfs[0].index.names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-files", nargs="+", default=None)
    parser.add_argument("-output")

    args = parser.parse_args()

    files = args.files
    if files is None:
        files = glob.glob(data_path + "/mean_df/mean_df_*.pl")
    outputfile = args.output
    dfs = []
    print("Merging mean dataframes")
    for fl in files:
        print(fl)
        df = pd.read_pickle(fl)
        df.index.rename([name + "_idx" for name in df.index.names], inplace=True)
        dfs.append(df)

    df = merge_means(dfs)

    if outputfile is None:
        outputfile = data_path + "/mean_df/df_means_merged.pl"
    print(f"Saving at {outputfile}")
    df.to_pickle(outputfile)
    print("Done")
