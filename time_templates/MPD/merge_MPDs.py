import glob
import pandas as pd

from time_templates import data_path

dfs = []

print(f"merging MPD dfs at {data_path}+/MPD/")

for fl in glob.glob(data_path + "/MPD/df_MPD*.pl"):
    print(fl)
    dfs.append(pd.read_pickle(fl))


df = pd.concat(dfs)

df = df[~df["shower_id"].duplicated()]

print(f"Saving at {data_path}/MPD/df_MPD.pl")
df.to_pickle(data_path + "/MPD/df_MPD.pl")
