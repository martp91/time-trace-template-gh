#!/usr/bin/env sh

for primary in proton iron; do
        for energy in 19_19.5 19.5_20; do
                python bin_df.py -energy=$energy -primary=$primary --force --Xmumax
        done
done
