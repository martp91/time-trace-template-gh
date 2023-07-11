#!/usr/bin/env sh

datadir=$(head -n 1 data_directory.txt)

python ./preprocessing/read_tree_to_df.py \
        $datadir/trees/tree_observer_icrc19_Golden_SdlgE18.8.root --is_data \
        --do_Rc_fit
python ./preprocessing/read_tree_to_df.py \
        $datadir/trees/tree_observer_icrc19_SD_SdlgE19_theta53.root --is_data \
        --do_Rc_fit
