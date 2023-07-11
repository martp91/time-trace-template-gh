#!/usr/bin/env sh

key='MCTask_Offline_v3r99p2a'
datadir=$(head -n 1 data_directory.txt)
echo $datadir
HIM=EPOSLHC

for primary in proton iron; do
	for energy in 19.0_19.5 19.5_20.0; do
		python ./preprocessing/read_tree_to_df.py \
			$datadir/trees/tree_${key}_${HIM}_${primary}_${energy}.root --do_Rc_fit
	done
done
