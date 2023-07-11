#!/usr/bin/env sh

key='new_UUB_SSD_rcola'
#key='rcola_Offlinev3r3p4'
datadir=$(head -n 1 data_directory.txt)
echo $datadir

#for HIM in EPOSLHC QGSII04 SIB23; do
for HIM in EPOS_LHC QGSJET-II.04; do
	for primary in proton iron; do
		for energy in 19_19.5 19.5_20; do # 20_20.2; do
			python ./preprocessing/read_tree_to_df.py \
				$datadir/trees/tree_${key}_${HIM}_${primary}_${energy}.root --do_Rc_fit
		done
	done
done
