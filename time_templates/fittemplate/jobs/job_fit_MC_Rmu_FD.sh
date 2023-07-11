#!/usr/bin/env sh

nevents=2000
#key='rcola_Offlinev3r3p4'
key='new_UUB_SSD_rcola'
HIM=$1
primary=$2
energy=$3

datadir=$(head -n 1 ../data_directory.txt)
pickle_filename=df_tree_${key}_${HIM}_${primary}_${energy}.pl

inputfile=${datadir}/$pickle_filename
outputdir=${datadir}/fitted_csv/

tmp=$TMPDIR

#if [ ! -f "$TMPDIR" ]; then
#        tmp=/tmp
#else
#        tmp=$TMPDIR
#fi

if [ ! -f "$inputfile" ]; then
        python ../preprocessing/read_tree_to_df.py $datadir/trees/tree_${key}_${HIM}_${primary}_${energy}.root
fi

cp -v $inputfile $tmp

output_filename=events_fitted_${key}_${HIM}_${primary}_${energy}_Rmu_FD.csv

python fit_events.py -i=$tmp/$pickle_filename -o=$tmp/$output_filename \
        -HIM=$HIM -key=$key -nevents=$nevents --MC \
       	--MClgE --fix_lgE --useFdlgE \
	--MCXmax --useFdXmax --fix_Xmax \
	--only_t0_fit

mv -vf $tmp/$output_filename ${outputdir}

rm $tmp/$pickle_filename
