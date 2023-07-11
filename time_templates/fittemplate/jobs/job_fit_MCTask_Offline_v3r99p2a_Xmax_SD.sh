#!/usr/bin/env sh

nevents=4000
key='MCTask_Offline_v3r99p2a'
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

output_filename=events_fitted_${key}_${HIM}_${primary}_${energy}_Xmax_SD.csv

function cleanup
{
	echo "Cleaning up, moving files"
	mv -vf $tmp/$output_filename $outputdir
	rm $tmp/$pickle_filename
	exit -1
}

#when the job is stopped or exited cleanup and move the files
trap cleanup SIGTERM
trap cleanup EXIT

python fit_events.py -i=$tmp/$pickle_filename -o=$tmp/$output_filename \
        -HIM=$HIM -key=$key -nevents=$nevents --MC \
	--fix_Rmu_timefit \
       	--MClgE --fix_lgE 


