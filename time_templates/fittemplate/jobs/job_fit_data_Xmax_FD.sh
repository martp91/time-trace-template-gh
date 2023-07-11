#!/usr/bin/env sh

datadir=$(head -n 1 ../data_directory.txt)

filename=df_tree_observer_icrc19_Golden_SdlgE18.8.pl

inputfile=${datadir}/${filename}
outputdir=${datadir}/fitted_csv/

tmp=$TMPDIR

if [ ! -f "$TMPDIR" ]; then
        tmp=/tmp
else
        tmp=$TMPDIR
fi

cp -v $inputfile $tmp

output_filename="events_fitted_observer_icrc19_Golden_SdlgE19_theta53.csv"

python fit_events.py -i=$tmp/$filename -o=$tmp/$output_filename \
        --is_data \
        --fix_lgE \
        --fix_Rmu_timefit

mv -vf $tmp/$output_filename ${outputdir}

rm $tmp/$filename
