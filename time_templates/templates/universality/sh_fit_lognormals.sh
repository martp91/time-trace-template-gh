#!/usr/bin/env sh
#python merge_mean_df.py
#python fit_lognormal_traces.py
#python lognormal_templates.py

datapath=/home/mart/auger/data/time_templates

#for fl in ${datapath}/binned_df/*.pl; do
#        echo $fl
#        filename=$(basename -- "$fl")
#        filename="${filename%.*}"
#        python fit_lognormals.py $fl -outfile=$datapath/lognormal_fit/${filename}_fitted_lognormals.pl
#done

python fit_lognormals.py ${datapath}/binned_df/*.pl -outfile=${datapath}/lognormal_fit/df_fitted_lognormals.pl
