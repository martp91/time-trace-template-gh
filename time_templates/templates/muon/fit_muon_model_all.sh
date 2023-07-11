#!/usr/bin/env sh

#for fl in ../data/mean_traces_binned_cos_theta*.pl; do
#	python fit_muon_model_mean_traces.py $fl 
#done
#
#echo "done fitting Xmumax and lambda"

for fl in ../data/mean_traces_binned_cos_theta*.pl; do
	python fit_muon_model_mean_traces.py $fl --fix_lambda
done

echo "done fitting Xmumax for fixed lambda"
