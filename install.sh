#!/usr/bin/env bash

pv=$(python -c 'import sys; print(sys.version_info[0])')

if [[ ${pv} != 3 ]]; then
    echo "You really need python 3 (3.9 even I think)"
    echo "Exiting"
    exit
fi

echo
echo "PLEASE use a virtual env, like with anaconda or similar"
echo
echo "The following will install some external python packages like:"
echo "numpy, pandas, matplotlib, uproot, iminuit, numba, scipy, scikit-learn"


echo "Setting up data directory and getting DB if necessary"
echo "Set the data directory to somewhere where you want to store some processed adst files etc."
echo "Needs a few 10s/100s of GBs i think"

echo

read -r -p "Are you sure you want to continue? [Y/n] " yn
while true; do
        case "${yn:-Y}" in
                [Yy]* ) break;;
                [Nn]* ) exit;;
                * ) echo "type y or n";;
        esac
done




dir=$PWD
cd setup/

#ask for data directory
./set_data_dir.py

datadir=$(head -n 1 data_directory.txt)
dbdir_default=$datadir/dbs
echo
echo "If you already have the database you can just skip this, but make sure it is in the specified directory"
echo


read -p "Download dbs? [Y/n]" yn

echo $yn


if [[ "${yn:-Y}" =~ [Yy] ]]; then
        read -p "Where do you want to store the dbs [default = ${dbdir_default}]?, if no password just enter through" dbdir
        dbdir=${dbdir:-$dbdir_default}
        mkdir -p $dbdir
        echo "only getting molecular db"
        ./fetchSQLiteDBs.sh $dbdir
fi

echo
echo "Installing time templates so you can load it into your python scripts as import time_templates"
echo

cd $dir



pip install -e .

if [[ -z "$AUGEROFFLINEROOT" ]]; then
        if [[ -z "$ADSTROOT" ]]; then
                echo "WARNING you need offline/adst to use this to read ADST files"
                echo "If you already have to processed adst files you can skip/ignore this"
                exit 0
        else
                echo "ADST at ${ADSTROOT}"
        fi
else
        echo "Offline at ${AUGEROFFLINEROOT}"
fi

echo
echo "Making adst to tree reader"
echo
cd $dir/adst_to_tree/
make

cd $dir
echo "Running test, should work without any errors"
./run_ttt_fit.py --test



#TODO, FIXME, needs work below

#Below is if you want to create the whole model again from scratch, you probably don't
#This script should run all scripts that are needed to create data
#echo "Will not try to read data files and setup everything"
#echo "e.g. create interpolating tables, create detector time response from sims,"
#echo "Setting up fits of universality and signal time distribution etc..."
#echo "This can take up to about 30 min or so?"
#
#echo "Make sure you have the MC simulations and data trees made with /adst_to_tree"
#echo "[TODO: where to get them from?]"
#
#echo "BEWARE: you need something like 16GB of RAM for this to run OK"

#cd $dir/signalmodel
#echo
#echo "Reading detector time response trees and fitting model"
#echo
#./detector_time_response.py
##
#
##cd $dir/templates
##Uncomment if you want to interpolate MPD energy spec
##echo
##echo "Interpolating E muon spec from MPD. This takes some time and you might want to comment this out"
##echo
##./interpolate_energy_integral.py
##echo
#
#
#cd $dir/templates/universality
#echo
#echo "Making S1000 model"
#echo
#./make_S1000_binned_model.py
#
#echo
#echo "Making signal density model"
#./make_rho_model.py --force #should force
#echo "Again now with cut pred"
#./make_rho_model.py --cut_pred --force
##Maybe do again?
#echo
#cd $dir/templates
##
###Might want to run this alway when refitting the model because total signal pred might change
###And so the cut on below WCDTotalSignal_pred can be different
#echo "Binning simulations in theta, DX, r, psi. This takes a lot of time and you might want to comment this out"
#echo "Except if you want to redo everything from scratch"
#echo
#echo "This also creates dense .pl with traces"
#echo
#
#./sh_bin_df.sh
#
#echo
#echo "Fitting lognormal traces"
#cd $dir/templates/universality
#
#./sh_fit_lognormals.sh
#
#echo "...done?"
