#!/usr/bin/env sh

adst_files_dir=/dcache/auger/martp/observer/icrc-2019/Golden/Root/

config=ICRC2019/hybrid.cfg #sd1500_vertical.cfg

output_file=/dcache/auger/martp/observer/merged/golden_observer_icrc19_SdlgE19_theta53.root

$AUGEROFFLINEROOT/bin/selectADSTEvents $adst_files_dir/**/**/*.root -c $config -o $output_file
