#!/usr/bin/env sh

shebang="#!/usr/bin/env sh"
#HIM=QGSJET-II.04
HIM=EPOS_LHC

for primary in proton iron; do
	for energy in 18.5_19 19_19.5 19.5_20; do
		line="python fit_read_hist.py /dcache/auger/martp/UUB_sims/MPD_new2/${HIM}/${primary}/${energy}/*.root -o=df_MPD_${HIM}_${primary}_${energy}.pl"
		echo $shebang > job.sh
		echo ${line} >> job.sh
		qsub job.sh -q generic -V -d . -N MPD_${HIM}_${primary}_${energy} -j oe -o MPD_${HIM}_${primary}_${energy}.log
		wait 
		rm job.sh
	done
done
