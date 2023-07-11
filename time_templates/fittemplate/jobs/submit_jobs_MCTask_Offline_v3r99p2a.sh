#!/usr/bin/env sh

jobfile=job_fit_MCTask_Offline_v3r99p2a_Xmax_SD.sh

HIM=EPOSLHC
for primary in proton iron; do
	for energy in 19.0_19.5 19.5_20.0; do
		echo "./${jobfile} ${HIM} ${primary} ${energy}" | \
			qsub -V -j oe -d $PWD -q short -o ttt_${HIM}_${primary}_${energy}_Xmax_SD.log -N ttt_${HIM}_${primary}_${energy}
	done
done

