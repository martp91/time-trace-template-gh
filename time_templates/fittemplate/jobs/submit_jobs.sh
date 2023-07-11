#!/usr/bin/env sh

#jobfile=job_fit_MC_Xmumax_FD.sh
#jobfile=job_fit_MC_Rmu_FD.sh
jobfile=job_fit_MC_Xmax_SD.sh


#for HIM in EPOS_LHC QGSJET-II.04; do
for HIM in EPOSLHC QGSII04 SIB23; do
	for primary in proton iron; do
		for energy in 19_19.5 19.5_20 20_20.2; do
			echo "./${jobfile} ${HIM} ${primary} ${energy}" | \
				qsub -V -j oe -d $PWD -q short -o ttt_${HIM}_${primary}_${energy}_Xmax_SD.log -N ttt_${HIM}_${primary}_${energy}
		done
	done
done

