#!/usr/bin/env bash

#loop adst files seperate jobs
queue=short
base_dir=/dcache/auger/martp/pauger/Simulations/libraries/MCTask/Offline_v3r99p2a/IdealMC_CORSIKA/Hybrid_NapoliCORSIKA/
nfiles=2
HIM=EPOSLHC

for primary in proton iron; do
	for lgE in 19.0_19.5 19.5_20.0; do

		output_dir=$base_dir/$HIM/$lgE/$primary/tree

		directory=$base_dir/$HIM/$lgE/$primary
		mkdir -p $output_dir

		output_key=tree_MCTask_Offline_v3r99p2a_${HIM}_${primary}_${lgE}.root
		output_file=$output_dir/$output_key

		n=0
		strfiles=""
		for file in $directory/*.root; do
			if [[ $n -lt $nfiles ]]
			then
				n=$((n+1))
				strfiles="${strfiles} ${file}"
			fi
		done

		echo "Input files: $strfiles"
		echo "Output file: ${output_file}"
		echo "./read_adst_boost.sh ${strfiles} -o ${output_file}" | qsub -V -j oe -d $PWD -q ${queue} -N $output_key -o ${output_file}.log

		#./read_adst_boost.sh ${strfiles} -o ${output_file}
	done
done

