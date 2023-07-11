#!/usr/bin/env bash

#loop adst files seperate jobs
queue=short

base_dir=/dcache/auger/martp/UUB_sims/new
#HIM=QGSJET-II.04
HIM=EPOS_LHC

for primary in proton iron; do  
	for lgE in 19_19.5 19.5_20; do
		prod=$HIM/$primary/$lgE

		output_dir=$base_dir/$prod/tree

		directory=$base_dir/$prod/
		output_log=$output_dir/logs
		mkdir -p $output_dir
		mkdir -p $output_log
		output_key=tree_new_UUB_SSD_rcola_${HIM}_${primary}_${lgE}.root
		output_file=$output_dir/$output_key
		strfiles=""
		for file in $directory/*.root; do
			strfiles="${strfiles} ${file}"
	#		if [ -f "${output_dir}/${output_key}" ]; then
	#			echo "File ${output_file} exists, skipping"
	#			continue
	#		fi
		done

		echo "Input files: $strfiles"
		echo "Output file: ${output_file}"
		echo "Output log: ${output_log}/${output_key}.log"
		echo "./read_adst_boost.sh ${strfiles} -o ${output_file}" | qsub -V -j oe -d $PWD -q ${queue} -N $output_key #-o ${output_log}/${output_file}.log
	done
done


