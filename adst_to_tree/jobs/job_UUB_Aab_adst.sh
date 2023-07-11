#!/usr/bin/env bash

#loop adst files seperate jobs
queue=short

base_dir=/dcache/auger/martp/CorSims_aab/UUB_recs/36/
output_dir=$base_dir/tree
output_log=$output_dir/logs
mkdir -p $output_dir
mkdir -p $output_log

for him in EPOS QGSII; do
	for primary in proton iron; do
		output_key=tree_UUB_SSD_Aab_${him}_${primary}_36_rings.root
		output_file=$output_dir/$output_key
		strfiles=""
		for file in $base_dir/$him/$primary/*.root; do
			strfiles="${strfiles} ${file}"
		done

		echo "Input files: $strfiles"
		echo "Output file: ${output_file}"
		echo "./read_adst_boost.sh ${strfiles} -o ${output_file}" | qsub -V -j oe -d $PWD -q ${queue} -N $output_key -o ${output_log}/${output_key}.log
#		./read_adst_boost.sh ${strfiles} -o ${output_file}
	done
done

