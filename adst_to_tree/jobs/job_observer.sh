#!/usr/bin/env bash

#loop adst files seperate jobs
queue=short

outputdir=/dcache/auger/martp/time_templates/trees/

key=SD_observer_icrc19_SdlgE19_theta53

outputfile=${outputdir}/tree_${key}.root

strfiles=/dcache/auger/martp/observer/merged/${key}.root

echo "Input file: $strfiles"
echo "Output file: ${outputfile}"
echo "./read_adst_boost.sh ${strfiles} -o ${outputfile}" | qsub -V -j oe -d $PWD -q ${queue} -N ${key} -o ${key}.log

#./read_adst_boost.sh ${strfiles} -o ${outputfile}
