#!/bin/bash
now=$(date +"%T")

while read P1 P2 P3 P4 P5 P6 P7 P8 P9
do
JOB= sbatch ./scripts/truba_GCN_single.sh ${P1} ${P2} ${P3} ${P4} ${P5} ${P6} ${P7} ${P8} ${P9}
echo "JobID = ${JOB} for parameters ${P1} ${P2} ${P3} ${P4} ${P5} ${P6} ${P7} ${P8} ${P9} submitted on date $now"
done < ./scripts/params.txt