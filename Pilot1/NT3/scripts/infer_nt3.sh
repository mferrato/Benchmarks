#!/bin/bash

module load singularity openmpi/3.0.0/gcc-7.3.0

export SINGULARITY_BINDPATH="/data/:/data,/gpfs:/gpfs,/spin1:/spin1,/gs3:/gs3,/gs4:/gs4,/gs5:/gs5,/gs6:/gs6,/gs7:/gs7,/gs8:/gs8,/gs9:/gs9,/gs10:/gs10,/gs11:/gs11,/scratch:/scratch,/fdb:/fdb"

#cd ../../
touch nt3_inference_results.csv
singularity exec --nv /data/classes/candle/candle-gpu.img python /data/ferratomh/fork/Benchmarks/Pilot1/NT3/infer.py 5
