#!/bin/sh

#load singularity module
module load singularity

#bind paths
export SINGULARITY_BINDPATH="/data:/data,/gpfs:/gpfs,/spin1:/spin1,/gs3:/gs3,/gs4:/gs4,/gs5:/gs5,/gs6:/gs6,/gs7:/gs7,/gs8:/gs8,/gs9:/gs9,/gs10:/gs10,/gs11:/gs11,/scratch:/scratch,/fdb:/fdb"

#NT3
singularity exec --nv /data/classes/candle/candle-gpu.img python Benchmarks/Pilot1/NT3/nt3_baseline_keras2.py

