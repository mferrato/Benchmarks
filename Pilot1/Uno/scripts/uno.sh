#!/bin/bash

#load modules

#Singularity
module load singularity

#MPI
module load openmpi/3.0.0/gcc-7.3.0

#bind paths
export SINGULARITY_BINDPATH="/data:/data,/gpfs:/gpfs,/spin1:/spin1,/gs3:/gs3,/gs4:/gs4,/gs5:/gs5,/gs6:/gs6,/gs7:/gs7,/gs8:/gs8,/gs9:/gs9,/gs10:/gs10,/gs11:/gs11,/scratch:/scratch,/fdb:/fdb"

#run UNO with multiple mpi ranks
mpiexec -n $SLURM_NTASKS -x NCCL_P2P_DISABLE=1 singularity exec --nv <HOROVOD-IMAGE> python Benchmarks/Pilot1/Uno/uno_baseline_keras2.py --epoch 5 --warmup_lr --optimizer adam --lr 0.00001 --use_landmark_genes --save_path save --model_file uno-adam-0.00001-gpu.p100.$SLURM_NTASKS
