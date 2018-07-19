#!/bin/bash
#COBALT -n 8 
#COBALT -q debug-cache-quad
#COBALT -A CSC249ADOA01
#COBALT -t 1:00:00

set -x

# CONDA_ENV=py3.6_tf1.4

# source activate $CONDA_ENV
# module load darshan
module load miniconda-3.6/conda-4.4.10

export KMP_BLOCKTIME=0
export KMP_SETTINGS=0
export KMP_AFFINITY="granularity=fine,compact,1,0"
export OMP_NUM_THREADS=62
export NUM_INTER_THREADS=1
export NUM_INTRA_THREADS=62

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR

# aprun -N 1 -n 8 -cc none -b python $DIR/uno_baseline_keras2.py --weights "246776/uno.A=relu.B=32.E=5.O=adam.LR=0.0001.CF=r.DF=df.wu_lr.L1000.D1=1000.D2=1000.D3=1000.weights.h5" --epochs 2 --cache uno-landmark.cache-1 -v -l 8.log.$COBALT_JOBID --use_landmark_genes --cp --save_path save_8 --loss mae --model_file 8 --optimizer sgd --warmup_lr --lr 0.01


aprun -N 1 -n 8 -cc none -b python $DIR/uno_baseline_keras2.py --epochs 2 --cache uno-landmark.cache-1 -v -l 8.log.$COBALT_JOBID --use_landmark_genes --cp --save_path save_8 --loss mae --model_file 8 --optimizer sgd --warmup_lr --lr 0.01

# source deactivate
