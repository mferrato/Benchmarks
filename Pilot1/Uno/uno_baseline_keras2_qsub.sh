#!/bin/bash
#COBALT -n 384 
#COBALT -q default
#COBALT -A CSC249ADOA01
#COBALT -t 9:00:00

set -x

# CONDA_ENV=py3.6_tf1.4

# source activate $CONDA_ENV
module load darshan
module load miniconda-3.6/conda-4.4.10

export KMP_BLOCKTIME=0
export KMP_SETTINGS=0
export KMP_AFFINITY="granularity=fine,compact,1,0"
export OMP_NUM_THREADS=128
export NUM_INTER_THREADS=2
export NUM_INTRA_THREADS=128

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR

# -cc depth –d 128 –j 2 –N 1 –n 1024
# OMP_NUM_INTRA_THREADS=128
# OMP_NUM_INTER_THREADS=1


#aprun -N 1 -n 128 -cc depth -d 128 -j 4 -b python $DIR/uno_baseline_keras2.py --epochs 5 --cache uno-landmark.cache-1 -v --warmup_lr --optimizer adam --lr 1.0        -l 1.0.log.$COBALT_JOBID --use_landmark_genes  --cp --save_path save --model_file uno-adam-1 &
#sleep 10

#aprun -N 1 -n 128 -cc depth -d 128 -j 4 -b python $DIR/uno_baseline_keras2.py --epochs 5 --cache uno-landmark.cache-1 -v --warmup_lr --optimizer adam --lr 0.1        -l 0.1.log.$COBALT_JOBID --use_landmark_genes  --cp --save_path save --model_file uno-adam-0.1 &
#sleep 10

#aprun -N 1 -n 128 -cc depth -d 128 -j 4 -b python $DIR/uno_baseline_keras2.py --epochs 5 --cache uno-landmark.cache-1 -v --warmup_lr --optimizer adam --lr 0.01       -l 0.01.log.$COBALT_JOBID --use_landmark_genes --cp --save_path save --model_file uno-adam-0.01 &
#sleep 10

#aprun -N 1 -n 128 -cc depth -d 128 -j 4 -b python $DIR/uno_baseline_keras2.py --epochs 5 --cache uno-landmark.cache-1 -v --warmup_lr --optimizer adam --lr 0.001      -l 0.001.log.$COBALT_JOBID --use_landmark_genes --cp --save_path save --model_file uno-adam-0.001 &
#sleep 10

#aprun -N 1 -n 128 -cc depth -d 128 -j 4 -b python $DIR/uno_baseline_keras2.py --epochs 5 --cache uno-landmark.cache-1 -v --warmup_lr --optimizer adam --lr 0.0001     -l 0.0001.log.$COBALT_JOBID --use_landmark_genes --cp --save_path save --model_file uno-adam-0.0001 &
#sleep 10

aprun -N 1 -n 128 -cc depth -d 128 -j 4 -b python $DIR/uno_baseline_keras2.py --epochs 5 --cache uno-landmark.cache-1 -v --warmup_lr --optimizer adam --lr 0.00001    -l 0.00001.log.$COBALT_JOBID --use_landmark_genes --cp --save_path save --model_file uno-adam-0.00001 &
sleep 10

aprun -N 1 -n 128 -cc depth -d 128 -j 4 -b python $DIR/uno_baseline_keras2.py --epochs 5 --cache uno-landmark.cache-1 -v --warmup_lr --optimizer adam --lr 0.000001   -l 0.000001.log.$COBALT_JOBID --use_landmark_genes --cp --save_path save --model_file uno-adam-0.000001 &
sleep 10

aprun -N 1 -n 128 -cc depth -d 128 -j 4 -b python $DIR/uno_baseline_keras2.py --epochs 5 --cache uno-landmark.cache-1 -v --warmup_lr --optimizer adam --lr 0.0000001  -l 0.0000001.log.$COBALT_JOBID --use_landmark_genes --cp --save_path save --model_file uno-adam-0.0000001
sleep 10

#aprun -N 1 -n 128 -cc depth -d 128 -j 4 -b python $DIR/uno_baseline_keras2.py --epochs 105 --cache uno-landmark.cache-1 -v --warmup_lr --optimizer adam --lr 0.01      -l 0.01.stop.log.$COBALT_JOBID --use_landmark_genes --cp --save_path save --model_file uno-adam-1stop
#sleep 10

# source deactivate
