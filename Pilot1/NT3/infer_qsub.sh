#!/bin/bash
#COBALT -n 8 
#COBALT -q debug-cache-quad
#COBALT -A CSC249ADOA01
#COBALT -t 1:00:00

set -x

# CONDA_ENV=py3.6_tf1.4
# source activate $CONDA_ENV
module load miniconda-3.6/conda-4.4.10

# -d <depth> specifices num CPUS (cores) per processing element.
# -j <num> specifies num CPUS (cores) per compute unit.
# -n <num> specifies num processing elements needed by the application
# -N <num> specifies num processing elements per node.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR

# aprun -n 64 -N 1 -d 256 -j 4 -cc depth -e KMP_SETTINGS=1 -e KMP_BLOCKTIME=30 -e KMP_AFFINITY="granularity=fine,verbose,compact,1,0" -e OMP_NUM_THREADS=144 -b $DIR/nt3_baseline_keras2.sh
aprun -n 8 -N 1 -d 256 -j 4 -cc depth -e KMP_SETTINGS=1 -e KMP_BLOCKTIME=30 -e KMP_AFFINITY="granularity=fine,verbose,compact,1,0" -e OMP_NUM_THREADS=144 -b $DIR/infer.py


# source deactivate


