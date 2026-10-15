#!/bin/bash
#SBATCH --job-name=r007-ocpu
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --output=${REPO}/log/ocpu_%j.out
# usage: sbatch --cpus-per-task=C --mem=M r007_orders_cpurank.sbatch PREPDIR   (CPU backend of gpu_rank.py: the same
# blocked elimination with threaded float64 BLAS; fallback while the H200 queue is blocked; writes rank_cpu.json)
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
python3 -c "import numpy, time; a=numpy.random.rand(4096,4096); t=time.time(); a@a; print('dgemm 4096: %.2fs' % (time.time()-t))"
python3 -u gpu_rank.py prep/$1 --cpu --blk 256
cat prep/$1/rank_cpu.json
echo "CPU RANK DONE $(date)"
