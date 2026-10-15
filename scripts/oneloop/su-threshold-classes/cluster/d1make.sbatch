#!/bin/bash
#SBATCH --job-name=d1-make
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=320G
#SBATCH --time=08:00:00
#SBATCH --output=${REPO}/log/d1make_%j.out
# R-007 order 2, option 1: t_k from kernel_vec_aug (x, z), check eps(t) = -z.f, Q_1 t_k and controls Q_1 Q_0 u
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
python3 -u d1_make.py --prep prep/r007_o2_out_codd --x xker_aug.npy --z zker_aug.npy --nvec 12 --out classes/r007_d1 --workers $SLURM_CPUS_PER_TASK
echo "D1 MAKE DONE $(date)"
