#!/bin/bash
#SBATCH --job-name=d1-check
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=${REPO}/log/d1check_%j.out
# Q_0 t_k = 0 and Q_0 Q_1 t_k = 0 (and Q_0 of the controls) at fresh random points
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
python3 -u d1_check.py classes/r007_d1 --workers $SLURM_CPUS_PER_TASK
echo "D1 CHECK DONE $(date)"
