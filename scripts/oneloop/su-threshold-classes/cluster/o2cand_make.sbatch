#!/bin/bash
#SBATCH --job-name=o2c-make
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=${REPO}/log/o2c_make_%j.out
# R-007 order 2: candidate t = sum x_i h_i, D t and the controls (what the (2,6) exactness prep needs)
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
python3 -u build_cand.py --stage make --nvec 2 --workers $SLURM_CPUS_PER_TASK
echo "MAKE DONE $(date)"
