#!/bin/bash
#SBATCH --job-name=o2c-build
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=${REPO}/log/o2c_build_%j.out
# R-007 order 2: candidate t = sum x_i h_i, checks at fresh points, D t and controls for the (2,6) exactness test
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
python3 -u build_cand.py --nvec 2 --workers $SLURM_CPUS_PER_TASK
echo "BUILD DONE $(date)"
