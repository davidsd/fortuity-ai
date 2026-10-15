#!/bin/bash
#SBATCH --job-name=r007-spsl
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=${REPO}/log/spsl_%A_%a.out
# usage: sbatch --array=0-(K-1) --export=ALL,CASE=r007_o2_out,K=8 r007_orders_span_slice.sbatch
# C-odd spanning certificate of the groups with index = SLURM_ARRAY_TASK_ID mod K (from the hw cache)
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
python3 -u fullpar.py $CASE --csec odd --span-only --span-slice $SLURM_ARRAY_TASK_ID/$K
echo "SPAN SLICE DONE $(date)"
