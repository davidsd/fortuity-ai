#!/bin/bash
#SBATCH --job-name=t040-ao
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=${REPO}/log/t040ao_%A_%a.out
# T-040 all orders: one rank-only case per array task (line SLURM_ARRAY_TASK_ID of t040_allorders_cases.txt)
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" t040_allorders_cases.txt)
echo "node $(hostname) case $LINE"
python3 -u fullpar.py $LINE --workers $SLURM_CPUS_PER_TASK --workdir $TMPDIR --span
echo "T040AO DONE $(date)"
