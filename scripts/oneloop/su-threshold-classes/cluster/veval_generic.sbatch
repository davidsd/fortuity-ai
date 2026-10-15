#!/bin/bash
#SBATCH --job-name=vevalg
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=${REPO}/log/vevalg_%j.out
# R-007 order 2 (summary/fortuity-summary.pdf): veval_generic
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
python3 -u veval.py "$@" --workers $SLURM_CPUS_PER_TASK --slice-words 100000
echo "vevalg DONE $(date)"
