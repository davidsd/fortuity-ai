#!/bin/bash
#SBATCH --job-name=jt
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=${REPO}/log/jt_%j.out
# R-007 order 2 (summary/fortuity-summary.pdf): jt_make
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
python3 -u jt_make.py --src classes/r007_d1 "$@"
echo "jt DONE $(date)"
