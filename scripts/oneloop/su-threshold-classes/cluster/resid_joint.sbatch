#!/bin/bash
#SBATCH --job-name=joint
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=${REPO}/log/joint_%j.out
# R-007 order 2 (summary/fortuity-summary.pdf): resid_joint
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
python3 -u resid_ff.py --joint "$@"
echo "joint DONE $(date)"
