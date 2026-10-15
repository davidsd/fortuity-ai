#!/bin/bash
#SBATCH --job-name=hw-only
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=${REPO}/log/hwonly_%j.out
# usage: sbatch --cpus-per-task=C --mem=M hwonly.sbatch CASE [fullpar flags]   (caches the compressed hw rows, then stops)
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
python3 -u fullpar.py "$@" --hw-only
echo "HW DONE $(date)"
