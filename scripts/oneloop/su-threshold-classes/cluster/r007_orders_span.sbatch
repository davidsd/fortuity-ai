#!/bin/bash
#SBATCH --job-name=r007-span
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=${REPO}/log/span_%j.out
# usage: sbatch --cpus-per-task=C --mem=M r007_orders_span.sbatch CASE   (C-odd spanning certificate from the hw cache)
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
ls -la cache/ | grep "fullpar_hw_$1" || true
python3 -u fullpar.py $1 --csec odd --span-only
echo "SPAN DONE $(date)"
