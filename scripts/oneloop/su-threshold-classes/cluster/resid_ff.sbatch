#!/bin/bash
#SBATCH --job-name=resid
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=${REPO}/log/resid_%j.out
# R-007 order 2 (summary/fortuity-summary.pdf): resid_ff
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
python3 -c "from fortuity.hwcoh import ffblas; assert ffblas.HAVE_FFBLAS"; python3 -u resid_ff.py prep/$1 --m 48
echo "resid DONE $(date)"
