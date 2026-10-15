#!/bin/bash
#SBATCH --job-name=r007-ord
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=${REPO}/log/orders_%j.out
# usage: sbatch --cpus-per-task=C --mem=M r007_orders.sbatch CASE [CASE ...]   (rank-only cases r007_o{n}_{in,out})
# Runs in a separate copy of the code (oneloop_orders) so that running T-031 jobs are not touched.
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
python3 -c "from fortuity.hwcoh import ffblas; assert ffblas.HAVE_FFBLAS; print('ffblas (MKL) loaded')"
echo "node $(hostname), cpus $SLURM_CPUS_PER_TASK"
for CASE in "$@"; do
  echo "=== $CASE $(date)"
  python3 fullpar.py $CASE --workers $SLURM_CPUS_PER_TASK --xcap 12e9 --workdir $TMPDIR --span
  cat classes/$CASE/fullpar_seed41.json
done
echo "ORDERS DONE $(date)"
