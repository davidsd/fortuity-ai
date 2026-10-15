#!/bin/bash
#SBATCH --job-name=r007-oprep
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --output=${REPO}/log/oprep_%j.out
# usage: sbatch --cpus-per-task=C --mem=M r007_orders_prep.sbatch CASE   -> oneloop_orders/prep/CASE_codd (C-odd sector)
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
python3 -c "from fortuity.hwcoh import ffblas; assert ffblas.HAVE_FFBLAS; print('ffblas (MKL) loaded')"
echo "node $(hostname), cpus $SLURM_CPUS_PER_TASK, TMPDIR $TMPDIR ($(df -h $TMPDIR | tail -1 | awk '{print $4}') free)"
CASE=$1; shift      # extra arguments (e.g. --cartan 4000) are passed to fullpar.py
rm -rf prep/${CASE}_codd
python3 fullpar.py $CASE --csec odd --workers $SLURM_CPUS_PER_TASK --xcap 12e9 --workdir $TMPDIR --prep-dir prep/${CASE}_codd "$@"
test -s prep/${CASE}_codd/meta.json
echo "PREP DONE $(date)"
