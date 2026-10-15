#!/bin/bash
#SBATCH --job-name=r007-prep
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --output=${REPO}/log/prep_%x_%j.out
# usage: sbatch --cpus-per-task=C --mem=M r007_prep.sbatch CASE [CASE ...]
# For each case: preimage hw vectors, Q_0, evaluation at the target functionals -> prep/CASE/{B,V}.i32, meta.json
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
python3 -c "import scipy" 2>/dev/null || pip install --quiet scipy
echo "node $(hostname), cpus $SLURM_CPUS_PER_TASK, TMPDIR $TMPDIR ($(df -h $TMPDIR | tail -1 | awk '{print $4}') free)"
python3 -c "from fortuity.hwcoh import ffblas; assert ffblas.HAVE_FFBLAS; print('ffblas (MKL) loaded')"
for CASE in "$@"; do
  echo "=== $CASE $(date)"
  rm -rf prep/$CASE
  python3 fullpar.py $CASE --workers $SLURM_CPUS_PER_TASK --xcap 12e9 --workdir $TMPDIR --prep-dir prep/$CASE
  test -s prep/$CASE/meta.json
done
echo "PREP DONE $(date)"
