#!/bin/bash
#SBATCH --job-name=membership
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=${REPO}/log/memb_%j.out
# usage: sbatch --cpus-per-task=C --mem=M --time=T membership.sbatch eval|rank   (R-007 order 2: are the abelian LW
# vectors Cartan images of T_2 cocycles?  membership.py, summary/fortuity-summary.pdf)
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
if [ "$1" = eval ]; then
  python3 -u membership.py eval prep/r007_o2_out_codd cache/fullpar_hw_r007_o2_out_codd_cartan.pkl lw90_T2.pkl 4000
else
  python3 -c "from fortuity.hwcoh import ffblas; assert ffblas.HAVE_FFBLAS; print('ffblas (MKL) loaded')"
  python3 -u membership.py rank prep/r007_o2_out_codd
fi
echo "MEMBERSHIP $1 DONE $(date)"
