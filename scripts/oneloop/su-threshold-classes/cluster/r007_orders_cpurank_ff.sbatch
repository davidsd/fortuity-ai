#!/bin/bash
#SBATCH --job-name=r007-offf
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --output=${REPO}/log/offf_%j.out
# usage: sbatch --cpus-per-task=C --mem=M r007_orders_cpurank_ff.sbatch PREPDIR [--with-cartan]
# exact rank mod p with the MKL streaming basis (fortuity.hwcoh.ffblas.StreamBasis), as in T-031's CPU cross-check
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
python3 -c "from fortuity.hwcoh import ffblas; assert ffblas.HAVE_FFBLAS; print('ffblas (MKL) loaded')"
python3 -u cpu_rank.py prep/$1 $2
echo "CPU FF RANK DONE $(date)"
