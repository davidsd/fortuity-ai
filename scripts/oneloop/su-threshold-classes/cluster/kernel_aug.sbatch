#!/bin/bash
#SBATCH --job-name=kernel-aug
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --output=${REPO}/log/kaug_%j.out
# usage: sbatch --cpus-per-task=C --mem=M kernel_cpu.sbatch PREPDIR [kernel_vec_cpu flags]  -> prep/PREPDIR/xker.npy
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
python3 -c "from fortuity.hwcoh import ffblas; assert ffblas.HAVE_FFBLAS; print('ffblas (MKL) loaded')"
P=$1; shift
python3 -u kernel_vec_aug.py prep/$P "$@"
echo "KERNEL AUG DONE $(date)"
