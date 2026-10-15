#!/bin/bash
#SBATCH --job-name=t025-col
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=${REPO}/log/col_%A_%a.out
# usage: sbatch --array=1-K --cpus-per-task=C --mem=M --time=T --export=ALL,TASKS=<file> col.sbatch
# line i of TASKS: N DIMS GEN_NMAX OUT col [col ...]   (every 2J of each column; run_sectors skips finished sectors)
set -eo pipefail
source ${REPO}/env.sh
set -- $(sed -n "${SLURM_ARRAY_TASK_ID}p" $TASKS)
N=$1; DIMS=$2; export GEN_NMAX=$3; OUT=$4; shift 4
echo "node $(hostname) cpus $SLURM_CPUS_PER_TASK N=$N GEN_NMAX=$GEN_NMAX columns: $*"
python3 -c "from fortuity.hwcoh import ffblas, ffla; print('ffblas', ffblas.HAVE_FFBLAS, 'flint', ffla.HAVE_FLINT)"
python3 -u -m fortuity.hwcoh.run_sectors $N $DIMS results/$OUT "$@"
echo "TASK DONE $(date)"
