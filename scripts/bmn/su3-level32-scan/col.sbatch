#!/bin/bash
#SBATCH --job-name=t042-col
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --output=${SCRATCH}/t042/log/col_%A_%a.out
# usage: sbatch --array=1-K --cpus-per-task=C --mem=M --export=ALL,TASKS=<file>,HWPAR_WORKERS=W col.sbatch
# line i of TASKS: OUT col [col ...]   (every 2J of each column; run_sectors skips finished sectors)
set -eo pipefail
source ${SCRATCH}/t042/env.sh
set -- $(sed -n "${SLURM_ARRAY_TASK_ID}p" $TASKS)
OUT=$1; shift
echo "node $(hostname) cpus $SLURM_CPUS_PER_TASK workers ${HWPAR_WORKERS:-auto} columns: $*"
python3 -c "from fortuity.hwcoh import ffblas, ffla; print('ffblas', ffblas.HAVE_FFBLAS, 'flint', ffla.HAVE_FLINT)"
python3 -u -m fortuity.hwcoh.run_sectors 3 dims_su3_j32.pkl results/$OUT "$@"
echo "TASK DONE $(date)"
