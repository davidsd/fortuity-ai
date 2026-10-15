#!/bin/bash
#SBATCH --job-name=slice-prep
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=${REPO}/log/sprep_%A_%a.out
# usage: sbatch --array=0-$((K-1)) --cpus-per-task=C --mem=M slice_prep.sbatch NAME K CASE [fullpar flags]
#   one column slice per array task -> oneloop_orders/prep/NAME/slice_<i>of<K>; then merge.sbatch NAME K
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
NAME=$1; K=$2; shift 2
I=$SLURM_ARRAY_TASK_ID
echo "node $(hostname), cpus $SLURM_CPUS_PER_TASK, slice $I/$K, TMPDIR $TMPDIR ($(df -h $TMPDIR | tail -1 | awk '{print $4}') free)"
rm -rf prep/$NAME/slice_${I}of${K}
python3 -u fullpar.py "$@" --workers $SLURM_CPUS_PER_TASK --xcap 12e9 --workdir $TMPDIR --col-slice $I/$K --prep-dir prep/$NAME/slice_${I}of${K}
test -s prep/$NAME/slice_${I}of${K}/meta.json
echo "SLICE DONE $(date)"
