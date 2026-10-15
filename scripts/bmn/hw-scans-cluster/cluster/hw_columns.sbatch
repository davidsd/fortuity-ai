#!/bin/bash
# Array job: each task runs the hw-complex columns listed on line $SLURM_ARRAY_TASK_ID+1 of $TASKFILE.
# Submit from scripts/bmn/hw-scans-cluster:
#   sbatch --array=0-<ntasks-1> --export=ALL,N=3,RUN=su3_j30,TASKFILE=cluster/tasks_su3_j30.txt,MAXLEN=6 cluster/hw_columns.sbatch
# Output: $FORT_STORE/results/$RUN/task_<id>.jsonl (restartable: finished sectors are skipped on resubmission).
#SBATCH --job-name=hwcol
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=${STORE}/logs/%x-%A_%a.out
set -euo pipefail
W=${REPO}/scripts/bmn/hw-scans-cluster
ulimit -c 0
cd "$W"; source env.sh
export FLINT_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=4 BLIS_NUM_THREADS=4
DIMS=${DIMS:-$FORT_STORE/results/dimsnf_N${N}_j${JD:-30}.pkl}
mkdir -p "$FORT_STORE/results/$RUN"
TASKS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKFILE")
echo "node $(hostname) task $SLURM_ARRAY_TASK_ID: $TASKS"
python3 -u scripts/run_sectors.py "$N" "$DIMS" "$FORT_STORE/results/$RUN/task_${SLURM_ARRAY_TASK_ID}.jsonl" $TASKS
