#!/bin/bash
# Production array with the parallel engine from code/ (fortuity.hwcoh.run_sectors, ENGINE=par).
# Each array task runs the TASKS on line $SLURM_ARRAY_TASK_ID+1 of $TASKFILE; output $FORT_STORE/results/$RUN/task_<id>.jsonl
#SBATCH --job-name=hwpar
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=${STORE}/logs/%x-%A_%a.out
set -euo pipefail
W=${REPO}/scripts/bmn/hw-scans-cluster
source $W/env.sh
ulimit -c 0                      # never write core files (they landed in home once)
cd "${TMPDIR:-/tmp}"             # work in node-local scratch; the code is found via PYTHONPATH (env.sh)
export ENGINE=par OMP_NUM_THREADS=1 FF_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p "$FORT_STORE/results/$RUN"
OUTF="$FORT_STORE/results/$RUN/task_${SLURM_ARRAY_TASK_ID}.jsonl"
# optional: pre-seed with finished sectors from another run (they are then skipped)
if [ -n "${SEED_JSONL:-}" ] && [ ! -e "$OUTF" ]; then cp "$SEED_JSONL" "$OUTF"; fi
TASKS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKFILE")
echo "node $(hostname) task $SLURM_ARRAY_TASK_ID tmp $TMPDIR: $TASKS"
python3 -u -m fortuity.hwcoh.run_sectors "$N" "$DIMS" "$FORT_STORE/results/$RUN/task_${SLURM_ARRAY_TASK_ID}.jsonl" $TASKS
