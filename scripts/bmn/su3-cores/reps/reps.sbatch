#!/bin/bash
#SBATCH --job-name=su3reps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%A_%a.out
# T-032 follow-up: explicit non-graviton representatives of one SU(3) sector and their SU(3) > SU(2) x U(1) channels
# (fortuity.hwcoh.classreps_par via run_reps.py).  One task-file line per array task:  LAM 2J CHANNELS RECORDS
# Submit from this directory, after `mkdir -p logs` (Slurm opens the log file before the script runs); memmaps on the
# lab's global scratch, as for the T-032 column:
#   sbatch -p yin,shared --array=1 --cpus-per-task=32 --mem=170G --time=2-00:00:00 \
#     --export=ALL,TASKS=tasks_reps_heavy.txt,TMPBASE=/n/netscratch/yin_lab/Lab,REQUIRE_TMP_GB=200 reps.sbatch
#   sbatch -p yin,shared --array=1-5 --cpus-per-task=16 --mem=96G --time=1-00:00:00 \
#     --export=ALL,TASKS=tasks_reps_medium.txt,TMPBASE=/n/netscratch/yin_lab/Lab,REQUIRE_TMP_GB=120 reps.sbatch
# Output: out/reps_<lam>_<2J>.json (summary, small) and out/*.json.gz (representatives, cores, ~10-100 MB each).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ../cluster/env.sh
ulimit -c 0
mkdir -p logs out
TASK=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS")
read -r LAM N2J CHANNELS RECORDS <<< "$TASK"
export TMPDIR=${TMPBASE:-${LOCAL_SCRATCH:-/scratch}}/$USER/su3reps_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
FREE_GB=$(df -BG --output=avail "$TMPDIR" | tail -1 | tr -dc '0-9')
if [ "${REQUIRE_TMP_GB:-0}" -gt 0 ] && [ "$FREE_GB" -lt "$REQUIRE_TMP_GB" ]; then
  echo "$(date) only ${FREE_GB} GB free in $TMPDIR, need ${REQUIRE_TMP_GB} GB: stopping"; exit 3
fi
export MAXLEN=6 MAXLEN_UP=6 GEN_NMAX=3 SEED=${SEED:-1} TARGET_SEED=${TARGET_SEED:-17} NCHECK=${NCHECK:-1}
export FF_THREADS=$SLURM_CPUS_PER_TASK HWPAR_WORKERS=${HWPAR_WORKERS:-4}
echo "$(date) node $(hostname) task: $LAM 2J=$N2J channels $CHANNELS  cpus $SLURM_CPUS_PER_TASK  tmp $TMPDIR (${FREE_GB} GB free)"
python3 -u run_reps.py "$LAM" "$N2J" "$CHANNELS" "$RECORDS" ../cluster/dims_nf_su3_36.pkl out
echo "$(date) done"
