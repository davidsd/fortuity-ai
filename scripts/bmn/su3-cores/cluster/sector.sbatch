#!/bin/bash
#SBATCH --job-name=su3core
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%A_%a.out
# T-032: one SU(3) hw sector per array task (fortuity.hwcoh.run_sectors, parallel engine). Submit from this directory:
#   sbatch -p yin --array=1-4 --cpus-per-task=32 --mem=170G --time=3-00:00:00 --export=ALL,TASKS=tasks_854_heavy.txt sector.sbatch
# Second-seed recheck (output in ../results_seed2/): add SEED=2 to --export, e.g. TASKS=tasks_854_recheck.txt,SEED=2
# (yin nodes: 48 cpus, ~188 GB, 3-day limit; do not ask for more than ~184G per task)
# Output: ../results/su3_<q1>_<q2>_<q3>_<2J>.jsonl (restartable: finished sectors are skipped on resubmission).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source env.sh
ulimit -c 0
TASK=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS")
# SEED != 1 (a recheck) writes to ../results_seed<SEED>/, since run_sectors skips sectors already in OUT whatever the seed
SEED=${SEED:-1}
RESDIR=../results
if [ "$SEED" != 1 ]; then RESDIR=../results_seed$SEED; mkdir -p "$RESDIR"; fi
OUT=$RESDIR/su3_$(echo "$TASK" | tr ',:' '__').jsonl
# Scratch for the memory-mapped matrices: (hw rows of 2J and 2J+1) x (functionals) x 4 bytes, plus up to 40 GB of
# word evaluations. That is 120-170 GB for 2J = 3, 4, 5 of (8,5,4), more than the ~62 GB free on node-local /scratch
# (the first attempts died with "Bus error"). So set TMPBASE to a big filesystem, e.g. the lab's global scratch:
#   --export=ALL,TASKS=...,TMPBASE=/n/netscratch/<lab>/Lab,REQUIRE_TMP_GB=200
# REQUIRE_TMP_GB makes the task stop at once (instead of hours later) if the scratch has less free space than that.
export TMPDIR=${TMPBASE:-${LOCAL_SCRATCH:-/scratch}}/$USER/su3core_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
FREE_GB=$(df -BG --output=avail "$TMPDIR" | tail -1 | tr -dc '0-9')
if [ "${REQUIRE_TMP_GB:-0}" -gt 0 ] && [ "$FREE_GB" -lt "$REQUIRE_TMP_GB" ]; then
  echo "$(date) only ${FREE_GB} GB free in $TMPDIR, need ${REQUIRE_TMP_GB} GB: stopping"; exit 3
fi
export GROUP=SU ENGINE=par MAXLEN=6 MAXLEN_UP=6 GEN_NMAX=3 SEED
export FF_THREADS=$SLURM_CPUS_PER_TASK HWPAR_WORKERS=${HWPAR_WORKERS:-4}
echo "$(date) node $(hostname) task $TASK -> $OUT  cpus $SLURM_CPUS_PER_TASK  tmp $TMPDIR ($(df -h --output=avail $TMPDIR | tail -1) free)"
python3 -u -m fortuity.hwcoh.run_sectors 3 dims_nf_su3_36.pkl "$OUT" "$TASK"
echo "$(date) done"
