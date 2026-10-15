#!/bin/bash
# T-023 scan v2 at j=28 (only after). Sized by jobs 564512/564515 (log 2026-09-28).
# Job array: task a of NA takes every NA-th sector of the cost-ranked list (make_sector_tasks.py; ranked by exact hw
# multiplicity m), so the heavy sectors are spread over the nodes. On each node PAR sectors run at once with CORES cores
# each (evaluation workers = mod-p kernel threads = CORES): the node-local /tmp (100 GB) holds the memmaps of ~2 heavy
# j=28 sectors (up to ~50 GB each), which is what limits PAR. Every sector task runs SO(7), Sp(3), G2 in turn (the hw
# vectors are computed once and served from HW_CACHE). Mod-p kernel: libffblas.so built against MKL (build_ffblas_mkl.sh).
# Restartable: run_sectors skips sectors already in the output files. The SO(7)->G2 restriction is a separate step.
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --job-name=t023-j28
#SBATCH --output=${REPO}/runs/%x-%A_%a.out
set -uo pipefail
V=${REPO}
PY=$V/venv/bin/python
J=${J:-28}; PAR=${PAR:-2}; CORES=${CORES:-40}
NA=${SLURM_ARRAY_TASK_COUNT:-1}; A=${SLURM_ARRAY_TASK_ID:-0}
OUT=$V/runs/scan_j$J; mkdir -p $OUT
cd "$V/code"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ENGINE=par HW_CACHE=$V/runs/hwcache GEN_NMAX=8
export FORTUITY_FFBLAS=$V/ffblas_mkl/libffblas.so
# remove this job's memmaps from the node-local disk whatever happens (a killed job leaves tens of GB behind)
trap 'find /tmp -maxdepth 1 -user $USER -name "[ABGX]_*_*.i32" -newer $OUT/.start_$A -delete 2>/dev/null' EXIT
touch $OUT/.start_$A
echo "[$(date)] host=$(hostname) task $A of $NA, PAR=$PAR CORES=$CORES, code $(cat $V/CODE_COMMIT)"
$PY -c "from fortuity.hwcoh import ffblas; assert ffblas.HAVE_FFBLAS and ffblas._mkl_set is not None; print('ffblas (MKL) loaded')" || exit 1
[ -s $OUT/tasks.txt ] || $PY -u $V/make_sector_tasks.py $V/runs/dims_so7_j$J.pkl $J --heavy 0 > $OUT/tasks.txt
sector_task() {
  lam=$1; n=$2; tag=${lam//,/_}_n$n
  for g in SO_odd:3:so7 Sp:3:usp6 G2:0:g2; do IFS=: read G N nm <<< "$g"
    SLURM_CPUS_PER_TASK=$CORES FF_THREADS=$CORES HWPAR_WORKERS=$CORES GROUP=$G $PY -u -m fortuity.hwcoh.run_sectors \
      $N $V/runs/dims_${nm}_j$J.pkl $OUT/${nm}_a$A.jsonl "$lam:$n" >> $OUT/${nm}_$tag.log 2>&1
  done
  echo "[$(date)] done $lam $n"
}
export -f sector_task; export PY V OUT J A CORES
awk -v na=$NA -v a=$A 'NR % na == a {print $2, $3}' $OUT/tasks.txt | xargs -P $PAR -n 2 bash -c 'sector_task "$0" "$1"'
echo "[$(date)] task $A done"
