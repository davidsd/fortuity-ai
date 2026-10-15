#!/bin/bash
# T-034: recheck the direct Coulomb count of the (8,3,3) [5,0] column at 2J=2, j=28 (job 567130 gave 41 graviton classes,
# 40 Coulomb-type, for USp(6) and SO(7), with the +64 functional margin and one seed; an underestimated Cartan rank would
# fake a non-Coulomb class). Margins 256 and 1024, two other seeds, both groups: 4 tasks in parallel.
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --job-name=t034-coul
#SBATCH --output=${REPO}/runs/%x-%j.out
set -uo pipefail
V=${REPO}
PY=$V/venv/bin/python; OUT=$V/runs/j28_checks; mkdir -p $OUT
cd "$V/code"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FF_THREADS=4 HW_CACHE=$V/runs/hwcache
export FORTUITY_FFBLAS=$V/ffblas_mkl/libffblas.so
echo "[$(date)] host=$(hostname) code $(cat $V/CODE_COMMIT)"
task() {
  IFS=- read G nm extra seed <<< "$1"
  $PY -u $V/g2/extra_checks.py coulomb $G 3 8,3,3 2 $V/runs/dims_${nm}_j28.pkl $extra $seed > $OUT/recheck-$1.log 2>&1
  echo "[$(date)] done $1: $(grep -h '^COULOMB' $OUT/recheck-$1.log)"
}
export -f task; export PY V OUT
printf '%s\n' Sp-usp6-256-23 Sp-usp6-1024-37 SO_odd-so7-256-23 SO_odd-so7-1024-37 | xargs -P 4 -n 1 bash -c 'task "$0"'
echo "[$(date)] done"
