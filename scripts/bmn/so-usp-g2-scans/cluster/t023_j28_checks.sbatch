#!/bin/bash
# T-023 / T-034 j=28 checks (), one node, tasks in parallel:
# (A) robustness of the two new SO(7) non-graviton classes at j=28 ((5,5,4) [0,1] and (6,4,4) [2,0], both at 2J=4):
#     SEED=2, GEN_NMAX=10 on 2J=3,4,5 of both columns;
# (B) direct Coulomb (Cartan-letter) graviton count on the (8,3,3) [5,0] column at 2J=2,4,6 for USp(6) and SO(7): the kernel
#     index gives T=+1 there (one Coulomb-invisible class), but its Coulomb count is a signed sum of sampled weight
#     dimensions (review, C19), so the "non-Coulomb graviton" reading needs this direct hw count.
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --job-name=t023-j28chk
#SBATCH --output=${REPO}/runs/%x-%j.out
set -uo pipefail
V=${REPO}
PY=$V/venv/bin/python; OUT=$V/runs/j28_checks; mkdir -p $OUT
cd "$V/code"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HW_CACHE=$V/runs/hwcache
export FORTUITY_FFBLAS=$V/ffblas_mkl/libffblas.so
trap 'find /tmp -maxdepth 1 -user $USER -name "[ABGX]_*_*.i32" -newer $OUT/.start -delete 2>/dev/null' EXIT
touch $OUT/.start
echo "[$(date)] host=$(hostname) code $(cat $V/CODE_COMMIT)"
task() {
  case $1 in
    rob554) SLURM_CPUS_PER_TASK=24 FF_THREADS=24 HWPAR_WORKERS=24 ENGINE=par SEED=2 GEN_NMAX=10 GROUP=SO_odd \
              $PY -u -m fortuity.hwcoh.run_sectors 3 $V/runs/dims_so7_j28.pkl $OUT/rob_so7_554.jsonl 5,5,4:3,4,5 ;;
    rob644) SLURM_CPUS_PER_TASK=24 FF_THREADS=24 HWPAR_WORKERS=24 ENGINE=par SEED=2 GEN_NMAX=10 GROUP=SO_odd \
              $PY -u -m fortuity.hwcoh.run_sectors 3 $V/runs/dims_so7_j28.pkl $OUT/rob_so7_644.jsonl 6,4,4:3,4,5 ;;
    coul-*) IFS=- read _ G nm n <<< "$1"; N=3
            FF_THREADS=4 GEN_NMAX=8 $PY -u $V/g2/extra_checks.py coulomb $G $N 8,3,3 $n $V/runs/dims_${nm}_j28.pkl ;;
  esac > $OUT/$1.log 2>&1
  echo "[$(date)] done $1: $(grep -hE '^COULOMB' $OUT/$1.log)"
}
export -f task; export PY V OUT
printf '%s\n' rob554 rob644 coul-Sp-usp6-2 coul-Sp-usp6-4 coul-Sp-usp6-6 coul-SO_odd-so7-2 coul-SO_odd-so7-4 coul-SO_odd-so7-6 \
  | xargs -P 8 -n 1 bash -c 'task "$0"'
echo "[$(date)] done"
