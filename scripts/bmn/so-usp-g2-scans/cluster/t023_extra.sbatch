#!/bin/bash
# T-023 extra checks (): (A) the j=26 SO(7) class at (5,4,4) 2J=3 - seed 2 / GEN_NMAX 10 on 2J=2,3,4 and the
# hairy-product test (j=18 class x w_2); (B) the extra SO(7) graviton at (6,3,3) 2J=2, j=24 - seed 2 / GEN_NMAX 10 on
# 2J=1,2,3 for SO(7) and Sp(3), and the Coulomb-type count (Cartan letters) for both groups. Parallel tasks, one node.
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --job-name=t023-extra
#SBATCH --output=${REPO}/runs/%x-%j.out
set -uo pipefail
V=${REPO}
PY=$V/venv/bin/python; OUT=$V/runs/extra; mkdir -p $OUT
cd "$V/code"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ENGINE=par HW_CACHE=$V/runs/hwcache HWPAR_WORKERS=4
DSO=$V/runs/dims_so7_j26.pkl; DSP=$V/runs/dims_usp6_j24.pkl
echo "[$(date)] host=$(hostname)"
task() {
  case $1 in
    robso7_544) SEED=2 GEN_NMAX=10 GROUP=SO_odd $PY -u -m fortuity.hwcoh.run_sectors 3 $DSO $OUT/rob_so7_544.jsonl 5,4,4:2,3,4 ;;
    robso7_633) SEED=2 GEN_NMAX=10 GROUP=SO_odd $PY -u -m fortuity.hwcoh.run_sectors 3 $DSO $OUT/rob_so7_633.jsonl 6,3,3:1,2,3 ;;
    robsp3_633) SEED=2 GEN_NMAX=10 GROUP=Sp $PY -u -m fortuity.hwcoh.run_sectors 3 $DSP $OUT/rob_sp3_633.jsonl 6,3,3:1,2,3 ;;
    coulso7)    $PY -u $V/g2/extra_checks.py coulomb SO_odd 3 6,3,3 2 $DSO ;;
    coulsp3)    $PY -u $V/g2/extra_checks.py coulomb Sp 3 6,3,3 2 $DSP ;;
    hairy)      $PY -u $V/g2/extra_checks.py hairy $DSO ;;
  esac > $OUT/$1.log 2>&1
  echo "[$(date)] done $1: $(grep -hE '^(COULOMB|HAIRY)' $OUT/$1.log)"
}
export -f task; export PY V OUT DSO DSP
printf '%s\n' robso7_544 hairy robso7_633 robsp3_633 coulso7 coulsp3 | xargs -P 6 -n 1 bash -c 'task "$0"'
echo "[$(date)] done"
