#!/bin/bash
# T-023 checks (): (1) robustness of the SO(7) non-graviton sectors at j=20,22,24 and their 2J neighbours with
# SEED=2 and GEN_NMAX=10; (3) complete Sp(3) hw scan at j=20,22,24 (S-dual comparison with SO(7); the +[3,0] at j=24
# in the SO(7)-Sp(3) BMN index difference). Parallel tasks on one node, shared hw cache.
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --job-name=t023-checks
#SBATCH --output=${REPO}/runs/%x-%j.out
set -uo pipefail
V=${REPO}
PY=$V/venv/bin/python; OUT=$V/runs/checks; mkdir -p $OUT
cd "$V/code"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ENGINE=par HW_CACHE=$V/runs/hwcache HWPAR_WORKERS=3 FORTUITY_WORKERS=40
echo "[$(date)] host=$(hostname)"
[ -s $V/runs/dims_usp6_j24.pkl ] || $PY -u -c "
import pickle; from fortuity.hwcoh.dims import partition_dims_nf
pickle.dump(partition_dims_nf(3, 24, group='Sp'), open('$V/runs/dims_usp6_j24.pkl', 'wb')); print('usp6 dims done', flush=True)"
task() {
  kind=$1; arg=$2; tag=${arg//[,:]/_}
  if [ "$kind" = rob ]; then
    SEED=2 GEN_NMAX=10 GROUP=SO_odd $PY -u -m fortuity.hwcoh.run_sectors 3 $V/runs/dims_so7_j26.pkl $OUT/rob_so7_$tag.jsonl $arg > $OUT/rob_so7_$tag.log 2>&1
  else
    GEN_NMAX=8 GROUP=Sp $PY -u -m fortuity.hwcoh.run_sectors 3 $V/runs/dims_usp6_j24.pkl $OUT/usp6_$tag.jsonl $arg > $OUT/usp6_$tag.log 2>&1
  fi
  echo "[$(date)] done $kind $arg"
}
export -f task; export PY V OUT
{
  for s in 4,3,3:1,2,3 4,4,3:2,3,4 4,4,4:3,4,5; do echo rob $s; done
  $PY -c "
tasks = []
for j in (24, 22, 20):
    s = j // 2
    for q1 in range(s, -1, -1):
        for q2 in range(min(q1, s - q1), -1, -1):
            q3 = s - q1 - q2
            if q3 <= q2:
                tasks.append((j, (q1, q2, q3)))
tasks.sort(key=lambda t: (-t[0], -t[1][2], -t[1][1]))
for j, l in tasks:
    print('sp', ','.join(map(str, l)))
"
} | xargs -P 26 -n 2 bash -c 'task "$0" "$1"'
echo "[$(date)] done"
