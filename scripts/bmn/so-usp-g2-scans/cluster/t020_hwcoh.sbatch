#!/bin/bash
# T-020: SO(11) / USp(10) BMN cohomology at the first index difference (j=26), on PI Symmetry (Slurm).
# Only after Submit from the login node, in $R:  sbatch --export=ALL,GROUP=SO_odd t020_hwcoh.sbatch  (or GROUP=Sp)
# Layout: R=${SCRATCH} $R/code = git ls-files of code/ at the commit in $R/CODE_COMMIT;
# $R/venv (python/3.12 module + pip install -e code/[dev]); outputs in $R/runs.
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --job-name=t020
#SBATCH --output=${SCRATCH}
set -euo pipefail
: "${GROUP:?set GROUP=SO_odd or GROUP=Sp}"
N=5; JMAX=26
R=${SCRATCH}
PY=$R/venv/bin/python
OUT=$R/runs; mkdir -p "$OUT"
cd "$R/code"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# workers: $WORKERS if given, else the best throughput for this group in the benchmark output ($BENCH), else 40
if [ -z "${WORKERS:-}" ] && [ -n "${BENCH:-}" ] && [ -s "$BENCH" ]; then
  WORKERS=$(awk -v g="$GROUP:" '$1==g{on=1;next} /^[A-Za-z]/{on=0} on && $2=="workers:"{if($3>best){best=$3;w=$1}} END{print w}' "$BENCH")
fi
export FORTUITY_WORKERS=${WORKERS:-40}
DIMS="$OUT/dims_${GROUP}${N}_j${JMAX}_2J3.pkl"   # 2J <= 3 only: enough for the columns below (2J = 0,1,2 need dims up to 2J = 3)
echo "[$(date)] host=$(hostname) group=$GROUP N=$N workers=$FORTUITY_WORKERS"
if [ ! -s "$DIMS" ]; then
  $PY -u -c "
import pickle, time, sys
from fortuity.hwcoh.dims import partition_dims_nf
t = time.time(); d = partition_dims_nf($N, $JMAX, group='$GROUP', twoJ_max=3, progress=True)
pickle.dump(d, open('$DIMS.tmp', 'wb')); print('dims done', len(d), f'{time.time()-t:.0f}s', flush=True)"
  mv "$DIMS.tmp" "$DIMS"
fi
echo "[$(date)] sectors"
# columns lam = (7,3,3) [4,0] and (5,5,3) [0,2], 2J = 0,1,2 (restartable: finished sectors are skipped)
# SEED (default 1) selects the random evaluation points; a non-default seed writes to its own file
SEED=${SEED:-1}
if [ "$SEED" = "1" ]; then TAG=""; else TAG="_seed$SEED"; fi
SEED=$SEED GROUP=$GROUP GEN_NMAX=12 $PY -u -m fortuity.hwcoh.run_sectors $N "$DIMS" "$OUT/hw_${GROUP}${N}${TAG}.jsonl" 7,3,3:0,1,2 5,5,3:0,1,2
echo "[$(date)] done"
