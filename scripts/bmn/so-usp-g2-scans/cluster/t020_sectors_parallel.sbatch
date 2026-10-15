#!/bin/bash
# T-020: the six hw sectors (columns (7,3,3), (5,5,3) x 2J = 0,1,2) as six parallel single-core processes, reusing the
# exact dims pickle of the first run. For extra seeds. Only after
# Submit in $R:  sbatch --export=ALL,GROUP=SO_odd,SEED=2 t020_sectors_parallel.sbatch   (or GROUP=Sp)
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=6:00:00
#SBATCH --job-name=t020-par
#SBATCH --output=${SCRATCH}
set -uo pipefail
: "${GROUP:?set GROUP=SO_odd or GROUP=Sp}"; : "${SEED:?set SEED}"
N=5; JMAX=26
R=${SCRATCH}
PY=$R/venv/bin/python; OUT=$R/runs
DIMS="$OUT/dims_${GROUP}${N}_j${JMAX}_2J3.pkl"
[ -s "$DIMS" ] || { echo "missing $DIMS"; exit 1; }
cd "$R/code"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[$(date)] host=$(hostname) group=$GROUP seed=$SEED"
pids=()
for lam in 7,3,3 5,5,3; do
  for n in 0 1 2; do
    tag="${lam//,/}_n${n}"
    SEED=$SEED GROUP=$GROUP GEN_NMAX=12 $PY -u -m fortuity.hwcoh.run_sectors $N "$DIMS" \
      "$OUT/hw_${GROUP}${N}_seed${SEED}_${tag}.jsonl" "$lam:$n" > "$OUT/log_${GROUP}${N}_seed${SEED}_${tag}.txt" 2>&1 &
    pids+=($!)
  done
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
grep -h SECTOR "$OUT"/log_${GROUP}${N}_seed${SEED}_*.txt
echo "[$(date)] done rc=$rc"
exit $rc
