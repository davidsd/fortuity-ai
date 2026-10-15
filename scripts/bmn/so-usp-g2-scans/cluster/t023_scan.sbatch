#!/bin/bash
# T-023: complete SU(3)_R hw scans of SO(7) and G2 at j = 22 and 24, as parallel as possible on one node, then the
# SO(7)->G2 restriction test on every SO(7) non-graviton sector. Only after
# Tasks = columns (q1>=q2>=q3, sum q = j/2), heaviest (near-singlet) first; each task runs SO(7) then G2 for its column
# (G2 reuses SO(7)'s cached hw vectors). PAR tasks at once, WPT workers each (PAR*WPT ~ cores).
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --job-name=t023-scan
#SBATCH --output=${REPO}/runs/%x-%j.out
set -uo pipefail
V=${REPO}
PY=$V/venv/bin/python; OUT=$V/runs/scan; mkdir -p $OUT
cd "$V/code"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ENGINE=par GEN_NMAX=8 HW_CACHE=$V/runs/hwcache HWPAR_WORKERS=${WPT:-3}
PAR=${PAR:-26}
JS=${JS:-"22 24"}
echo "[$(date)] host=$(hostname) PAR=$PAR WPT=$HWPAR_WORKERS JS=$JS"
col_task() {
  j=$1; lam=$2; tag=${lam//,/_}
  GROUP=SO_odd $PY -u -m fortuity.hwcoh.run_sectors 3 $V/runs/dims_so7_j26.pkl $OUT/so7_j${j}_${tag}.jsonl $lam > $OUT/so7_j${j}_${tag}.log 2>&1
  GROUP=G2 $PY -u -m fortuity.hwcoh.run_sectors 0 $V/runs/dims_g2_j26.pkl $OUT/g2_j${j}_${tag}.jsonl $lam > $OUT/g2_j${j}_${tag}.log 2>&1
  echo "[$(date)] done j=$j lam=$lam"
}
export -f col_task; export PY V OUT
$PY -c "
js = [int(x) for x in '$JS'.split()]
tasks = []
for j in js:
    s = j // 2
    for q1 in range(s, -1, -1):
        for q2 in range(min(q1, s - q1), -1, -1):
            q3 = s - q1 - q2
            if q3 <= q2:
                tasks.append((j, (q1, q2, q3)))
tasks.sort(key=lambda t: (-t[0], -t[1][2], -t[1][1]))       # larger j and near-singlet columns first
for j, l in tasks:
    print(j, ','.join(map(str, l)))
" | xargs -P $PAR -n 2 bash -c 'col_task "$0" "$1"'
echo "[$(date)] scans done; restriction test"
FORTUITY_WORKERS=$PAR $PY -u $V/restrict_scan.py $V/runs/dims_so7_j26.pkl $V/runs/dims_g2_j26.pkl $OUT/restrict.jsonl $OUT/so7_j2*.jsonl
echo "[$(date)] done"
