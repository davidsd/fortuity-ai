#!/bin/bash
#SBATCH -p amdq
#SBATCH -N 1 --exclusive
#SBATCH -t 01:00:00
#SBATCH --mem=16G
#SBATCH -J t041-extended
#SBATCH -o ${SCRATCH}/slurm_logs/%x_%j.out
# T-041: the part of the black-hole domain sigma2(Q) > 0 that scan12 (Q_I > 0) misses, one process per ray
# (physics/scan_extended.py, independent of the t038 code). Usage: export REPO=...; sbatch --export=ALL run_extended.sbatch
set -u
: "${REPO:?set REPO}"
set +u; source /etc/profile.d/modules.sh; module load anaconda3/2023.07-2; set -u
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
OUT=${SCRATCH}/t041/extended/${SLURM_JOB_ID:-manual}; mkdir -p "$OUT"
cd "$REPO/scripts/entropy/lifting-excess/physics" || exit 1
python scan_extended.py selftest > "$OUT/selftest.out" 2>&1
for K in $(seq 0 12); do ( python scan_extended.py ray $K 12 --from-edge > "$OUT/ray$K.tsv" 2>&1; echo "# exit $?" >> "$OUT/ray$K.tsv" ) & done
wait; echo "done $(date)"
