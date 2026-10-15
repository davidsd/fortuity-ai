#!/bin/bash
# T-041 independent check: selftest, free-energy verification and the 23 comparison points of points_wedge.txt.
# Serial, one core, about 2 min on an M1 Max core (selftest 11 s, verify 2 s, points ~110 s); allow 3-6 min here.
# Usage (symmetry; survey first: sinfo -h -p amddebugq,debugq -o '%P %t %D' | awk '$2=="idle"'):
#   export PATH=/cm/shared/apps/slurm/current/bin:$PATH
#   mkdir -p ${SCRATCH}/slurm_logs
#   REPO=/path/to/fortuity sbatch --export=ALL run_indep_points.sbatch
#SBATCH -p amddebugq
#SBATCH -N 1 -n 1 -c 1
#SBATCH --mem=2G
#SBATCH -t 00:30:00
#SBATCH -J t041_indep_points
#SBATCH -o ${SCRATCH}/slurm_logs/%x_%j.out
set -euo pipefail
: "${REPO:?set REPO to the repository clone on this machine (it must contain scripts/entropy/lifting-excess/indep)}"
OUT=${OUT:-${SCRATCH}/t041/indep}
mkdir -p "$OUT"
set +u; source /etc/profile.d/modules.sh; set -u
set +u; module load anaconda3/2023.07-2; set -u
cd "$REPO/scripts/entropy/lifting-excess/indep"
echo "host $(hostname)  start $(date)  python $(python -c 'import sys; print(sys.version.split()[0])')  mpmath $(python -c 'import mpmath; print(mpmath.__version__)')"
python indep_delta_r1.py selftest              > "$OUT/selftest.out" 2>&1
python verify_freeenergy.py                    > "$OUT/verify_freeenergy.out" 2>&1
python indep_delta_r1.py --file points_wedge.txt > "$OUT/indep_wedge.tsv" 2>&1
tail -n 1 "$OUT/selftest.out" "$OUT/verify_freeenergy.out" "$OUT/indep_wedge.tsv"
echo "end $(date)"
