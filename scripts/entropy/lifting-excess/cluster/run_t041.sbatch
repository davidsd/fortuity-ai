#!/bin/bash
#SBATCH -p amdq
#SBATCH -N 1 --exclusive
#SBATCH -t 03:00:00
#SBATCH --mem=32G
#SBATCH -J t041-main
#SBATCH -o ${SCRATCH}/slurm_logs/%x_%j.out
# T-041 main runs with the patched scripts in t041/code (the T-038 originals plus audit/patches/*.patch):
#   1. full map, patched: edge points, theta = 0 bounded at r = 0.8, both saddles continued, full precision
#   2. holonomy dominance with rim-aware verdicts: nine presets plus nine extra points, six groups
#   3. grid convergence of the dominance scans at four far points
#   4. finer map (2.5 deg, dr = 0.01)
# Usage on symmetry:  export REPO=/path/to/fortuity; sbatch --export=ALL $REPO/scripts/entropy/lifting-excess/cluster/run_t041.sbatch
set -u
: "${REPO:?set REPO to the repo checkout}"
set +u; source /etc/profile.d/modules.sh; module load anaconda3/2023.07-2; set -u
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=${SCRATCH}/t041/main/${SLURM_JOB_ID:-manual}; mkdir -p "$OUT"
cd "$REPO/scripts/entropy/lifting-excess/code" || exit 1
echo "host $(hostname), out $OUT, $(python --version 2>&1), numpy $(python -c 'import numpy; print(numpy.__version__)'), mpmath $(python -c 'import mpmath; print(mpmath.__version__)'), $(date)"
run() { local name=$1; shift; ( t0=$(date +%s); python "$@" > "$OUT/$name" 2>&1; echo "# exit $? after $(( $(date +%s) - t0 )) s" >> "$OUT/$name" ) & }
python cardy_torus_unequal.py selftest > "$OUT/torus_selftest.out" 2>&1
python rf/scan_kklp_charges.py selftest > "$OUT/scan_selftest.out" 2>&1
run scan12_patched.tsv rf/scan_kklp_charges.py scan 12 0.8 0.02
for g in su2 su3 g2 so7 usp6; do run torus_${g}_allext.out cardy_torus_unequal.py $g all+ext; done
run torus_su4_allext.out cardy_torus_unequal.py su4 all+ext
run conv_su3_r1d.out  cardy_torus_unequal.py su3 r1d 1/32 1/120
run conv_g2_r5a.out   cardy_torus_unequal.py g2 r5a 1/32 1/120
run conv_so7_r4b.out  cardy_torus_unequal.py so7 r4b 1/32 1/120
run conv_usp6_r1e.out cardy_torus_unequal.py usp6 r1e 1/32 1/120
run scan24_fine.tsv rf/scan_kklp_charges.py scan 24 0.8 0.01
wait
echo "done $(date)"; grep -H "# exit" "$OUT"/*
