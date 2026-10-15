#!/bin/bash
# T-041 (optional): the whole grid of scan_kklp_charges.py scan 12, computed independently: 184 points (the TH = 0
# ray, which has no edge, cut at R <= RMAX0 = 1.0), split round-robin into one chunk per physical core.
# About 12 min of CPU on M1 Max cores in total; with 32 or more chunks about 1-2 min wall.  Allow 30 min.
# Usage (symmetry; survey first: sinfo -h -p amddebugq,debugq -o '%P %t %D' | awk '$2=="idle"'):
#   export PATH=/cm/shared/apps/slurm/current/bin:$PATH
#   mkdir -p ${SCRATCH}/slurm_logs
#   REPO=/path/to/fortuity sbatch --export=ALL run_indep_grid.sbatch
# Afterwards: (head -n 2 $OUT/grid12_chunk0.tsv | tail -n 1; grep -h -v -e '^#' -e '^theta' $OUT/grid12_chunk*.tsv | sort -t$'\t' -k1,1g -k2,2g) > $OUT/indep_grid12.tsv
#SBATCH -p amddebugq
#SBATCH -N 1 --exclusive
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -J t041_indep_grid
#SBATCH -o ${SCRATCH}/slurm_logs/%x_%j.out
set -euo pipefail
: "${REPO:?set REPO to the repository clone on this machine (it must contain scripts/entropy/lifting-excess/indep)}"
OUT=${OUT:-${SCRATCH}/t041/indep}
RMAX0=${RMAX0:-1.0}
mkdir -p "$OUT"
set +u; source /etc/profile.d/modules.sh; set -u
set +u; module load anaconda3/2023.07-2; set -u
cd "$REPO/scripts/entropy/lifting-excess/indep"
phys_alloc() {   # distinct physical cores in this process's allowed CPU set (hpc-clusters rule 7)
  local a; a=$(grep -m1 Cpus_allowed_list /proc/self/status | cut -f2)
  lscpu -p=CPU,CORE | grep -v '^#' | awk -F, -v a="$a" '
    BEGIN{n=split(a,r,","); for(i=1;i<=n;i++){m=split(r[i],b,"-"); lo=b[1]; hi=(m==2?b[2]:b[1]); for(c=lo;c<=hi;c++) ok[c]=1}}
    ($1 in ok) && !($2 in seen){seen[$2]=1; k++} END{print k+0}'
}
PA=$(phys_alloc); CAP=${SLURM_CPUS_ON_NODE:-1}
K=$(( PA < CAP ? PA : CAP )); K=$(( K > 48 ? 48 : K ))
(( K >= 1 )) || { echo "no cores" >&2; exit 64; }
echo "host $(hostname)  start $(date)  chunks K=$K  RMAX0=$RMAX0"
for (( k = 0; k < K; k++ )); do
  python indep_delta_r1.py --grid 12 --rmax0 "$RMAX0" --chunk "$k/$K" > "$OUT/grid12_chunk$k.tsv" 2>&1 &
done
wait
grep -h '^# fail' "$OUT"/grid12_chunk*.tsv || echo "no failed points"
echo "end $(date)"
