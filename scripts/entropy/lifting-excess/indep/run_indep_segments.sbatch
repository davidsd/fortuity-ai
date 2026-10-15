#!/bin/bash
# T-041: independent check of the segments with one negative charge. indep_delta_r1.py (the independent code; not
# physics/classical_max.py, which produced outputs/extended/) at the 245 points of points_segments.txt, split
# round-robin into one chunk per physical core (at most 64). About 1-15 s per point; with 64 chunks well under a
# minute of wall time.  Allow 30 min.
# The finite-eps step is --h 1e-5, not the default 0.002: near the arc the protected black hole at fixed xi exists
# only for eps < eps_b ~ 0.68 (S_prot/K)^2 (0.0032 at the end of ray 1), and the r1 stencil reaches 4h.
# Usage (symmetry; survey first: sinfo -h -p amddebugq,amdq -o '%P %t %D' | awk '$2=="idle"'):
#   export PATH=/cm/shared/apps/slurm/current/bin:$PATH
#   mkdir -p ${SCRATCH}/slurm_logs
#   REPO=/path/to/fortuity sbatch --export=ALL run_indep_segments.sbatch
# Output: $OUT/seg_chunk<k>.tsv, k = 0..K-1, with OUT = ${SCRATCH}/t041/indep_seg/<jobid> by default, a fresh
# directory per job.  Afterwards (K is printed in the Slurm log as "chunks K=..."):
#   python merge_grid.py $OUT K --points points_segments.txt --prefix seg_chunk     # -> $OUT/indep_points_segments.tsv
#   python compare_segments.py $OUT/indep_points_segments.tsv ../outputs/extended
#SBATCH -p amddebugq
#SBATCH -N 1 --exclusive
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -J t041_indep_seg
#SBATCH -o ${SCRATCH}/slurm_logs/%x_%j.out
set -euo pipefail
: "${REPO:?set REPO to the repository clone on this machine (it must contain scripts/entropy/lifting-excess/indep)}"
OUT=${OUT:-${SCRATCH}/t041/indep_seg/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
set +u; source /etc/profile.d/modules.sh; set -u
set +u; module load anaconda3/2023.07-2; set -u
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
cd "$REPO/scripts/entropy/lifting-excess/indep"
phys_alloc() {   # distinct physical cores in this process's allowed CPU set (hpc-clusters rule 7)
  local a; a=$(grep -m1 Cpus_allowed_list /proc/self/status | cut -f2)
  lscpu -p=CPU,CORE | grep -v '^#' | awk -F, -v a="$a" '
    BEGIN{n=split(a,r,","); for(i=1;i<=n;i++){m=split(r[i],b,"-"); lo=b[1]; hi=(m==2?b[2]:b[1]); for(c=lo;c<=hi;c++) ok[c]=1}}
    ($1 in ok) && !($2 in seen){seen[$2]=1; k++} END{print k+0}'
}
PA=$(phys_alloc); CAP=${SLURM_CPUS_ON_NODE:-1}
K=$(( PA < CAP ? PA : CAP )); K=$(( K > 64 ? 64 : K ))
(( K >= 1 )) || { echo "no cores" >&2; exit 64; }
echo "host $(hostname)  start $(date)  chunks K=$K  out $OUT  $(python --version 2>&1)  mpmath $(python -c 'import mpmath; print(mpmath.__version__)')"
md5sum indep_delta_r1.py points_segments.txt "$0"      # $0: the copy of this script that Slurm runs
H=1e-5
declare -a pids
for (( k = 0; k < K; k++ )); do
  python indep_delta_r1.py --file points_segments.txt --h "$H" --chunk "$k/$K" > "$OUT/seg_chunk$k.tsv" 2>&1 &
  pids[$k]=$!
done
bad=0
for (( k = 0; k < K; k++ )); do
  wait "${pids[$k]}" || { echo "chunk $k exited with status $?"; bad=1; }
done
grep -h '^# fail' "$OUT"/seg_chunk*.tsv && bad=1 || echo "no failed points"
missing=$(grep -L '^# total' "$OUT"/seg_chunk*.tsv || true)
[[ -z "$missing" ]] || { echo "chunks without the '# total' trailer: $missing"; bad=1; }
echo "end $(date)  h=$H  bad=$bad"
exit $bad
