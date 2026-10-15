#!/bin/bash
# Launcher: SU(4) nf-refined dims to j=30, task lists (heavy columns per sector, light ones packed), submit arrays.
#SBATCH --job-name=su4j30-launch
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=${STORE}/logs/%x-%j.out
set -euo pipefail
W=${REPO}/scripts/bmn/hw-scans-cluster
source $W/env.sh
ulimit -c 0                      # never write core files (they landed in home once)
cd "${TMPDIR:-/tmp}"             # work in node-local scratch; the code is found via PYTHONPATH (env.sh)
DIMS=$FORT_STORE/results/dimsnf_N4_j30.pkl
if [ ! -s $DIMS ]; then
  FORTUITY_WORKERS=32 python3 -u -c "
import pickle, time, os
from fortuity.hwcoh.dims import partition_dims_nf
t=time.time(); d=partition_dims_nf(4, 30); print('N=4 j=30 nf dims', len(d), f'{time.time()-t:.0f}s', flush=True)
pickle.dump(d, open('$DIMS','wb'))"
fi
python3 -u - <<'PY'
import pickle, os
from fortuity.hwcoh.dims import collapse
from fortuity.hwcoh.column import hw_mult
W = '${REPO}/scripts/bmn/hw-scans-cluster/cluster/'
dims = collapse(pickle.load(open(os.environ['FORT_STORE'] + '/results/dimsnf_N4_j30.pkl', 'rb')))
s = 15
cols = []
for q1 in range(s, -1, -1):
    for q2 in range(min(q1, s - q1), -1, -1):
        q3 = s - q1 - q2
        if q3 > q2 or q3 < 0:
            continue
        lam = (q1, q2, q3)
        per = {n: dims.get((n,) + lam, 0) * max(hw_mult(dims, n, lam), 1) for n in range(0, 46) if dims.get((n,) + lam, 0)}
        if per:
            cols.append((sum(per.values()), lam, per))
cols.sort(reverse=True)
tot = sum(c for c, _, _ in cols)
heavy, light, acc = [], [], 0
for c, lam, per in cols:
    (heavy if acc < 0.85 * tot else light).append((c, lam, per)); acc += c
lam_s = lambda l: ",".join(map(str, l))
hl = [f"{lam_s(lam)}:{n}" for c, lam, per in heavy for n in sorted(per)]
nb = 24
bins = [[0, []] for _ in range(nb)]
for c, lam, per in light:
    b = min(bins, key=lambda x: x[0]); b[0] += c; b[1].append(lam_s(lam))
open(W + 'tasks_su4_j30_heavy.txt', 'w').write("\n".join(hl) + "\n")
open(W + 'tasks_su4_j30_light.txt', 'w').write("\n".join(" ".join(b[1]) for b in bins if b[1]) + "\n")
print(len(cols), "columns:", len(heavy), "heavy ->", len(hl), "sector tasks;", len(light), "light in", sum(1 for b in bins if b[1]), "tasks")
PY
NH=$(wc -l < $W/cluster/tasks_su4_j30_heavy.txt); NL=$(wc -l < $W/cluster/tasks_su4_j30_light.txt)
cd $W; module load slurm   # env.sh purges modules, which removes sbatch
sbatch --array=0-$((NH-1)) --cpus-per-task=32 --mem=200G --time=48:00:00 --export=ALL,N=4,DIMS=$DIMS,RUN=su4_j30,TASKFILE=$W/cluster/tasks_su4_j30_heavy.txt,MAXLEN=10 --job-name=hwpar-su4j30-heavy cluster/hw_par.sbatch
sbatch --array=0-$((NL-1)) --cpus-per-task=16 --mem=64G --time=24:00:00 --export=ALL,N=4,DIMS=$DIMS,RUN=su4_j30l,TASKFILE=$W/cluster/tasks_su4_j30_light.txt,MAXLEN=10 --job-name=hwpar-su4j30-light cluster/hw_par.sbatch
