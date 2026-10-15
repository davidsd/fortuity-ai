#!/bin/bash
#SBATCH --job-name=d2-transp
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=${REPO}/log/d2_transp_%j.out
# R-007 order 2, d_2 (summary/fortuity-summary.pdf): F = the V row Q_1 x of the one-loop prep (VC.i32), and the transposed B
# for kernel_vec_aug.py (c B = -Q_1 x)
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
P1=../oneloop/prep/su3_j30
python3 - $P1 <<'PY'
import json, sys
import numpy as np
d = sys.argv[1]
m = json.load(open(d + '/meta.json'))
V = np.fromfile(d + '/V.i32', dtype=np.int32).reshape(m['nV'], m['ncols'])
r = m['n_ctrl']
V[r:r + 1].tofile(d + '/VC.i32')
json.dump(dict(nVC=1, width=m['ncols'], row=r, what='Q_1 x, x = vector 0 of classes/su3_j30 (R-007)'), open(d + '/VC.json', 'w'))
print('VC: V row', r, 'nonzero', int(np.count_nonzero(V[r])), 'of', m['ncols'], flush=True)
PY
python3 -u transpose_prep.py $P1
echo "D2 TRANSPOSE DONE $(date)"
