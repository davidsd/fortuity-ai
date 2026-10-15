#!/bin/bash
#SBATCH --job-name=d2-kgpu
#SBATCH --account=${ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --output=${REPO}/log/d2_kgpu_%j.out
# R-007 d_2, GPU route: 26 random left-kernel vectors of A' = [B; Q_1 x] on one H200 (kernel_vec.py; the rank must be
# 76781, i.e. Q_1 x in the row space of B).  The last component is z; the result is written as xker_aug.npy /
# zker_aug.npy in prep/su3_j30_c, a directory with the original meta.json, for d2_make.py.
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
NVLIBS=$(ls -d ${VENV}/lib*/python*/site-packages/nvidia/*/lib 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH=$NVLIBS$LD_LIBRARY_PATH
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -c "import cupy; cupy.arange(8).sum(); print('GPU smoke test ok')" || { echo "GPU STACK BROKEN"; exit 1; }
A=../oneloop/prep/su3_j30_aug; C=../oneloop/prep/su3_j30_c
python3 -u kernel_vec.py $A --k 26 --seed 2029 --blk 2048
mkdir -p $C
ln -sf ../su3_j30/meta.json $C/meta.json
python3 - $A $C <<'PY'
import json, sys
import numpy as np
a, c = sys.argv[1], sys.argv[2]
res = json.load(open(a + '/xker.json'))
assert res['rank'] == 76781, res['rank']
X = np.load(a + '/xker.npy')
nB = X.shape[0] - 1
Z = X[nB].reshape(-1, 1).copy()
print('z nonzero:', int(np.count_nonzero(Z)), 'of', Z.size, flush=True)
assert Z.all()
np.save(c + '/xker_aug.npy', np.ascontiguousarray(X[:nB]))
np.save(c + '/zker_aug.npy', Z)
json.dump(dict(res, source=a, nB=nB), open(c + '/xker_aug.json', 'w'), indent=1)
PY
echo "D2 KERNEL GPU DONE $(date)"
