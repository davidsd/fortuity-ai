#!/bin/bash
#SBATCH --job-name=d2-aug
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=${REPO}/log/d2_aug_%j.out
# R-007 d_2, GPU route for c B = -Q_1 x: the augmented matrix A' = [B; Q_1 x] ((nB + 1) x c) and its transpose as a
# prep directory for kernel_vec.py (left-kernel vectors of A' on one H200).  Streaming copies, no large memory.
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes
S=prep/su3_j30; A=prep/su3_j30_aug
mkdir -p $A
python3 - $S $A <<'PY'
import json, os, sys, time
import numpy as np
s, a = sys.argv[1], sys.argv[2]
m = json.load(open(s + '/meta.json'))
nB, c = m['nB'], m['ncols']
v = np.fromfile(s + '/VC.i32', dtype=np.int32)
assert v.size == c
t0 = time.time()
BT = np.memmap(s + '/BT.i32', dtype=np.int32, mode='r', shape=(c, nB))
with open(a + '/BT.i32.part', 'wb') as fh:
    for r0 in range(0, c, 4096):
        r1 = min(c, r0 + 4096)
        np.ascontiguousarray(np.hstack([np.asarray(BT[r0:r1]), v[r0:r1, None]])).tofile(fh)
        if (r0 // 4096) % 8 == 0:
            print(f"  BT' rows {r1}/{c} ({time.time()-t0:.0f}s)", flush=True)
assert os.path.getsize(a + '/BT.i32.part') == 4 * c * (nB + 1)
os.replace(a + '/BT.i32.part', a + '/BT.i32')
json.dump(dict(m, nB=nB + 1, nB_without_V=nB, note="B with the V row Q_1 x (VC.i32 of prep/su3_j30) appended as last row"),
          open(a + '/meta.json', 'w'), indent=1)
print(f"wrote BT' {c} x {nB + 1} ({time.time()-t0:.0f}s)", flush=True)
PY
cat $S/B.i32 $S/VC.i32 > $A/B.i32.part
python3 -c "
import os, json
m = json.load(open('$A/meta.json')); assert os.path.getsize('$A/B.i32.part') == 4 * m['nB'] * m['ncols']"
mv $A/B.i32.part $A/B.i32
ls -la $A
echo "D2 AUG DONE $(date)"
