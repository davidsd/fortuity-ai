"""Transpose a fullpar prep matrix B.i32 (nB x c row-major, c = ncols + ncart) into BT.i32 (c x nB row-major), so that
kernel_vec.py can stream the columns of B as rows.  Loads B into memory (needs ~4 nB c bytes of RAM).
usage: python3 transpose_prep.py PREPDIR [--blk 4096]"""
import argparse, json, os, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('prep_dir')
ap.add_argument('--blk', type=int, default=4096)
a = ap.parse_args()
meta = json.load(open(os.path.join(a.prep_dir, 'meta.json')))
nB, c = meta['nB'], meta['ncols'] + int(meta.get('ncart', 0) or 0)
t0 = time.time()
B = np.fromfile(os.path.join(a.prep_dir, 'B.i32'), dtype=np.int32)
assert B.size == nB * c, (B.size, nB, c)
B = B.reshape(nB, c)
print(f"loaded B {nB} x {c} ({time.time() - t0:.0f}s)", flush=True)
tmp = os.path.join(a.prep_dir, 'BT.i32.part')
with open(tmp, 'wb') as fh:
    for c0 in range(0, c, a.blk):
        c1 = min(c, c0 + a.blk)
        np.ascontiguousarray(B[:, c0:c1].T).tofile(fh)
        if (c0 // a.blk) % 8 == 0:
            print(f"  columns {c1}/{c} ({time.time() - t0:.0f}s)", flush=True)
assert os.path.getsize(tmp) == 4 * nB * c
os.replace(tmp, os.path.join(a.prep_dir, 'BT.i32'))
print(f"wrote BT {c} x {nB} ({time.time() - t0:.0f}s)", flush=True)
