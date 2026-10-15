"""Reorder (or select) the test rows of a prep's V.i32, for a second pass of the sequential rank increments.

cpu_rank.py / gpu_rank.py add the test rows one by one after the controls, so the increments depend on the order: with
rows [z, d1_1.., t_1..] the first increment says whether z is Q_0-exact, with [d1_1.., z, ..] whether it is exact
modulo the d_1 images.  This writes OUT/V.i32 with the controls first and the test rows in the given order, OUT/meta.json
(nV, n_vecs updated, 'row_order' recorded) and a symlink OUT/B.i32.
usage: python3 vperm.py PREP OUT --order 1-24,0,25-39     (indices of test rows, 0 = first row after the controls)"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('prep')
ap.add_argument('out')
ap.add_argument('--order', required=True)
a = ap.parse_args()
meta = json.load(open(os.path.join(a.prep, 'meta.json')))
nV, nc, c = meta['nV'], meta['n_ctrl'], meta['ncols']
order = []
for part in a.order.split(','):
    if '-' in part:
        lo, hi = part.split('-')
        order += list(range(int(lo), int(hi) + 1))
    else:
        order.append(int(part))
assert len(set(order)) == len(order) and all(0 <= i < nV - nc for i in order), order
V = np.fromfile(os.path.join(a.prep, 'V.i32'), dtype=np.int32).reshape(nV, c)
os.makedirs(a.out, exist_ok=True)
np.ascontiguousarray(V[list(range(nc)) + [nc + i for i in order]]).tofile(os.path.join(a.out, 'V.i32'))
bl = os.path.join(a.out, 'B.i32')
if not os.path.lexists(bl):
    os.symlink(os.path.realpath(os.path.join(a.prep, 'B.i32')), bl)
m2 = dict(meta, nV=nc + len(order), n_vecs=len(order), prep_dir=a.out, row_order=order, row_order_of=a.prep)
json.dump(m2, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
print(f"wrote {a.out}: {nc} controls + {len(order)} test rows in order {order}")
