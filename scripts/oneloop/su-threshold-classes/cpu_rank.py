"""T-031: the rank test of gpu_rank.py on the CPU with the BLIS/MKL kernel (fortuity.hwcoh.ffblas.StreamBasis), as an
independent implementation next to the GPU one.  Usage: python3 cpu_rank.py PREP_DIR  -> PREP_DIR/rank_cpu_ffblas.json"""
import os, sys, json, time
import numpy as np
from fortuity.hwcoh import ffblas
from fortuity.bmn.modp import P
d = sys.argv[1]
meta = json.load(open(os.path.join(d, 'meta.json')))
assert ffblas.HAVE_FFBLAS, "libffblas.so not loaded (set FORTUITY_FFBLAS)"
assert meta.get('p', P) == P
nB, nV, c, nc = meta['nB'], meta['nV'], meta['ncols'], meta['n_ctrl']
# B may carry Cartan columns after the ncols target functionals (fullpar --cartan): the file is nB x (ncols + ncart).
# By default only the first ncols columns are used (rank B); with --with-cartan all of them (rank [B | E]).
ncart = int(meta.get('ncart', 0) or 0)
with_cartan = '--with-cartan' in sys.argv[2:]
Bfull = np.memmap(os.path.join(d, 'B.i32'), dtype=np.int32, mode='r', shape=(max(nB, 1), c + ncart))
if with_cartan:
    c = c + ncart
B = Bfull[:, :c]
V = np.memmap(os.path.join(d, 'V.i32'), dtype=np.int32, mode='r', shape=(max(nV, 1), meta['ncols']))
if with_cartan and ncart:
    V = np.hstack([np.asarray(V), np.zeros((V.shape[0], ncart), dtype=np.int32)])
t0 = time.time()
SB = ffblas.StreamBasis(c)
for r0 in range(0, nB, 4096):
    SB.add(np.asarray(B[r0:min(nB, r0 + 4096)], dtype=np.int64))
    if (r0 // 4096) % 5 == 0:
        print(f"  rows {min(nB, r0 + 4096)}/{nB}: rank {SB.rank} ({time.time() - t0:.0f}s)", flush=True)
rB = SB.rank
if nc:
    SB.add(np.asarray(V[:nc], dtype=np.int64))
rC = SB.rank
inc = []
for k in range(nc, nV):
    b = SB.rank
    SB.add(np.asarray(V[k:k + 1], dtype=np.int64))
    inc.append(int(SB.rank - b))
res = dict(backend='cpu_ffblas', nB=nB, nV=nV, ncols=c, ncart_used=ncart if with_cartan else 0, n_ctrl=nc, rank_B=int(rB), rank_after_controls=int(rC),
           controls_exact=bool(rC == rB), lifted_increments=inc, n_lifted=int(sum(inc)), time=round(time.time() - t0),
           case=meta.get('case'))
print(json.dumps(res), flush=True)
json.dump(res, open(os.path.join(d, 'rank_cpu_ffblas%s.json' % ('_cartan' if with_cartan else '')), 'w'), indent=1)
