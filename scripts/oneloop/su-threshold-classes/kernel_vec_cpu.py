"""CPU (MKL, fortuity.hwcoh.ffblas) version of kernel_vec.py: random left-kernel vectors x of a prep matrix A = B.i32
(nB x c), x^T A = 0 mod p, written to PREP/xker.npy like the GPU version.

StreamBasis.null_space() returns the whole null space, which for A^T of the R-007 T_2 prep (nullity ~114k, length
219k) would not fit in memory.  So x is restricted to a random subset R of the rows of A (a fraction --frac): every x
supported on R with x^T A[R] = 0 is a left-kernel vector of A, and the null space of A[R]^T has dimension
|R| - rank(A[R]).  The span of the rows h_R meets ker Q_0 n ker(Cartan) in a subspace of that dimension that is generic
in it (the compressed rows are random combinations), so a random x has a random component along a non-exact class when
there is one.  Every x is checked against the original B.
usage: python3 kernel_vec_cpu.py PREP [--frac 0.7] [--k 4] [--seed 2026]   (reads PREP/BT.i32 from transpose_prep.py)"""
import argparse, json, os, time
import numpy as np
from fortuity.hwcoh import ffblas
from fortuity.bmn.modp import P, matmul_mod

ap = argparse.ArgumentParser()
ap.add_argument('prep')
ap.add_argument('--frac', type=float, default=0.7)
ap.add_argument('--k', type=int, default=4)
ap.add_argument('--seed', type=int, default=2026)
ap.add_argument('--blk', type=int, default=4096)
a = ap.parse_args()
assert ffblas.HAVE_FFBLAS, "libffblas.so not loaded (set FORTUITY_FFBLAS)"
meta = json.load(open(os.path.join(a.prep, 'meta.json')))
nB, c = meta['nB'], meta['ncols'] + int(meta.get('ncart', 0) or 0)
rng = np.random.default_rng(a.seed)
R = np.sort(rng.choice(nB, size=int(a.frac * nB), replace=False))
t0 = time.time()
BT = np.memmap(os.path.join(a.prep, 'BT.i32'), dtype=np.int32, mode='r', shape=(c, nB))
SB = ffblas.StreamBasis(len(R))
for r0 in range(0, c, a.blk):
    r1 = min(c, r0 + a.blk)
    SB.add(np.ascontiguousarray(np.asarray(BT[r0:r1])[:, R], dtype=np.int64))
    if (r0 // a.blk) % 5 == 0:
        print(f"  rows of A^T {r1}/{c}: rank {SB.rank} ({time.time() - t0:.0f}s)", flush=True)
rank = SB.rank
nul = len(R) - rank
print(f"rank A[R] = {rank} with |R| = {len(R)} of {nB} rows; null space dimension {nul} ({time.time() - t0:.0f}s)",
      flush=True)
N = SB.null_space()                              # nul x |R|, rows x with A[R]^T x = 0
del SB
print(f"null space computed ({time.time() - t0:.0f}s)", flush=True)
XR = matmul_mod(rng.integers(0, P, (a.k, nul), dtype=np.int64), N % P)    # k random combinations
del N
X = np.zeros((nB, a.k), dtype=np.int64)
X[R] = XR.T
# check x^T B = 0 against the original prep matrix, streamed
B = np.memmap(os.path.join(a.prep, 'B.i32'), dtype=np.int32, mode='r', shape=(nB, c))
acc = np.zeros((a.k, c), dtype=np.int64)
for r0 in range(0, nB, a.blk):
    r1 = min(nB, r0 + a.blk)
    if X[r0:r1].any():
        acc = (acc + matmul_mod(np.ascontiguousarray(X[r0:r1].T), np.asarray(B[r0:r1], dtype=np.int64))) % P
nz = int(np.count_nonzero(acc))
print(f"check x^T B = 0: {nz} nonzero entries of {a.k} x {c}; x nonzero entries per vector "
      f"{[int(np.count_nonzero(X[:, j])) for j in range(a.k)]} ({time.time() - t0:.0f}s)", flush=True)
assert nz == 0, "kernel vectors fail the check against B"
assert all(np.count_nonzero(X[:, j]) for j in range(a.k))
np.save(os.path.join(a.prep, 'xker.npy'), X)
res = dict(backend='cpu_ffblas', nB=nB, c=c, rows_used=int(len(R)), frac=a.frac, rank=int(rank), kernel_dim_R=int(nul),
           k=a.k, seed=a.seed, check_nonzero=nz, time=round(time.time() - t0), case=meta.get('case'))
json.dump(res, open(os.path.join(a.prep, 'xker.json'), 'w'), indent=1)
print(json.dumps(res), flush=True)
