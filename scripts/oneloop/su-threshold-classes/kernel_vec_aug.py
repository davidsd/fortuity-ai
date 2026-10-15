"""Random left-kernel vectors of the augmented prep matrix [A; F] (CPU, MKL): (x, z) with x^T A + z^T F = 0.

A = B.i32 (nB x c: Q_0-image functionals | Cartan functionals) of the T_2 outbound prep; F = VC.i32 (nVC x c), the rows
[0 | e_f] of the abelian LW polynomials f (membership.py).  Then t = sum_i x_i h_i is a Q_0-cocycle with
eps(t) = -sum_j z_j f_j: random kernel vectors with z != 0 give representatives of the Coulomb-type classes whose
Cartan image is LW (8 of them at T_2, plus exact and non-Coulomb parts).  As in kernel_vec_cpu.py, x is restricted to
a random subset R of the rows of A (all rows of F are kept).
usage: python3 kernel_vec_aug.py PREP [--frac 0.7] [--k 12] [--seed 2027]  -> PREP/xker_aug.npy, zker_aug.npy, .json"""
import argparse, json, os, time
import numpy as np
from fortuity.hwcoh import ffblas
from fortuity.bmn.modp import P, matmul_mod

ap = argparse.ArgumentParser()
ap.add_argument('prep')
ap.add_argument('--frac', type=float, default=0.7)
ap.add_argument('--k', type=int, default=12)
ap.add_argument('--seed', type=int, default=2027)
ap.add_argument('--blk', type=int, default=4096)
a = ap.parse_args()
assert ffblas.HAVE_FFBLAS
meta = json.load(open(os.path.join(a.prep, 'meta.json')))
vmeta = json.load(open(os.path.join(a.prep, 'VC.json')))
nB, c, nF = meta['nB'], meta['ncols'] + int(meta.get('ncart', 0) or 0), vmeta['nVC']
assert vmeta['width'] == c
rng = np.random.default_rng(a.seed)
R = np.sort(rng.choice(nB, size=int(a.frac * nB), replace=False))
t0 = time.time()
BT = np.memmap(os.path.join(a.prep, 'BT.i32'), dtype=np.int32, mode='r', shape=(c, nB))
F = np.fromfile(os.path.join(a.prep, 'VC.i32'), dtype=np.int32).reshape(nF, c)
SB = ffblas.StreamBasis(len(R) + nF)
for r0 in range(0, c, a.blk):
    r1 = min(c, r0 + a.blk)
    blk = np.hstack([np.asarray(BT[r0:r1])[:, R], F[:, r0:r1].T]).astype(np.int64)
    SB.add(np.ascontiguousarray(blk))
    if (r0 // a.blk) % 5 == 0:
        print(f"  rows of [A;F]^T {r1}/{c}: rank {SB.rank} ({time.time() - t0:.0f}s)", flush=True)
rank = SB.rank
nul = len(R) + nF - rank
print(f"rank [A[R]; F] = {rank}; null space dimension {nul} ({time.time() - t0:.0f}s)", flush=True)
N = SB.null_space()
del SB
NZ = N[:, len(R):] % P
from fortuity.hwcoh import ffla
print(f"null space computed; exact rank of its F-part (dimension of the new directions) {int(ffla.rank(NZ))} "
      f"({time.time() - t0:.0f}s)", flush=True)
XR = matmul_mod(rng.integers(0, P, (a.k, nul), dtype=np.int64), N % P)
del N
X = np.zeros((nB, a.k), dtype=np.int64)
X[R] = XR[:, :len(R)].T
Z = XR[:, len(R):] % P
# check x^T B + z^T F = 0
B = np.memmap(os.path.join(a.prep, 'B.i32'), dtype=np.int32, mode='r', shape=(nB, c))
acc = matmul_mod(np.ascontiguousarray(Z), F.astype(np.int64) % P)
for r0 in range(0, nB, a.blk):
    r1 = min(nB, r0 + a.blk)
    if X[r0:r1].any():
        acc = (acc + matmul_mod(np.ascontiguousarray(X[r0:r1].T), np.asarray(B[r0:r1], dtype=np.int64))) % P
nz = int(np.count_nonzero(acc))
print(f"check x^T A + z^T F = 0: {nz} nonzero of {acc.size}; z nonzero per vector "
      f"{[int(np.count_nonzero(Z[j])) for j in range(a.k)]} ({time.time() - t0:.0f}s)", flush=True)
assert nz == 0
np.save(os.path.join(a.prep, 'xker_aug.npy'), X)
np.save(os.path.join(a.prep, 'zker_aug.npy'), Z)
res = dict(nB=nB, c=c, nF=nF, rows_used=int(len(R)), rank=int(rank), kernel_dim=int(nul), k=a.k, seed=a.seed,
           check_nonzero=nz, time=round(time.time() - t0))
json.dump(res, open(os.path.join(a.prep, 'xker_aug.json'), 'w'), indent=1)
print(json.dumps(res), flush=True)
