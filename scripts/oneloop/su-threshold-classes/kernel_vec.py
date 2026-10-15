"""Random left-kernel vectors of a fullpar prep matrix A = B.i32 (nB x c, c = ncols + ncart), i.e. x with x^T A = 0 mod p.

Rows of the 'out' prep of a sector T are its compressed hw rows h_i, and its columns are the Q_0-image functionals plus
the Cartan (Coulomb-branch) functionals.  So x^T A = 0 says that t = sum_i x_i h_i is Q_0-closed and vanishes on the
Cartan at those functionals: a random x gives a random element of (ker Q_0 n ker Cartan) of T, i.e. a representative of
the non-Coulomb classes plus a random exact vector.

Method: stream the rows of A^T (BT.i32 from transpose_prep.py) into the blocked elimination of gpu_rank.py (each basis
block in RREF on its own pivots and zero on the pivots of earlier blocks), then back-substitute: free entries of x
random, x[piv_j] = -(E_j x) for the blocks in reverse order.  x is uniform in the right kernel of A^T.  Every x is
checked against the original B (x^T B = 0, streamed), independently of the transposition.
usage: python3 kernel_vec.py PREPDIR [--k 4] [--seed 2026]   -> PREPDIR/xker.npy (nB x k int64), xker.json"""
import argparse, json, os, time
import numpy as np
from gpu_rank import backend, ModBasis, P

SPLIT, KMAX = 1 << 13, 1 << 13


def matmul_split(xp, A, X):
    """(A @ X) mod p exactly: A (m x n) int32 entries < p, X (n x k) int64 entries < p; X split into 13-bit halves."""
    acc = xp.zeros((A.shape[0], X.shape[1]), dtype=xp.int64)
    for k0 in range(0, A.shape[1], KMAX):
        k1 = min(A.shape[1], k0 + KMAX)
        Af = A[:, k0:k1].astype(xp.float64)
        Xh = (X[k0:k1] >> 13).astype(xp.float64)
        Xl = (X[k0:k1] & (SPLIT - 1)).astype(xp.float64)
        T1 = xp.fmod(Af @ Xh, P)
        T2 = xp.fmod(Af @ Xl, P)
        acc = (acc + xp.fmod(T1 * SPLIT + T2, P).astype(xp.int64)) % P
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('prep_dir')
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--blk', type=int, default=2048)
    ap.add_argument('--cpu', action='store_true')
    a = ap.parse_args()
    xp, name = backend(a.cpu)
    meta = json.load(open(os.path.join(a.prep_dir, 'meta.json')))
    nB, c = meta['nB'], meta['ncols'] + int(meta.get('ncart', 0) or 0)
    BT = np.memmap(os.path.join(a.prep_dir, 'BT.i32'), dtype=np.int32, mode='r', shape=(c, nB))
    t0 = time.time()
    basis = ModBasis(xp, nB)
    for r0 in range(0, c, a.blk):
        r1 = min(c, r0 + a.blk)
        basis.add(xp.asarray(np.asarray(BT[r0:r1], dtype=np.int64)) % P)
        if (r0 // a.blk) % 10 == 0:
            print(f"  rows of A^T {r1}/{c}: rank {basis.rank} ({time.time() - t0:.0f}s)", flush=True)
    r = basis.rank
    print(f"rank A^T = {r}; left kernel of A has dimension {nB - r} ({time.time() - t0:.0f}s)", flush=True)
    # back-substitution
    rng = np.random.default_rng(a.seed)
    is_piv = np.zeros(nB, dtype=bool)
    for _, piv in basis.blocks:
        is_piv[xp.asnumpy(piv) if name == 'gpu' else piv] = True
    X = rng.integers(0, P, (nB, a.k), dtype=np.int64)
    X[is_piv] = 0
    X = xp.asarray(X)
    for E, piv in reversed(basis.blocks):
        X[piv] = (-matmul_split(xp, E, X)) % P
    Xh = xp.asnumpy(X) if name == 'gpu' else X
    del basis
    # check x^T B = 0 against the original prep matrix (row-major B.i32), streamed
    B = np.memmap(os.path.join(a.prep_dir, 'B.i32'), dtype=np.int32, mode='r', shape=(nB, c))
    acc = xp.zeros((a.k, c), dtype=xp.int64)
    for r0 in range(0, nB, a.blk):
        r1 = min(nB, r0 + a.blk)
        Bb = xp.asarray(np.asarray(B[r0:r1]))
        acc = (acc + matmul_split(xp, Bb.T.copy(), xp.asarray(Xh[r0:r1])).T) % P
    nz = int(xp.count_nonzero(acc))
    print(f"check x^T B = 0: {nz} nonzero entries of {a.k} x {c} ({time.time() - t0:.0f}s)", flush=True)
    assert nz == 0, "kernel vectors fail the check against B"
    np.save(os.path.join(a.prep_dir, 'xker.npy'), Xh)
    res = dict(backend=name, nB=nB, c=c, rank=int(r), kernel_dim=int(nB - r), k=a.k, seed=a.seed,
               check_nonzero=nz, time=round(time.time() - t0), case=meta.get('case'))
    json.dump(res, open(os.path.join(a.prep_dir, 'xker.json'), 'w'), indent=1)
    print(json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
