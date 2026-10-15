import os, sys, json, numpy as np
from fortuity.hwcoh import ffblas
from fortuity.bmn.modp import P, matmul_mod
from fortuity.hwcoh import ffla
print("HAVE_FFBLAS", ffblas.HAVE_FFBLAS, flush=True)
rng = np.random.default_rng(5)
ok = True
for trial in range(6):
    m, k, n = rng.integers(50, 700, 3)
    A = rng.integers(0, P, (m, k)); B = rng.integers(0, P, (k, n))
    ok &= np.array_equal(ffblas.matmul(A, B), matmul_mod(A, B, P))
    B32 = rng.integers(0, P, (k, n)).astype(np.int32); C = np.zeros((m, n), np.int32)
    ffblas.matmul_into32(A, B32, C)
    ok &= np.array_equal(C.astype(np.int64), matmul_mod(A, B32.astype(np.int64), P))
    # rank-deficient pair: A1 of rank r1, A2 adding r2 more directions
    c = int(rng.integers(300, 900)); r1 = int(rng.integers(10, 200)); r2 = int(rng.integers(0, 100))
    U = rng.integers(0, P, (r1 + r2, c))
    A1 = matmul_mod(rng.integers(0, P, (r1 + 40, r1)), U[:r1], P).astype(np.int32)
    A2 = matmul_mod(rng.integers(0, P, (r2 + 30, r1 + r2)), U, P).astype(np.int32) if r2 else np.zeros((0, c), np.int32)
    got = ffblas.rank_pair32(A1, A2, blk=64)
    ok &= got == (r1, r1 + r2)
    N = ffblas.nullspace_right(A1.astype(np.int64))
    ok &= N.shape[0] == c - r1 and not matmul_mod(A1.astype(np.int64), N.T % P, P).any()
    print(trial, (m, k, n), got, (r1, r1 + r2), N.shape[0], c - r1, ok, flush=True)
print("ALL_OK" if ok else "MISMATCH")
