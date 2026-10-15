"""Exact sector dimensions refined by the number of f letters: dim V_(2J, q1, q2, q3; n_f).

Same Haar-integral method as fortuity.bmn.fullindex.partition_dims (SU(N) torus grid, exact), with an extra
grading by n_f (the fermion degree of a word in sector 2J is n_psi = 2J - 2 n_f).  Summing over n_f reproduces
partition_dims (checked in tests).
"""
import math
import numpy as np
from fortuity.bmn.fullindex import orbits, LETTERS


def partition_dims_nf(N, jmax, chunk=16, group='SU', workers=None, twoJ_max=None, progress=False):
    """twoJ_max (SO/Sp path): keep only 2J <= twoJ_max (and n_f <= twoJ_max // 2). Exact for those sectors, since
    letters only add to 2J and n_f; much less memory traffic."""
    if group != 'SU':
        return _partition_dims_nf_general(group, N, jmax, chunk, workers, twoJ_max, progress)
    L = jmax // 2
    M = L + N
    A = L
    n = N - 1
    omega = np.exp(2j * np.pi / M)
    JM = jmax // 2
    FM = jmax // 6
    total = np.zeros((jmax + 1, 2 * A + 1, 2 * A + 1, JM + 1, FM + 1), dtype=np.complex128)
    orbs = list(orbits(M, n))
    # letters: (j, a, b, ferm, twoJ increment, nf increment)
    letters = []
    for (jm, am, bm, ferm) in LETTERS:
        tj = 1 if ferm else (2 if jm == 6 else 0)
        nf = 1 if jm == 6 else 0
        letters.append((jm, am, bm, ferm, tj, nf))
    for s in range(0, len(orbs), chunk):
        block = orbs[s:s + chunk]
        C = len(block)
        R = np.array([list(ms) + [0] for ms, _ in block], dtype=np.int64)
        w = np.array([sz for _, sz in block], dtype=np.float64)
        z = omega ** R
        vand = np.ones(C, dtype=np.complex128)
        for a in range(N):
            for b in range(N):
                if a != b:
                    vand *= (1 - z[:, a] / z[:, b])
        roots = [(z[:, a] / z[:, b]) for a in range(N) for b in range(N) if a != b]
        roots += [np.ones(C, dtype=np.complex128)] * (N - 1)
        S = np.zeros((C, jmax + 1, 2 * A + 1, 2 * A + 1, JM + 1, FM + 1), dtype=np.complex128)
        S[:, 0, A, A, 0, 0] = 1.0
        for (jm, am, bm, ferm, tj, nf) in letters:
            a_lo, a_hi = max(0, am), min(2 * A + 1, 2 * A + 1 + am)
            b_lo, b_hi = max(0, bm), min(2 * A + 1, 2 * A + 1 + bm)
            for r in roots:
                cc = r[:, None, None, None, None]
                rng_j = range(jmax, jm - 1, -1) if ferm else range(jm, jmax + 1)
                for j in rng_j:
                    S[:, j, a_lo:a_hi, b_lo:b_hi, tj:, nf:] += cc * S[:, j - jm, a_lo - am:a_hi - am,
                                                                      b_lo - bm:b_hi - bm, :JM + 1 - tj, :FM + 1 - nf]
        total += np.einsum('c,cjabtf->jabtf', w * vand, S)
    total /= (M ** n) * math.factorial(N)
    Zr = np.rint(total.real).astype(np.int64)
    assert np.abs(total - Zr).max() < 1e-4
    dims = {}
    for j in range(0, jmax + 1, 2):
        for ia, ib, tj, f in zip(*np.nonzero(Zr[j])):
            a, b = ia - A, ib - A
            q3 = (j // 2 - a + b) / 3
            if q3 != int(q3):
                continue
            q3 = int(q3)
            dims[(int(tj), q3 + a, q3 - b, q3, int(f))] = int(Zr[j, ia, ib, tj, f])
    return dims


def _progress(on, i, n, t0):
    import time
    if on and (i == n or i % max(1, n // 50) == 0):
        el = time.time() - t0
        print(f"[dims] {i}/{n} chunks, {el:.0f}s elapsed, ~{el * (n - i) / i:.0f}s left", flush=True)


def _dims_chunk(arg):
    block, R_roots, M, jmax, A, JM, FM, letters = arg
    omega = np.exp(2j * np.pi / M)
    r = R_roots.shape[1]
    C = len(block)
    Rz = np.array([list(ms) for ms, _ in block], dtype=np.int64)
    w = np.array([sz for _, sz in block], dtype=np.float64)
    zr = omega ** ((Rz @ R_roots.T) % M)
    vand = np.prod(1 - zr, axis=1)
    weights = [zr[:, k] for k in range(R_roots.shape[0])] + [np.ones(C, dtype=np.complex128)] * r
    S = np.zeros((C, jmax + 1, 2 * A + 1, 2 * A + 1, JM + 1, FM + 1), dtype=np.complex128)
    S[:, 0, A, A, 0, 0] = 1.0
    for (jm, am, bm, ferm, tj, nf) in letters:
        a_lo, a_hi = max(0, am), min(2 * A + 1, 2 * A + 1 + am)
        b_lo, b_hi = max(0, bm), min(2 * A + 1, 2 * A + 1 + bm)
        for c in weights:
            cc = c[:, None, None, None, None]
            rng_j = range(jmax, jm - 1, -1) if ferm else range(jm, jmax + 1)
            for j in rng_j:
                S[:, j, a_lo:a_hi, b_lo:b_hi, tj:, nf:] += cc * S[:, j - jm, a_lo - am:a_hi - am,
                                                                  b_lo - bm:b_hi - bm, :JM + 1 - tj, :FM + 1 - nf]
    return np.einsum('c,cjabtf->jabtf', w * vand, S)


def _partition_dims_nf_general(group, N, jmax, chunk=16, workers=None, twoJ_max=None, progress=False):
    """Same quantity for SO(2N+1) ('SO_odd') or USp(2N) ('Sp'): Weyl integration formula over the maximal torus
    with the root systems of fortuity.bmn.groupindex (exact grid average; the integrand is symmetric under
    permutations of the torus variables, so the grid is summed over multisets)."""
    from fortuity.bmn.groupindex import root_system
    roots, r, W = root_system(group, N)
    L = jmax // 2
    wmax = max(abs(c) for a in roots for c in a)
    vpos = max(sum(max(0, a[i]) for a in roots) for i in range(r))
    M = L * wmax + vpos + 1
    A = L
    omega = np.exp(2j * np.pi / M)
    JM = jmax // 2
    FM = jmax // 6
    if twoJ_max is not None:
        JM = min(JM, twoJ_max)
        FM = min(FM, twoJ_max // 2)
    R_roots = np.array(roots, dtype=np.int64)
    total = np.zeros((jmax + 1, 2 * A + 1, 2 * A + 1, JM + 1, FM + 1), dtype=np.complex128)
    letters = []
    for (jm, am, bm, ferm) in LETTERS:
        tj = 1 if ferm else (2 if jm == 6 else 0)
        nf = 1 if jm == 6 else 0
        letters.append((jm, am, bm, ferm, tj, nf))
    import time
    t0 = time.time()
    orbs = list(orbits(M, r))
    args = [(orbs[s:s + chunk], R_roots, M, jmax, A, JM, FM, letters) for s in range(0, len(orbs), chunk)]
    from fortuity.bmn.groupindex import _n_workers
    workers = _n_workers(workers)
    if workers > 1 and len(args) > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(workers) as pool:
            for i, part in enumerate(pool.imap_unordered(_dims_chunk, args, chunksize=max(1, len(args) // (64 * workers)))):
                total += part
                _progress(progress, i + 1, len(args), t0)
    else:
        for i, a in enumerate(args):
            total += _dims_chunk(a)
            _progress(progress, i + 1, len(args), t0)
    total /= (M ** r) * W
    Zr = np.rint(total.real).astype(np.int64)
    assert np.abs(total - Zr).max() < 1e-4
    dims = {}
    for j in range(0, jmax + 1, 2):
        for ia, ib, tj, f in zip(*np.nonzero(Zr[j])):
            a, b = ia - A, ib - A
            q3 = (j // 2 - a + b) / 3
            if q3 != int(q3):
                continue
            q3 = int(q3)
            dims[(int(tj), q3 + a, q3 - b, q3, int(f))] = int(Zr[j, ia, ib, tj, f])
    return dims


def collapse(dims_nf):
    out = {}
    for (tj, q1, q2, q3, f), d in dims_nf.items():
        out[(tj, q1, q2, q3)] = out.get((tj, q1, q2, q3), 0) + d
    return out
