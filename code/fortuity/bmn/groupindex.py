"""BMN-sector superconformal index for a general compact gauge group, by an exact torus-grid Haar integral.

Same index and letter weights as `fullindex.py`:
    Z(t,x,y) = Tr (-1)^F t^{2 sum_I (R_I+J)} x^{q1-q3} y^{q3-q2},   q_I = R_I + J,
    phi_I -> t^2 x_I (boson), psi_I -> -t^4 / x_I (fermion), f -> t^6 (boson),   (x_1,x_2,x_3) = (x, 1/y, y/x).

Weyl integration formula over the maximal torus (rank r, variables z_1..z_r):
    Z = 1/|W| * oint prod_i dz_i/(2 pi i z_i)  prod_{alpha in roots} (1 - z^alpha)  PE[ f(t,x,y) chi_adj(z) ],
with chi_adj = sum_{alpha in roots} z^alpha + r. The contour integral is the average over an M^r grid of
M-th roots of unity, exact once M exceeds the largest |exponent| of any z_i in the truncated integrand.
The integrand is symmetric under permutations of the z_i for all groups implemented here, so the grid is
summed over multisets (as in `fullindex.py`).

Groups: 'SU' (rank N-1, weights e_a - e_b with z_N = 1), 'SO_odd' (SO(2N+1), type B_N),
'Sp' (Sp(N) = USp(2N), type C_N), 'G2' (N ignored). Adjoint characters as in arXiv:2506.13887 eq. (4.4).
"""
import math
import itertools
import numpy as np

from .fullindex import LETTERS, SeriesBatch, orbits


def root_system(group, N):
    """Return (roots, rank, weyl_order). Roots are integer tuples of length rank."""
    if group == 'SU':
        r = N - 1
        e = [tuple(1 if k == a else 0 for k in range(r)) if a < r else (0,) * r for a in range(N)]
        roots = [tuple(e[a][k] - e[b][k] for k in range(r)) for a in range(N) for b in range(N) if a != b]
        return roots, r, math.factorial(N)
    if group == 'G2':
        # G2 roots in the basis of two short simple-like roots: short +-(1,0), +-(0,1), +-(1,1);
        # long +-(2,1), +-(1,-1), +-(1,2). |W| = 12. N is ignored. Symmetric under swapping the two variables.
        short = [(1, 0), (0, 1), (1, 1)]
        long_ = [(2, 1), (1, -1), (1, 2)]
        roots = [s_ for v in short + long_ for s_ in (v, tuple(-c for c in v))]
        return roots, 2, 12
    r = N
    unit = [tuple(1 if k == i else 0 for k in range(r)) for i in range(r)]
    roots = []
    for i in range(r):
        for j in range(i + 1, r):
            for si in (1, -1):
                for sj in (1, -1):
                    roots.append(tuple(si * unit[i][k] + sj * unit[j][k] for k in range(r)))
    long_or_short = 1 if group == 'SO_odd' else 2 if group == 'Sp' else None
    if long_or_short is None:
        raise ValueError(group)
    for i in range(r):
        for s in (1, -1):
            roots.append(tuple(s * long_or_short * unit[i][k] for k in range(r)))
    return roots, r, (2 ** N) * math.factorial(N)


def group_dim(group, N):
    roots, r, _ = root_system(group, N)
    return len(roots) + r


def _n_workers(workers):
    import os
    if workers is None:
        workers = int(os.environ.get('FORTUITY_WORKERS', '1'))
    return max(1, workers)


def _index_chunk(arg):
    """Partial grid sum over one block of torus-grid orbits (top level, so that it can run in a worker process)."""
    block, R_roots, M, jmax, A, letters = arg
    omega = np.exp(2j * np.pi / M)
    r = R_roots.shape[1]
    C = len(block)
    Rz = np.array([list(ms) for ms, _ in block], dtype=np.int64)      # (C, r) residues
    w = np.array([sz for _, sz in block], dtype=np.float64)
    zr = omega ** ((Rz @ R_roots.T) % M)                              # z^alpha
    vand = np.prod(1 - zr, axis=1)
    weights = [zr[:, k] for k in range(R_roots.shape[0])] + [np.ones(C, dtype=np.complex128)] * r
    sb = SeriesBatch(C, jmax, A)
    for (jm, am, bm, ferm) in letters:
        for c in weights:
            sb.mul_factor(c, jm, am, bm, ferm)
    return np.einsum('c,cjab->jab', w * vand, sb.S)


def bmn_index(group, N, jmax, chunk=64, unrefined=False, verbose=False, workers=None):
    """Return (Z, A): Z[j, a+A, b+A] = coefficient of t^j x^a y^b of the BMN index, j <= jmax.
    If unrefined, x = y = 1 and A = 0. workers: processes for the grid sum (default: env FORTUITY_WORKERS or 1)."""
    roots, r, W = root_system(group, N)
    L = jmax // 2                                   # max number of letters
    wmax = max(abs(c) for a in roots for c in a)
    vpos = max(sum(max(0, a[i]) for a in roots) for i in range(r))
    M = L * wmax + vpos + 1                         # exceeds every |exponent| in the truncated integrand
    A = 0 if unrefined else L
    omega = np.exp(2j * np.pi / M)
    R_roots = np.array(roots, dtype=np.int64)       # (nroots, r)
    total = np.zeros((jmax + 1, 2 * A + 1, 2 * A + 1), dtype=np.complex128)
    orbs = list(orbits(M, r))
    if verbose:
        print(f"[bmn_index] {group}({N}) rank={r} |W|={W} jmax={jmax} M={M} orbits={len(orbs)}", flush=True)
    letters = LETTERS if not unrefined else [(j, 0, 0, f) for (j, a, b, f) in LETTERS]
    blocks = [orbs[s:s + chunk] for s in range(0, len(orbs), chunk)]
    workers = _n_workers(workers)
    args = [(b, R_roots, M, jmax, A, letters) for b in blocks]
    if workers > 1 and len(blocks) > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(workers) as pool:
            for part in pool.imap_unordered(_index_chunk, args, chunksize=max(1, len(args) // (8 * workers))):
                total += part
    else:
        for a in args:
            total += _index_chunk(a)
    total /= (M ** r) * W
    Zr = np.rint(total.real).astype(np.int64)
    err = np.abs(total - Zr).max()
    if verbose:
        print(f"[bmn_index] done, max rounding error {err:.2e}", flush=True)
    assert err < 1e-4, err
    return Zr, A


def unrefined_series(group, N, jmax, **kw):
    """Coefficients c_j of t^j (x = y = 1), j = 0..jmax."""
    Z, _ = bmn_index(group, N, jmax, unrefined=True, **kw)
    return [int(v) for v in Z[:, 0, 0]]


def compare_dual_pair(N, jmax_unrefined, jmax_refined, verbose=False):
    """SO(2N+1) vs Sp(N) BMN indices: unrefined series to jmax_unrefined and the SU(3)_R content of
    the refined difference to jmax_refined. Returns a JSON-serialisable dict."""
    from .fullindex import su3_multiplicities
    so = unrefined_series('SO_odd', N, jmax_unrefined, verbose=verbose)
    sp = unrefined_series('Sp', N, jmax_unrefined, verbose=verbose)
    Zs, A = bmn_index('SO_odd', N, jmax_refined, verbose=verbose)
    Zp, _ = bmn_index('Sp', N, jmax_refined, verbose=verbose)
    D = Zs - Zp
    reps = {}
    for j in range(jmax_refined + 1):
        if D[j].any():
            reps[str(j)] = {f"[{p},{q}]": int(m) for (p, q), m in sorted(su3_multiplicities(D[j], A).items())}
    return {
        "N": N,
        "groups": [f"SO({2 * N + 1})", f"Sp({N})"],
        "jmax_unrefined": jmax_unrefined,
        "jmax_refined": jmax_refined,
        "index_SO_unrefined": so,
        "index_Sp_unrefined": sp,
        "difference_SO_minus_Sp_unrefined": [a - b for a, b in zip(so, sp)],
        "difference_SO_minus_Sp_su3_reps": reps,
    }


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="SO(2N+1) vs Sp(N) BMN index comparison, written as JSON")
    ap.add_argument("N", type=int)
    ap.add_argument("jmax_unrefined", type=int)
    ap.add_argument("jmax_refined", type=int)
    ap.add_argument("out")
    a = ap.parse_args()
    res = compare_dual_pair(a.N, a.jmax_unrefined, a.jmax_refined, verbose=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("difference:", res["difference_SO_minus_Sp_unrefined"])
    print("reps:", res["difference_SO_minus_Sp_su3_reps"])
