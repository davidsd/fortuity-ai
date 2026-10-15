"""BMN index of a gauge group at fixed numerical SU(3)_R fugacities (x, y) (T-043).
The fully refined expansion of fortuity.bmn.groupindex.bmn_index costs ~(2A+1)^2 times the unrefined one (~1000x at
j = 30). Evaluating at fixed (x, y) on the unit circle costs the same as the unrefined series; the level-j coefficient is
then the SU(3)_R character of that level at z = (x, 1/y, y/x) (the phi_I fugacities of fullindex.LETTERS), which can be
compared with sums of Schur polynomials `schur(irrep(p, q), z)` of candidate irreps [p, q].
CLI (from code/): python -m fortuity.bmn.fugacityindex N j [npoints]   (SO(2N+1) - Sp(N) difference at t^j)"""
import sys, cmath, math, random
import numpy as np
from .groupindex import root_system, _n_workers
from .fullindex import LETTERS, SeriesBatch, orbits


def _chunk(arg):
    block, R_roots, M, jmax, letters = arg
    omega = np.exp(2j * np.pi / M)
    r = R_roots.shape[1]
    C = len(block)
    Rz = np.array([list(ms) for ms, _ in block], dtype=np.int64)
    w = np.array([sz for _, sz in block], dtype=np.float64)
    zr = omega ** ((Rz @ R_roots.T) % M)
    vand = np.prod(1 - zr, axis=1)
    weights = [zr[:, k] for k in range(R_roots.shape[0])] + [np.ones(C, dtype=np.complex128)] * r
    sb = SeriesBatch(C, jmax, 0)
    for (jm, fl, ferm) in letters:
        for c in weights:
            sb.mul_factor(c * fl, jm, 0, 0, ferm)
    return np.einsum('c,cj->j', w * vand, sb.S[:, :, 0, 0])


def index_at(group, N, jmax, x, y, chunk=4096, workers=None):
    roots, r, W = root_system(group, N)
    L = jmax // 2
    wmax = max(abs(c) for a in roots for c in a)
    vpos = max(sum(max(0, a[i]) for a in roots) for i in range(r))
    M = L * wmax + vpos + 1
    letters = [(jm, x ** a * y ** b, f) for (jm, a, b, f) in LETTERS]
    orbs = list(orbits(M, r))
    args = [([*orbs[s:s + chunk]], np.array(roots, dtype=np.int64), M, jmax, letters) for s in range(0, len(orbs), chunk)]
    workers = _n_workers(workers)
    total = np.zeros(jmax + 1, dtype=np.complex128)
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(workers) as pool:
            for part in pool.imap_unordered(_chunk, args):
                total += part
    else:
        for a in args:
            total += _chunk(a)
    return total / (M ** r * W)


def schur(lam, z):
    l = [lam[i] + 2 - i for i in range(3)]
    num = np.linalg.det(np.array([[zi ** li for zi in z] for li in l]))
    den = np.linalg.det(np.array([[zi ** (2 - i) for zi in z] for i in range(3)]))
    return num / den


def irrep(p, q):            # Dynkin [p, q] -> partition (p + q, q, 0)
    return (p + q, q, 0)


if __name__ == "__main__":
    N, j = int(sys.argv[1]), int(sys.argv[2])
    npts = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    pred = {3: [(0, 0)], 4: [(2, 0)], 5: [(4, 0), (0, 2)], 6: [(6, 0), (2, 2), (0, 0)]}[N]
    # alternatives with the same dimension at N = 6, to show the points discriminate
    alts = {5: [[(4, 0), (0, 2)], [(0, 4), (2, 0)]], 6: [[(6, 0), (2, 2), (0, 0)], [(0, 6), (2, 2), (0, 0)], [(6, 0), (0, 3), (3, 0), (1, 1), (0, 0)]]}.get(N, [pred])
    rng = random.Random(1)
    for _ in range(npts):
        x, y = cmath.exp(2j * math.pi * rng.random()), cmath.exp(2j * math.pi * rng.random())
        z = (x, 1 / y, y / x)
        d = index_at('SO_odd', N, j, x, y)[j] - index_at('Sp', N, j, x, y)[j]
        line = f"x={x:.4f} y={y:.4f}: diff at t^{j} = {d:.6f}"
        for alt in alts:
            v = -sum(schur(irrep(*pq), z) for pq in alt)
            line += f" | -{alt}: {v:.6f} (|diff| {abs(d - v):.1e})"
        print(line, flush=True)
