"""Memory-light version of abelian_lw.py for large sectors (the R-007 order-2 sector T_2 has 154k abelian monomials).

Stage 1 (exact, per block): E1, E2 (SU(3)_R raising), D (SU(2)_R lowering) and e12 (raising of the U(2) that rotates
the two Cartan species; it commutes with every superconformal generator, so it suffices to count U(2)-highest-weight LW
vectors) all preserve the multiset of (a + b, |S|) over the letters.  So the subspace annihilated by them is the direct
sum over those blocks of small dense null spaces.
Stage 2 (energy-lowering K_+-, S_a, T_{al,a}): the stacked images of the stage-1 vectors, compressed by independent
random linear maps on each target space (k = n + margin functionals, in column chunks).  Compression can only lower the
rank, so  LW dim <= (stage-1 dim) - (computed rank)  rigorously; 0 is a proof.
usage: python3 abelian_lw2.py 2J1 2J2 2q1 2q2 2q3 Y [--pts-extra 64]"""
import sys, time, math, argparse
from itertools import combinations
from collections import defaultdict
import numpy as np
from fortuity.bmn.modp import P, matmul_mod
from fortuity.hwcoh import ffla
from fortuity.full.letters import is_ferm
import abelian_lw as A

log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)


def coarse(mono):
    return tuple(sorted((L[1] + L[2], len(L[3])) for L in mono))


def e12(mono):
    """Raising of the species U(2): letter of species 2 -> species 1 (an even derivation)."""
    out = defaultdict(int)
    for pos, L in enumerate(mono):
        if L[0] == 2:
            sgn, m = A.canon(mono[:pos] + ((1,) + L[1:],) + mono[pos + 1:])
            if sgn:
                out[m] += sgn
    return out


def stage1(basis):
    """Basis (list of sparse dicts) of the SU(3)_R-hw, SU(2)_R-singlet subspace."""
    ops = [op for name, op in A.OPS if name in ('E1', 'E2', 'D')] + [e12]
    groups = defaultdict(list)
    for m in basis:
        groups[coarse(m)].append(m)
    out = []
    for g, ms in groups.items():
        rows, ent = {}, []
        for j, m in enumerate(ms):
            for op in ops:
                for m2, c in op(m).items():
                    if c % P:
                        ent.append((rows.setdefault((id(op), m2), len(rows)), j, c % P))
        if not rows:
            out.extend({m: 1} for m in ms)
            continue
        M = np.zeros((len(rows), len(ms)), dtype=np.int64)
        for r, j, c in ent:
            M[r, j] = (M[r, j] + c) % P
        N = ffla.nullspace_right(M) if hasattr(ffla, 'nullspace_right') else None
        if N is None:
            from fortuity.hwcoh.hw import nullspace_left
            N = nullspace_left(np.ascontiguousarray(M.T))
        for v in N:
            out.append({ms[j]: int(v[j]) for j in np.flatnonzero(v % P)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sec', type=int, nargs=5)
    ap.add_argument('Y', type=int)
    ap.add_argument('--pts-extra', type=int, default=64)
    ap.add_argument('--save', default=None)
    a = ap.parse_args()
    sec = tuple(a.sec)
    t0 = time.time()
    basis = A.monomials(sec, a.Y)
    log(f"sector {sec} Y={a.Y}: {len(basis)} abelian monomials ({time.time()-t0:.0f}s)")
    V = stage1(basis)
    n = len(V)
    log(f"stage 1: SU(3)_R hw, SU(2)_R singlet, species-U(2) hw subspace of dimension {n} ({time.time()-t0:.0f}s)")
    if n == 0:
        log("LW dimension 0 (empty after stage 1)")
        return
    lowering = [(name, op) for name, op in A.OPS if name not in ('E1', 'E2', 'D')]
    import scipy.sparse as sp
    idx = {m: i for i, m in enumerate(basis)}
    k = n + a.pts_extra
    Os = []
    for name, op in lowering:
        tgt, rs, cs, vs = {}, [], [], []
        for j, m in enumerate(basis):
            for m2, c in op(m).items():
                if c:
                    rs.append(tgt.setdefault(m2, len(tgt))); cs.append(j); vs.append(int(c))
        if tgt:
            O = sp.csr_matrix((np.array(vs, dtype=np.int64), (np.array(rs), np.array(cs))), shape=(len(tgt), len(basis)))
            assert int(abs(O).max()) < 1 << 20
            Os.append((name, O.T.tocsr()))
        log(f"  {name}: {len(tgt)} target monomials, {len(vs)} entries ({time.time()-t0:.0f}s)")
    # V^T as a sparse matrix split into 13-bit halves (exact products with a dense mod-p matrix in int64)
    rr, cc, vv = [], [], []
    for r, v in enumerate(V):
        for m, c in v.items():
            rr.append(r); cc.append(idx[m]); vv.append(c % P)
    vv = np.array(vv, dtype=np.int64)
    Vh = sp.csr_matrix((vv >> 13, (rr, cc)), shape=(n, len(basis)))
    Vl = sp.csr_matrix((vv & 8191, (rr, cc)), shape=(n, len(basis)))

    def smul(W):
        Wh, Wl = W >> 13, W & 8191
        hh = (Vh @ Wh) % P
        mid = ((Vh @ Wl) % P + (Vl @ Wh) % P) % P
        ll = (Vl @ Wl) % P
        return (hh * ((1 << 26) % P) % P + mid * 8192 % P + ll) % P

    C = np.zeros((n, k), dtype=np.int64)
    chunk = max(1, int(4e8 // max(1, len(basis))))
    for c0 in range(0, k, chunk):
        c1 = min(k, c0 + chunk)
        W = np.zeros((len(basis), c1 - c0), dtype=np.int64)
        for i, (name, OT) in enumerate(Os):
            R = np.random.default_rng(7919 * i + c0).integers(0, P, (OT.shape[1], c1 - c0), dtype=np.int64)
            W = (W + (OT @ R) % P) % P            # O^T R: small integer entries of O, exact in int64
        C[:, c0:c1] = smul(W)
    rk = int(ffla.rank(C))
    log(f"stage 2: rank of the stacked lowering images {rk} of {n}; LW dimension <= {n - rk} ({time.time()-t0:.0f}s)")
    if a.save:
        import pickle
        pickle.dump(dict(sec=sec, Y=a.Y, V=V, C=C, rank=rk), open(a.save, 'wb'), protocol=4)
    if n - rk:
        analyse_kernel(V, C, lowering, t0)


def analyse_kernel(V, C, lowering, t0):
    """Explicit LW candidates: left kernel x of C, f = sum x_r V_r; check every generator kills f exactly, and split
    the space by species weight (n1, n2) (the operators preserve it)."""
    from fortuity.hwcoh.hw import nullspace_left
    X = nullspace_left(C % P)
    fs = []
    for x in X:
        f = defaultdict(int)
        for r in np.flatnonzero(x % P):
            for m, c in V[r].items():
                f[m] = (f[m] + int(x[r]) * c) % P
        fs.append({m: c for m, c in f.items() if c})
    bad = 0
    for f in fs:
        for name, op in list(A.OPS) + [('e12', e12)]:
            img = defaultdict(int)
            for m, c in f.items():
                for m2, c2 in op(m).items():
                    img[m2] = (img[m2] + c * c2) % P
            bad += any(v % P for v in img.values())
    log(f"explicit check: {len(fs)} kernel vectors, nonzero images under the generators: {bad} (must be 0) "
        f"({time.time()-t0:.0f}s)")
    # species-weight decomposition of the kernel space
    wts = sorted({(sum(1 for L in m if L[0] == 1), sum(1 for L in m if L[0] == 2)) for f in fs for m in f})
    from fortuity.hwcoh import ffla
    dims = {}
    for w in wts:
        rows = [{m: c for m, c in f.items() if (sum(1 for L in m if L[0] == 1), sum(1 for L in m if L[0] == 2)) == w}
                for f in fs]
        keys = sorted({m for r in rows for m in r})
        M = np.array([[r.get(m, 0) for m in keys] for r in rows], dtype=np.int64)
        dims[w] = int(ffla.rank(M)) if M.size else 0
    log(f"LW (species-hw) dimension by species weight (n1, n2): {dims}")
    # S_3 (Weyl) invariants in the GL(2) irreps generated: Sym^(n1-n2) (x) det^n2, det = sign on S_3
    inv = lambda m: sum(1 for j in range(m // 3 + 1) if (m - 3 * j) % 2 == 0)
    sgn = lambda m: inv(m - 3) if m >= 3 else 0
    tot = 0
    for (n1, n2), d in dims.items():
        k = inv(n1 - n2) if n2 % 2 == 0 else sgn(n1 - n2)
        tot += d * k
        log(f"  species weight ({n1},{n2}): {d} hw vectors x {k} Weyl invariants in Sym^{n1 - n2} (x) det^{n2}")
    log(f"Weyl-invariant LW dimension (upper bound on Coulomb-type LW classes): {tot}")


if __name__ == '__main__':
    main()
