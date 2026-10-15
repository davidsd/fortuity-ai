"""T-023 extra checks.
  coulomb G N lam n DIMS    : number of graviton classes (hw, mod exact) and of Coulomb-type ones (rank of graviton rows
                              on Cartan-valued letters, where exact operators vanish) in sector (n, lam)
  hairy DIMS                : is the product (SO(7) j=18 class) x w_2^1 a non-graviton class of SO(7) in (3;5,4,4)?
Run from code/.  Validation: `coulomb SU 3 4,4,4 3 <SU(3) dims>` must give 29 graviton classes, 28 Coulomb (R-002)."""
import sys, pickle, math, copy
import numpy as np
from collections import defaultdict
from fortuity.bmn.modp import RREFBasis, P
from fortuity.bmn.words import add_term, w_gen
from fortuity.hwcoh.sector import Sector, frame_k
from fortuity.hwcoh.evalg import Frame, DegFrames, nferm
from fortuity.hwcoh.gravhw import GravitonAlgebra


def rank(M):
    if not M.shape[0]:
        return 0
    b = RREFBasis(M.shape[1]); b.add_batched(M % P); return b.rank


def coulomb(group, N, lam, n, dims, gen_nmax=8, npts_extra=64, seed=11):
    GA = GravitonAlgebra(gen_nmax)
    S = Sector(N, lam, n, dims, log=lambda s: None, graviton_algebra=GA, group=group)
    S.prepare()
    A, B, G = S.evaluate()
    grav = rank(np.concatenate([B, G])) - rank(B)
    gb = S.gblocks
    need = {d: G.shape[0] + npts_extra for d in sorted(S.m_d)}
    frames = {}
    cols = []
    for d in sorted(S.m_d):
        k = frame_k(N, d, group=group)
        npts = max(1, math.ceil(need[d] / max(1, math.comb(k, d))))
        fr = Frame(N, npts, k, [d], seed=seed + d, group=group, cartan=True)
        df = DegFrames.__new__(DegFrames); df.frames, df.off, df.ncols = {d: fr}, {d: 0}, fr.ncols
        cols.append(GA.rows_from_blocks(gb, df))
    Gc = np.concatenate(cols, axis=1) if cols else np.zeros((G.shape[0], 0), dtype=np.int64)
    return dict(group=group, lam=list(lam), n=n, graviton_classes=int(grav), coulomb_graviton_classes=int(rank(Gc)))


def hairy(dims):
    from fortuity.bmn.g2_checks import so7_class as operator
    O = operator()
    w = w_gen((0,), 1)                                    # w_2 with upper index 1: highest weight of [1,0]
    Pexpr = defaultdict(int)
    for m1, c1 in O.items():
        for m2, c2 in w.items():
            add_term(Pexpr, tuple(m1) + tuple(m2), c1 * c2)
    Pexpr = {m: c for m, c in Pexpr.items() if c}
    GA = GravitonAlgebra(8)
    S = Sector(3, (5, 4, 4), 3, dims, log=lambda s: None, graviton_algebra=GA, group='SO_odd')
    S.prepare()
    A, B, G = S.evaluate()
    # evaluate P at the sector's functionals, degree by degree
    byd = defaultdict(list)
    for m, c in Pexpr.items():
        byd[nferm(m)].append((m, c))
    v = np.zeros(A.shape[1], dtype=np.int64)
    col = 0
    for d, fr in S.colblocks:
        terms = byd.get(d, [])
        if terms:
            X = fr.eval_words([m for m, _ in terms])
            coef = np.array([c % P for _, c in terms], dtype=np.int64)
            v[col:col + fr.ncols] = (v[col:col + fr.ncols] + (coef[:, None] * X % P).sum(axis=0)) % P
        col += fr.ncols
    EG = np.concatenate([B, G])
    r0, r1 = rank(EG), rank(np.concatenate([EG, v[None, :]]))
    return dict(product_terms=len(Pexpr), P_nonzero=bool(v.any()), P_is_nongraviton=bool(r1 == r0 + 1),
                degrees=sorted(byd))


if __name__ == "__main__":
    if sys.argv[1] == 'coulomb':
        g, N, lam, n, dp = sys.argv[2], int(sys.argv[3]), tuple(map(int, sys.argv[4].split(','))), int(sys.argv[5]), sys.argv[6]
        extra = int(sys.argv[7]) if len(sys.argv) > 7 else 64          # functional margin (recheck positives with more)
        seed = int(sys.argv[8]) if len(sys.argv) > 8 else 11
        res = coulomb(g, N, lam, n, pickle.load(open(dp, 'rb')), npts_extra=extra, seed=seed)
        print("COULOMB", res, f"npts_extra={extra} seed={seed}", flush=True)
    elif sys.argv[1] == 'hairy':
        print("HAIRY", hairy(pickle.load(open(sys.argv[2], 'rb'))), flush=True)
