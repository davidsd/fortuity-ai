"""Restriction of non-graviton (fortuitous-candidate) cohomology classes from a group G to a subgroup H, per
SU(3)_R highest-weight sector (n = 2J, lam).

If the Lie algebra h is a subalgebra of g (e.g. g2 inside so(7)), evaluating the letters on h-valued matrices is a chain
map: Q is built from commutators, so it commutes with the restriction. It also commutes with SU(3)_R (which acts on
flavours, not on gauge indices), so it maps SU(3)_R highest-weight vectors to highest-weight vectors, and the
hw vectors of a sector are formal combinations of words that do not depend on the group. The induced map on
non-graviton classes, H_G / (exact + graviton) -> H_H / (exact + graviton), is well defined because restriction maps
G-exact operators to H-exact ones and G-graviton polynomials to H-graviton polynomials.

restrict_sector returns, for one sector:
  bh_G         number of non-graviton classes of G (hw multiplicity),
  bh_G_to_H    rank of their image in the non-graviton classes of H (how many survive as non-graviton),
  bh_H         number of non-graviton classes of H in the same sector (for comparison).
"""
import numpy as np
from fortuity.bmn.words import Q_multi
from fortuity.bmn.modp import RREFBasis, matmul_mod, P
from .sector import Sector, hw_mult_nf
from .evalg import DegFrames, nferm
from .hw import left_kernel


def _lower_q_rows(S, dims, margin=64, seed=4242, r_cert=None, max_rounds=5):
    """Q-images of the hw vectors of sector S (rows aligned with S.hw rows), evaluated at random matrices of the sector
    one step down (2J - 1). Rows that are zero there are exactly the Q-closed hw combinations -- provided the evaluation
    is faithful on the span of the images. That was not checked before 2026-10-01 (review, Part B.1: SU(3)
    (4,4,2) 2J=3 gave rank 88 instead of the certified 89 for every seed at margin 64, hence a fake closed combination).
    Now the margin grows x4 per round (fresh seed) until the rank is stable over two consecutive rounds and, if r_cert
    (the certified rank of Q from sector n to n-1, e.g. r_up of sector n-1 from the engine) is given, equals it. A random
    evaluation can only underestimate the rank, so rank == r_cert certifies faithfulness."""
    from . import ffla
    prev = None
    for k in range(max_rounds):
        L = _lower_q_rows_once(S, dims, margin * 4 ** k, seed + 7919 * k)
        r = ffla.rank(L % P) if L.size else 0
        if r_cert is not None:
            if r == r_cert:
                return L
            if r > r_cert:
                raise RuntimeError(f"lower rank {r} exceeds the certified {r_cert}: inconsistent inputs")
        elif prev is not None and r == prev:
            return L
        prev = r
    raise RuntimeError(f"_lower_q_rows: rank not certified after {max_rounds} rounds (last {prev}, certified {r_cert})")


def _lower_q_rows_once(S, dims, margin=64, seed=4242):
    """One evaluation of the lower Q-images (see _lower_q_rows, which checks faithfulness)."""
    blocks, targets = [], {}
    for widx, H in S.hw:
        ent = []
        for r, wi in enumerate(widx):
            for m2, c in Q_multi(S.ws[int(wi)]).items():
                ent.append((r, targets.setdefault(m2, len(targets)), c))
        blocks.append((H, ent))
    nrows = sum(H.shape[0] for H, _ in blocks)
    if not targets:
        return np.zeros((nrows, 0), dtype=np.int64)
    tw = list(targets)
    degs = sorted({nferm(w) for w in tw})
    n1 = S.n - 1
    need = {d: max(1, hw_mult_nf(dims, n1, S.lam, (n1 - d) // 2)) + margin for d in degs}
    fr = DegFrames(S.N, need, seed=seed, group=S.group)
    X = fr.eval_words(tw)                                     # (len(tw), ncols)
    out = np.zeros((nrows, X.shape[1]), dtype=np.int64)
    r0 = 0
    for H, ent in blocks:
        cols = sorted({j for _, j, _ in ent})
        if cols:
            loc = {j: a for a, j in enumerate(cols)}
            Qm = np.zeros((H.shape[1], len(cols)), dtype=np.int64)
            for r, j, c in ent:
                Qm[r, loc[j]] += c
            out[r0:r0 + H.shape[0]] = matmul_mod(matmul_mod(H, Qm % P), X[cols])
        r0 += H.shape[0]
    return out


def _rank_increase(base_rows, new_rows):
    """Indices of new_rows that are independent modulo the row space of base_rows (greedy, in order)."""
    ncols = base_rows.shape[1] if base_rows.size else new_rows.shape[1]
    b = RREFBasis(ncols)
    if base_rows.shape[0]:
        b.add_batched(base_rows % P)
    flags = b.add_batched(new_rows % P) if new_rows.shape[0] else np.zeros(0, dtype=bool)
    return np.flatnonzero(flags)


def certified_lower_rank(N, lam, n, dims, GA, group, seed=1):
    """Rank of Q from sector (n, lam) to (n-1, lam) on hw vectors, certified by the engine (r_up of sector n-1, whose run
    checks rank = exact multiplicity)."""
    if n == 0:
        return 0
    res = Sector(N, lam, n - 1, dims, log=lambda s: None, graviton_algebra=GA, group=group, seed=seed).run()
    if not res.get('check'):
        raise RuntimeError(f"engine self-check failed in sector {n - 1} {lam}")
    return int(res['r_up'])


def _sector_data(N, lam, n, dims, GA, group, seed):
    S = Sector(N, lam, n, dims, log=lambda s: None, graviton_algebra=GA, group=group, seed=seed)
    S.prepare()
    A, B, G = S.evaluate()
    if n > 0:
        L = _lower_q_rows(S, dims, r_cert=certified_lower_rank(N, lam, n, dims, GA, group, seed))
    else:
        L = np.zeros((A.shape[0], 0), dtype=np.int64)
    return S, A, B, G, L


def restrict_sector(lam, n, G_spec, H_spec, GA, seed=1):
    """G_spec, H_spec: (N, group, dims). Returns dict with bh_G, bh_G_to_H, bh_H and consistency checks."""
    (NG, gG, dG), (NH, gH, dH) = G_spec, H_spec
    SG, AG, BG, GG, LG = _sector_data(NG, lam, n, dG, GA, gG, seed)
    SH, AH, BH, GH, LH = _sector_data(NH, lam, n, dH, GA, gH, seed + 1)
    assert len(SG.hw) == len(SH.hw) and all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                                            for a, b in zip(SG.hw, SH.hw)), "hw vectors must be identical"
    nA = AG.shape[0]
    # G-closed combinations of hw vectors: left kernel of the lower Q-rows (all hw vectors if L has no columns)
    Kc = left_kernel(LG) if LG.shape[1] else np.eye(nA, dtype=np.int64)
    # non-graviton representatives of G: closed combinations independent modulo exact + graviton at G's functionals
    EGg = np.concatenate([BG, GG], axis=0) if (BG.size or GG.size) else np.zeros((0, AG.shape[1]), dtype=np.int64)
    closed_vals = matmul_mod(Kc, AG) if Kc.shape[0] else np.zeros((0, AG.shape[1]), dtype=np.int64)
    idx = _rank_increase(EGg, closed_vals)
    R = Kc[idx]                                               # coefficient rows (over hw vectors) of the representatives
    # checks: the representatives are closed in H too (restriction is a chain map)
    closed_in_H = (not R.shape[0]) or (not LH.shape[1]) or not (matmul_mod(R, LH) % P).any()
    # image in H: evaluate the same formal combinations at H's functionals, count independent ones mod H exact + graviton
    EGh = np.concatenate([BH, GH], axis=0) if (BH.size or GH.size) else np.zeros((0, AH.shape[1]), dtype=np.int64)
    img = matmul_mod(R, AH) if R.shape[0] else np.zeros((0, AH.shape[1]), dtype=np.int64)
    surv = len(_rank_increase(EGh, img))
    # H's own non-graviton classes in this sector
    KcH = left_kernel(LH) if LH.shape[1] else np.eye(nA, dtype=np.int64)
    bhH = len(_rank_increase(EGh, matmul_mod(KcH, AH) if KcH.shape[0] else np.zeros((0, AH.shape[1]), dtype=np.int64)))
    return dict(lam=list(lam), n=n, m_G=SG.m, m_H=SH.m, bh_G=int(R.shape[0]), bh_G_to_H=int(surv), bh_H=int(bhH),
                closed_in_H=bool(closed_in_H))
