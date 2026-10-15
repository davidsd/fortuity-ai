"""T-019 (SO/Sp part), step 1: reproduce ChoiLee2025Konishi -- the SO(7) fortuitous class of R-005 is lifted at one loop.
O_18 = eq. (4.14) of arXiv:2506.13887 (SO7_CLASS_TERMS of code/fortuity/bmn/g2_checks.py), converted to
superfield letters: phi_i -> (0,0,(i,)), psi_1 -> +(0,0,(1,2)), psi_2 -> -(0,0,(0,2)), psi_3 -> +(0,0,(0,1)), f -> -(0,0,(0,1,2))
(signs fixed by matching Q_0: letters.Q_letter against fortuity.bmn.words.Q_multi).
Lifting test: is Q_1 O_18 Q_0-exact in the target sector (2J1,2J2;2q) = (0,0;6,6,6), Y = 8, d_BMN = 1?  The Q_0-exact
operators there are Q_0 of all words of (1,1;6,6,6), Y = 7, d_BMN = 1. Not exact <=> lifted (the class pairs with the
class [Q_1 O_18] at one loop).
Run: cd scripts/oneloop/so-odd-classes && PYTHONPATH=../../../code python so7_lift.py"""
import sys
import numpy as np
from collections import defaultdict
from fortuity.bmn.g2_checks import SO7_CLASS_TERMS as TERMS
from fortuity.bmn.words import PHI as BPHI, PSI as BPSI, F as BF
from fortuity.full.words import add_term, Q_expr, sector_of
from fortuity.full.q1 import Q1_expr
from fortuity.full.evalg import Frame, nferm

CONV = {BPHI[0]: (1, (0, 0, (0,))), BPHI[1]: (1, (0, 0, (1,))), BPHI[2]: (1, (0, 0, (2,))),
        BPSI[0]: (1, (0, 0, (1, 2))), BPSI[1]: (-1, (0, 0, (0, 2))), BPSI[2]: (1, (0, 0, (0, 1))),
        BF: (-1, (0, 0, (0, 1, 2)))}


def O18():
    O = defaultdict(int)
    for c, traces in TERMS:
        sign, multi = c, []
        for tr_ in traces:
            w = []
            for l in tr_:
                s, L = CONV[l]
                sign *= s
                w.append(L)
            multi.append(tuple(w))
        add_term(O, tuple(multi), sign)
    return {m: c for m, c in O.items() if c}


def values(expr, N, group, npts=6, seed=5, cartan=False):
    words = list(expr)
    letters = sorted({L for m in words for w in m for L in w})
    degs = sorted({nferm(m) for m in words})
    fr = Frame(N, npts, max(degs) + 2, degs, letters, seed=seed, group=group, cartan=cartan)
    M = fr.eval_words(words, dtype=np.int64) % fr.p
    c = np.array([expr[m] % fr.p for m in words], dtype=np.int64)
    # accumulate mod p in chunks: c @ M over ~1e5 terms of size ~p^2 would overflow int64
    out = np.zeros(M.shape[1], dtype=np.int64)
    for a0 in range(0, len(words), 512):
        out = (out + ((c[a0:a0 + 512, None] * M[a0:a0 + 512]) % fr.p).sum(axis=0)) % fr.p
    return out


if __name__ == "__main__":
    O = O18()
    print(f"O_18: {len(O)} canonical terms; sectors {set(sector_of(m) for m in O)}")
    QO = Q_expr(O)
    print("Q_0 O_18 = 0 on so(7):", not values(QO, 3, 'SO_odd').any(), "| on so(9):", not values(QO, 4, 'SO_odd').any())
    Q1O = Q1_expr(O, 7, 'SO_odd')
    secs = defaultdict(int)
    for m in Q1O:
        secs[(sector_of(m), sum(len(w) for w in m), sum(L[0] + L[1] for w in m for L in w))] += 1
    print(f"Q_1 O_18: {len(Q1O)} terms; (sector, Y, total derivatives): {dict(secs)}")
    print("Q_1 O_18 nonzero on so(7):", values(Q1O, 3, 'SO_odd').any(),
          "| vanishes on the Coulomb branch:", not values(Q1O, 3, 'SO_odd', cartan=True).any())


def exactness_test(v_expr, controls, N, group, src_sec, tgt_sec, Y_src, D, seeds=(11, 12), log=print):
    """Is v (an expression in the target sector) in the span of Q_0(source words)?  Returns (rank_T, rank_B, rank_Bv)
    per seed, with a point-doubling check that the evaluation is faithful on the target space."""
    from sector_words import words
    from fortuity.bmn.modp import RREFBasis
    import math
    S = words(src_sec, Y_src, D)
    T = words(tgt_sec, Y_src + 1, D)
    tindex = {m: a for a, m in enumerate(T)}
    degs = sorted({nferm(m) for m in T})
    # Q_0 of source words over the target words (formal; missing words -> extra columns evaluated separately)
    extra = {}
    rows = []
    for m in S:
        q = Q_expr({m: 1})
        row = {}
        for m2, c in q.items():
            if m2 not in tindex:
                extra.setdefault(m2, len(T) + len(extra))
            k = tindex.get(m2, extra.get(m2))
            row[k] = row.get(k, 0) + c
        rows.append(row)
    vec_exprs = [v_expr] + list(controls)
    for e in vec_exprs:
        for m2 in e:
            if m2 not in tindex:
                extra.setdefault(m2, len(T) + len(extra))
    allw = T + sorted(extra, key=extra.get)
    letters = sorted({L for m in allw for w in m for L in w})
    out = []
    for seed in seeds:
        res = []
        for factor in (1, 2):
            need = factor * int(1.5 * len(T)) + 64
            k = 10
            npts = max(1, math.ceil(need / math.comb(k, degs[0])))
            fr = Frame(N, npts, k, degs, letters, seed=seed, group=group)
            p = fr.p
            M = fr.eval_words(allw, dtype=np.int64) % p
            bT = RREFBasis(M.shape[1]); bT.add_batched(M[:len(T)] % p); rT = bT.rank
            B = np.zeros((len(rows), M.shape[1]), dtype=np.int64)
            for r_, row in enumerate(rows):
                acc = np.zeros(M.shape[1], dtype=np.int64)
                for kk, c in row.items():
                    acc = (acc + (c % p) * M[kk]) % p
                B[r_] = acc
            bB = RREFBasis(M.shape[1]); bB.add_batched(B % p); rB = bB.rank
            ranks = []
            for e in vec_exprs:
                vv = np.zeros(M.shape[1], dtype=np.int64)
                for m2, c in e.items():
                    kk = tindex.get(m2, extra.get(m2))
                    vv = (vv + (c % p) * M[kk]) % p
                import copy
                b2 = copy.deepcopy(bB)
                inc = int(b2.add_batched(vv[None, :] % p).sum())
                ranks.append(inc)
            res.append((M.shape[1], rT, rB, ranks))
            log(f"  seed {seed}, {M.shape[1]} functionals: rank(target words) {rT}, rank(Q_0 source) {rB}, "
                f"rank increase by [Q_1 O, controls...] {ranks}")
        out.append(res)
    return len(S), len(T), out


if __name__ == "__main__":
    # control: Q_1 Q_0 u is Q_0-exact for any u (here BMN words of (2,2;6,6,6), Y=7, D=0: Q_0 u is in the sector of O_18)
    from sector_words import words as W
    import random
    U = W((2, 2, 6, 6, 6), 7, 0)
    rng = random.Random(3)
    controls = []
    for _ in range(2):
        u = {m: rng.randint(1, 50) for m in rng.sample(U, 5)}
        controls.append(Q1_expr(Q_expr(u), 7, 'SO_odd'))
    QQ = Q_expr(Q1O)
    print("Q_0 Q_1 O_18 = 0 on so(7):", (not QQ) or not values(QQ, 3, 'SO_odd').any(), "(formally zero)" if not QQ else "")
    nS, nT, out = exactness_test(Q1O, controls, 3, 'SO_odd', (1, 1, 6, 6, 6), (0, 0, 6, 6, 6), 7, 2)
    print(f"source words {nS}, target words {nT}")
    lifted = all(r[3][0] == 1 for res in out for r in res)
    ctrl_exact = all(all(x == 0 for x in r[3][1:]) for res in out for r in res)
    faithful = all(res[0][1] == res[1][1] and res[0][2] == res[1][2] for res in out)
    print(f"VERDICT: Q_1 O_18 not Q_0-exact (O_18 lifted): {lifted}; controls exact: {ctrl_exact}; "
          f"ranks stable under point doubling: {faithful}")
