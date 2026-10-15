"""T-019 (SO/Sp part): the one-loop lifting test of a Q_0-cohomology class O of SO(2N+1), including gravitons.
O is lifted at one loop iff Q_1 O is NOT in  span( Q_0(all words of the source sector) , Q_1 g for every graviton
polynomial g in O's sector ).  (Adding exact terms or gravitons to O changes Q_1 O by elements of this span; the SU(3)_R
rep is not used, so the test is conservative: a larger span can only make 'lifted' harder to establish.)
Faithfulness: the target-word rank and all ranks must be stable under doubling the evaluation points; two seeds."""
import math, copy
import numpy as np
from fortuity.bmn.modp import RREFBasis
from fortuity.full.words import Q_expr
from fortuity.full.q1 import Q1_expr
from fortuity.full.evalg import Frame, nferm
from sector_words import words


def lift_test(O, tgt_sec, Y, D, N, group='SO_odd', gravs=(), controls=(), seeds=(11, 12), log=print):
    n = 2 * N + 1
    src_sec = (tgt_sec[0] + 1, tgt_sec[1] + 1) + tuple(tgt_sec[2:])
    S = words(src_sec, Y - 1, D)
    T = words(tgt_sec, Y, D)
    span_exprs = [Q_expr({m: 1}) for m in S] + [Q1_expr(g, n, group) for g in gravs]
    test_exprs = [Q1_expr(O, n, group)] + list(controls)
    allw = {m: a for a, m in enumerate(T)}
    for e in span_exprs + test_exprs:
        for m in e:
            allw.setdefault(m, len(allw))
    wl = sorted(allw, key=allw.get)
    letters = sorted({L for m in wl for w in m for L in w})
    degs = sorted({nferm(m) for m in T})
    results = []
    for seed in seeds:
        per = []
        for factor in (1, 2):
            k = 10
            npts = math.ceil((factor * int(1.5 * len(T)) + 64) / math.comb(k, degs[0]))
            fr = Frame(N, npts, k, degs, letters, seed=seed, group=group)
            p = fr.p
            M = fr.eval_words(wl, dtype=np.int64) % p
            bT = RREFBasis(M.shape[1]); bT.add_batched(M[:len(T)]); rT = bT.rank

            def vec(e):
                v = np.zeros(M.shape[1], dtype=np.int64)
                for m, c in e.items():
                    v = (v + (c % p) * M[allw[m]]) % p
                return v
            nB = len(S)
            bE = RREFBasis(M.shape[1]); bE.add_batched(np.array([vec(e) for e in span_exprs[:nB]]) % p); rE = bE.rank
            bS = copy.deepcopy(bE)
            if len(span_exprs) > nB:
                bS.add_batched(np.array([vec(e) for e in span_exprs[nB:]]) % p)
            rS = bS.rank
            incs = []
            for e in test_exprs:
                b2 = copy.deepcopy(bS)
                incs.append(int(b2.add_batched(vec(e)[None, :] % p).sum()))
            per.append((M.shape[1], rT, rE, rS, incs))
            log(f"  seed {seed}, {M.shape[1]} functionals: rank target words {rT}, exact {rE}, "
                f"exact + Q_1(gravitons) {rS}, rank increase by [Q_1 O, controls] {incs}")
        results.append(per)
    lifted = all(r[4][0] == 1 for per in results for r in per)
    ctrl = all(all(x == 0 for x in r[4][1:]) for per in results for r in per)
    stable = all(per[0][1:4] == per[1][1:4] for per in results)
    return dict(source_words=len(S), target_words=len(T), lifted=lifted, controls_exact=ctrl, stable=stable,
                ranks=[per[0][1:4] for per in results])


def lift_test_span(test_exprs, span_extra, tgt_sec, Y, D, N, group='SO_odd', seeds=(11, 12), log=print):
    """Rank test in group(N)'s sector tgt_sec (Y letters, D derivatives): is each of test_exprs in
    span(Q_0(all source words), span_extra)?  Reports the rank increase of each test expression (1 = outside)."""
    src_sec = (tgt_sec[0] + 1, tgt_sec[1] + 1) + tuple(tgt_sec[2:])
    S = words(src_sec, Y - 1, D)
    T = words(tgt_sec, Y, D)
    span_exprs = [Q_expr({m: 1}) for m in S] + list(span_extra)
    allw = {m: a for a, m in enumerate(T)}
    for e in span_exprs + list(test_exprs):
        for m in e:
            allw.setdefault(m, len(allw))
    wl = sorted(allw, key=allw.get)
    letters = sorted({L for m in wl for w in m for L in w})
    degs = sorted({nferm(m) for m in T})
    results = []
    for seed in seeds:
        per = []
        for factor in (1, 2):
            k = 10
            npts = math.ceil((factor * int(1.5 * len(T)) + 64) / math.comb(k, degs[0]))
            fr = Frame(N, npts, k, degs, letters, seed=seed, group=group)
            p = fr.p
            M = fr.eval_words(wl, dtype=np.int64) % p
            bT = RREFBasis(M.shape[1]); bT.add_batched(M[:len(T)]); rT = bT.rank

            def vec(e):
                v = np.zeros(M.shape[1], dtype=np.int64)
                for m, c in e.items():
                    v = (v + (c % p) * M[allw[m]]) % p
                return v
            nB = len(S)
            bE = RREFBasis(M.shape[1]); bE.add_batched(np.array([vec(e) for e in span_exprs[:nB]]) % p); rE = bE.rank
            bS = copy.deepcopy(bE)
            if len(span_exprs) > nB:
                bS.add_batched(np.array([vec(e) for e in span_exprs[nB:]]) % p)
            rS = bS.rank
            incs = []
            for e in test_exprs:
                b2 = copy.deepcopy(bS)
                incs.append(int(b2.add_batched(vec(e)[None, :] % p).sum()))
            per.append((M.shape[1], rT, rE, rS, incs))
            log(f"  seed {seed}, {M.shape[1]} functionals: rank target words {rT}, exact {rE}, exact + extra {rS}, "
                f"rank increase by tests {incs}")
        results.append(per)
    stable = all(per[0][1:4] == per[1][1:4] and per[0][4] == per[1][4] for per in results)
    same = all(results[0][0][4] == per[0][4] for per in results)
    return dict(source_words=len(S), target_words=len(T), increases=results[0][0][4], stable=stable, seeds_agree=same,
                ranks=[per[0][1:4] for per in results])


if __name__ == "__main__":
    import random
    from so7_lift import O18, values
    from gravitons_bmn import graviton_exprs
    O = O18()
    G = [e for _, e in graviton_exprs((1, 3, 3, 3))]
    closed = sum(1 for g in G if (lambda q: (not q) or not values(q, 3, 'SO_odd').any())(Q_expr(g)))
    print(f"{len(G)} graviton polynomials in (1;3,3,3); Q_0-closed on so(7): {closed}/{len(G)}")
    U = words((2, 2, 6, 6, 6), 7, 0)
    rng = random.Random(3)
    ctrls = [Q1_expr(Q_expr({m: rng.randint(1, 50) for m in rng.sample(U, 5)}), 7, 'SO_odd') for _ in range(2)]
    res = lift_test(O, (0, 0, 6, 6, 6), 8, 2, 3, gravs=G, controls=ctrls)
    print("SO(7) O_18:", res)
