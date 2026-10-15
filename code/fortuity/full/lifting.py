"""One-loop lifting tests of SO(2N+1) BMN fortuitous classes (T-019), promoted from scripts/oneloop/so-odd-classes
(so7_lift, lift_test, gravitons_bmn, sector_words, so9_block_lift, so11_block_lift) on 2026-10-04.

A Q_0-class O is lifted at one loop iff Q_1 O is not in span(Q_0(words one step up), Q_1 g for every graviton polynomial g
of O's sector). For SO(9), SO(11) the test is the one-way block test: the coefficient of x_1^2 (one SO(2) block), resp.
x_1^2 y_1^2 or x_1^2 y_2^2 (two blocks), is a Q_0 chain map into SO(7), so a non-exact image proves lifting.
Q_1 is fortuity.full.q1 with group='SO_odd' (T-039(a), tests code/tests/test_q1_groups.py).
CLI (from code/):  python -m fortuity.full.lifting so7 | so9 | so11     (~0.5 min, ~2 min, ~35 min on a laptop core)
"""
import json, math, copy, random, time
import numpy as np
from collections import defaultdict
from fortuity.bmn.modp import RREFBasis, P
from fortuity.bmn.words import PHI as BPHI, PSI as BPSI, F as BF, graviton_generators, mul_expr, PHI, PSI, F
from fortuity.bmn.g2_checks import SO7_CLASS_TERMS
from .words import add_term, Q_expr, sector_of, canon_multi, contents_of_sector, multitraces_with_letters
from .q1 import Q1_expr
from .evalg import Frame, nferm

TR11 = ((0, 0, (0,)), (0, 0, (0,)))


def words(sec, Y, D, fast=True):
    """fast=True: FKM necklaces with fixed content (wordenum_full, same canonical forms); fast=False: the permutation
    enumeration of words_full (slow beyond ~8 letters)."""
    from .wordenum import multitraces_with_content
    out = []
    for c in contents_of_sector(sec, Y):
        if len(c) == Y and sum(L[0] + L[1] for L in c) == D:
            out += multitraces_with_content(c, Y) if fast else multitraces_with_letters(c, Y)
    return out


CONV = {BPHI[0]: (1, (0, 0, (0,))), BPHI[1]: (1, (0, 0, (1,))), BPHI[2]: (1, (0, 0, (2,))),
        BPSI[0]: (1, (0, 0, (1, 2))), BPSI[1]: (-1, (0, 0, (0, 2))), BPSI[2]: (1, (0, 0, (0, 1))),
        BF: (-1, (0, 0, (0, 1, 2)))}


def O18():
    O = defaultdict(int)
    for c, traces in SO7_CLASS_TERMS:
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


CH = {PHI[0]: (0, (1, 0, 0)), PHI[1]: (0, (0, 1, 0)), PHI[2]: (0, (0, 0, 1)),
      PSI[0]: (1, (0, 1, 1)), PSI[1]: (1, (1, 0, 1)), PSI[2]: (1, (1, 1, 0)), F: (2, (1, 1, 1))}


def sector(expr):
    m = next(iter(expr))
    tj, q = 0, [0, 0, 0]
    for w in m:
        for l in w:
            a, b = CH[l]
            tj += a
            q = [x + y for x, y in zip(q, b)]
    return (tj,) + tuple(q)


def monomials(sec, nmax=8):
    gens = [(name, e) for name, e in graviton_generators(nmax, 1, qmax=sec) if e]
    gsec = [sector(e) for _, e in gens]
    gpar = [s[0] % 2 for s in gsec]          # fermionic iff 2J odd
    out = []

    def rec(start, rem, chosen):
        if not any(rem):
            out.append(tuple(chosen)); return
        for i in range(start, len(gens)):
            if all(a <= b for a, b in zip(gsec[i], rem)):
                rec(i + 1 if gpar[i] else i, tuple(b - a for a, b in zip(gsec[i], rem)), chosen + [i])
    rec(0, tuple(sec), [])
    return gens, out


def to_full(expr):
    O = defaultdict(int)
    for m, c in expr.items():
        sign, multi = c, []
        for w in m:
            ww = []
            for l in w:
                s, L = CONV[l]
                sign *= s
                ww.append(L)
            multi.append(tuple(ww))
        add_term(O, tuple(multi), sign)
    return {m: c for m, c in O.items() if c}


def graviton_exprs(sec, nmax=8):
    gens, monos = monomials(sec, nmax)
    for mono in monos:
        e = gens[mono[0]][1]
        for gi in mono[1:]:
            e = mul_expr(e, gens[gi][1])
        e = {m: c for m, c in e.items() if c}
        if e:
            yield mono, to_full(e)


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




def load_rep(path):
    d = json.load(open(path))
    e = defaultdict(int)
    for c, traces in d['terms']:
        e[tuple(tuple(t) for t in traces)] = (e[tuple(tuple(t) for t in traces)] + c) % P
    return to_full({m: c for m, c in e.items() if c}), d['info']


def rho2(expr):
    out = defaultdict(int)
    for m, c in expr.items():
        for a, w in enumerate(m):
            if w == TR11:
                rest = m[:a] + m[a + 1:]
                s, cm = canon_multi(list(rest))
                if s:
                    out[cm] = (out[cm] - 2 * s * c) % P
    return {m: c for m, c in out.items() if c}


TR22 = ((0, 0, (1,)), (0, 0, (1,)))


def rho_pair(expr, wx, wy):
    out = defaultdict(int)
    for m, c in expr.items():
        for a, w in enumerate(m):
            if w != wx:
                continue
            for b, w2 in enumerate(m):
                if b == a or w2 != wy:
                    continue
                rest = tuple(x for k, x in enumerate(m) if k not in (a, b))
                s, cm = canon_multi(list(rest))
                if s:
                    out[cm] = (out[cm] + 4 * s * c) % P
    return {m: c for m, c in out.items() if c}


def run(path, sec_bmn, wx, wy, label, src_bmn_ctrl):
    t0 = time.time()
    O, info = load_rep(path)
    print(f"\n== {label}: {len(O)} terms, sector {set(sector_of(m) for m in O)}, info {info}", flush=True)
    q0 = Q_expr(O)
    print("Q_0 O = 0 on so(11):", (not q0) or not values(q0, 5, 'SO_odd').any(),
          "| on so(13):", (not q0) or not values(q0, 6, 'SO_odd').any(), flush=True)
    Q1O = Q1_expr(O, 11, 'SO_odd')
    r = rho_pair(Q1O, wx, wy)
    print(f"Q_1 O: {len(Q1O)} terms; rho(Q_1 O): {len(r)} terms, sectors {set(sector_of(m) for m in r)}, "
          f"nonzero on so(7): {bool(r) and values(r, 3, 'SO_odd').any()}  ({time.time() - t0:.0f} s)", flush=True)
    span_extra, ng = [], 0
    for _, g in graviton_exprs(sec_bmn, 10):
        ng += 1
        e = rho_pair(Q1_expr(g, 11, 'SO_odd'), wx, wy)
        if e:
            span_extra.append(e)
    print(f"{ng} SO(11) graviton polynomials; nonzero rho(Q_1 g): {len(span_extra)}  ({time.time() - t0:.0f} s)", flush=True)
    U = words(src_bmn_ctrl, 11, 0)
    rng = random.Random(7)
    ctrls = [rho_pair(Q1_expr(Q_expr({m: rng.randint(1, 50) for m in rng.sample(U, 8)}), 11, 'SO_odd'), wx, wy)
             for _ in range(3)]
    ctrls = [c for c in ctrls if c]
    print(f"controls with nonzero rho: {len(ctrls)}", flush=True)
    res = lift_test_span([r] + ctrls, span_extra, (0, 0, 6, 6, 6), 8, 2, 3)
    print(f"{label} two-block test:", res, flush=True)
    Q1O18 = Q1_expr(O18(), 7, 'SO_odd')
    res2 = lift_test_span([r], [Q1O18], (0, 0, 6, 6, 6), 8, 2, 3, seeds=(13,))
    print(f"{label}: rho(Q_1 O) in span(exact, Q_1 O_18): {res2['increases'][0] == 0}  ({time.time() - t0:.0f} s)", flush=True)
    return res, res2



import os as _os
_DATA = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', 'data', 'so_sp')


def so7_main():
    """SO(7): O_18 (R-005) in (0,0;6,6,6), Y=8, d_BMN=1, with the 324 graviton polynomials of (1;3,3,3) in the span."""
    O = O18()
    G = [e for _, e in graviton_exprs((1, 3, 3, 3))]
    U = words((2, 2, 6, 6, 6), 7, 0)
    rng = random.Random(3)
    ctrls = [Q1_expr(Q_expr({m: rng.randint(1, 50) for m in rng.sample(U, 5)}), 7, 'SO_odd') for _ in range(2)]
    res = lift_test(O, (0, 0, 6, 6, 6), 8, 2, 3, gravs=G, controls=ctrls)
    print(f"{len(G)} graviton polynomials in (1;3,3,3)")
    print("SO(7) O_18:", res)
    return res


def so9_main():
    """SO(9): the R-006 class by the one-way block test (x_1^2 coefficient), gravitons of (1;5,3,3) in the span."""
    O9, info = load_rep(_os.path.join(_DATA, 'so9_nongraviton_rep_1533.json'))
    r = rho2(Q1_expr(O9, 9, 'SO_odd'))
    G9 = [e for _, e in graviton_exprs((1, 5, 3, 3))]
    span_extra = [e for e in (rho2(Q1_expr(g, 9, 'SO_odd')) for g in G9) if e]
    U = words((2, 2, 10, 6, 6), 9, 0)
    rng = random.Random(5)
    ctrls = [c for c in (rho2(Q1_expr(Q_expr({m: rng.randint(1, 50) for m in rng.sample(U, 8)}), 9, 'SO_odd'))
                         for _ in range(2)) if c]
    print(f"rho_2(Q_1 O_9): {len(r)} terms; {len(G9)} SO(9) graviton polynomials; controls {len(ctrls)}")
    res = lift_test_span([r] + ctrls, span_extra, (0, 0, 6, 6, 6), 8, 2, 3)
    print("SO(9) block test:", res)
    Q1O18 = Q1_expr(O18(), 7, 'SO_odd')
    res2 = lift_test_span([r], [Q1O18], (0, 0, 6, 6, 6), 8, 2, 3, seeds=(13,))
    print("rho_2(Q_1 O_9) in span(exact, Q_1 O_18):", res2['increases'][0] == 0)
    return res, res2


def so11_main():
    """SO(11): both R-011 classes by the two-block test."""
    out = []
    out.append(run(_os.path.join(_DATA, 'so11_nongraviton_rep_1733.json'), (1, 7, 3, 3), TR11, TR11,
                   "SO(11) [4,0] (1;7,3,3)", (2, 2, 14, 6, 6)))
    out.append(run(_os.path.join(_DATA, 'so11_nongraviton_rep_1553.json'), (1, 5, 5, 3), TR11, TR22,
                   "SO(11) [0,2] (1;5,5,3)", (2, 2, 10, 10, 6)))
    return out


if __name__ == "__main__":
    import sys
    {'so7': so7_main, 'so9': so9_main, 'so11': so11_main}[sys.argv[1] if len(sys.argv) > 1 else 'so7']()
