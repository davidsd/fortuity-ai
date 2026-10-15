"""Does block restriction commute with the one-loop supercharge in cohomology?  Test on the SO(9) fused class.

rho_2 = coefficient of x_1^2 under SO(9) > SO(7) x SO(2) (R-016).  rho_2 is a Q_0 chain map but not a Q_1 chain map
(W-boson exchange).  The order-1 defect on a class v is
    Delta(v) = rho_2(Q_1^{SO(9)} v) - Q_1^{SO(7)} rho_2(v),
a Q_0-closed element of the SO(7) target sector; [Delta] = rho_* d_1 - d_1' rho_* on E_1.  "Intertwining at order 1"
on v means [Delta(v)] = 0.  For R-007 this is automatic (d_1 = 0 on both sides), so we test it where d_1 != 0:
the SO(9) class O_9 (R-006), whose restriction is a x_1^2 (x) O_18 + gravitons + exact and whose core O_18 lifts
(Choi-Lee; T-019).  Writing rho_2(Q_1 O_9) = c Q_1 O_18 and Q_1 rho_2(O_9) = a Q_1 O_18 modulo
(exact + Q_1 gravitons), intertwining <=> c = a.

Uses the SO one-loop code (scripts/oneloop/so-odd-classes: Q_1 of Budzik-Kulp for SO(2N+1),
full-sector words and evaluation frames, rho_2 on full-sector words).
Run from the repo root:  python3 scripts/oneloop/protection/code/so9_intertwining.py
"""
import sys, os, math, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path[:0] = [os.path.join(ROOT, 'scripts', 'oneloop', 'so-odd-classes'), os.path.join(ROOT, 'code')]
from fortuity.bmn.modp import RREFBasis, P
from fortuity.bmn.classify import ratio
from fortuity.full.words import Q_expr, sector_of
from fortuity.full.q1 import Q1_expr
from fortuity.full.evalg import Frame, nferm
from sector_words import words
from so9_block_lift import load_rep, rho2
from so7_lift import O18, values
from gravitons_bmn import graviton_exprs


def lin(*pairs):
    out = {}
    for c, e in pairs:
        for m, v in e.items():
            out[m] = (out.get(m, 0) + c * v) % P
    return {m: v for m, v in out.items() if v}


def ratrec(a, p=P):
    """rational reconstruction of a mod p (small numerator/denominator), as a string"""
    a %= p
    r0, r1, s0, s1 = p, a, 0, 1
    while r1 * r1 * 2 > p:
        q = r0 // r1
        r0, r1, s0, s1 = r1, r0 - q * r1, s1, s0 - q * s1
    num, den = (r1, s1) if s1 > 0 else (-r1, -s1)
    return f"{num}/{den}" if den != 1 else str(num)


def residues(tests, span_extra, tgt, Y, D, N, seed, factor, group='SO_odd'):
    src = (tgt[0] + 1, tgt[1] + 1) + tuple(tgt[2:])
    S, T = words(src, Y - 1, D), words(tgt, Y, D)
    span = [Q_expr({m: 1}) for m in S]
    allw = {m: a for a, m in enumerate(T)}
    for e in span + list(span_extra) + list(tests):
        for m in e:
            allw.setdefault(m, len(allw))
    wl = sorted(allw, key=allw.get)
    letters = sorted({L for m in wl for w in m for L in w})
    degs = sorted({nferm(m) for m in T})
    k = 10
    npts = math.ceil((factor * int(1.5 * len(T)) + 64) / math.comb(k, degs[0]))
    fr = Frame(N, npts, k, degs, letters, seed=seed, group=group)
    p = fr.p
    M = fr.eval_words(wl, dtype=np.int64) % p

    def vec(e):
        v = np.zeros(M.shape[1], dtype=np.int64)
        for m, c in e.items():
            v = (v + (c % p) * M[allw[m]]) % p
        return v
    bT = RREFBasis(M.shape[1]); bT.add(M[:len(T)])
    bE = RREFBasis(M.shape[1]); bE.add(np.array([vec(e) for e in span]) % p)
    rE = bE.rank
    if span_extra:
        bE.add(np.array([vec(e) for e in span_extra]) % p)
    R = bE.reduce(np.array([vec(e) for e in tests]) % p)
    return dict(functionals=M.shape[1], rank_target=bT.rank, rank_exact=rE, rank_span=bE.rank,
                source_words=len(S), target_words=len(T)), R


if __name__ == '__main__':
    t0 = time.time()
    O9, info = load_rep(os.path.join(ROOT, 'data', 'so_sp', 'so9_nongraviton_rep_1533.json'))
    r0 = rho2(O9)
    print(f"O_9: {len(O9)} terms; rho_2(O_9): {len(r0)} terms in {set(sector_of(m) for m in r0)}")
    q0 = Q_expr(r0)
    print("rho_2(O_9) Q_0-closed on so(7):", (not q0) or not values(q0, 3, 'SO_odd').any())
    r1 = rho2(Q1_expr(O9, 9, 'SO_odd'))
    Q1r0 = Q1_expr(r0, 7, 'SO_odd')
    Q1O18 = Q1_expr(O18(), 7, 'SO_odd')
    Delta = lin((1, r1), (-1, Q1r0))
    print(f"rho_2(Q_1 O_9): {len(r1)} terms; Q_1 rho_2(O_9): {len(Q1r0)} terms; Delta: {len(Delta)} terms "
          f"in {set(sector_of(m) for m in Delta)}  [{time.time() - t0:.0f}s]")
    G7 = [e for _, e in graviton_exprs((1, 3, 3, 3))]
    Q1G = [e for e in (Q1_expr(g, 7, 'SO_odd') for g in G7) if e]
    print(f"{len(G7)} SO(7) graviton polynomials in (1;3,3,3); nonzero Q_1 g: {len(Q1G)}  [{time.time() - t0:.0f}s]")
    tgt, Y, D, N = (0, 0, 6, 6, 6), 8, 2, 3
    names = ['Delta', 'rho2 Q1 O9', 'Q1 rho2 O9', 'Q1 O18']
    tests = [Delta, r1, Q1r0, Q1O18]
    for label, extra in (('exact', []), ('exact + Q_1 gravitons', Q1G)):
        for seed in (11, 12):
            for factor in (1, 2):
                info_, R = residues(tests, extra, tgt, Y, D, N, seed, factor)
                nz = [bool(np.any(r)) for r in R]
                rat = {names[i]: ratio(R[i], R[3]) for i in range(3)}
                rr = {k: (ratrec(v) if v is not None else None) for k, v in rat.items()}
                print(f"[mod {label}] seed {seed} x{factor}: {info_}\n   nonzero residue {dict(zip(names, nz))}\n"
                      f"   ratio to Q_1 O_18 (mod p): {rat}  rational: {rr}  [{time.time() - t0:.0f}s]")
