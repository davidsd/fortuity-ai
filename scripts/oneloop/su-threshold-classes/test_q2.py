"""Checks of q2bmn.U_multi (two-loop supercharge on derivative-free letters, up to the constant K_2).

1. Graded symmetry of the vertex under exchange of the two outer letters (adding the exchanged assignment doubles U).
2. Nilpotency at order g^4:  Q_1^2 w + K_2 {Q_0, U} w = 0 on random derivative-free multitraces, N = 2, 3, 4, with ONE
   constant K_2 for all words and all N.  Q_0 preserves derivative-free words, so only U on such words enters.
   Negative control: the same fit with a wrong relative weight in D2 (3 instead of -2 for the (1,1)(1,1) term).
3. Symmetries on derivative-free words: U commutes with SU(3)_R (E_i, F_i), with charge conjugation C, and
   anticommutes with the odd generators that keep words derivative-free (d/dtheta_a and S_a), as Q_1 does.
usage: python3 test_q2.py [nwords]"""
import random, sys
from collections import defaultdict
from itertools import combinations
NW = int(sys.argv[1]) if len(sys.argv) > 1 else 40
sys.argv = ['x']
import q1d
import q2bmn
import build_cand as bc
import gravitons_full as G
import c_parity as cp
from fortuity.full.q1 import Q1_multi
from fortuity.full.words import Q_multi, canon_word, canon_multi
from fortuity.full.hw import E_multi
from fortuity.full.letters import admissible
from fortuity.bmn.modp import P


def app(op, v):
    out = defaultdict(int)
    for w, c in v.items():
        for m2, c2 in op(w).items():
            out[m2] = (out[m2] + c * c2) % P
    return {m: c for m, c in out.items() if c}


def random_words(rng, letters, n, lens=(2, 3, 4), ntr=(1, 2)):
    out = []
    while len(out) < n:
        ws = [tuple(rng.choice(letters) for _ in range(rng.choice(lens))) for _ in range(rng.choice(ntr))]
        if not all(canon_word(w)[0] for w in ws):
            continue
        s, cm = canon_multi([canon_word(w)[1] for w in ws])
        if s:
            out.append(cm)
    return out


def lincomb(*terms):
    out = defaultdict(int)
    for c, v in terms:
        for m, x in v.items():
            out[m] = (out[m] + c * x) % P
    return {m: x for m, x in out.items() if x}


def ratrec(x):
    """small rational a/b = x mod P"""
    for b in range(1, 2000):
        a = x * b % P
        if a > P // 2:
            a -= P
        if abs(a) < 2000:
            return f"{a}/{b}" if b > 1 else str(a)
    return f"{x} (mod p)"


def fit(words, N, U):
    """K_2 with Q_1^2 w + K_2 {Q_0, U} w = 0: returns (set of fitted values, #words with Q_1^2 != 0, #violations)"""
    Q1 = lambda m: q1d.Q1d_multi(m, N)
    ks, nz, bad = set(), 0, 0
    for w in words:
        a = app(Q1, app(Q1, {w: 1}))
        b = lincomb((1, app(Q_multi, app(U, {w: 1}))), (1, app(U, app(Q_multi, {w: 1}))))
        nz += bool(a)
        if not a and not b:
            continue
        if not b or not a:
            bad += 1
            continue
        m0 = next(iter(b))
        k = (-a.get(m0, 0)) * pow(b[m0], P - 2, P) % P
        if lincomb((1, a), (k, b)):
            bad += 1
        else:
            ks.add(k)
    return ks, nz, bad


if __name__ == '__main__':
    nw = NW
    rng = random.Random(3)
    Ss = [tuple(c) for r in (1, 2, 3) for c in combinations((0, 1, 2), r)]
    bmn = [(0, 0, S) for S in Ss]
    words = random_words(rng, bmn, nw, lens=(2, 3, 4, 5), ntr=(1, 2))
    print(f"{nw} random derivative-free multitraces (1-2 traces of 2-5 letters)", flush=True)
    bad = sum(1 for w in words for N in (2, 3)
              if lincomb((2, q2bmn.U_multi(w, N)), (-1, q2bmn.U_multi(w, N, both_orders=True))))
    print(f"1. exchange symmetry of the outer letters: {bad} violations (N = 2, 3)", flush=True)
    allk = set()
    for N in (2, 3, 4):
        ks, nz, bad = fit(words, N, lambda m: q2bmn.U_multi(m, N))
        allk |= ks
        print(f"2. N={N}: Q_1^2 != 0 on {nz} words; violations of Q_1^2 + K_2 {{Q_0,U}} = 0: {bad}; "
              f"fitted K_2: {sorted(ratrec(k) for k in ks)}", flush=True)
    print(f"   distinct K_2 over all N: {sorted(ratrec(k) for k in allk)}", flush=True)
    saved = q2bmn.DERIV
    q2bmn.DERIV = ((4, (2, 0), (0, 2)), (3, (1, 1), (1, 1)), (4, (0, 2), (2, 0)))
    ks, nz, bad = fit(words, 3, lambda m: q2bmn.U_multi(m, 3))
    print(f"   negative control (wrong D2 weight), N=3: violations {bad}, distinct fitted constants {len(ks)}", flush=True)
    q2bmn.DERIV = saved
    # 3. symmetries
    U = lambda m: q2bmn.U_multi(m, 3)
    Q1 = lambda m: {k: v % P for k, v in Q1_multi(m, 3).items() if v % P}
    Cop = lambda m: (lambda sc: {sc[1]: sc[0] % P} if sc[0] else {})(cp.C_multi(m))
    even = [('E_1', lambda m: E_multi(m, 1)), ('E_2', lambda m: E_multi(m, 2)), ('F_1', bc.F1_multi), ('F_2', bc.F2_multi),
            ('C', Cop)]
    odd = ([(f'd_theta{a}', G._Qtheta(a)) for a in range(3)] + [(f'S_{a}', bc._odd_derivation(bc._S_letter(a))) for a in range(3)])
    print("3. graded commutators on the same words, N=3 (violations for U | for Q_1 | words where J w != 0):", flush=True)
    for sign, gens in ((-1, even), (+1, odd)):
        for name, J in gens:
            res = []
            for op in (U, Q1):
                bad = 0
                for w in words:
                    l, r = app(op, app(J, {w: 1})), app(J, app(op, {w: 1}))
                    bad += bool(lincomb((1, l), (sign, r)))
                res.append(bad)
            nzj = sum(1 for w in words if app(J, {w: 1}))
            print(f"   {name:9s}: {res[0]:3d} | {res[1]:3d} | {nzj}", flush=True)
