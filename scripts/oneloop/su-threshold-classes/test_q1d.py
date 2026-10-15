"""Validation of q1d.Q1d_multi (Q_1 on derivative letters)."""
import random, sys
from collections import defaultdict
from itertools import combinations
import q1d
from fortuity.full.q1 import Q1_multi
from fortuity.full.words import canon_word, canon_multi, Q_multi
from fortuity.full.letters import admissible
from fortuity.bmn.modp import P


def app(op, v):
    out = defaultdict(int)
    for w, c in v.items():
        for m2, c2 in op(w).items():
            out[m2] = (out[m2] + c * c2) % P
    return {m: c for m, c in out.items() if c}


def deriv(al):
    """Translation d_al on letters (a, b, S) -> (a+1) (a+1, b, S) [al = 0], an even derivation."""
    def op(multi):
        out = defaultdict(int)
        for i, w in enumerate(multi):
            for pos, (a, b, S) in enumerate(w):
                nl, c = ((a + 1, b, S), a + 1) if al == 0 else ((a, b + 1, S), b + 1)
                s, cw = canon_word(w[:pos] + (nl,) + w[pos + 1:])
                if s:
                    sm, cm = canon_multi(multi[:i] + (cw,) + multi[i + 1:])
                    if sm:
                        out[cm] += c * s * sm
        return {m: c for m, c in out.items() if c}
    return op


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


def check(name, lhs_op, rhs_op, words, sign):
    bad = 0
    for w in words:
        l, r = app(lhs_op, {w: 1}), app(rhs_op, {w: 1})
        bad += any((l.get(k, 0) + sign * r.get(k, 0)) % P for k in set(l) | set(r))
    print(f"{name}: fails on {bad} of {len(words)}", flush=True)
    return bad


rng = random.Random(7)
Ss = [()] + [tuple(c) for r in (1, 2, 3) for c in combinations((0, 1, 2), r)]
bmn = [(0, 0, S) for S in Ss if admissible(0, 0, S)]
full = [(a, b, S) for a in range(3) for b in range(3) for S in Ss if admissible(a, b, S)]
fails = 0
for N in (2, 3):
    Q1 = lambda m, N=N: q1d.Q1d_multi(m, N)
    # 1. agreement with fortuity.full.q1 on derivative-free words
    ws = random_words(rng, bmn, 40)
    bad = sum(1 for w in ws if app(Q1, {w: 1}) != {m: c % P for m, c in Q1_multi(w, N).items() if c % P})
    print(f"N={N} agreement with Q1_multi on 40 BMN words: {bad} differ", flush=True)
    fails += bad
    ws = random_words(rng, full, 60)
    # 2. {Q_0, Q_1} = 0 (formal identity on canonical multitraces)
    fails += check(f"N={N} {{Q_0, Q_1}} = 0", lambda m: app(Q_multi, app(Q1, {m: 1})), lambda m: app(Q1, app(Q_multi, {m: 1})), ws, +1)
    # 3. [d_al, Q_1] = 0
    for al in (0, 1):
        d = deriv(al)
        fails += check(f"N={N} [d_{al+1}, Q_1] = 0", lambda m, d=d: app(d, app(Q1, {m: 1})), lambda m, d=d: app(Q1, app(d, {m: 1})), ws, -1)
    # 4. [D (SU(2)_R), Q_1] = 0
    sys.argv = ['x']
    import build_cand as bc
    fails += check(f"N={N} [D, Q_1] = 0", lambda m: app(bc.D_multi, app(Q1, {m: 1})), lambda m: app(Q1, app(bc.D_multi, {m: 1})), ws, -1)
print('TOTAL FAILURES', fails)
