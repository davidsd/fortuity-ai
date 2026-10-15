import os, sys
sys.path.insert(0, os.environ['REPO'] + '/code')
# (a) Q_+^m annihilates w_2 and its two SU(3)_R partners, so w_2^k Q_+^S O_0 = Q_+^S (w_2^k O_0).
# (b) {Q_0, Q_+^m} = 0 on gauge-invariant words containing derivative letters and gauginos: L1 applies Q_+ to the
#     Q_0-preimage L of Q_1 v, which sits at d_BMN = 1, outside the BMN letters of d_L1_qplus.py.
import random
from collections import defaultdict
from fortuity.full.words import Q_expr, add_term, canon_word, word_parity
from fortuity.full.letters import is_ferm, theta_sign, letters_up_to


def qplus(expr, m):  # same Q_+ as d_L1_qplus.py
    out = defaultdict(int)
    for multi, c in expr.items():
        par = 0
        for i, w in enumerate(multi):
            nf = 0
            for pos, (a, b, S) in enumerate(w):
                if m not in S:
                    s0, cw = canon_word(w[:pos] + ((a, b, tuple(sorted(S + (m,)))),) + w[pos + 1:])
                    if s0:
                        add_term(out, multi[:i] + (cw,) + multi[i + 1:], c * s0 * theta_sign(S, (m,)) * (-1) ** (par + nf))
                nf += is_ferm((a, b, S))
            par += word_parity(w)
    return {k: v for k, v in out.items() if v}


def plus(A, B):
    t = defaultdict(int)
    for e in (A, B):
        for k, v in e.items():
            t[k] += v
    return {k: v for k, v in t.items() if v}


f = (0, 0, (0, 1, 2))
phi = [(0, 0, (i,)) for i in range(3)]
psi = {1: (0, 0, (1, 2)), 2: (0, 0, (0, 2)), 3: (0, 0, (0, 1))}
for name, (ph, pa, pb) in {'w(1)': (phi[0], psi[2], psi[3]), 'w(2)': (phi[1], psi[1], psi[3]),
                            'w(3)': (phi[2], psi[1], psi[2])}.items():
    ok = None
    for c in (1, -1):
        w = defaultdict(int); add_term(w, ((f, ph),), 1); add_term(w, ((pa, pb),), c)
        if not Q_expr(w):
            ok = (c, w)
    assert ok is not None, name
    c, w = ok
    print(name, 'closed with relative sign', c, '; Q_+^m w =', [qplus(w, m) for m in range(3)])

random.seed(7)
L = letters_up_to(1, 1)
bad = n = nd = 0
for _ in range(400):
    lets = [random.choice(L) for _ in range(random.randint(2, 6))]
    cut = random.randint(2, len(lets) - 2) if len(lets) >= 4 and random.random() < 0.5 else len(lets)
    e = defaultdict(int); add_term(e, tuple(p for p in (tuple(lets[:cut]), tuple(lets[cut:])) if p), 1)
    if not e:
        continue
    has_der = any(a + b for (a, b, S) in lets)
    for m in range(3):
        r = plus(Q_expr(qplus(e, m)), qplus(Q_expr(e), m))
        bad += bool(r); n += 1; nd += has_der
print(f'{{Q_0,Q_+}} != 0 on {bad} of {n} (word, m) cases; {nd} of them with derivative letters or gauginos')
