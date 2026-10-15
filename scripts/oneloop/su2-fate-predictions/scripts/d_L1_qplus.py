import os, sys, random
from collections import defaultdict
sys.path.insert(0, os.environ['REPO'] + '/code')
from fortuity.full.words import Q_expr, add_term, canon_word, word_parity
from fortuity.full.letters import is_ferm, theta_sign
from fortuity.full.q1 import Q1_expr
def qplus(expr, m):                    # theta-shift Psi_S -> sign(theta^S theta^m) Psi_{S u m}, a graded derivation
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
BMN = [(0, 0, (i,)) for i in range(3)] + [(0, 0, (1, 2)), (0, 0, (0, 2)), (0, 0, (0, 1)), (0, 0, (0, 1, 2))]
random.seed(1); bad0 = bad1 = n = 0
for _ in range(300):
    lets = [random.choice(BMN) for _ in range(random.randint(2, 7))]
    cut = random.randint(2, len(lets) - 2) if len(lets) >= 4 and random.random() < 0.5 else len(lets)
    e = defaultdict(int); add_term(e, tuple(p for p in (tuple(lets[:cut]), tuple(lets[cut:])) if p), 1)
    if not e:
        continue
    for m in range(3):
        bad0 += bool(plus(Q_expr(qplus(e, m)), qplus(Q_expr(e), m)))
        for N in (2, 3):
            bad1 += bool(plus(Q1_expr(qplus(e, m), N), qplus(Q1_expr(e, N), m))); n += 1
print(f'{{Q_0,Q_+}} != 0 in {bad0} of {n // 2} cases; {{Q_1,Q_+}} != 0 in {bad1} of {n} cases')
