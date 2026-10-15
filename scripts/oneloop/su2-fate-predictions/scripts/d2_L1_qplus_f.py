import os, sys, random
from collections import defaultdict
sys.path.insert(0, os.environ['REPO'] + '/code')
from fortuity.full.words import Q_expr, add_term, canon_word, word_parity
from fortuity.full.letters import is_ferm, theta_sign
from fortuity.full.q1 import Q1_expr
# L1 on f-heavy words: {Q_+^m, Q_0} = 0 and {Q_+^m, Q_1} = 0 on multi-trace BMN words containing one or two f's
# (exercises the pair rules with f, including (f, f)). Same Q_+ as d_L1_qplus.py.
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
F = (0, 0, (0, 1, 2))
BMN = [(0, 0, (i,)) for i in range(3)] + [(0, 0, (1, 2)), (0, 0, (0, 2)), (0, 0, (0, 1)), F]
random.seed(2); bad0 = bad1 = n0 = n1 = words = two_f = 0
while words < 400:
    lets = [F] * (2 if random.random() < 0.775 else 1) + [random.choice(BMN) for _ in range(random.randint(1, 5))]
    random.shuffle(lets)
    ntr = random.randint(1, min(3, len(lets) // 2))          # up to three traces, each of length >= 2
    cuts = sorted(random.sample(range(2, len(lets) - 1), ntr - 1)) if ntr > 1 else []
    bounds = [0] + cuts + [len(lets)]
    traces = [tuple(lets[bounds[i]:bounds[i + 1]]) for i in range(ntr)]
    if any(len(t) < 2 for t in traces):
        continue
    e = defaultdict(int); add_term(e, tuple(traces), 1)
    e = {k: v for k, v in e.items() if v}
    if not e:
        continue
    words += 1; two_f += lets.count(F) >= 2
    for m in range(3):
        bad0 += bool(plus(Q_expr(qplus(e, m)), qplus(Q_expr(e), m))); n0 += 1
        for N in (2, 3):
            bad1 += bool(plus(Q1_expr(qplus(e, m), N), qplus(Q1_expr(e, N), m))); n1 += 1
print(f'{words} words ({two_f} with two or more f): {{Q_0,Q_+}} != 0 in {bad0} of {n0} cases; '
      f'{{Q_1,Q_+}} != 0 in {bad1} of {n1} cases')
