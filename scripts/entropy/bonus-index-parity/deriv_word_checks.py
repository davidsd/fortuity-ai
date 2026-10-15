"""T-038 derivation, word-level checks of the weight ratios rho_n used in the protection table (n = 0, 1).

On formal multitrace words of the repo (fortuity.full.words, fortuity.full.q1; read-only use):
  (a) (-1)^{F+Y} = (-1)^{|q| - 3 d} on every word (d = sum (a+b)/2);
  (b) Q_0 (Q_multi): every output term has Y+1, the same d, J_R, q, and opposite F;
  (c) Q_1 (Q1_multi, BMN words only, N = 3): every output term has the same Y, d+1, the same J_R, q, opposite F;
  (d) C Q_0 = Q_0 C and C Q_1 = Q_1 C as formal expressions, with C(tr X_1..X_L) = (-1)^L (-1)^{k(k-1)/2} tr(X_L..X_1)
      (k = number of Grassmann-odd letters), the rule of code/fortuity/bmn/cparity.py extended to full letters;
  hence P = C(-1)^Y anticommutes with Q_0 and commutes with Q_1.
"""
import sys, time, random
from fractions import Fraction as Fr
from collections import defaultdict

import os  # repo root: $REPO, or three levels up from scripts/entropy/bonus-index-parity/
sys.path.insert(0, os.path.join(os.environ.get('REPO') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')), 'code'))
from fortuity.full.letters import is_ferm, charges2
from fortuity.full.words import words_of_sector, Q_multi, canon_word, canon_multi, sectors_at_level
from fortuity.full.q1 import Q1_multi

t0 = time.time()
random.seed(1)


def grades(multi):
    Y = sum(len(w) for w in multi)
    F = sum(1 for w in multi for L in w if is_ferm(L)) % 2
    two_d = sum(L[0] + L[1] for w in multi for L in w)            # 2 d_BMN
    two_JR = sum(L[0] - L[1] for w in multi for L in w)           # 2 J_R
    tq = [0, 0, 0]
    for w in multi:
        for L in w:
            c = charges2(L)
            for j in range(3):
                tq[j] += c[2 + j]
    return Y, F, Fr(two_d, 2), Fr(two_JR, 2), tuple(Fr(x, 2) for x in tq)


def conj_word(w):
    k = sum(1 for L in w if is_ferm(L))
    s0 = (-1) ** len(w) * (-1) ** (k * (k - 1) // 2)
    s, cw = canon_word(tuple(reversed(w)))
    if s == 0:
        return 0, None
    return s0 * s, cw


def conj_expr(expr):
    out = defaultdict(int)
    for multi, c in expr.items():
        sign, ws = 1, []
        for w in multi:
            s, cw = conj_word(w)
            if s == 0:
                sign = 0
                break
            sign *= s
            ws.append(cw)
        if not sign:
            continue
        s, m = canon_multi(ws)
        if s:
            out[m] += sign * s * c
    return {m: v for m, v in out.items() if v}


def apply(op, expr):
    out = defaultdict(int)
    for m, c in expr.items():
        for m2, c2 in op(m).items():
            out[m2] += c * c2
    return {m: v for m, v in out.items() if v}


def add(e1, e2, s=1):
    out = defaultdict(int)
    for m, c in e1.items():
        out[m] += c
    for m, c in e2.items():
        out[m] += s * c
    return {m: v for m, v in out.items() if v}


# sectors: a few with gauginos/derivatives (for Q_0) and BMN ones (for Q_1)
full_secs = [s for s in sectors_at_level(12) if s[0] >= 0 and s[1] >= 0][:12] + \
            [s for s in sectors_at_level(14) if s[0] >= 0 and s[1] >= 0][::7][:8]
bmn_secs = [(5, 5, 8, 8, 8), (4, 4, 6, 6, 6), (3, 3, 6, 4, 4), (6, 6, 8, 8, 8)]

n_words = n_q0 = n_q1 = 0
bad_a = bad_b = bad_c = bad_d0 = bad_d1 = 0
for sec in full_secs:
    ws = words_of_sector(sec, maxlen=4, Ymax=7)
    for m in ws:
        n_words += 1
        Y, F, d, JR, q = grades(m)
        if (-1) ** (F + Y) != (-1) ** int(sum(q) - 3 * d):
            bad_a += 1
    for m in random.sample(ws, min(25, len(ws))):
        Y, F, d, JR, q = grades(m)
        Qm = {k: v for k, v in Q_multi(m).items() if v}
        for m2 in Qm:
            n_q0 += 1
            Y2, F2, d2, JR2, q2 = grades(m2)
            if not (Y2 == Y + 1 and d2 == d and JR2 == JR and q2 == q and F2 != F):
                bad_b += 1
        lhs = conj_expr(Qm)
        rhs = apply(lambda mm: {k: v for k, v in Q_multi(mm).items() if v}, conj_expr({m: 1}))
        if add(lhs, rhs, -1):
            bad_d0 += 1

for sec in bmn_secs:
    ws = [m for m in words_of_sector(sec, maxlen=5, Ymax=8) if all(L[0] == L[1] == 0 for w in m for L in w)]
    for m in random.sample(ws, min(12, len(ws))):
        n_words += 1
        Y, F, d, JR, q = grades(m)
        Q1m = Q1_multi(m, 3)
        for m2 in Q1m:
            n_q1 += 1
            Y2, F2, d2, JR2, q2 = grades(m2)
            if not (Y2 == Y and d2 == d + 1 and JR2 == JR and q2 == q and F2 != F):
                bad_c += 1
        lhs = conj_expr(Q1m)
        rhs = apply(lambda mm: Q1_multi(mm, 3), conj_expr({m: 1}))
        if add(lhs, rhs, -1):
            bad_d1 += 1

print(f'words checked: {n_words}; (a) (-1)^(F+Y) = (-1)^(|q|-3d) failures: {bad_a}')
print(f'(b) Q_0 output terms: {n_q0}; selection-rule failures: {bad_b}; C Q_0 != Q_0 C on {bad_d0} words')
print(f'(c) Q_1 output terms (N=3, BMN words): {n_q1}; selection-rule failures: {bad_c}; C Q_1 != Q_1 C on {bad_d1} words')
print('time %.1fs' % (time.time() - t0))
