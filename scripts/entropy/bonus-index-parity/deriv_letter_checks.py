"""T-038 derivation, per-letter sanity checks with the repo's fortuity.full.letters (read-only use).

Checks, for every admissible letter (a, b, S) with a, b <= AMAX:
  (1) KKLP exponents (arXiv:2606.27955 eq. 4.1, T = R1+R2+R3-Y): J1+T = a, J2+T = b, R_j-T = [j in S];
  (2) KKLP d_BMN (eq. 2.21) = R1+R2+R3+(J1+J2)/2-Y = (a+b)/2, and equals the repo's d_BMN = (a+b)/2;
  (3) the same exponents in exact-charge variables: (d+J_R, d-J_R, q_j-d), J_R = (J1-J2)/2, q_j = R_j + J_L;
      KKLP weight p^{J1+T} q^{J2+T} prod u_j^{R_j-T} = x^{-d} (p/q)^{J_R} prod u_j^{q_j},  x = u1 u2 u3/(p q);
  (4) statistics: is_ferm <=> |S| even; (-1)^F = (-1)^{2J1} = (-1)^{2J2}; (-1)^{F+Y} = (-1)^{|S|}; |S| = |q| - 3d;
      KKLP bonus charge 3(R-T) = |S|; Y = |q| - 2 J_L - d (= 1 per letter);
  (5) single-letter index: sum_L (-1)^F p^a q^b prod u_j^{[j in S]} = 1 - (1-u)(1-v)(1-w)/((1-p)(1-q))
      (the integrand of KKLP eq. 4.1), as a truncated power series;
  (6) Q_0 on letters (Q_letter): every term (L1, L2) has a1+a2=a, b1+b2=b, S1 u S2 = S disjoint, so Q_0 preserves
      sum a, sum b, n_j (all five KKLP exponents, hence d, J_R, q), raises Y by 1 and flips F;
      charges of Q: Delta J1 = Delta J2 = -1/2, Delta R_j = +1/2, Delta q_j = 0, Delta J_R = 0.
"""
import sys, time
from fractions import Fraction as Fr
from itertools import combinations

import os  # repo root: $REPO, or three levels up from scripts/entropy/bonus-index-parity/
sys.path.insert(0, os.path.join(os.environ.get('REPO') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')), 'code'))
from fortuity.full.letters import letters_up_to, charges2, is_ferm, Q_letter, admissible, THETA

t0 = time.time()
AMAX = 8
H = Fr(1, 2)


def ch(L):
    """Exact charges from the repo's charges2 (doubled) -> (J1, J2, q1, q2, q3) as Fractions, plus derived ones."""
    tJ1, tJ2, tq1, tq2, tq3 = charges2(L)
    J1, J2 = Fr(tJ1, 2), Fr(tJ2, 2)
    q = (Fr(tq1, 2), Fr(tq2, 2), Fr(tq3, 2))
    JL, JR = (J1 + J2) / 2, (J1 - J2) / 2
    R = tuple(qj - JL for qj in q)            # q_j = R_j + J_L
    return J1, J2, q, JL, JR, R


Ls = letters_up_to(AMAX, AMAX)
nfail = 0
for L in Ls:
    a, b, S = L
    s = len(S)
    J1, J2, q, JL, JR, R = ch(L)
    Y = 1
    # docstring formulas of letters.py (lines 6-8) against charges2
    assert J1 == -H + a + Fr(s, 2) and J2 == -H + b + Fr(s, 2)
    assert all(R[j] == H - Fr(s, 2) + (1 if j in S else 0) for j in THETA)
    assert JR == Fr(a - b, 2)
    T = sum(R) - Y
    # (1)
    ok1 = (J1 + T == a) and (J2 + T == b) and all(R[j] - T == (1 if j in S else 0) for j in THETA)
    # (2)
    d_kklp = sum(R) + (J1 + J2) / 2 - Y
    d_repo = Fr(a + b, 2)
    ok2 = d_kklp == d_repo
    d = d_repo
    # (3)
    ok3 = (J1 + T == d + JR) and (J2 + T == d - JR) and all(R[j] - T == q[j] - d for j in THETA)
    # (4)
    F = 1 if is_ferm(L) else 0
    ok4 = (F == (1 if s % 2 == 0 else 0)) and ((-1) ** F == (-1) ** int(2 * J1) == (-1) ** int(2 * J2)) \
        and ((-1) ** (F + Y) == (-1) ** s) and (s == sum(q) - 3 * d) and (3 * (sum(R) / 3 - T) == s) \
        and (Y == sum(q) - 2 * JL - d)
    if not (ok1 and ok2 and ok3 and ok4):
        nfail += 1
        print('FAIL', L, ok1, ok2, ok3, ok4)
print(f'(1)-(4): {len(Ls)} letters with a,b <= {AMAX}: failures = {nfail}')

# (5) single-letter index as a truncated series in p, q (degree <= K each), exact polynomial in u, v, w
import sympy as sp
p, qq, u, v, w = sp.symbols('p q u v w')
K = 5
lhs = 0
for L in letters_up_to(K, K):
    a, b, S = L
    F = 1 if is_ferm(L) else 0
    mon = (-1) ** F * p ** a * qq ** b
    for j, t in zip(THETA, (u, v, w)):
        if j in S:
            mon *= t
    lhs += mon
geo_p = sum(p ** k for k in range(K + 1))
geo_q = sum(qq ** k for k in range(K + 1))
rhs = sp.expand(1 - (1 - u) * (1 - v) * (1 - w) * geo_p * geo_q)
# truncate rhs to p-degree <= K, q-degree <= K (already) and compare
diff = sp.expand(lhs - rhs)
print('(5) single-letter index = 1 - (1-u)(1-v)(1-w)/((1-p)(1-q)) through p^%d q^%d:' % (K, K), diff == 0)

# (6) Q_0 on letters
nterms = 0
bad = 0
for L in Ls:
    a, b, S = L
    J1, J2, q, JL, JR, R = ch(L)
    F = 1 if is_ferm(L) else 0
    for c, L1, L2 in Q_letter(L):
        nterms += 1
        (a1, b1, S1), (a2, b2, S2) = L1, L2
        c1, c2 = ch(L1), ch(L2)
        F12 = (1 if is_ferm(L1) else 0) + (1 if is_ferm(L2) else 0)
        good = (a1 + a2 == a and b1 + b2 == b and set(S1) | set(S2) == set(S) and not (set(S1) & set(S2))
                and admissible(*L1) and admissible(*L2)
                and (F12 - F) % 2 == 1
                and c1[0] + c2[0] - J1 == -H and c1[1] + c2[1] - J2 == -H
                and all(c1[5][j] + c2[5][j] - R[j] == H for j in THETA)
                and all(c1[2][j] + c2[2][j] == q[j] for j in THETA)
                and c1[4] + c2[4] == JR
                and Fr(a1 + b1 + a2 + b2, 2) == Fr(a + b, 2))
        if not good:
            bad += 1
            print('Q0 FAIL', L, L1, L2)
print(f'(6) Q_0 letter splittings: {nterms} terms over {len(Ls)} letters, failures = {bad}')
print('time %.2fs' % (time.time() - t0))
