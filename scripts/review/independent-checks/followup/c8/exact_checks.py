import os as _os, sys as _sys
REPO = (_os.environ.get('REPO') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *['..'] * 5))).rstrip('/') + '/'
_sys.path.insert(0, REPO + 'code'); _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
"""Exact (rational, symbolic) checks of the dmk claims, independent of random evaluation."""
import sys
import time
import random
from fractions import Fraction as Fr
from itertools import permutations, product

pass  # symeval is found next to this script
import symeval as S
from fortuity.bmn import dmk
from fortuity.bmn import words as W
from fortuity.bmn.blockrestrict import restrict_u1
from fortuity.bmn.modp import P

t0 = time.time()
A2, A3 = S.Alg(2), S.Alg(3)

# ---------------------------------------------------------------- A. conventions
random.seed(1)
bad = 0
for _ in range(300):
    L = random.randint(2, 6)
    w = tuple(random.randrange(6) for _ in range(L))
    s, cw = W.canon_word(w)
    lhs = A3.trace(w)
    rhs = {} if s == 0 else {k: s * v for k, v in A3.trace(cw).items()}
    bad += (lhs != rhs)
print('A1 repo canon_word sign convention vs direct N=3 trace (300 random words): mismatches =', bad)
bad = 0
for _ in range(20):
    E = {}
    for _ in range(4):
        ws = tuple(tuple(random.randrange(6) for _ in range(random.randint(2, 4))) for _ in range(random.randint(1, 2)))
        s, m = W.canon_multi([W.canon_word(x)[1] for x in ws]) if all(W.canon_word(x)[0] for x in ws) else (0, None)
        if s:
            E[m] = E.get(m, 0) + random.randint(-3, 3)
    E = {k: v for k, v in E.items() if v}
    bad += (A3.expr(W.Q_expr(E)) != A3.Q(A3.expr(E)))
print('A2 repo Q_expr vs independent symbolic Q at N=3 (20 random expressions): mismatches =', bad)

# ---------------------------------------------------------------- B. exact SU(2) x U(1) restriction at N=2
O = dmk.o330()
X = {k: S.restrict_x1(O, k) for k in (0, 1, 2, 3)}
Xm = restrict_u1(dmk.to_modp(O), 2, (3, 0, 0))
mine_modp = {m: int(Fr(v).numerator * pow(Fr(v).denominator, P - 2, P) % P) for m, v in X[3].items()}
mine_modp = {m: v for m, v in mine_modp.items() if v}
print('B1 independent exact x1^3 restriction: %d SU(2) multitraces; equals repo restrict_u1 mod p: %s'
      % (len(X[3]), mine_modp == Xm))


def v2p(m, q):
    t = A2.trace((m, 3 + q))
    if m == q:
        for a in range(3):
            t = S.padd(t, A2.trace((a, 3 + a)), Fr(-1, 3))
    return t


O224 = {}
for p_, q, r in permutations(range(3)):
    e = S.eps(p_, q, r)
    for m in range(3):
        for n in range(3):
            s3 = {}
            for a, b, c in permutations((m, n, p_)):
                s3 = S.padd(s3, A2.trace((3 + a, 3 + b, 3 + c)), Fr(1, 6))
            O224 = S.padd(O224, S.pmul(S.pmul(v2p(m, q), v2p(n, r)), s3), e)
print('B2 independent O^2_24 (eq. 2.6) at N=2: %d monomials; equals repo dmk.o2_24() at N=2: %s'
      % (len(O224), A2.expr(dmk.o2_24()) == O224))
pX3 = A2.expr(X[3])
k0 = next(iter(O224))
ratio = Fr(pX3.get(k0, 0)) / Fr(O224[k0])
print('B3 exact x1^3 coefficient at N=2: %d monomials; ratio on first monomial = %s; '
      'x1^3 coeff == ratio * O^2_24 identically: %s' % (len(pX3), ratio, pX3 == {k: ratio * v for k, v in O224.items()}))
for k in (0, 1, 2):
    pk = A2.expr(X[k])
    print('B4 exact x1^%d coefficient at N=2: formal terms %d, value at N=2 has %d monomials' % (k, len(X[k]), len(pk)))
print('time %.1fs' % (time.time() - t0))

# ---------------------------------------------------------------- C. exact ingredients of closedness at N=3
gbad = 0
for name, nidx in (('u2', 2), ('u3', 3), ('v2', 2), ('v3', 3)):
    for I in product(range(3), repeat=nidx):
        g = dmk.tensor(name)(*I)
        if W.Q_expr(g) or A3.Q(A3.expr(g)):
            gbad += 1
print('C1 gravitons u2,u3,v2,v3: components with Q != 0 (formal or at N=3):', gbad)
HW = {'r12_03': ((2, 2, 2), None), 'r12_11': ((0, 2), (1, 1)), 'r12_22': ((0, 0, 2, 2), (2, 2)),
      'r14_13': ((0, 2, 2, 2), (1, 3)), 'r16_01': ((2,), None), 'r16_23': ((0, 0, 2, 2, 2), (2, 3))}
for name, (I, pq) in HW.items():
    if pq is not None:
        assert dmk.projector(*pq)[I] == [(Fr(1), I)], name          # traceless projection is the identity at hw
    r_raw, R_raw = dmk.tensor(name, project=False)(*I), dmk.tensor(dmk.R_OF_r[name], project=False)(*I)
    assert r_raw == dmk.tensor(name)(*I)
    D = S.padd(A3.Q(A3.expr(r_raw)), A3.expr(R_raw), -1)
    print('C2 %-7s hw component %s: Q r - R at N=3 has %d monomials (exact); |Q r| = %d monomials'
          % (name, I, len(D), len(A3.Q(A3.expr(r_raw)))))
# a trace component, as printed: Q r != R at N=3 (exact)
D = S.padd(A3.Q(A3.expr(dmk.tensor('r12_11', project=False)(0, 0))),
           A3.expr(dmk.tensor('R12_11', project=False)(0, 0)), -1)
print('C3 printed r12_11 trace component (0,0): Q r - R at N=3 has %d monomials' % len(D))
print('time %.1fs' % (time.time() - t0))
