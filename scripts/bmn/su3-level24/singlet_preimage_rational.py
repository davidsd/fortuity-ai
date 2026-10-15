"""Stage B: exact rational coefficients of the covariant f-free preimage of eq. (4.8) by multi-modular solving + CRT,
then verification at fresh primes/points and formal/N=4 checks."""
import sys, json, time, pickle, itertools, numpy as np, sympy
from fractions import Fraction
from math import gcd, lcm
from collections import defaultdict
from build_O import eval_term, P_, S_
from gens_util import load_O, canon_expr
from fortuity.bmn.words import Q_expr, expr_sector
from fortuity.bmn.evaluate import Points
from fortuity.bmn.modp import RREFBasis
t0 = time.time(); log = lambda *a: print('[%6.0fs]' % (time.time() - t0), *a, flush=True)
d = pickle.load(open('runs/singlet_stageA.pkl', 'rb'))
basis, (label, S, T) = d['basis'], d['best']
sel = [j for j, _ in T]
log('support:', len(sel), 'covariant singlets; ordering used:', label)
exprs = [basis[j][0] for j in sel]
qex = [Q_expr(e) for e in exprs]
O, _ = load_O('O_cckll.json')           # = 12 * O_paper
def ratrec(a, M):
    a %= M
    r0, r1, s0, s1 = M, a, 0, 1
    bound = int((M // 2) ** 0.5)
    while r1 > bound:
        q = r0 // r1
        r0, r1, s0, s1 = r1, r0 - q * r1, s1, s0 - q * s1
    if s1 == 0 or abs(s1) > bound or gcd(r1, abs(s1)) != 1: return None
    return Fraction(r1, s1) if s1 > 0 else Fraction(-r1, -s1)
def solve_mod(p, seed, npts=28):
    pd = Points(3, npts, 7, [3], seed=seed, p=p)
    M = np.array([pd.expr_vector(q) % p for q in qex], dtype=np.int64)
    v = pd.expr_vector(O) % p
    A = np.concatenate([M, np.eye(len(sel), dtype=np.int64)], axis=1)
    full = RREFBasis(A.shape[1], p); ind = full.add_batched(A)
    assert ind.all(), 'selected Q-images not independent at prime %d' % p
    r = full.reduce(np.concatenate([v, np.zeros(len(sel), dtype=np.int64)])[None, :])[0]
    assert not np.any(r[:pd.ncols]), 'not in span at prime %d' % p
    return [int(x) for x in (-r[pd.ncols:]) % p]
primes = []; q = 2 ** 26
while len(primes) < 40:
    q = int(sympy.prevprime(q)); primes.append(q)
Mod = 1; res = [0] * len(sel); prev = None; stable = 0; used = 0
for p in primes[:34]:
    y = solve_mod(p, 9000 + used)
    inv = pow(Mod, -1, p)
    res = [(r + Mod * (((yy - r) * inv) % p)) for r, yy in zip(res, y)]
    Mod *= p; used += 1
    rat = [ratrec(r, Mod) for r in res]
    ok = all(x is not None for x in rat)
    if ok and rat == prev: stable += 1
    else: stable = 0
    prev = rat if ok else None
    log('primes used %2d (modulus ~ 10^%d): reconstruction %s, stable for %d extra primes' % (used, len(str(Mod)) - 1, 'complete' if ok else 'incomplete (%d of %d)' % (sum(x is not None for x in rat), len(rat)), stable))
    if stable >= 3: break
assert prev is not None and stable >= 3, 'rational reconstruction did not stabilise'
y = prev
den = 1
for x in y: den = lcm(den, x.denominator)
log('exact rational coefficients found. common denominator:', den, '| max |numerator| after clearing:', max(abs(int(x * den)) for x in y))
# lambda_0 (integer-cleared):  den * lambda_0 = sum_j (den*y_j) e_j ;  claim  Q(lambda_0) = O_stored = 12 O_paper
lam = defaultdict(int)
for x, e in zip(y, exprs):
    c = int(x * den)
    for m, v in e.items(): lam[m] += c * v
lam = {m: v for m, v in lam.items() if v}
Qlam = canon_expr(Q_expr(lam))
Oc = canon_expr(O)
diff = defaultdict(int)
for m, v in Qlam.items(): diff[m] += v
for m, v in Oc.items(): diff[m] -= den * v
diff = {m: v for m, v in diff.items() if v}
log('lambda_0: canonical multi-trace terms', len(lam), '| sector', expr_sector(lam))
log('Q(lambda_0) - O as a formal expression: %d canonical terms (%s)' % (len(diff), 'vanishes formally' if not diff else 'nonzero formally, as expected for an identity that uses N=3 trace relations'))
for N, p, seed in ((3, primes[36], 1), (3, primes[37], 2), (3, primes[38], 3), (4, primes[36], 4), (5, primes[36], 5)):
    pts = Points(N, 40, 7, [3], seed=seed, p=p)
    z = not np.any(pts.expr_vector(diff)) if diff else True
    log('   Q(lambda_0) - O on random traceless %dx%d matrices, fresh prime %d, 40 points: %s' % (N, N, p, 'ZERO' if z else 'nonzero'))
pickle.dump({'sel': sel, 'y': y, 'den': den, 'basis_items': [basis[j] for j in sel]}, open('runs/singlet_rational.pkl', 'wb'))
log('saved')
