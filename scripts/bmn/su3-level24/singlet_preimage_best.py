"""Stage C: among many choices of independent covariant singlets, pick the preimage of (4.8) with the smallest
support and smallest coefficients; express it in covariant monomials; verify; write LaTeX and JSON."""
import sys, json, time, pickle, itertools, random, numpy as np, sympy
from fractions import Fraction
from math import gcd, lcm
from collections import defaultdict
from build_O import eval_term, P_, S_
from gens_util import load_O, canon_expr
from fortuity.bmn.words import Q_expr, expr_sector
from fortuity.bmn.evaluate import Points
from fortuity.bmn.modp import RREFBasis
t0 = time.time(); log = lambda *a: print('[%6.0fs]' % (time.time() - t0), *a, flush=True)
basis = pickle.load(open('runs/singlet_stageA.pkl', 'rb'))['basis']
n = len(basis)
O, _ = load_O('O_cckll.json')            # 12 * (4.8)
# covariant monomial C_j (unnormalised) and factor f_j with C_j = f_j * e_j
def covariant(P, perm):
    factors = []; ip = 0; isx = 0
    for tr in P:
        slots = []
        for ch in tr:
            if ch == 'p': slots.append(P_('a%d' % ip)); ip += 1
            else: slots.append(S_('a%d' % perm[isx])); isx += 1
        factors.append(('tr', slots))
    return canon_expr({m: int(c) for m, c in eval_term(1, factors, []).items()})
fac = []
for e, P, perm in basis:
    C = covariant(P, perm); k0 = min(e)
    f = Fraction(C[k0], e[k0]); assert all(C[m] == f * e[m] for m in e) and set(C) == set(e)
    fac.append(f)
qex = [Q_expr(e) for e, _, _ in basis]
primes = []; q = 2 ** 26
while len(primes) < 12:
    q = int(sympy.prevprime(q)); primes.append(q)
NP = 5
ev = []
for i, p in enumerate(primes[:NP]):
    pd = Points(3, 28, 7, [3], seed=7000 + i, p=p)
    ev.append((p, np.array([pd.expr_vector(x) % p if x else np.zeros(pd.ncols, dtype=np.int64) for x in qex], dtype=np.int64), pd.expr_vector(O) % p, pd.ncols))
log('Q-images of the %d basis singlets evaluated at %d primes' % (n, NP))
def ratrec(a, M):
    a %= M
    r0, r1, s0, s1 = M, a, 0, 1
    bound = int((M // 2) ** 0.5)
    while r1 > bound:
        qq = r0 // r1
        r0, r1, s0, s1 = r1, r0 - qq * r1, s1, s0 - qq * s1
    if s1 == 0 or abs(s1) > bound or gcd(r1, abs(s1)) != 1: return None
    return Fraction(r1, s1) if s1 > 0 else Fraction(-r1, -s1)
def select(order):
    p, M, v, nc = ev[0]
    b = RREFBasis(nc, p); S = []
    for j in order:
        if b.add(M[j][None, :])[0]: S.append(j)
        if not np.any(b.reduce(v[None, :])[0]): break
    return S
def solve(S):
    Mod = 1; res = [0] * len(S)
    for p, M, v, nc in ev:
        A = np.concatenate([M[S], np.eye(len(S), dtype=np.int64)], axis=1)
        full = RREFBasis(A.shape[1], p)
        if not full.add_batched(A).all(): return None
        r = full.reduce(np.concatenate([v, np.zeros(len(S), dtype=np.int64)])[None, :])[0]
        if np.any(r[:nc]): return None
        y = [int(x) for x in (-r[nc:]) % p]
        inv = pow(Mod, -1, p)
        res = [(a + Mod * (((yy - a) * inv) % p)) for a, yy in zip(res, y)]
        Mod *= p
    rat = [ratrec(a, Mod) for a in res]
    if any(x is None for x in rat): return None
    return rat
strc = [tuple(sorted(len(t) for t in basis[j][1])) for j in range(n)]
orders = {'short expressions first': sorted(range(n), key=lambda j: (len(basis[j][0]), j)),
          'more traces first': list(range(n)),
          '(4,4) first': sorted(range(n), key=lambda j: (0 if strc[j] == (4, 4) else 1, j)),
          'more traces, then short': sorted(range(n), key=lambda j: (-len(strc[j]), len(basis[j][0]), j)),
          'short, then more traces': sorted(range(n), key=lambda j: (len(basis[j][0]), -len(strc[j]), j))}
rng = random.Random(12345)
for t in range(60):
    o = list(range(n)); rng.shuffle(o); orders['random %d' % t] = o
best = None
for label, order in orders.items():
    S = select(order); y = solve(S)
    if y is None: continue
    c = [(S[k], y[k] / fac[S[k]] / 12) for k in range(len(S)) if y[k]]       # coefficients on covariant monomials for (4.8) itself
    D = 1
    for _, x in c: D = lcm(D, x.denominator)
    height = max(abs(int(x * D)) for _, x in c)
    score = (len(c), D * height)
    if best is None or score < best[0]:
        best = (score, label, c, D, height)
        log('%-28s support %3d | common denominator %d | max |numerator| %d' % (label, len(c), D, height))
score, label, c, D, height = best
log('chosen:', label, '| support', len(c), '| denominator', D, '| max numerator', height)
# assemble lambda_0 (times D) and verify at fresh primes
lam = defaultdict(int)
for j, x in c:
    C = covariant(basis[j][1], basis[j][2])
    for m, v in C.items(): lam[m] += int(x * D) * v
lam = {m: v for m, v in lam.items() if v}
Qlam = canon_expr(Q_expr(lam)); Oc = canon_expr(O)
diff = defaultdict(int)
for m, v in Qlam.items(): diff[m] += 12 * v          # 12 * D * Q(lambda_0)
for m, v in Oc.items(): diff[m] -= D * v             # - D * 12 * (4.8)
diff = {m: v for m, v in diff.items() if v}
log('lambda_0: %d canonical multi-trace terms; Q(lambda_0) - (4.8) formally: %d terms' % (len(lam), len(diff)))
for N, p, seed in ((3, primes[8], 1), (3, primes[9], 2), (3, primes[10], 3), (4, primes[8], 4), (5, primes[8], 5)):
    pts = Points(N, 40, 7, [3], seed=seed, p=p)
    log('   Q(lambda_0) - (4.8) on random traceless %dx%d matrices, fresh prime %d, 40 points: %s' % (N, N, p, 'ZERO' if not np.any(pts.expr_vector(diff)) else 'nonzero'))
# LaTeX
IDX = 'abcd'
def tex_mono(P, perm):
    out = ''; ip = 0; isx = 0
    for tr in P:
        out += '\\tr('
        for ch in tr:
            if ch == 'p': out += '\\phi^{%s}' % IDX[ip]; ip += 1
            else: out += '\\psi_{%s}' % IDX[perm[isx]]; isx += 1
        out += ')'
    return out
terms = []
for j, x in sorted(c, key=lambda t: (tuple(sorted(len(tt) for tt in basis[t[0]][1])), basis[t[0]][1], basis[t[0]][2])):
    k = int(x * D); sgn = '-' if k < 0 else '+'; mag = '' if abs(k) == 1 else str(abs(k))
    terms.append('%s %s\\,%s' % (sgn, mag, tex_mono(basis[j][1], basis[j][2])))
open('lambda0_tex.txt', 'w').write(' \\\\\n&\\quad '.join(' '.join(terms[i:i + 2]) for i in range(0, len(terms), 2)))
json.dump({'statement': '(4.8) of arXiv:2312.16443 = Q_code(lambda_0) = i Q_paper(lambda_0) on 3x3 traceless matrices; lambda_0 = (1/D) sum_j n_j C_j',
           'D': D, 'terms': [{'n': int(x * D), 'pattern': [''.join(t) for t in basis[j][1]], 'psi_index_perm': list(basis[j][2]), 'tex': tex_mono(basis[j][1], basis[j][2])} for j, x in c],
           'convention': "pattern: traces as strings over p (phi, upper index) and s (psi, lower index); phi slots carry indices a,b,c,d in order of appearance; the k-th psi slot carries the index of phi slot psi_index_perm[k]; all indices summed over 1..3",
           'lambda0_times_D_canonical': [[v, [list(w) for w in m]] for m, v in sorted(lam.items())]}, open('lambda0_preimage.json', 'w'))
open('lambda0_meta.txt', 'w').write('%d %d %d %d' % (len(c), D, height, len(lam)))
log('written lambda0_tex.txt, lambda0_preimage.json')
print(open('lambda0_tex.txt').read()[:900])
