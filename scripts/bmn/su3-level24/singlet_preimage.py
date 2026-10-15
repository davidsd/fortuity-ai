"""Covariant (SU(3)-singlet) f-free preimage of eq. (4.8) of arXiv:2312.16443.
Singlets with content (4 phi, 4 psi): multi-trace patterns with the four upper indices of the phi's contracted with the
four lower indices of the psi's by a permutation.  Stage A: build them, extract a basis, test (4.8) in Q(basis), sparsify."""
import sys, json, time, copy, pickle, itertools, numpy as np
from math import gcd
from collections import defaultdict
from build_O import eval_term, P_, S_
from gens_util import load_O, canon_expr
from fortuity.bmn.words import words_of_sector, Q_expr
from fortuity.bmn.evaluate import Points
from fortuity.bmn.modp import RREFBasis
p = int(sys.argv[1]) if len(sys.argv) > 1 else 67108859
seed = 4711
t0 = time.time(); log = lambda *a: print('[%6.0fs]' % (time.time() - t0), *a, flush=True)
up = (4, 4, 4, 4)
def canon_cyc(t):
    return min(t[i:] + t[:i] for i in range(len(t)))
pats = set()
for w in words_of_sector(up, 100):
    if any(l == 6 for tr in w for l in tr): continue
    pats.add(tuple(sorted(canon_cyc(tuple('p' if l < 3 else 's' for l in tr)) for tr in w)))
pats = sorted(pats, key=lambda P: (tuple(sorted(len(t) for t in P)), P))
log('f-free multi-trace letter patterns with 4 phi, 4 psi:', len(pats))
def normalise(e):
    e = {m: int(c) for m, c in e.items() if c}
    if not e: return None
    g = 0
    for c in e.values(): g = gcd(g, abs(c))
    k0 = min(e)
    s = 1 if e[k0] > 0 else -1
    return {m: s * c // g for m, c in e.items()}
singlets = {}     # key -> (expr, pattern, perm)
for P in pats:
    for perm in itertools.permutations(range(4)):
        factors = []; ip = 0; isx = 0
        for tr in P:
            slots = []
            for ch in tr:
                if ch == 'p': slots.append(P_('a%d' % ip)); ip += 1
                else: slots.append(S_('a%d' % perm[isx])); isx += 1
            factors.append(('tr', slots))
        e = normalise(canon_expr({m: int(c) for m, c in eval_term(1, factors, []).items()}))
        if e is None: continue
        key = frozenset(e.items())
        if key not in singlets:
            singlets[key] = (e, P, perm)
S_all = list(singlets.values())
log('distinct nonzero covariant singlets:', len(S_all))
structs = defaultdict(int)
for e, P, perm in S_all: structs[tuple(sorted(len(t) for t in P))] += 1
log('by trace structure:', dict(sorted(structs.items())))
# basis of the singlet space at N=3
pu = Points(3, 14, 8, [4], seed=seed, p=p)
def order_key(x):
    e, P, perm = x
    return (-len(P), tuple(sorted(len(t) for t in P)), len(e))
S_all.sort(key=order_key)
eb = RREFBasis(pu.ncols, p); basis = []
for s0 in range(0, len(S_all), 128):
    chunk = S_all[s0:s0 + 128]
    ind = eb.add(np.array([pu.expr_vector(e) for e, _, _ in chunk], dtype=np.int64))
    basis += [x for x, f in zip(chunk, ind) if f]
log('dimension of the span of covariant singlets at N=3 (f-free, (4;4,4,4)):', len(basis), '| functionals', pu.ncols)
# Q-images
pd = Points(3, 28, 7, [3], seed=seed + 1, p=p)
O, _ = load_O('O_cckll.json'); vO = pd.expr_vector(O) % p
t1 = time.time()
Qrows = []
for e, P, perm in basis:
    q = Q_expr(e)
    Qrows.append(pd.expr_vector(q) % p if q else np.zeros(pd.ncols, dtype=np.int64))
Qrows = np.array(Qrows, dtype=np.int64)
log('Q-images of the singlet basis evaluated (%d functionals)' % pd.ncols)
ex = RREFBasis(pd.ncols, p); ind = ex.add_batched(Qrows)
log('rank of Q on the singlet space:', ex.rank, '| (4.8) in the span of Q(singlets):', not np.any(ex.reduce(vO[None, :])[0]))
def solve(order, label):
    b = RREFBasis(pd.ncols, p); S = []
    for j in order:
        f = b.add(Qrows[j][None, :])[0]
        if f: S.append(j)
        if not np.any(b.reduce(vO[None, :])[0]): break
    M = Qrows[S]
    A = np.concatenate([M, np.eye(len(S), dtype=np.int64)], axis=1)
    full = RREFBasis(A.shape[1], p); full.add_batched(A)
    r = full.reduce(np.concatenate([vO, np.zeros(len(S), dtype=np.int64)])[None, :])[0]
    assert not np.any(r[:pd.ncols])
    y = (-r[pd.ncols:]) % p
    T = [(S[j], int(y[j])) for j in range(len(S)) if y[j]]
    st = defaultdict(int)
    for j, _ in T: st[tuple(sorted(len(t) for t in basis[j][1]))] += 1
    log('%-40s words used %4d | support %4d | %s' % (label, len(S), len(T), dict(sorted(st.items()))))
    return S, T
n = len(basis)
strc = [tuple(sorted(len(t) for t in basis[j][1])) for j in range(n)]
orders = {'more traces first': list(range(n)),
          'fewer traces first': sorted(range(n), key=lambda j: (len(strc[j]), strc[j], j)),
          '(2,2,4),(2,3,3),(2,2,2,2) first': sorted(range(n), key=lambda j: (0 if strc[j] in ((2, 2, 4), (2, 3, 3), (2, 2, 2, 2)) else 1, j)),
          '(2,3,3) first': sorted(range(n), key=lambda j: (0 if strc[j] == (2, 3, 3) else 1, j)),
          '(2,2,4) first': sorted(range(n), key=lambda j: (0 if strc[j] == (2, 2, 4) else 1, j)),
          '(4,4) first': sorted(range(n), key=lambda j: (0 if strc[j] == (4, 4) else 1, j)),
          'short expressions first': sorted(range(n), key=lambda j: (len(basis[j][0]), j))}
best = None
for label, order in orders.items():
    S, T = solve(order, label)
    if best is None or len(T) < len(best[2]): best = (label, S, T)
log('best:', best[0], 'support', len(best[2]))
pickle.dump({'basis': basis, 'best': best, 'prime': p}, open('runs/singlet_stageA.pkl', 'wb'))
log('saved')
