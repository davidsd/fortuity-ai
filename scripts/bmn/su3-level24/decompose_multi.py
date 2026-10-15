"""Exactness / graviton-class analysis of several closed operators in the SU(3) BMN sector (3;4,4,4),
Weyl-symmetric subspace, over F_p.  usage: python3 decompose_multi.py <prime> <seed> <npts> <out.json> <target.json>...
For each target: Q-closed at N=3?  exact?  coefficients on the 29 graviton pivot classes (rational reconstruction)."""
import sys, json, time, copy, numpy as np
from collections import defaultdict
from fortuity.bmn.symdirect import SymDirect, symmetrize_expr
from fortuity.bmn.words import Q_multi, Q_expr, contents_of_sector
from fortuity.bmn.evaluate import Points
from fortuity.bmn.modp import RREFBasis
from gens_util import load_O, poly_from_terms, ratrec

p, seed, npts, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
targets = sys.argv[5:]
sec, up, low = (3, 4, 4, 4), (4, 4, 4, 4), (2, 4, 4, 4)
t0 = time.time(); log = lambda *a: print('[%6.0fs]' % (time.time() - t0), *a, flush=True)
D = SymDirect(N=3, gen_nmax=3, seed=seed, verbose=False); D.guard.log = False
k, degs, cpp = D._sector_k(sec)
pts = Points(3, npts, k, degs, seed=seed, p=p)
degs_low = sorted({sum(c[3:6]) for c in contents_of_sector(low)})
ptsl = Points(3, 8, max(degs_low) + 2, degs_low, seed=seed + 7, p=p)
ops = {}
for path in targets:
    e, meta = load_O(path); ops[path] = e
dN = json.load(open('../../../data/su3/graviton_class_vanishing_on_diagonal_j24.json'))
ops['Sym(R-002 polynomial)'] = symmetrize_expr(poly_from_terms([(t['coeff'], t['monomial']) for t in dN['operator']]))
vals = {}
for name, e in ops.items():
    q = Q_expr(e)
    closed = (not q) or not np.any(ptsl.expr_vector(q))
    v = pts.expr_vector(e)
    vals[name] = v
    log(name, ': terms', len(e), '| Q-closed at N=3:', closed, '| nonzero at N=3:', bool(np.any(v)))
sw_up = D.symwords(up)
qexprs = []
for e in sw_up:
    q = defaultdict(int)
    for m, c in e.items():
        for m2, c2 in Q_multi(m).items():
            q[m2] += c * c2
    qexprs.append({m: c for m, c in q.items() if c})
Qrows = np.array([pts.expr_vector(q) if q else np.zeros(pts.ncols, dtype=np.int64) for q in qexprs], dtype=np.int64)
exact = RREFBasis(pts.ncols, p); exact.add_batched(Qrows)
log('rank exact =', exact.rank)
G, nm = D.graviton_rows(sec, pts)
gb = copy.deepcopy(exact); gind = gb.add_batched(G); piv = np.nonzero(gind)[0]
log('graviton classes mod exact =', len(piv))
Mg = G[piv] % p
A = np.concatenate([Mg, np.eye(len(piv), dtype=np.int64)], axis=1)
# basis: exact rows first (they carry zero identity part), then pivot gravitons with identity tags
full = RREFBasis(A.shape[1], p)
full.add_batched(np.concatenate([Qrows % p, np.zeros((Qrows.shape[0], len(piv)), dtype=np.int64)], axis=1))
full.add_batched(A)
res = {'prime': p, 'seed': seed, 'rank_exact': int(exact.rank), 'graviton_classes': int(len(piv)), 'targets': {}}
target_vecs = []
for name, v in vals.items():
    t = copy.deepcopy(exact); nonexact = bool(t.add(v[None, :])[0]); del t
    r = full.reduce(np.concatenate([v % p, np.zeros(len(piv), dtype=np.int64)])[None, :])[0]
    in_span = not np.any(r[:pts.ncols])
    x = (-r[pts.ncols:]) % p
    xr = [ratrec(int(c), p) for c in x]
    nz = [(int(i), str(xr[i]) if xr[i] is not None else 'mod p: %d' % int(x[i])) for i in range(len(piv)) if x[i]]
    log(name, ': non-exact:', nonexact, '| graviton+exact:', in_span, '| nonzero pivot coefficients:', len(nz))
    res['targets'][name] = dict(nonexact=nonexact, in_span=in_span, coeffs=nz)
    target_vecs.append(x)
# rank of the span of the classes
T = np.array(target_vecs, dtype=np.int64) % p
rb = RREFBasis(T.shape[1], p); rb.add_batched(T)
log('rank of the span of the target classes in graviton cohomology:', rb.rank)
res['class_span_rank'] = int(rb.rank)
json.dump(res, open(out, 'w'), indent=1); log('saved', out)
