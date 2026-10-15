"""Exactness of eq. (4.8) of arXiv:2312.16443 tested in the FULL complex (no Weyl symmetrisation): plain multi-trace
word basis of sector (4;4,4,4) -> Q-images -> RREF basis over F_p; targets reduced against it. Injectivity of the
evaluation functionals on the full sector (3;4,4,4) is certified by comparing the rank of all its words with the exact
sector dimension from the partition function (fullindex.partition_dims).
usage: python3 full_complex_check.py <prime> <seed> <npts> <out.json>"""
import sys, json, time, copy, numpy as np
from collections import defaultdict
from fortuity.bmn.words import words_of_sector, Q_multi, Q_expr, contents_of_sector
from fortuity.bmn.evaluate import Points
from fortuity.bmn.modp import RREFBasis
from fortuity.bmn.fullindex import partition_dims
from fortuity.bmn.symdirect import symmetrize_expr
from gens_util import load_O, poly_from_terms

p, seed, npts, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
sec, up, low = (3, 4, 4, 4), (4, 4, 4, 4), (2, 4, 4, 4)
t0 = time.time(); log = lambda *a: print('[%6.0fs]' % (time.time() - t0), *a, flush=True)
dims = partition_dims(3, 24)
log('exact sector dimensions from the partition function: dim(3;4,4,4) =', dims.get(sec), ', dim(4;4,4,4) =', dims.get(up))
degs = sorted({sum(c[3:6]) for c in contents_of_sector(sec)})
k = max(degs) + 4
pts = Points(3, npts, k, degs, seed=seed, p=p)
log('prime', p, 'points', npts, 'frame', k, 'degrees', degs, 'functionals', pts.ncols)
# targets
O, _ = load_O('O_cckll.json')
O1, _ = load_O('O1_control.json')
dN = json.load(open('../../../data/su3/graviton_class_vanishing_on_diagonal_j24.json'))
Pn = poly_from_terms([(t['coeff'], t['monomial']) for t in dN['operator']])
targets = {'(4.8) of 2312.16443': O, 'O_1^{(0,3)} (exact by construction)': O1, 'R-002 polynomial P': Pn, 'Sym(P)': symmetrize_expr(Pn)}
degs_low = sorted({sum(c[3:6]) for c in contents_of_sector(low)})
ptsl = Points(3, 8, max(degs_low) + 2, degs_low, seed=seed + 1, p=p)
vals = {}
for name, e in targets.items():
    q = Q_expr(e)
    vals[name] = pts.expr_vector(e) % p
    log(name, ': Q-closed at N=3:', (not q) or not np.any(ptsl.expr_vector(q)), '| nonzero:', bool(np.any(vals[name])))
# injectivity: rank of all words of the sector
ws = words_of_sector(sec, 100)
log('words of (3;4,4,4):', len(ws))
eb = RREFBasis(pts.ncols, p)
for s0 in range(0, len(ws), 256):
    eb.add(np.array([pts.expr_vector({w: 1}) for w in ws[s0:s0 + 256]], dtype=np.int64))
log('rank of all words =', eb.rank, '| equals exact dimension:', eb.rank == dims.get(sec))
del eb
# exact space
wu = words_of_sector(up, 100)
log('words of (4;4,4,4):', len(wu))
exact = RREFBasis(pts.ncols, p)
batch = []
for i, w in enumerate(wu):
    q = Q_multi(w)
    batch.append(pts.expr_vector(q) if q else np.zeros(pts.ncols, dtype=np.int64))
    if len(batch) == 256:
        exact.add(np.array(batch, dtype=np.int64)); batch = []
    if i % 5000 == 4999: log('  ', i + 1, 'Q-images processed, rank so far', exact.rank)
if batch: exact.add(np.array(batch, dtype=np.int64))
log('rank of the Q-exact subspace (full complex) =', exact.rank)
res = {'prime': p, 'seed': seed, 'npts': npts, 'ncols': int(pts.ncols), 'dim_sector': dims.get(sec), 'rank_exact_full': int(exact.rank), 'targets': {}}
for name, v in vals.items():
    r = exact.reduce(v[None, :])[0]
    ex = not np.any(r)
    log(name, ': exact =', ex)
    res['targets'][name] = ex
json.dump(res, open(out, 'w'), indent=1); log('saved', out)
