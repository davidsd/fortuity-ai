"""Independent verification of the explicit SU(3) multi-graviton class (results/su3_graviton_class.pkl)."""
import pickle, copy, numpy as np
from collections import defaultdict
from fortuity.bmn.words import graviton_generators, mul_expr, Q_expr, expr_sector, content, NAMES
from fortuity.bmn.symdirect import SymDirect, symmetrize_expr
from fortuity.bmn.evaluate import Points
from fortuity.bmn.modp import RREFBasis
d = pickle.load(open('results/su3_graviton_class.pkl', 'rb'))
gens = dict((name, e) for name, e in graviton_generators(3, 1))
# build the operator as a multi-trace expression
op = defaultdict(int)
for coef, mono in d['terms']:
    e = {(): 1}
    for g in mono:
        e = mul_expr(e, gens[d['names'][g]])
    for m, c in e.items():
        op[m] += coef * c
op = {m: c for m, c in op.items() if c}
sec = expr_sector(op)
print('operator: %d multi-trace terms, sector %s' % (len(op), sec), flush=True)
# 1. Q-closed
q = Q_expr(op)
low = (sec[0]-1,)+sec[1:]
pl = Points(3, 6, 7, [1, 3], seed=31)
print('Q(op) == 0 :', (not q) or not np.any(pl.expr_vector(q)), flush=True)
# 2. vanishes on diagonal matrices, nonzero on generic ones
pd = Points(3, 8, 7, [1, 3], seed=32, diagonal=True); pg = Points(3, 8, 7, [1, 3], seed=33)
print('vanishes on diagonal:', not np.any(pd.expr_vector(op)), '; nonzero generic:', bool(np.any(pg.expr_vector(op))), flush=True)
# 3. Weyl-invariant?  compare symmetrized (sum over 6 Weyl elements) with 6*op
sym = symmetrize_expr(op)
diff = defaultdict(int)
for m, c in sym.items(): diff[m] += c
for m, c in op.items(): diff[m] -= 6 * c
diff = {m: c for m, c in diff.items() if c}
print('Weyl-invariant (sym == 6*op):', not diff, flush=True)
# 4. non-exact: add to the Q-image basis of the symmetric up-sector with fresh seed
D = SymDirect(N=3, seed=2024); D.guard.log = False
up = (sec[0]+1,)+sec[1:]
r = D.rankQ(up); qb, qpts = D._Qrows[up]
tb = copy.deepcopy(qb)
print('symmetrized op: %d terms; vanishes on diagonal: %s' % (len(sym), not np.any(pd.expr_vector(sym))), flush=True)
v = qpts.expr_vector(sym)
print('rank Q(up) =', r, '; symmetrized op nonzero:', bool(np.any(v)), '; independent of exact operators:', bool(tb.add(v[None, :])[0]), flush=True)
# readable form
print('\nExplicit form (X,Y,Z = phi_1,2,3; u^{ij}=tr(phi_i phi_j); v^i_j = 3 tr(phi_i psi_j) - delta^i_j tr(phi_a psi_a);')
print(' w^{ij} = 2 tr(f phi_(i phi_j)) + eps-terms, summed over orderings; see bmn/words.py):')
def nm(name):
    L = 'XYZ'
    if name[0] == 'u': return 'u^{%s}' % ''.join(L[i] for i in name[2])
    if name[0] == 'v': return 'v^{%s}_{%s}' % (L[name[2][0]], L[name[3]])
    return 'w^{%s}' % ''.join(L[i] for i in name[2])
print(' + '.join('(%d) %s' % (c, ' '.join(nm(d['names'][g]) for g in mono)) for c, mono in d['terms']))
