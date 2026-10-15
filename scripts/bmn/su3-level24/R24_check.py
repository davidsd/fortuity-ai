"""R_24^{(0,3)} = eps_{a1a2a3} u^{a1 b} v^{a2}_b v^{a3 c}_d v^d_c  (arXiv:2312.16443 eq. (A.4), t^24 [0,0]):
listed there as a fundamental trace relation (graviton polynomial vanishing on the diagonal, assumed Q-exact)."""
import json, numpy as np
from fractions import Fraction
from build_O6 import build, to_int, save
from gens_util import canon_expr, weyl_image, same
from fortuity.bmn.words import Q_expr, expr_sector
from fortuity.bmn.evaluate import Points
from fortuity.bmn.symdirect import PERMS, perm_sign
R24 = build([(1, [('eps', ['a1', 'a2', 'a3']), ('u2', ['a1', 'b']), ('v2', ['a2', 'b']), ('v3', ['a3', 'c', 'd']), ('v2', ['d', 'c'])], [])])
Ri, sc = to_int(R24); Rc = canon_expr(Ri)
print('R_24^{(0,3)}: canonical multi-trace terms', len(Rc), '| sector', expr_sector(Rc), '| stored = %s * defined' % sc)
print('formally Q-closed (polynomial in closed generators):', not canon_expr(Q_expr(Rc)))
for N in (2, 3, 4, 5):
    pg = Points(N, 16, 5, [1, 3], seed=300 + N); pd = Points(N, 16, 5, [1, 3], seed=400 + N, diagonal=True)
    print('  N=%d: on diagonal matrices: %-8s | on generic matrices: %s' % (N, 'zero' if not np.any(pd.expr_vector(Rc)) else 'NONZERO', 'zero' if not np.any(pg.expr_vector(Rc)) else 'nonzero'))
rels = []
for s in PERMS:
    img = canon_expr(weyl_image(Rc, s, with_sign=False))
    rels.append('+' if same(img, Rc, 1) else ('-' if same(img, Rc, -1) else '?'))
print('under the 6 flavour relabellings (perm signs %s): %s' % ([perm_sign(s) for s in PERMS], rels))
save(R24, 'R24.json', 'R_24^{(0,3)} of arXiv:2312.16443 eq. (A.4)')
print('saved R24.json')
