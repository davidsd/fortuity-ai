"""Check that symmetrized graviton monomials lie in the Weyl-symmetric subspace V^sym of a sector."""
import sys, numpy as np
from fortuity.bmn.symdirect import SymDirect
from fortuity.bmn.modp import RREFBasis
sec = tuple(int(x) for x in sys.argv[1].split(','))
D = SymDirect(N=3); D.guard.log = False
sw = D.symwords(sec)
eb, pts = D._basis_doubling(sec, len(sw), lambda p: (p.expr_vector(e) for e in sw), 'E')
print('sector', sec, 'dim V^sym =', eb.rank, 'cols', pts.ncols, flush=True)
G, nm = D.graviton_rows(sec, pts)
print('graviton monomials', nm, 'rows', G.shape, flush=True)
indep = eb.add_batched(G)
print('graviton rows outside V^sym:', int(indep.sum()), flush=True)
# also: rank of G alone, and of G within closed subspace
gb = RREFBasis(pts.ncols); r = int(gb.add_batched(G).sum()); print('rank of symmetrized graviton rows:', r)
