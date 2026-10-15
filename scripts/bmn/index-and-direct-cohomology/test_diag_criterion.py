"""Rank of symmetrized graviton monomials in a sector: (a) modulo Q-exact at generic points, (b) on diagonal points.
If (a) > (b), some multi-graviton class vanishes on the diagonal without being Q-exact."""
import sys, math, numpy as np
from fortuity.bmn.symdirect import SymDirect
from fortuity.bmn.evaluate import Points
from fortuity.bmn.modp import RREFBasis
sec = tuple(int(x) for x in sys.argv[1].split(','))
nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 3
D = SymDirect(N=3, gen_nmax=nmax); D.guard.log = False
k, degs, cpp = D._sector_k(sec)
# (b) diagonal points, adaptive
prev = None; n = 8
while True:
    pd = Points(3, n, k, degs, seed=99 + n, diagonal=True)
    G, nm = D.graviton_rows(sec, pd)
    r = int(RREFBasis(pd.ncols).add_batched(G).sum())
    print(f'diagonal: n={n} cols={pd.ncols} monomials={nm} rank={r}', flush=True)
    if prev == r and r < pd.ncols - 16: break
    prev = r; n *= 2
# (a) generic points: rank of gravitons modulo Q-exact = analyse()'s grav
up = (sec[0] + 1,) + sec[1:]
rQup = D.rankQ(up)
qb, qpts = D._Qrows[up]
import copy
gb = copy.deepcopy(qb)
G, nm = D.graviton_rows(sec, qpts)
grav = int(gb.add_batched(G).sum())
print(f'generic: gravitons modulo Q-exact = {grav}  (rank Q_up = {rQup}, cols {qpts.ncols})', flush=True)
gr = int(RREFBasis(qpts.ncols).add_batched(G).sum())
print(f'generic: rank of graviton rows themselves = {gr}', flush=True)
