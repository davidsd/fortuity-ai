"""Graviton monomials in a sector: rank modulo Q-exact (generic points) vs rank on diagonal matrices.
usage: test_diag_criterion_N.py N 2J,q1,q2,q3 dims_pickle [gen_nmax]"""
import sys, math, copy, pickle, numpy as np
from fortuity.bmn.symdirect import SymDirect
from fortuity.bmn.evaluate import Points
from fortuity.bmn.modp import RREFBasis
N = int(sys.argv[1]); sec = tuple(int(x) for x in sys.argv[2].split(',')); dims = pickle.load(open(sys.argv[3], 'rb'))
nmax = int(sys.argv[4]) if len(sys.argv) > 4 else N
D = SymDirect(N=N, gen_nmax=nmax, symmetrize=False, maxlen=30); D.guard.log = True; D.dimV = dims
k, degs, cpp = D._sector_k(sec)
print('sector', sec, 'dim V =', dims.get(sec), 'k', k, 'degs', degs, 'cpp', cpp, flush=True)
# diagonal rank (adaptive)
n = 8; prev = None
while True:
    pd = Points(N, n, k, degs, seed=99 + n, diagonal=True)
    G, nm = D.graviton_rows(sec, pd)
    r = int(RREFBasis(pd.ncols).add_batched(G).sum())
    print(f'diagonal: n={n} cols={pd.ncols} monomials={nm} rank={r}', flush=True)
    if prev == r and r < pd.ncols - 16: break
    prev = r; n *= 2
# generic: modulo Q-exact
up = (sec[0] + 1,) + sec[1:]
rQup = D.rankQ(up); qb, qpts = D._Qrows[up]
gb = copy.deepcopy(qb)
G, nm = D.graviton_rows(sec, qpts)
grav = int(gb.add_batched(G).sum())
gr = int(RREFBasis(qpts.ncols).add_batched(G).sum())
print(f'generic: rank Q(up) = {rQup} (cols {qpts.ncols}); gravitons modulo Q-exact = {grav}; rank of graviton rows = {gr}', flush=True)
print('RESULT: diagonal rank', r, ' vs gravitons mod exact', grav, flush=True)
