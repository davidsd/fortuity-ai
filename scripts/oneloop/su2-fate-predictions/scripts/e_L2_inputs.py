import os, sys
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.environ['REPO'] + '/code')
from fortuity.full.words import Q_expr, Q_multi, add_term, words_of_sector, contents_of_sector
from fortuity.full.q1 import Q1_expr
from fortuity.full.letters import is_ferm
from fortuity.full.evalpoints import PointsFull
from fortuity.bmn.modp import rank_mod
phi1, psi2, psi3, f = (0, 0, (0,)), (0, 0, (0, 2)), (0, 0, (0, 1)), (0, 0, (0, 1, 2))
w2 = defaultdict(int); add_term(w2, ((f, phi1),), 1); add_term(w2, ((psi2, psi3),), 1)   # tr(f phi1) + tr(psi2 psi3)
assert not Q_expr(w2)                                       # Q_0-closed, (2J1,2J2;2q) = (2,2;4,2,2), Y = 2
print('Q_1 w2 =', dict(Q1_expr(w2, 2)), dict(Q1_expr(w2, 3)))
sq = defaultdict(int)
for a, ca in w2.items():
    for b, cb in w2.items():
        add_term(sq, a + b, ca * cb)
v = Q1_expr(sq, 2)                                          # = {w2, w2}, since Q_1 w2 = 0 (96 terms, nonzero)
T, UP = (3, 3, 8, 4, 4), (4, 4, 8, 4, 4)
up = [m for m in words_of_sector(UP, 4) if sum(map(len, m)) == 3]   # Q_0-preimage words, Y = 3
tw = [m for m in words_of_sector(T, 4) if sum(map(len, m)) == 4]    # target words, Y = 4
conts = [c for c in contents_of_sector(T) if len(c) == 4]
degs = sorted({sum(map(is_ferm, c)) for c in conts})
letters = {L for c in conts for L in c} | {L for m in v for w in m for L in w}
for seed, npts in [(5, 32), (6, 64), (7, 128)]:
    P = PointsFull(2, npts, max(degs) + 4, degs, letters, seed=seed)
    M = np.array([P.expr_vector(Q_multi(m)) for m in up])
    dimT = rank_mod(np.array([P.expr_vector({m: 1}) for m in tw]))
    r0, r1 = rank_mod(M), rank_mod(np.vstack([M, P.expr_vector(v)[None, :]]))
    print(f'seed {seed}, {P.ncols} functionals: dim V_T = {dimT}, exact rank {r0}, with {{w2,w2}} {r1} -> exact: {r0 == r1}')
