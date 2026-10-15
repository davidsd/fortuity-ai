import os, sys
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.environ['REPO'] + '/code')
from fortuity.bmn.words import add_term, Q_expr, Qplus_expr, expr_sector
from fortuity.bmn.direct import Direct
from fortuity.bmn.modp import rank_mod
X, Y, Z, p1, p2, p3 = range(6)
O0 = defaultdict(int)   # Choi-Kim-Lee-Park operator, as in scripts/bmn/index-and-direct-cohomology/test_threshold_ops.py:14-17
for (a, b, c, s1, s2, s3) in [(X, Y, Z, p1, p2, p3), (Y, Z, X, p2, p3, p1), (Z, X, Y, p3, p1, p2)]:
    add_term(O0, ((s1, a), (s3, a), (s2, s1, s1)), 1); add_term(O0, ((s2, b), (s3, a), (s2, s1, s1)), -1)
for seed, margin in [(12345, 16), (777, 16), (4242, 3000)]:
    D = Direct(N=2, maxlen=3, gen_nmax=2, seed=seed, margin=margin, verbose=False)
    for name, d in [('O_0', O0), ('Q+^2Q+^1 O_0', Qplus_expr(Qplus_expr(O0, 0), 1)),
                    ('Q+^3Q+^2Q+^1 O_0', Qplus_expr(Qplus_expr(Qplus_expr(O0, 0), 1), 2))]:
        sec = expr_sector(d); up = (sec[0] + 1,) + sec[1:]; low = (sec[0] - 1,) + sec[1:]
        closed = not np.any(D.points(low).expr_vector(Q_expr(d)))
        r0 = D.rankQ(up); r1 = rank_mod(np.vstack([D._Qimg_rows[up], D.points(sec).expr_vector(d)[None, :]]))
        print(f'seed {seed} margin {margin}: {name} in {sec}: closed {closed}, exact rank {r0}, with it {r1} -> non-exact: {r1 > r0}')
