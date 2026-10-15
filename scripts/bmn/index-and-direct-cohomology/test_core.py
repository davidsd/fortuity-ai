import numpy as np, time
from collections import defaultdict
from fortuity.bmn.words import *
from fortuity.bmn.evaluate import Points

def vec(expr, N, seed=1, npts=6):
    sec = expr_sector(expr)
    degs = sorted({sum(content(m)[3:6]) for m in expr})
    k = max(degs)
    pts = Points(N, npts, max(k,1), degs, seed=seed)
    return pts.expr_vector(expr)

# 1. Q^2 = 0 on a bunch of random multi-trace words, for N=2,3
rng = np.random.default_rng(0)
for N in (2,3):
    for trial in range(20):
        L = rng.integers(2,6); w = tuple(rng.integers(0,7,L))
        L2 = rng.integers(2,5); w2 = tuple(rng.integers(0,7,L2))
        s,cw = canon_word(w); s2,cw2 = canon_word(w2)
        if not (s and s2): continue
        sgn, m = canon_multi((cw,cw2))
        if not sgn: continue
        e = {m:1}
        qq = Q_expr(Q_expr(e))
        if qq:
            v = vec(qq, N)
            assert not np.any(v), (N, m, qq)
print("Q^2 = 0 OK")

# 2. closure of generators; determine psipsi coefficient in w
for coeff in (1,-1,2,-2):
    gens = graviton_generators(3, coeff)
    bad = 0
    for name,e in gens:
        q = Q_expr(e)
        if q:
            for N in (2,3):
                if np.any(vec(q,N)): bad += 1
    print("coeff",coeff,"non-closed count:",bad, "of", len(gens))
