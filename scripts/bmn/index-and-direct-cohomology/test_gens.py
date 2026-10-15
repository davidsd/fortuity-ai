import numpy as np
from fortuity.bmn.words import *
from fortuity.bmn.evaluate import Points
def vec(expr, N, seed=1, npts=6):
    degs = sorted({sum(content(m)[3:6]) for m in expr}); k = max(degs)
    return Points(N, npts, max(k,1), degs, seed=seed).expr_vector(expr)
for coeff in (1,):
    for name,e in graviton_generators(3, coeff):
        q = Q_expr(e)
        res = []
        for N in (2,3):
            res.append(bool(np.any(vec(q,N))) if q else False)
        if any(res): print(name, res, "nterms", len(e))
