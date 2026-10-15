"""Is Q_1 v nonzero as an operator at rank N (generic points), and does it vanish on the Coulomb branch (Cartan
points, where every Q_0-exact operator vanishes)?  Small independent check next to the exactness test."""
import sys, pickle
import numpy as np
from fortuity.bmn.modp import matmul_mod, P
from fortuity.full.evalg import Frame, nferm
N = {'su3_j24': 3, 'su2_j36': 2}[sys.argv[1]]
q1s = pickle.load(open(f'classes/{sys.argv[1]}/q1.pkl', 'rb'))
allw = sorted({w for q in q1s for w in q})
letters = sorted({L for w in allw for n in w for L in n})
degs = sorted({nferm(w) for w in allw})
widx = {w: i for i, w in enumerate(allw)}
for cartan in (False, True):
    frames = [(d, Frame(N, 6, max(d + 4, 6) if d else 0, [d], letters, seed=515 + d, cartan=cartan)) for d in degs]
    ncols = sum(fr.ncols for _, fr in frames)
    X = np.zeros((len(allw), ncols), dtype=np.int64)
    deg = np.array([nferm(w) for w in allw])
    col = 0
    for d, fr in frames:
        idx = np.flatnonzero(deg == d)
        X[idx, col:col + fr.ncols] = fr.eval_words([allw[i] for i in idx])
        col += fr.ncols
    for k, q in enumerate(q1s):
        rows = np.array([widx[w] for w in q]); cs = np.array([c % P for c in q.values()], dtype=np.int64)
        val = matmul_mod(cs[None, :], X[rows])[0]
        print(f"{sys.argv[1]} class {k}: Q_1 v at {'Cartan' if cartan else 'generic'} points: nonzero {int(np.count_nonzero(val))} of {val.size}", flush=True)
