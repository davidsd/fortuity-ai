import pickle, time
import numpy as np
from fortuity.hwcoh.sector import Sector
from fortuity.hwcoh.gravhw import GravitonAlgebra
from fortuity.hwcoh.classreps import hw_nongraviton_reps
from fortuity.bmn.words import Q_expr
from fortuity.bmn.modp import P
dims = pickle.load(open('dims_su3_20.pkl','rb'))
GA = GravitonAlgebra(3)
N, lam, n = 3, (4,4,2), 3
res = {k: Sector(N, lam, k, dims, log=lambda s: None, graviton_algebra=GA).run() for k in (n-1, n)}
h = res[n]['m'] - res[n-1]['r_up'] - res[n]['r_up']
print('engine: m', res[n]['m'], 'r_n', res[n-1]['r_up'], 'r_{n+1}', res[n]['r_up'], 'h', h, 'grav', res[n]['grav'], 'BH', h - res[n]['grav'], 'checks', res[n-1]['check'], res[n]['check'])
exprs, info = hw_nongraviton_reps(N, 'SU', lam, n, dims, GA, log=lambda s: None)
print('classreps info:', info)
# test Q-closedness of returned reps by evaluation at random SU(3) points via a DegFrames evaluation of Q(rep)
from fortuity.hwcoh.evalg import DegFrames, nferm
for e in exprs:
    q = Q_expr({m: c for m, c in e.items()})
    q = {m: c % P for m, c in q.items() if c % P}
    ws = list(q)
    need = {d: 400 for d in sorted({nferm(w) for w in ws})}
    fr = DegFrames(N, need, target=12, seed=99)
    X = fr.eval_words(ws)
    v = np.zeros(X.shape[1], dtype=object)
    vals = (np.array([q[w] for w in ws], dtype=object)[:, None] * X.astype(object)).sum(axis=0) % P
    print('rep with', len(e), 'terms: Q(rep) formally', len(q), 'terms; nonzero at random SU(3) points:', bool(np.any(vals != 0)))
