"""SU(3) BMN, SU(3)_R singlet column lam = (6,6,6) at j = 36: hw cohomology and graviton classes per 2J.

Question (Table 1 of arXiv:2312.16443, tower F_0): is the fermionic [0,0] entry at j = 36 a graviton, like the ones
at j = 24 (R_24^{(0,3)}) and j = 30 (R_30^{(0,5)})?  The diagonal graviton count of this column is 0 (R-017), so every
graviton class here vanishes on the diagonal.  Per sector n = 2J, with the hw complex of fortuity.hwcoh
(as hw_relation_test.py):  m = hw multiplicity (certified by the faithfulness check), r_up(n) = rank of Q(hw of n+1),
grav(n) = rank [B; G] - rank B = graviton classes.  Then h(n) = m(n) - r_up(n) - r_up(n-1) and BH(n) = h(n) - grav(n).
usage: python3 singlet_column_j36.py [list of 2J] -> runs/singlet_j36_n<2J>.json (one file per sector)"""
import sys, json, time, pickle, os
import numpy as np
from fortuity.hwcoh.sector import Sector
from fortuity.hwcoh.gravhw import GravitonAlgebra
from fortuity.hwcoh import ffla

LAM = (6, 6, 6)
dims = pickle.load(open('runs/dims_nf_36.pkl', 'rb'))
order = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [7, 6, 5, 8, 4, 9, 3, 10, 2, 11, 1, 12, 0]
for n in order:
    out = 'runs/singlet_j36_n%d.json' % n
    if os.path.exists(out):
        continue
    t0 = time.time()
    log = lambda *a: print('[n=%d %6.0fs]' % (n, time.time() - t0), *a, flush=True)
    S = Sector(3, LAM, n, dims, seed=1, log=lambda s: log(s), graviton_algebra=GravitonAlgebra(3), maxlen=6)
    S.prepare()
    A, B, G = S.evaluate()
    ok, col, rk = True, 0, {}
    for d in sorted(S.m_d):
        w = sum(fr.ncols for dd, fr in S.colblocks if dd == d)
        rows = np.flatnonzero(A[:, col:col + w].any(axis=1))
        rk[d] = int(ffla.rank(A[rows, col:col + w])) if rows.size else 0
        ok &= (rk[d] == S.m_d[d])
        col += w
    r_up = int(ffla.rank(B)) if B.shape[0] else 0
    rg = int(ffla.rank(np.concatenate([B, G], axis=0))) if G.shape[0] else r_up
    res = dict(lam=LAM, twoJ=n, m=int(sum(S.m_d.values())), m_d={int(k): int(v) for k, v in S.m_d.items()}, faithful=bool(ok),
               r_up=r_up, grav=rg - r_up, n_graviton_rows=int(G.shape[0]), time=round(time.time() - t0))
    log(json.dumps(res))
    json.dump(res, open(out, 'w'), indent=1)
