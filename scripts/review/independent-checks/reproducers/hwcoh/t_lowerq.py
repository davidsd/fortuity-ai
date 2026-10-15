import time, sys, pickle, os
import numpy as np
from fortuity.hwcoh.dims import partition_dims_nf
from fortuity.hwcoh.sector import Sector, hw_mult_nf
from fortuity.hwcoh.restrict import _lower_q_rows
from fortuity.hwcoh import ffla
fn = 'dims_su3_20.pkl'
if os.path.exists(fn): dims = pickle.load(open(fn,'rb'))
else:
    dims = partition_dims_nf(3, 20); pickle.dump(dims, open(fn,'wb'))
N = 3
cases = [((4,3,3),3), ((4,3,3),2), ((4,4,2),3), ((5,3,2),3), ((4,4,2),4), ((5,3,2),4), ((4,3,3),4)]
for lam, n in cases:
    t = time.time()
    lo = Sector(N, lam, n-1, dims, log=lambda s: None).run()     # certified r_up(n-1) = rank Q: C_n -> C_{n-1}
    S = Sector(N, lam, n, dims, log=lambda s: None)
    S.prepare()
    L = _lower_q_rows(S, dims)
    rL = ffla.rank(L) if L.size else 0
    md = {d: hw_mult_nf(dims, n-1, lam, (n-1-d)//2) for d in range(n) if (n-1-d)%2==0}
    print(lam, n, 'lower m_d', md, 'certified r_n =', lo['r_up'], '(check', lo['check'], ') rank(_lower_q_rows) =', rL, 'ncols', L.shape, f'{time.time()-t:.1f}s', flush=True)
