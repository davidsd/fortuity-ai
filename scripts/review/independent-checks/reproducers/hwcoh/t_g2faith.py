import pickle, time
from fortuity.hwcoh.sector import Sector
from fortuity.hwcoh.classreps import _faithful
from fortuity.hwcoh.restrict import _lower_q_rows
from fortuity.hwcoh import ffla
dg = pickle.load(open('dims_g2_22.pkl','rb'))
t = time.time()
for lam, n in [((4,4,3), 3)]:
    S = Sector(0, lam, n, dg, log=lambda s: None, group='G2', seed=2)   # restrict_sector uses seed+1 = 2 for H
    S.prepare()
    A, B, G = S.evaluate()
    ok, rk = _faithful(S, A)
    print('G2', lam, n, 'm', S.m, 'm_d', S.m_d, 'faithful at attempt 0:', ok, rk, round(time.time()-t,1), flush=True)
    L = _lower_q_rows(S, dg)
    lo = Sector(0, lam, n-1, dg, log=lambda s: None, group='G2').run()
    print('  rank lower Q rows', ffla.rank(L), 'certified r_n', lo['r_up'], lo['check'], round(time.time()-t,1))
