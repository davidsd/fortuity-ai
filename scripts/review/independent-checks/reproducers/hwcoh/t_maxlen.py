import time
from fortuity.hwcoh.dims import partition_dims_nf
from fortuity.hwcoh.sector import Sector, hw_mult_nf
from fortuity.hwcoh.gravhw import GravitonAlgebra
t=time.time()
dims = partition_dims_nf(3, 20)
print('dims', time.time()-t)
GA = GravitonAlgebra(3)
for lam, n in [((5,5,0),5), ((5,5,0),4), ((5,5,0),3)]:
    exact = {d: hw_mult_nf(dims, n, lam, (n-d)//2) for d in range(n+1) if (n-d)%2==0}
    for ml in (None, 4):
        r = Sector(3, lam, n, dims, log=lambda s: None, graviton_algebra=GA, maxlen=ml).run()
        print(lam, n, 'maxlen', ml, 'exact m_d', {d:v for d,v in exact.items() if v}, '-> m', r['m'], 'm_d', r['m_d'], 'check', r['check'], 'r_up', r['r_up'], 'grav', r['grav'], 'words', r['words'])
