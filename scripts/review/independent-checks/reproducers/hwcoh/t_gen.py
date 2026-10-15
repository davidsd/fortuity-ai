import time
from fortuity.hwcoh.dims import partition_dims_nf
from fortuity.hwcoh.sector import Sector
from fortuity.hwcoh.gravhw import GravitonAlgebra
t = time.time()
dso = partition_dims_nf(3, 8, group='SO_odd'); dg2 = partition_dims_nf(0, 12, group='G2')
print('dims', round(time.time()-t,1))
for grp, N, dims, lam, n in [('SO_odd', 3, dso, (4,0,0), 0), ('G2', 0, dg2, (6,0,0), 0), ('G2', 0, dg2, (2,0,0), 0)]:
    for gmax in (N, 8):
        GA = GravitonAlgebra(gmax)
        r = Sector(N, lam, n, dims, log=lambda s: None, graviton_algebra=GA, group=grp).run()
        lo = 0
        print(grp, 'N', N, lam, '2J', n, 'GEN_NMAX', gmax, '-> m', r['m'], 'r_up', r['r_up'], 'h', r['m'] - r['r_up'], 'grav', r['grav'], 'BH', r['m'] - r['r_up'] - r['grav'], 'check', r['check'])
