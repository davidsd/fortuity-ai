import pickle
from fortuity.hwcoh.dims import partition_dims_nf
from fortuity.hwcoh.sector import Sector
from fortuity.hwcoh.gravhw import GravitonAlgebra
d14 = partition_dims_nf(3, 14)
d20 = pickle.load(open('dims_su3_20.pkl', 'rb'))
GA = GravitonAlgebra(3)
for lam, n in [((4,4,2), 2), ((4,4,2), 3)]:
    for name, dims in (('dims j<=14', d14), ('dims j<=20', d20)):
        r = Sector(3, lam, n, dims, log=lambda s: None, graviton_algebra=GA).run()
        print(lam, n, name, '-> m', r['m'], 'check', r['check'], 'r_up', r['r_up'], 'grav', r['grav'], 'words', r['words'])
