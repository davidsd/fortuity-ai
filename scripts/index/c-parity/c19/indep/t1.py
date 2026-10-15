import time
from weylcount import LET, count_N2, formula_N3, enumerate_N3
print(LET)
t=time.time(); E = enumerate_N3(13); F = formula_N3(13); print('enum vs formula N3 |q|<=13:', E == F, len(E), sum(E.values()), f'{time.time()-t:.1f}s')
# spot values
for k in [(1,1,1,1),(2,1,1,2),(3,1,2,2),(4,2,2,2),(0,2,0,0),(0,1,1,0),(0,3,0,0)]:
    print(k, 'N3', F.get(k,0), 'N2', count_N2(8).get(k,0))
