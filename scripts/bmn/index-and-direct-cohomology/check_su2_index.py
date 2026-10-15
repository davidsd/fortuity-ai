import json, numpy as np, time
from itertools import permutations
from fortuity.bmn.fullindex import full_index, su3_multiplicities
jmax = 28
t=time.time(); Z, A = full_index(2, jmax, verbose=False); print("full index time", time.time()-t)
# unrefined coefficients
unref = Z.sum(axis=(1,2))
print("SU(2) full BMN index (unrefined):", list(unref))
# Korean closed form for Z_grav SU(2)
import sympy as sp
t_ = sp.symbols('t')
num = 1+3*t_**4-8*t_**6-6*t_**10+10*t_**12+9*t_**14-9*t_**16+16*t_**18-18*t_**20-3*t_**22+t_**24-3*t_**26+9*t_**28-2*t_**30+3*t_**32-3*t_**34
Zg = sp.series(num/((1-t_**4)**3*(1-t_**8)**3), t_, 0, jmax+1).removeO()
zg = [int(Zg.coeff(t_, j)) for j in range(jmax+1)]
print("Korean Z_grav (3.4)              :", zg)
print("difference Z - Z_grav            :", [int(a-b) for a,b in zip(unref, zg)])
# direct counts, signed
import os   # R-001 direct cohomology as stored in data/
res = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "su2",
                                  "bmn_direct_cohomology_j28.json")))["sectors"]
from collections import defaultdict
sig = defaultdict(int); grav = defaultdict(int)
for r in res:
    sec = r['sector']; mult = len(set(permutations(sec[1:]))); s = (-1)**sec[0]
    sig[r['j']] += mult*s*r['coh']; grav[r['j']] += mult*s*r['grav']
print("direct signed coh                :", [sig[j] for j in range(jmax+1)])
print("direct signed grav               :", [grav[j] for j in range(jmax+1)])
for j in (24,26,28):
    print(j, su3_multiplicities(Z[j], A))
