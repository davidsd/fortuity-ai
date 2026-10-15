import os as _os  # added on copy into the repo: repo root from this file's location (or $FORT_ROOT)
REPO_ROOT = _os.environ.get('FORT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..')) + '/'
import sympy as sp, pickle
from collections import defaultdict
t=sp.symbols('t')
num = 1+3*t**4-8*t**6-6*t**10+10*t**12+9*t**14-9*t**16+16*t**18-18*t**20-3*t**22+t**24-3*t**26+9*t**28-2*t**30+3*t**32-3*t**34
jmax=48
Zg = sp.series(num/((1-t**4)**3*(1-t**8)**3), t, 0, jmax+1).removeO()
zg = [int(Zg.coeff(t, j)) for j in range(jmax+1)]
d=pickle.load(open(REPO_ROOT + 'scripts/index/c-parity/runs/cache/diag_dims_N2_j44.pkl','rb'))['dims']
lev=defaultdict(int)
for s,v in d.items(): lev[2*sum(s[1:])]+=(-1)**s[0]*v
print('j  closed(3.4)  computed-signed')
for j in range(0,45,2): print(j, zg[j], lev[j], 'OK' if zg[j]==lev[j] else 'MISMATCH')
