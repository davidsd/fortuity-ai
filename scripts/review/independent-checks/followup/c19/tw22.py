import os as _os, sys as _sys
REPO = (_os.environ.get('REPO') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *['..'] * 5))).rstrip('/') + '/'
_sys.path.insert(0, REPO + 'code'); _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
# Library twisted SU(3) index at |q|<=22 (read-only use of repo code): the six weights of (9,6,6) and (10,6,6),
# hw values, and the list of columns with (-1)^|lam| I_C^hw < 0.
import time
from fortuity.bmn.twistedindex import SUN, integrate, to_int, hw_multiplicities
t0=time.time()
tw, err = to_int(integrate(SUN(3), 22, True)[0])
print('rounding err', err, 'time', round(time.time()-t0,1), 'nonzero weights', len(tw))
terms=[(0,0,1),(1,0,-1),(0,1,-1),(2,1,1),(1,2,1),(2,2,-1)]
for lam in [(9,6,6),(10,6,6),(7,4,4),(8,5,4)]:
    ws=[(tuple(l+i*a+j*b for l,a,b in zip(lam,(1,-1,0),(0,1,-1))),s) for i,j,s in terms]
    print(lam, [(w,s,tw.get(w,0)) for w,s in ws])
H=hw_multiplicities(tw,22)
neg=sorted([(2*sum(l),l,(-1)**sum(l)*c) for l,c in H.items() if (-1)**sum(l)*c<0])
print('columns with (-1)^|lam| I_C^hw < 0:', neg)
