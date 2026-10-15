import os as _os  # added on copy into the repo: repo root from this file's location (or $FORT_ROOT)
REPO_ROOT = _os.environ.get('FORT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..')) + '/'
import pickle, json
from collections import defaultdict
from fortuity.bmn.twistedindex import hw_multiplicities
C=REPO_ROOT + 'scripts/index/c-parity/runs/cache/'
R=REPO_ROOT + ''
terms = [(0, 0, 1), (1, 0, -1), (0, 1, -1), (2, 1, 1), (1, 2, 1), (2, 2, -1)]
def shift(lam,i,j): return (lam[0]+i, lam[1]-i+j, lam[2]-j)
for pk, Q, datafile in [('diag_dims_N3_j40.pkl',20,'data/su3/kernel_parity_j40.json'),('diag_dims_N2_j44.pkl',22,'data/su2/kernel_parity_j44.json')]:
    dims=pickle.load(open(C+pk,'rb'))['dims']
    uns=defaultdict(int); per2J=defaultdict(lambda: defaultdict(int))
    for s,v in dims.items():
        q=tuple(s[1:])
        if sum(q)<=Q and v:
            uns[q]+=v; per2J[s[0]][q]+=v
    H=hw_multiplicities(dict(uns),Q)
    # per-2J hw multiplicities must be >= 0 (each 2J sector is an SU(3)_R rep)
    neg=[]
    for tJ,w in per2J.items():
        h=hw_multiplicities(dict(w),Q)
        neg += [(tJ,l,c) for l,c in h.items() if c<0]
    print('==',pk,' negative per-2J hw multiplicities:', len(neg), neg[:5])
    rows=json.load(open(R+datafile))['rows']
    print(' data rows with T<0; D recomputed from pickle; six weights (sign, unsigned dim summed over 2J):')
    for r in rows:
        if r['T']>=0: continue
        lam=tuple(r['lam'])
        Dre=H.get(lam,0)
        if 'diag_unsigned' in r: assert Dre==r['diag_unsigned'],(lam,Dre,r)
        six=[(sg, uns.get(shift(lam,i,j),0)) for i,j,sg in terms]
        negsum=sum(v for sg,v in six if sg<0)
        print(f"  j={r['j']} {lam} T={r['T']:3d} D={Dre:4d}  terms={six}  sum of (-) weights={negsum}")
