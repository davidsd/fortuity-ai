import os as _os  # added on copy into the repo: repo root from this file's location (or $FORT_ROOT)
REPO_ROOT = _os.environ.get('FORT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..')) + '/'
import pickle, math, sys
from collections import defaultdict
C=REPO_ROOT + 'scripts/index/c-parity/runs/cache/'
o=pickle.load(open(C+'diag_dims_N3_j40.pkl','rb'))['dims']
def cpp(k,tJ): return sum(math.comb(k,d) for d in range(tJ%2,tJ+1,2) if d<=k)
for f in sys.argv[1:]:
    r=pickle.load(open(f,'rb')); d=r['dims']; info=r['info']
    diff=[(s,d.get(s,0),o.get(s,0)) for s in set(d)|set(o) if d.get(s,0)!=o.get(s,0)]
    print('==',f,'seed',r['seed'],'p',r['p'],'sectors',len(d),'dims differing from cached pickle:',len(diff),diff[:5])
    rat=[]; full=0; tot_f=0
    for s,(dim,ncols,rank) in info.items():
        if ncols is None: continue
        assert rank==dim
        rat.append((dim/ncols,dim,ncols,ncols//cpp(6,s[0]),s))
        if ncols//cpp(6,s[0])>=4096: full+=1
    rat.sort(reverse=True)
    print(' sectors instrumented', len(rat), ' max dim/functionals', [(round(a,3),b,c,n,s) for a,b,c,n,s in rat[:6]])
    print(' sectors that used all 4096 points:', full)
    big=sorted(rat,key=lambda x:-x[1])[:5]
    print(' largest dims: (ratio, dim, functionals, points, sector)', [(round(a,3),b,c,n,s) for a,b,c,n,s in big])
    hist=defaultdict(int)
    for a,*_ in rat: hist[min(9,int(a*10))]+=1
    print(' histogram of dim/functionals by tenths:', dict(sorted(hist.items())))
