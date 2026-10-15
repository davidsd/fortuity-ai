import os as _os  # added on copy into the repo: repo root from this file's location (or $FORT_ROOT)
REPO_ROOT = _os.environ.get('FORT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..')) + '/'
import pickle, math, json, itertools
from collections import defaultdict
C=REPO_ROOT + 'scripts/index/c-parity/runs/cache/'
R=REPO_ROOT + ''
def cpp(k, twoJ):
    return sum(math.comb(k,d) for d in range(twoJ%2, twoJ+1, 2) if d<=k)
for name,N,k in [('diag_dims_N3_j40.pkl',3,6),('diag_dims_N2_j44.pkl',2,3),('diag_dims_N3_j44.pkl',3,6),('diag_dims_N3_j36.pkl',3,6)]:
    d=pickle.load(open(C+name,'rb'))
    dims=d['dims']
    print('==',name,'N',d['N'],'jmax',d['jmax'],'nsectors',len(dims))
    # max dim, ratio
    best=[]
    for s,v in dims.items():
        tJ=s[0]; c=cpp(k,tJ); best.append((v/(4096*c), v, c, s))
    best.sort(reverse=True)
    print(' max dim', max(dims.values()), 'top ratios dim/(4096*cpp):', [(round(r,4),v,c,s) for r,v,c,s in best[:5]])
    # per 2J max dims
    m=defaultdict(int)
    for s,v in dims.items(): m[s[0]]=max(m[s[0]],v)
    print(' max dim per 2J:', dict(sorted(m.items())))
    # Weyl symmetry check: dims invariant under permutations of q
    bad=0; tot=0
    for s,v in dims.items():
        for p in itertools.permutations(s[1:]):
            t=(s[0],)+p
            if t!=s:
                tot+=1
                if dims.get(t,0)!=v: bad+=1
    print(' S3 permutation asymmetries:', bad, 'of', tot)
    # level totals
    lev=defaultdict(int); levs=defaultdict(int)
    for s,v in dims.items():
        j=2*sum(s[1:]); lev[j]+=v; levs[j]+=(-1)**s[0]*v
    print(' unsigned per level (last 6):', [(j,lev[j]) for j in sorted(lev)][-6:])
    print(' signed per level (last 6):', [(j,levs[j]) for j in sorted(levs)][-6:])
    if N==2:
        oddY=[s for s,v in dims.items() if v and (sum(s[1:])-s[0])%2]
        print(' SU(2) sectors with odd Y and nonzero dim:', len(oddY))
    ref={2:'data/su2/bmn_index_j40.json',3:'data/su3/bmn_index_j36.json'}[N]
    refg=json.load(open(R+ref))['graviton_index_unrefined']
    mism=[(j,levs[j],refg[j]) for j in range(0,len(refg),2) if levs[j]!=refg[j]]
    print(' signed level index vs data', ref, 'mismatches:', mism, ' compared up to j=', len(refg)-1)
