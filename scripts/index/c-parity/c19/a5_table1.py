import os as _os  # added on copy into the repo: repo root from this file's location (or $FORT_ROOT)
REPO_ROOT = _os.environ.get('FORT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..')) + '/'
import pickle, json
from collections import defaultdict
from fortuity.bmn.twistedindex import hw_multiplicities
C=REPO_ROOT + 'scripts/index/c-parity/runs/cache/'
R=REPO_ROOT + ''
Q=20
pl=pickle.load(open('su3_index_q20.pkl','rb'))['pl']
dims=pickle.load(open(C+'diag_dims_N3_j40.pkl','rb'))['dims']
sig=defaultdict(int)
for s,v in dims.items():
    q=tuple(s[1:])
    if sum(q)<=Q: sig[q]+=(-1)**s[0]*v
diff=defaultdict(int)
for q in set(pl)|set(sig):
    diff[q]=pl.get(q,0)-sig.get(q,0)
def dyn(l): return f'[{l[0]-l[1]},{l[1]-l[2]}]'
H=hw_multiplicities({k:v for k,v in diff.items() if v},Q)
ref=json.load(open(R+'data/su3/bmn_index_j36.json'))['nongraviton_index_su3_reps']
print('Z - Zgrav per SU(3)_R rep (signed), from the pickle vs data (j<=36):')
for j in range(24,2*Q+1,2):
    mine={dyn(l):c for l,c in H.items() if 2*sum(l)==j}
    print(' j',j, mine, '| data', ref.get(str(j),'-'), '' if str(j) not in ref else ('OK' if mine==ref[str(j)] else 'MISMATCH'))
# core: multiply by prod_I 1/(1-t^2 x_I) and prod_I (1 - t^8 x_I)
cur=dict(diff)
E=[(1,0,0),(0,1,0),(0,0,1)]
for e in E:
    new=defaultdict(int)
    for q,v in cur.items():
        if not v: continue
        k=0
        while sum(q)+k<=Q:
            new[tuple(a+k*b for a,b in zip(q,e))]+=v; k+=1
    cur=new
for e in E:
    y=tuple(1+b for b in e)
    new=defaultdict(int)
    for q,v in cur.items():
        new[q]+=v
        qq=tuple(a+b for a,b in zip(q,y))
        if sum(qq)<=Q: new[qq]-=v
    cur=new
Hc=hw_multiplicities({k:v for k,v in cur.items() if v},Q)
print('core content (after descendant and w2 factors):')
for j in range(24,2*Q+1,2):
    print(' j',j, {dyn(l):c for l,c in sorted(Hc.items()) if 2*sum(l)==j})
