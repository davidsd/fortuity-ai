import os as _os  # added on copy into the repo: repo root from this file's location (or $FORT_ROOT)
REPO_ROOT = _os.environ.get('FORT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..')) + '/'
import pickle, json
from collections import defaultdict
from fortuity.bmn.twistedindex import hw_multiplicities
C=REPO_ROOT + 'scripts/index/c-parity/runs/cache/'
R=REPO_ROOT + ''
Q=20
ix=pickle.load(open('su3_index_q20.pkl','rb'))
dims=pickle.load(open(C+'diag_dims_N3_j40.pkl','rb'))['dims']
uns=defaultdict(int)
for s,v in dims.items():
    q=tuple(s[1:])
    if sum(q)<=Q and v: uns[q]+=v
Htw=hw_multiplicities(ix['tw'],Q); Hun=hw_multiplicities(dict(uns),Q)
T={l:(-1)**sum(l)*Htw.get(l,0)-Hun.get(l,0) for l in set(Htw)|set(Hun) if l!=(0,0,0)}
rows=json.load(open(R+'data/su3/kernel_parity_j40.json'))['rows']
datT={tuple(r['lam']):r['T'] for r in rows}
mine={l:t for l,t in T.items() if t}
print('nonzero T columns mine', len(mine), 'data rows with T!=0', sum(1 for t in datT.values() if t))
print('mismatch', [(l,mine.get(l,0),datT.get(l,0)) for l in set(mine)|set(datT) if mine.get(l,0)!=datT.get(l,0)])
print('A = (-1)^|lam| I_C for forced columns:', {l:(-1)**sum(l)*Htw.get(l,0) for l,t in sorted(mine.items()) if t<0})
