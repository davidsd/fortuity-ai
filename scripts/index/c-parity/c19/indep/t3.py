import os as _os  # added on copy into the repo: repo root from this file's location (or $FORT_ROOT)
REPO_ROOT = _os.environ.get('FORT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..', '..')) + '/'
import pickle
from collections import defaultdict
C = REPO_ROOT + 'scripts/index/c-parity/runs/cache/'
U = pickle.load(open('my_dup_q22.pkl','rb'))
for N, f in [(2,'diag_dims_N2_j44.pkl'), (3,'diag_dims_N3_j36.pkl'), (3,'diag_dims_N3_j40.pkl'), (3,'diag_dims_N3_j44.pkl')]:
    ck = pickle.load(open(C+f,'rb'))
    dims = ck['dims']
    up = U[f'N{N}']
    keys = list(dims)
    qmax = max(sum(k[1:]) for k in keys)
    # completeness: every (2J,q) with |q|<=qmax, 0<=2J<=|q| present?
    expected = {(t,a,b,s-a-b) for s in range(qmax+1) for a in range(s+1) for b in range(s+1-a) for t in range(s+1)}
    missing = expected - set(keys)
    extra = set(keys) - expected
    viol = [(k, dims[k], up.get(k,0)) for k in keys if dims[k] > up.get(k,0)]
    # Molien nonzero at 2J > |q|? (cache only covers 2J<=|q|)
    beyond = [k for k in up if sum(k[1:])<=qmax and k[0] > sum(k[1:])]
    gap = sum(1 for k in keys if up.get(k,0) > dims[k])
    eq0 = all(dims.get(k,0) == v for k, v in up.items() if k[0]==0 and sum(k[1:])<=qmax)
    neg = [k for k in keys if dims[k] < 0]
    print(f'N={N} {f}: keys={len(keys)} qmax={qmax} other ckpt fields={[x for x in ck if x!="dims"]}; missing={len(missing)} extra={len(extra)}; '
          f'violations d_comp>d_up: {len(viol)} {viol[:3]}; gaps {gap}; d_up nonzero with 2J>|q|: {len(beyond)}; 2J=0 exact: {eq0}; neg {len(neg)}')
# j40 vs j44 agreement
a = pickle.load(open(C+'diag_dims_N3_j40.pkl','rb'))['dims']; b = pickle.load(open(C+'diag_dims_N3_j44.pkl','rb'))['dims']
c = pickle.load(open(C+'diag_dims_N3_j36.pkl','rb'))['dims']
print('j40 vs j44 disagreements at |q|<=20:', sum(1 for k in a if a[k] != b.get(k)), ' j36 vs j44 at |q|<=18:', sum(1 for k in c if c[k] != b.get(k)))
