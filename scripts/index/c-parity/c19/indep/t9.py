import os as _os  # added on copy into the repo: repo root from this file's location (or $FORT_ROOT)
REPO_ROOT = _os.environ.get('FORT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..', '..')) + '/'
import pickle
o = pickle.load(open('t4_out.pkl','rb')); U = pickle.load(open('my_dup_q22.pkl','rb'))['N3']
C = REPO_ROOT + 'scripts/index/c-parity/runs/cache/'
dims = pickle.load(open(C+'diag_dims_N3_j40.pkl','rb'))['dims']
MINUS = [(0,1,-1),(1,-1,0),(2,0,-2)]
ws = set()
for r in o['o40']:
    lam = r['lam']
    for sh in MINUS: ws.add(tuple(lam[i]+sh[i] for i in range(3)))
n2J = {w: [t for t in range(0, 30) if U.get((t,)+w,0) or dims.get((t,)+w,0)] for w in ws}
print('distinct minus weights (j<=40):', len(ws), ' max #2J values with nonzero d_up or d_comp:', max(len(v) for v in n2J.values()))
print('max #2J with a gap:', max(sum(1 for t in v if U.get((t,)+w,0) > dims.get((t,)+w,0)) for w, v in n2J.items()))
print('max #2J with d_comp>0:', max(sum(1 for t in v if dims.get((t,)+w,0)) for w, v in n2J.items()),
      ' max #2J with d_up>0:', max(sum(1 for t in v if U.get((t,)+w,0)) for w, v in n2J.items()))
