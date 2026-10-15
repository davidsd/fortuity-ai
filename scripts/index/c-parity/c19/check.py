import os as _os  # added on copy into the repo: repo root from this file's location (or $FORT_ROOT)
REPO_ROOT = _os.environ.get('FORT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..')) + '/'
# Read-only checks for C19(a): sign of T in D, and S3 symmetry / hw nonnegativity of cached Coulomb dims.
import pickle, itertools, json
from fortuity.bmn.twistedindex import hw_multiplicities
C=REPO_ROOT + 'scripts/index/c-parity/runs/cache/'
# 1. toy: per-hw T monotone decreasing in D_hw; signed-sum error propagation
terms = [(0, 0, 1), (1, 0, -1), (0, 1, -1), (2, 1, 1), (1, 2, 1), (2, 2, -1)]
print('hw extraction coefficients (i*a12 + j*a23, sign):', terms)
for name, Q in [('diag_dims_N3_j36.pkl', 18), ('diag_dims_N3_j40.pkl', 20), ('diag_dims_N3_j44.pkl', 22), ('diag_dims_N2_j44.pkl', 22), ('diag_dims_N4_j28.pkl', 14)]:
    try:
        d = pickle.load(open(C + name, 'rb'))
    except Exception as e:
        print(name, 'ERR', e); continue
    dims = d['dims'] if isinstance(d, dict) and 'dims' in d else d
    # per (2J, q) S3 symmetry
    asym = []
    for sec, v in dims.items():
        if sum(sec[1:]) > Q: continue
        for p in itertools.permutations(sec[1:]):
            w = dims.get((sec[0],) + p)
            if w is not None and w != v:
                asym.append((sec, p, v, w))
    missing = [ (sec,p) for sec in dims if sum(sec[1:])<=Q for p in itertools.permutations(sec[1:]) if (sec[0],)+p not in dims]
    uns = {}
    for sec, v in dims.items():
        q = tuple(sec[1:])
        if sum(q) <= Q and v: uns[q] = uns.get(q, 0) + v
    H = hw_multiplicities(uns, Q)
    neg = {l: c for l, c in H.items() if c < 0}
    # per-2J hw multiplicities too
    neg2 = []
    for twoJ in sorted({s[0] for s in dims}):
        w = {tuple(s[1:]): v for s, v in dims.items() if s[0] == twoJ and sum(s[1:]) <= Q and v}
        for l, c in hw_multiplicities(w, Q).items():
            if c < 0: neg2.append((twoJ, l, c))
    print(name, 'Qmax', Q, 'sectors', sum(1 for s in dims if sum(s[1:]) <= Q), 'S3-asym', len(asym), asym[:3],
          'missing perms', len(missing), '| negative hw mult (all 2J):', len(neg), '| negative per-2J hw:', len(neg2), neg2[:3])
