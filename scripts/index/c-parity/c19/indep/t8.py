import os as _os  # added on copy into the repo: repo root from this file's location (or $FORT_ROOT)
REPO_ROOT = _os.environ.get('FORT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..', '..')) + '/'
import pickle
from types import SimpleNamespace
from collections import defaultdict
from fortuity.bmn.gravindex import GravitonCounter
from weylcount import count_vectors
C = REPO_ROOT + 'scripts/index/c-parity/runs/cache/'
U = pickle.load(open('my_dup_q22.pkl','rb'))
for N, f in [(2,'diag_dims_N2_j44.pkl'), (3,'diag_dims_N3_j40.pkl'), (3,'diag_dims_N3_j44.pkl')]:
    dims = pickle.load(open(C+f,'rb'))['dims']; up = U[f'N{N}']; Q = max(sum(k[1:]) for k in dims)
    per = defaultdict(lambda: [0,0,0])
    for k in dims:
        g = up.get(k,0) - dims[k]
        if up.get(k,0): per[k[0]][2] += 1
        if g: per[k[0]][0] += 1; per[k[0]][1] += g
    print(N, f, dict(sorted((k, tuple(v)) for k, v in per.items())))
    # does the weight admit an f-free invariant-carrying count vector (nf=0) with nonzero ... simply: admits a letter count vector with nf=0
    nf0 = set(key for n, key in count_vectors(N, Q) if n[6] == 0)
    gapw = [k for k in dims if up.get(k,0) > dims[k]]
    print('   gap weights', len(gapw), ' admitting an f-free count vector:', sum(1 for k in gapw if k in nf0))
    A = GravitonCounter.monomial_counts(SimpleNamespace(nmax=N), 2*Q)
    nz = [k for k in dims if dims[k]]
    print('   nonzero d_comp weights', len(nz), ' with min(Molien,free)==d_comp:', sum(1 for k in nz if min(up.get(k,0), A.get(k,0)) == dims[k]),
          ' free-count violations:', sum(1 for k in dims if dims[k] > A.get(k,0)))
