import os as _os  # added on copy into the repo: repo root from this file's location (or $FORT_ROOT)
REPO_ROOT = _os.environ.get('FORT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..')) + '/'
import pickle
from collections import Counter
from molien import molien
from brute import letter_counts
C = REPO_ROOT + 'scripts/index/c-parity/runs/cache/'
for N, f, Q in [(2, 'diag_dims_N2_j44.pkl', 22), (3, 'diag_dims_N3_j40.pkl', 20)]:
    dims = pickle.load(open(C + f, 'rb'))['dims']; M = molien(N, Q)
    gaps = [(k, dims.get(k, 0), v) for k, v in M.items() if v > dims.get(k, 0)]
    stat = Counter()
    for k, dc, du in gaps:
        lc = letter_counts(k[0], k[1:])
        lc = [c for c in lc if max(c[3:6]) <= N - 1]          # fermion type can appear at most N-1 times
        minf = min(c[6] for c in lc)
        stat[('min n_f', minf)] += 1
    stat2 = Counter()
    for k, dc, du in gaps:
        stat2[('2J', k[0], 'd_comp=0' if dc == 0 else 'd_comp>0')] += 1
    print(f'N={N}: {len(gaps)} gap weights; by minimal number of f letters any monomial at that weight needs: '
          f'{sorted(stat.items())}')
    print('   smallest gap weights:', sorted(gaps, key=lambda g: (sum(g[0][1:]), g[0]))[:8])
