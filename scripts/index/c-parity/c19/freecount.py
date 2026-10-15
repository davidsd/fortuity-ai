import os as _os  # added on copy into the repo: repo root from this file's location (or $FORT_ROOT)
REPO_ROOT = _os.environ.get('FORT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..')) + '/'
"""Second rigorous per-weight upper bound on the graviton-image dimension: number of monomials in the generator
components (GravitonCounter.monomial_counts, no relations; v_n counted before traceless projection, so an overcount)."""
import pickle, sys, json
from types import SimpleNamespace
from fortuity.bmn.gravindex import GravitonCounter
from molien import molien
CACHE = REPO_ROOT + 'scripts/index/c-parity/runs/cache/'
def free_counts(N, jmax):
    return GravitonCounter.monomial_counts(SimpleNamespace(nmax=N), jmax)
if __name__ == '__main__':
    for N, f in [(2, 'diag_dims_N2_j44.pkl'), (3, 'diag_dims_N3_j44.pkl')]:
        dims = pickle.load(open(CACHE + f, 'rb'))['dims']
        A = free_counts(N, 44); M = molien(N, 22)
        viol = [(k, v, A.get(k, 0)) for k, v in dims.items() if v > A.get(k, 0)]
        better = sum(1 for k in M if A.get(k, 0) < M[k])
        tight = sum(1 for k in dims if dims[k] and min(A.get(k, 0), M.get(k, 0)) == dims[k])
        print(f'N={N}: free-count violations (d_comp > A): {len(viol)} {viol[:5]}; weights where A < Molien: {better}; '
              f'weights with d_comp>0 where min(A,Molien)==d_comp: {tight} of {sum(1 for v in dims.values() if v)}')
        pickle.dump(dict(A), open(f'freecount_N{N}_j44.pkl', 'wb'))
