import os as _os  # added on copy into the repo: repo root from this file's location (or $FORT_ROOT)
REPO_ROOT = _os.environ.get('FORT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..')) + '/'
import pickle, json, sys
from collections import Counter, defaultdict
from molien import molien
CACHE = REPO_ROOT + 'scripts/index/c-parity/runs/cache/'
out = {}
for N, f in [(2, 'diag_dims_N2_j44.pkl'), (3, 'diag_dims_N3_j40.pkl'), (3, 'diag_dims_N3_j44.pkl')]:
    d = pickle.load(open(CACHE + f, 'rb'))
    dims = d['dims']; Q = d['jmax'] // 2
    M = molien(N, Q)
    viol, gap, eq = [], [], 0
    keys = set(dims) | set(M)
    for k in keys:
        if sum(k[1:]) > Q: continue
        dc, du = dims.get(k, 0), M.get(k, 0)
        if k not in dims and du:
            print('  weight missing from cache with d_up>0:', k, du)
        if du < dc: viol.append((k, dc, du))
        elif du > dc: gap.append((k, dc, du))
        else: eq += 1
    by2J = defaultdict(lambda: [0, 0, 0])   # 2J -> [#weights gap, sum gap, #weights total nonzero d_up]
    for k, dc, du in gap:
        by2J[k[0]][0] += 1; by2J[k[0]][1] += du - dc
    for k, du in M.items():
        if sum(k[1:]) <= Q and du: by2J[k[0]][2] += 1
    print(f'N={N} {f}: cached weights {len(dims)}, molien nonzero {len(M)}; violations d_up<d_comp: {len(viol)}; '
          f'd_up>d_comp at {len(gap)} weights; equal at {eq}')
    print('   violations:', viol[:10])
    print('   2J : #weights with gap, total gap, #weights with d_up>0  ->', {k: tuple(v) for k, v in sorted(by2J.items())})
    lowgap = sorted(gap, key=lambda x: (sum(x[0][1:]), x[0]))[:12]
    print('   lowest |q| gaps:', lowgap)
    out[f] = dict(N=N, Q=Q, violations=[list(map(list, [v[0]])) + [v[1], v[2]] for v in viol],
                  ngap=len(gap), neq=eq, by2J={k: v for k, v in by2J.items()})
json.dump(out, open('compare_cache_summary.json', 'w'), indent=1, default=str)
