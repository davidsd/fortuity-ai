import os as _os  # added on copy into the repo: repo root from this file's location (or $FORT_ROOT)
REPO_ROOT = _os.environ.get('FORT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..')) + '/'
"""Steps 2-4: reproduce stored T from the cached d_comp with the library pipeline, then the rigorous T_rig with
D_rig = sum_{+ terms} uns_comp - sum_{- terms} uns_up (uns_up from the exact Molien series)."""
import sys, json, pickle, time
import numpy as np
from fortuity.bmn.twistedindex import SUN, integrate, to_int, hw_multiplicities
from molien import molien

REPO = REPO_ROOT + ''
CACHE = REPO + 'scripts/index/c-parity/runs/cache/'
a12 = np.array([1, -1, 0]); a23 = np.array([0, 1, -1])
TERMS = [(0, 0, 1), (1, 0, -1), (0, 1, -1), (2, 1, 1), (1, 2, 1), (2, 2, -1)]

N, Q, cache, stored = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
t0 = time.time()
dims = pickle.load(open(CACHE + cache, 'rb'))['dims']
M = molien(N, Q)
uns_c, uns_u = {}, {}
for sec, d in dims.items():
    q = tuple(sec[1:])
    if sum(q) <= Q and d:
        uns_c[q] = uns_c.get(q, 0) + d
for sec, d in M.items():
    q = tuple(sec[1:])
    if sum(q) <= Q and d:
        uns_u[q] = uns_u.get(q, 0) + d
G = SUN(N)
twr, err = integrate(G, Q, True)[0], None
tw, err = to_int(twr)
info = dict(rounding_twisted=err)
if N == 2:
    pl, err2 = to_int(integrate(G, Q, False)[0])
    info['rounding_ordinary'] = err2
    info['twisted_equals_ordinary'] = (pl == tw)
print(f'SU({N}) qmax={Q}: integrals done {time.time()-t0:.0f}s, {info}', flush=True)
H_tw, H_c = hw_multiplicities(tw, Q), hw_multiplicities(uns_c, Q)

# stored T values
st = json.load(open(stored))
stored_T = {tuple(r['lam']): r['T'] for r in st['rows']}
# recompute T for all columns
lams = sorted(set(H_tw) | set(H_c), key=lambda l: (sum(l), l))
T = {}
for lam in lams:
    if lam == (0, 0, 0) or sum(lam) > Q: continue
    t = (-1) ** sum(lam) * H_tw.get(lam, 0) - H_c.get(lam, 0)
    if t: T[lam] = t
mis = [(l, T.get(l, 0), v) for l, v in stored_T.items() if T.get(l, 0) != v] + \
      [(l, v, stored_T.get(l, 0)) for l, v in T.items() if l not in stored_T and sum(l) <= max(sum(x) for x in stored_T)]
print(f'reproduce stored {stored.split("fortuity/")[-1]}: {len(stored_T)} stored nonzero rows, recomputed nonzero {len(T)}; '
      f'mismatches {mis}', flush=True)

def shifted(lam, i, j):
    return tuple(int(x) for x in np.array(lam) + i * a12 + j * a23)

rows = []
for lam in sorted([l for l, t in T.items() if t < 0], key=lambda l: (sum(l), l)):
    Dc = sum(sg * uns_c.get(shifted(lam, i, j), 0) for i, j, sg in TERMS)
    assert Dc == H_c.get(lam, 0)
    Drig = sum((uns_c if sg > 0 else uns_u).get(shifted(lam, i, j), 0) * sg for i, j, sg in TERMS)
    sgn = (-1) ** sum(lam)
    Trig = sgn * H_tw.get(lam, 0) - Drig
    minus = []
    for i, j, sg in TERMS:
        if sg < 0:
            w = shifted(lam, i, j)
            per2J = {}
            for tj in range(0, sum(w) + 1):
                dc, du = dims.get((tj,) + w, 0), M.get((tj,) + w, 0)
                if dc or du: per2J[tj] = (dc, du)
            minus.append(dict(w=w, comp=uns_c.get(w, 0), up=uns_u.get(w, 0), per2J=per2J))
    exact_minus = all(m['comp'] == m['up'] for m in minus)
    rows.append(dict(lam=lam, j=2 * sum(lam), T_stored=stored_T.get(lam), T=T[lam], twisted=H_tw.get(lam, 0),
                     D_comp=Dc, D_rig=Drig, T_rig=Trig, forced=max(0, -Trig), minus_exact=exact_minus,
                     minus_gap=sum(m['up'] - m['comp'] for m in minus), minus=minus))
print(f'{"lam":>12} {"j":>3} {"T_st":>5} {"T":>4} {"I_C":>6} {"D_comp":>7} {"D_rig":>8} {"T_rig":>8} forced minus_exact minus_gap')
for r in rows:
    print(f'{str(r["lam"]):>12} {r["j"]:3} {str(r["T_stored"]):>5} {r["T"]:4} {r["twisted"]:6} {r["D_comp"]:7} {r["D_rig"]:8} '
          f'{r["T_rig"]:8} {r["forced"]:6} {str(r["minus_exact"]):>11} {r["minus_gap"]:8}')
surv = [r for r in rows if r['T_rig'] < 0]
print(f'columns with T<0: {len(rows)};  T_rig<0 (rigorously forced): {len(surv)} {[(r["lam"], r["T_rig"]) for r in surv]}; '
      f'minus-weights exact in {sum(r["minus_exact"] for r in rows)} columns')
json.dump(dict(N=N, qmax=Q, cache=cache, stored=stored, info=info, reproduce_mismatches=mis, rows=rows),
          open(f'repair_N{N}_q{Q}.json', 'w'), indent=1, default=str)
