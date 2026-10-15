import os as _os  # added on copy into the repo: repo root from this file's location (or $FORT_ROOT)
REPO_ROOT = _os.environ.get('FORT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..')) + '/'
"""Variants of the rigorous bound, per T<0 column:
  T_rig    : D_rig = sum_+ uns_comp - sum_- uns_molien                  (task's repair)
  T_rig2   : same with uns_up = sum_2J min(Molien, free monomial count)  (both are valid upper bounds on the graviton image)
  T_triv   : D >= 0 (hw multiplicity of a genuine SU(3)_R rep), i.e. T_true <= (-1)^|lam| I_C
  T_best   : (-1)^|lam| I_C - max(0, D_rig2)
"""
import sys, json, pickle
import numpy as np
from fortuity.bmn.twistedindex import SUN, integrate, to_int, hw_multiplicities
from molien import molien
from freecount import free_counts
CACHE = REPO_ROOT + 'scripts/index/c-parity/runs/cache/'
a12 = np.array([1, -1, 0]); a23 = np.array([0, 1, -1])
TERMS = [(0, 0, 1), (1, 0, -1), (0, 1, -1), (2, 1, 1), (1, 2, 1), (2, 2, -1)]
N, Q, cache = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
dims = pickle.load(open(CACHE + cache, 'rb'))['dims']
M = molien(N, Q); A = free_counts(N, 2 * Q)
def agg(f):
    out = {}
    for sec, d in f.items():
        q = tuple(sec[1:])
        if sum(q) <= Q and d: out[q] = out.get(q, 0) + d
    return out
uc = agg(dims); um = agg(M)
u2 = agg({k: min(M.get(k, 0), A.get(k, 0)) for k in set(M) | set(A)})
tw, err = to_int(integrate(SUN(N), Q, True)[0])
Htw, Hc = hw_multiplicities(tw, Q), hw_multiplicities(uc, Q)
sh = lambda lam, i, j: tuple(int(x) for x in np.array(lam) + i * a12 + j * a23)
def Dlow(lam, up):
    return sum((uc if sg > 0 else up).get(sh(lam, i, j), 0) * sg for i, j, sg in TERMS)
rows = []
for lam in sorted(set(Htw) | set(Hc), key=lambda l: (sum(l), l)):
    if lam == (0, 0, 0): continue
    s = (-1) ** sum(lam) * Htw.get(lam, 0)
    T = s - Hc.get(lam, 0)
    if T >= 0: continue
    d1, d2 = Dlow(lam, um), Dlow(lam, u2)
    gap2 = sum(u2.get(sh(lam, i, j), 0) - uc.get(sh(lam, i, j), 0) for i, j, sg in TERMS if sg < 0)
    rows.append(dict(lam=lam, j=2 * sum(lam), T=T, signedIC=s, D_comp=Hc.get(lam, 0), D_rig=d1, T_rig=s - d1,
                     D_rig2=d2, T_rig2=s - d2, T_triv=s, T_best=s - max(0, d2), minus_gap_min=gap2))
print(f'SU({N}) qmax={Q} ({cache}); rounding {err:.1e}')
print(f'{"lam":>12} {"j":>3} {"T":>4} {"sI_C":>5} {"D_comp":>6} {"D_rig":>7} {"T_rig":>6} {"D_rig2":>7} {"T_rig2":>6} {"T_triv":>6} {"T_best":>6} gap2')
for r in rows:
    print(f'{str(r["lam"]):>12} {r["j"]:3} {r["T"]:4} {r["signedIC"]:5} {r["D_comp"]:6} {r["D_rig"]:7} {r["T_rig"]:6} '
          f'{r["D_rig2"]:7} {r["T_rig2"]:6} {r["T_triv"]:6} {r["T_best"]:6} {r["minus_gap_min"]:5}')
print('rigorously forced (T_best < 0):', [(r['lam'], -r['T_best']) for r in rows if r['T_best'] < 0])
json.dump(rows, open(f'repair2_N{N}_q{Q}.json', 'w'), indent=1, default=str)
