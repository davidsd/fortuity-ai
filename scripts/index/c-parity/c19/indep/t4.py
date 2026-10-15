import os as _os  # added on copy into the repo: repo root from this file's location (or $FORT_ROOT)
REPO_ROOT = _os.environ.get('FORT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..', '..')) + '/'
"""Independent T_rig: my own Weyl alternation (shifts rho - w rho over S3), my own d_up, s from stored data."""
import pickle, json, ast
from itertools import permutations
from types import SimpleNamespace
from fortuity.bmn.gravindex import GravitonCounter
R = REPO_ROOT + ''
C = R + 'scripts/index/c-parity/runs/cache/'
U = pickle.load(open('my_dup_q22.pkl','rb'))

def psign(p):
    s = 1
    for i in range(len(p)):
        for j in range(i+1, len(p)):
            if p[i] > p[j]: s = -s
    return s
rho = (2, 1, 0)
SHIFTS = []
for w in permutations(range(3)):
    wr = tuple(rho[w[i]] for i in range(3))  # a permutation of rho's entries
    SHIFTS.append((tuple(rho[i] - wr[i] for i in range(3)), psign(w)))
print('Weyl shifts (rho - w rho, sgn):', SHIFTS)

def agg(f, Q):
    out = {}
    for k, d in f.items():
        q = tuple(k[1:])
        if sum(q) <= Q and d: out[q] = out.get(q, 0) + d
    return out

def alt(lam, lo, hi):
    """lower bound on sum_w sgn(w) m(lam + shift): lower bound at + terms, upper bound at - terms"""
    tot = 0
    for sh, sg in SHIFTS:
        mu = tuple(lam[i] + sh[i] for i in range(3))
        tot += sg * (lo if sg > 0 else hi).get(mu, 0)
    return tot

def run(N, cachef, Q, rows, label):
    dims = pickle.load(open(C + cachef, 'rb'))['dims']
    uc = agg(dims, Q); uu = agg(U[f'N{N}'], Q)
    A = GravitonCounter.monomial_counts(SimpleNamespace(nmax=N), 2 * Q)
    u2 = agg({k: min(U[f'N{N}'].get(k, 0), A.get(k, 0)) for k in set(U[f'N{N}']) | set(A)}, Q)
    print(f'--- {label} ---')
    print(f'{"lam":>12} {"j":>3} {"T_st":>5} {"s":>4} {"Dcomp":>5} {"D_rig":>6} {"T_rig":>5} {"D_rig2":>6} {"T_rig2":>6} {"T_best":>6} gapM gap2')
    out = []
    for lam, Tst, s in rows:
        Dc = alt(lam, uc, uc)
        Dr = alt(lam, uc, uu); Dr2 = alt(lam, uc, u2)
        gapM = sum(uu.get(tuple(lam[i]+sh[i] for i in range(3)),0) - uc.get(tuple(lam[i]+sh[i] for i in range(3)),0) for sh, sg in SHIFTS if sg < 0)
        gap2 = sum(u2.get(tuple(lam[i]+sh[i] for i in range(3)),0) - uc.get(tuple(lam[i]+sh[i] for i in range(3)),0) for sh, sg in SHIFTS if sg < 0)
        Tbest = s - max(0, Dr2)
        flag = '' if Tst == s - Dc else '  <-- T_st != s - Dcomp'
        print(f'{str(lam):>12} {2*sum(lam):3} {Tst:5} {s:4} {Dc:5} {Dr:6} {s-Dr:5} {Dr2:6} {s-Dr2:6} {Tbest:6} {gapM:4} {gap2:4}{flag}')
        out.append(dict(lam=lam, Tst=Tst, s=s, Dc=Dc, Trig=s-Dr, Trig2=s-Dr2, Tbest=Tbest, gapM=gapM, gap2=gap2))
    return out

# SU(3) j<=40 from data/; s = T + diag_unsigned
d40 = json.load(open(R + 'data/su3/kernel_parity_j40.json'))
rows40 = [(tuple(r['lam']), r['T'], r['T'] + r['diag_unsigned']) for r in d40['rows'] if r['T'] < 0]
o40 = run(3, 'diag_dims_N3_j40.pkl', 20, rows40, 'SU(3) j<=40, data/su3/kernel_parity_j40.json')
d44 = json.load(open(R + 'scripts/index/c-parity/runs/su3_kernel_parity_j44.json'))
rows44 = [(tuple(r['lam']), r['T'], r['T'] + r['diag_unsigned']) for r in d44['rows'] if r['T'] < 0]
o44 = run(3, 'diag_dims_N3_j44.pkl', 22, rows44, 'SU(3) j<=44, workspace run')
# consistency of j40 rows between the two files
m40 = {tuple(r['lam']): (r['T'], r['diag_unsigned']) for r in d40['rows']}
m44 = {tuple(r['lam']): (r['T'], r['diag_unsigned']) for r in d44['rows'] if sum(r['lam']) <= 20}
print('data j40 vs workspace j44 rows at |lam|<=20 identical:', m40 == m44)
# SU(2): s = (-1)^|lam| H_I from su2_index_hw_j44.json
HI = {ast.literal_eval(k): v for k, v in json.load(open(C + 'su2_index_hw_j44.json')).items()}
d2 = json.load(open(R + 'data/su2/kernel_parity_j44.json'))
rows2 = [(tuple(r['lam']), r['T'], (-1) ** sum(r['lam']) * HI.get(tuple(r['lam']), 0)) for r in d2['rows'] if r['T'] < 0]
o2 = run(2, 'diag_dims_N2_j44.pkl', 22, rows2, 'SU(2) j<=44')
print('SU(2) total forced via s<0:', sum(-s for _, _, s in rows2 if s < 0), ' rows:', len(rows2), ' all T>=... any positive T stored:', [r['T'] for r in d2['rows'] if r['T'] > 0])
pickle.dump(dict(o40=o40, o44=o44, o2=o2), open('t4_out.pkl', 'wb'))
