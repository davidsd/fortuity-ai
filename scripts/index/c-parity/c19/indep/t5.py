import os as _os  # added on copy into the repo: repo root from this file's location (or $FORT_ROOT)
REPO_ROOT = _os.environ.get('FORT_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '..', '..')) + '/'
"""Independent exact SU(2) BMN index per weight q (|q|<=22) by constant-term extraction with integer arrays,
then hw multiplicities with my own alternation; compare s = (-1)^|lam| H_I(lam) with the stored file."""
import numpy as np, json, ast, pickle
from itertools import permutations
from weylcount import LET
Q = 22; ZM = 2 * Q; NZ = 2 * ZM + 1
A = np.zeros((Q + 1, Q + 1, Q + 1, NZ), dtype=np.int64)
A[0, 0, 0, ZM] = 1
def shift(B, q, a):
    out = np.zeros_like(B)
    s1, s2, s3 = q
    if a >= 0:
        out[s1:, s2:, s3:, a:] = B[:Q + 1 - s1, :Q + 1 - s2, :Q + 1 - s3, :NZ - a]
    else:
        out[s1:, s2:, s3:, :NZ + a] = B[:Q + 1 - s1, :Q + 1 - s2, :Q + 1 - s3, -a:]
    return out
for (L, ferm, twoJ, q) in LET:
    for a in (2, 0, -2):
        if ferm:
            A = A - shift(A, q, a)          # (1 - x^q z^a): fermion with (-1)^F
        else:
            # 1/(1 - m): B = A + m B, iterate to fixed point (degree bounded)
            B = A.copy()
            for _ in range(Q // sum(q) + 1):
                B = A + shift(B, q, a)
            A = B
# mask |q|<=Q
idx = {}
for q1 in range(Q + 1):
    for q2 in range(Q + 1 - q1):
        for q3 in range(Q + 1 - q1 - q2):
            F = A[q1, q2, q3]
            v2 = 2 * F[ZM] - F[ZM + 2] - F[ZM - 2]
            assert v2 % 2 == 0
            if v2: idx[(q1, q2, q3)] = v2 // 2
rho = (2, 1, 0)
def psign(p):
    return 1 if sum(1 for i in range(3) for j in range(i+1,3) if p[i] > p[j]) % 2 == 0 else -1
SH = [(tuple(rho[i] - rho[w[i]] for i in range(3)), psign(w)) for w in permutations(range(3))]
H = {}
for s in range(Q + 1):
    for a in range(s + 1):
        for b in range(s + 1 - a):
            lam = (a, b, s - a - b)
            if not (lam[0] >= lam[1] >= lam[2]): continue
            c = sum(sg * idx.get(tuple(lam[i] + sh[i] for i in range(3)), 0) for sh, sg in SH)
            if c: H[lam] = c
C = REPO_ROOT + 'scripts/index/c-parity/runs/cache/'
HI = {ast.literal_eval(k): v for k, v in json.load(open(C + 'su2_index_hw_j44.json')).items()}
print('my SU(2) hw index == stored su2_index_hw_j44.json:', H == HI, len(H), len(HI))
# unrefined check vs data/su2/bmn_index_j40.json
un = json.load(open(REPO_ROOT + 'data/su2/bmn_index_j40.json'))['full_index_unrefined']
mine = [0] * 41
for q, v in idx.items():
    if 2 * sum(q) <= 40: mine[2 * sum(q)] += v
print('unrefined vs data/su2/bmn_index_j40.json full_index_unrefined:', mine == un)
for lam in [(4,4,4),(5,5,5),(6,6,6),(7,6,6),(7,7,7),(8,7,7),(9,7,6)]:
    print(lam, 's =', (-1) ** sum(lam) * H.get(lam, 0))
pickle.dump(idx, open('my_su2_index_q22.pkl', 'wb'))
