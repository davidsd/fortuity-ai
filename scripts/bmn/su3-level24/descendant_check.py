"""Is (R_26^{(0,4)})^i of arXiv:2312.16443 (A.4) the SU(1|3) descendant Q_+^i R_24^{(0,3)} in cohomology?

Both are hw vectors of the sector (2J; lam) = (4; 5,4,4) (i = 1), whose hw cohomology is one-dimensional (R-007 data).
In the hw complex of hw_relation_test.py: v(T) = values of T at the functionals, B = values of the Q-exact hw space.
N = right null space of B.  T is exact iff v(T) N^T = 0, and R_26 = c Q_+ R_24 + exact iff the two residual vectors
are proportional with ratio c.
Normalisation: hw_relation_test.build stores each operator cleared of denominators (to_int: stored = scale * defined, with
the paper's u, v).  The ratio is computed on the stored vectors (c_stored) and converted: c = c_stored * scale(R_24) /
scale(R_26) is the coefficient in the paper's normalisation, with Q_+^m phi^n = eps^{mnp} psi_p, Q_+^m psi_n = delta^m_n f.
usage: python3 descendant_check.py [seed]"""
import sys, json, time, numpy as np
import hw_relation_test as H
from fortuity.hwcoh import ffla
from fortuity.hwcoh.sector import Sector
from fortuity.hwcoh.gravhw import GravitonAlgebra
from fortuity.hwcoh.evalg import nferm
from fortuity.bmn.modp import matmul_mod, P
import pickle, os
from fortuity.hwcoh.dims import partition_dims_nf


def ratrec(x, bound=5000):
    for b in range(1, bound):
        a = x * b % P
        if a > P // 2:
            a -= P
        if abs(a) < bound:
            return f"{a}/{b}" if b > 1 else str(a)
    return None


seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
cases = ['R26', 'QR24_1']
lam, n = H.CASES['R26'][0], H.CASES['R26'][1]
j = 2 * sum(lam)
cache = 'runs/dims_nf_%d.pkl' % j
dims = pickle.load(open(cache, 'rb')) if os.path.exists(cache) else partition_dims_nf(3, j)
scales = {}
_to_int = H.to_int
def _to_int_rec(e):
    r = _to_int(e)
    scales['last'] = r[1]
    return r
H.to_int = _to_int_rec
T = {}
for c in cases:
    T[c] = H.CASES[c][2]()
    scales[c] = scales['last']              # for QR24_1 this is the scale of the stored R_24
H.to_int = _to_int
t0 = time.time()
S = Sector(3, lam, n, dims, seed=seed, log=lambda s: None, graviton_algebra=GravitonAlgebra(3))
S.prepare()
A, B, G = S.evaluate()
r_up = int(ffla.rank(B))
vs = {}
for c in cases:
    words = list(T[c]); v = np.zeros(B.shape[1], dtype=np.int64); col = 0
    for d, fr in S.colblocks:
        idx = [i for i, w in enumerate(words) if nferm(w) == d]
        if idx:
            X = fr.eval_words([words[i] for i in idx])
            v[col:col + fr.ncols] = matmul_mod(np.array([[T[c][words[i]] % P for i in idx]], dtype=np.int64), X)[0]
        col += fr.ncols
    vs[c] = v
N = np.asarray(ffla.nullspace_right(B), dtype=np.int64) % P            # rows n: B n = 0
res = {c: matmul_mod(vs[c][None, :], np.ascontiguousarray(N.T))[0] for c in cases}
r1 = {c: int(ffla.rank(np.concatenate([B, vs[c][None, :]], axis=0))) - r_up for c in cases}
r12 = int(ffla.rank(np.concatenate([B, vs['R26'][None, :], vs['QR24_1'][None, :]], axis=0))) - r_up
k = int(np.flatnonzero(res['QR24_1'])[0])
cmod = int(res['R26'][k]) * pow(int(res['QR24_1'][k]), P - 2, P) % P
prop = not ((res['R26'] - cmod * res['QR24_1']) % P).any()
s26, s24 = scales['R26'], scales['QR24_1']
cpaper = cmod * (s24.numerator * s26.denominator) % P * pow(s24.denominator * s26.numerator, P - 2, P) % P
out = dict(sector=dict(lam=lam, twoJ=n), seed=seed, terms={c: len(T[c]) for c in cases}, exact_rank=r_up,
           null_space=int(N.shape[0]), rank_increment={**r1, 'both': r12}, proportional_mod_exact=bool(prop),
           stored_scale=dict(R26=str(s26), R24=str(s24)), c_stored_mod_p=cmod, c_stored=ratrec(cmod),
           c=ratrec(cpaper), statement='R_26^1 = c * Q_+^1 R_24 + Q(...) at N = 3, paper normalisation of u, v; '
           'c_stored is the same ratio for the stored integer forms (stored = scale * defined)',
           time=round(time.time() - t0))
print(json.dumps(out))
json.dump(out, open('runs/descendant_R26_QR24_seed%d.json' % seed, 'w'), indent=1)
