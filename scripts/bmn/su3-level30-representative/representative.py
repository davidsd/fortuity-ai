"""T-021: explicit representative of the non-graviton class of R-007, SU(3) BMN, sector (2J; q) = (5; 7,4,4),
SU(3)_R irrep [3,0] (highest-weight component), E = 25/2; with graviton generators u_n, v_n, w_n up to n = NMAX.

Steps (fortuity.hwcoh machinery, MAXLEN=6 as in R-007, faithfulness certified per fermion degree):
 1. sector 2J=4: functionals V_4; Q-images of the hw vectors of 2J=5 evaluated there (Qi).
 2. sector 2J=5: hw vectors of 2J=5 (same list) evaluated at functionals of V_5 (A5); Q-images of 2J=6 (B5);
    graviton hw vectors (G5).
 3. random combinations Y of the hw vectors (spanning set) -> closed combos = left kernel of Y Qi.
 4. closed vectors reduced against span(B5, G5): the ones outside are non-graviton; pick one -> representative.
 5. the representative as a multi-trace expression (coefficients mod p); checks: highest weight, Q = 0 on fresh
    3x3 points, Q != 0 on 4x4 and 5x5 points (lifetime of this representative).
usage: python3 representative.py NMAX SEED"""
import sys, time, json, gzip, pickle, numpy as np
from collections import defaultdict
from fortuity.hwcoh.sector import Sector
from fortuity.hwcoh.gravhw import GravitonAlgebra
from fortuity.hwcoh.hw import E_multi
from fortuity.hwcoh import ffla
from fortuity.bmn.words import Q_multi, contents_of_sector
from fortuity.bmn.modp import matmul_mod, P, RREFBasis
from fortuity.bmn.evaluate import Points

NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 11
lam = (7, 4, 4)
t0 = time.time()
def log(*a): print('[%6.0fs]' % (time.time() - t0), *a, flush=True)
dims = pickle.load(open('dims_nf_30.pkl', 'rb'))

def faithful(S, A):
    col, rk, ok = 0, {}, True
    for d in sorted(S.m_d):
        w = sum(fr.ncols for dd, fr in S.colblocks if dd == d)
        rows = np.flatnonzero(A[:, col:col + w].any(axis=1))
        rk[d] = ffla.rank(A[rows, col:col + w]) if rows.size else 0
        ok &= rk[d] == S.m_d[d]; col += w
    return ok, rk

def run_sector(S, name):
    S.prepare()
    for attempt, factor in enumerate((1, 2, 4)):
        A, B, G = S.evaluate(factor, attempt)
        ok, rk = faithful(S, A)
        log('%s: faithfulness %s (rank per degree %s vs exact %s)' % (name, 'OK' if ok else 'FAIL', rk, S.m_d))
        if ok: return A, B, G
    raise SystemExit('functionals not faithful for ' + name)

# ---- 2J = 4 ----
S4 = Sector(3, lam, 4, dims, seed=SEED, log=lambda s: log(s), maxlen=6)
A4, B4, _ = run_sector(S4, '2J=4')
del A4
hw5 = S4.hw_up                       # blocks (word indices into S4.ws_up, H)
R = sum(H.shape[0] for _, H in hw5)
Qi = np.zeros((R, B4.shape[1]), dtype=np.int64)
r0 = b0 = 0
for widx, H in hw5:
    # same criterion as Sector.prepare: a block contributes rows to B iff H @ (Q-matrix) has a nonzero column
    tg, ent = {}, []
    for r, wi in enumerate(widx):
        for m2, c in Q_multi(S4.ws_up[wi]).items():
            j = tg.setdefault(m2, len(tg)); ent.append((r, j, c))
    has = False
    if tg:
        Qm = np.zeros((len(widx), len(tg)), dtype=np.int64)
        for r, j, c in ent: Qm[r, j] += c
        has = bool(matmul_mod(H, Qm % P).any())
    if has:
        Qi[r0:r0 + H.shape[0]] = B4[b0:b0 + H.shape[0]]; b0 += H.shape[0]
    r0 += H.shape[0]
assert b0 == B4.shape[0], (b0, B4.shape)
del B4
log('Q-images of the %d hw vectors of 2J=5 placed (%d functionals of 2J=4)' % (R, Qi.shape[1]))
# ---- 2J = 5 ----
GA = GravitonAlgebra(NMAX)
S5 = Sector(3, lam, 5, dims, seed=SEED + 1, log=lambda s: log(s), maxlen=6, graviton_algebra=GA, pre=S4.handoff())
A5, B5, G5 = run_sector(S5, '2J=5')
assert A5.shape[0] == R
m5 = S5.m
r6 = ffla.rank(B5)
rg = ffla.rank(np.concatenate([B5, G5])) if G5.shape[0] else r6
log('m5 = %d | rank Q from 2J=6 (exact) = %d | graviton classes (generators n<=%d) = %d' % (m5, r6, NMAX, rg - r6))
# ---- closed combinations ----
rng = np.random.default_rng(SEED + 99)
K0 = m5 + 64
Y = rng.integers(0, P, (K0, R), dtype=np.int64)
YA = matmul_mod(Y, A5 % P); YQ = matmul_mod(Y, Qi % P)
del Qi
r5 = ffla.rank(YQ)
Kn = ffla.nullspace_right(np.ascontiguousarray(YQ.T))      # rows x with x YQ = 0
Cv = matmul_mod(Kn % P, YA)
closed = ffla.rank(Cv)
h5 = closed - r6
log('rank Q on 2J=5 = %d | closed dim = %d (m5 - r5 = %d) | cohomology h5 = %d' % (r5, closed, m5 - r5, h5))
# ---- non-graviton vectors ----
base = RREFBasis(Cv.shape[1], P)
base.add_batched(np.concatenate([B5, G5]) % P)
rb = base.rank
new = base.add_batched(Cv)
nong = int(new.sum())
log('closed vectors outside span(exact, gravitons n<=%d): %d  (= non-graviton classes)' % (NMAX, nong))
res = dict(lam=lam, twoJ=5, nmax=NMAX, seed=SEED, m5=int(m5), r5=int(r5), r6=int(r6), closed=int(closed), h5=int(h5), grav=int(rg - r6), nongrav=nong)
if nong == 0:
    json.dump(res, open('runs/rep_nmax%d_seed%d.json' % (NMAX, SEED), 'w'), indent=1); raise SystemExit('no non-graviton class')
k = int(np.flatnonzero(new)[0])
z = matmul_mod((Kn[k:k + 1] % P), Y)[0]                     # coefficients on the hw vectors of 2J=5
# ---- expression ----
rep = defaultdict(int)
r0 = 0
for widx, H in hw5:
    zb = z[r0:r0 + H.shape[0]]; r0 += H.shape[0]
    if not zb.any(): continue
    c = matmul_mod(zb[None, :] % P, H % P)[0]
    for wi, cc in zip(widx, c):
        if cc: rep[S4.ws_up[wi]] = (rep[S4.ws_up[wi]] + int(cc)) % P
rep = {m: c for m, c in rep.items() if c}
log('representative: %d multi-trace words (coefficients mod p)' % len(rep))
# highest weight
hw_ok = True
for i in (1, 2):
    img = defaultdict(int)
    for m, v in rep.items():
        for m2, c2 in E_multi(m, i).items(): img[m2] = (img[m2] + v * c2) % P
    hw_ok &= not any(img.values())
log('E1, E2 annihilate the representative (highest weight):', hw_ok)
# Q at N = 3, 4, 5 on fresh points
q = defaultdict(int)
for m, v in rep.items():
    for m2, c2 in Q_multi(m).items(): q[m2] = (q[m2] + v * c2) % P
q = {m: c for m, c in q.items() if c}
degs4 = sorted({sum(c[3:6]) for c in contents_of_sector((4,) + lam)})
life = {}
for N in (3, 4, 5):
    pts = Points(N, 24 if N == 3 else 8, max(degs4) + 2, degs4, seed=SEED + 500 + N)
    life[N] = bool(np.any(pts.expr_vector(q)))
    log('Q(representative) on random traceless %dx%d matrices: %s' % (N, N, 'NONZERO' if life[N] else 'zero'))
degs5 = sorted({sum(c[3:6]) for c in contents_of_sector((5,) + lam)})
pts3 = Points(3, 24, max(degs5) + 2, degs5, seed=SEED + 900)
res.update(rep_words=len(rep), formal_Q_terms=len(q), hw=hw_ok, Q_nonzero_at_N={str(k): v for k, v in life.items()},
           rep_nonzero_N3=bool(np.any(pts3.expr_vector(rep))))
json.dump(res, open('runs/rep_nmax%d_seed%d.json' % (NMAX, SEED), 'w'), indent=1)
with gzip.open('runs/rep_nmax%d_seed%d_expr.json.gz' % (NMAX, SEED), 'wt') as f:
    json.dump({'prime': P, 'sector': [5, 7, 4, 4], 'letters': '0,1,2 = phi_1,2,3; 3,4,5 = psi_1,2,3; 6 = f',
               'terms': [[c, [list(w) for w in m]] for m, c in sorted(rep.items())]}, f)
log('saved', res)
