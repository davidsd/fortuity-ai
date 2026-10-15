import os as _os, sys as _sys
REPO = (_os.environ.get('REPO') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *['..'] * 5))).rstrip('/') + '/'
_sys.path.insert(0, REPO + 'code'); _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
"""Independent exact check (no repo code imported) of
  - the SU(2) ordinary BMN index and its SU(3)_R hw content to |q| <= 22 (j <= 44),
  - the SU(3) C-twisted BMN index and its hw content to |q| <= 22,
  - closed form (3.4) of arXiv:2312.16443 against data/su2/bmn_index_j40.json,
  - stored R-017 data (twisted_index_j30, kernel_parity_j40, kernel_parity_j44, su2 nongraviton reps).
Method: exact integer power series. For an element g of the (twisted) torus with adjoint eigenvalues c*s^a (c=+-1),
PE = prod_{letters, eigenvalues} (1 - c s^a z^q)^{-1} (bosons) or (1 - c s^a z^q) (fermions, (-1)^F).
Twisted Weyl integration (normalised so the constant function integrates to 1)."""
import json, sys, numpy as np
from itertools import permutations
REPO = REPO + ''
Q = 22
SM = 2 * Q            # s-power offset
NS = 2 * SM + 1

# BMN letters (fermion?, q): phi_m at e_m, psi_m at (1,1,1)-e_m, f at (1,1,1)
LETTERS = [(False, (1, 0, 0)), (False, (0, 1, 0)), (False, (0, 0, 1)),
           (True, (0, 1, 1)), (True, (1, 0, 1)), (True, (1, 1, 0)), (False, (1, 1, 1))]


def _unused_shifted(A, q, a):
    """multiply by s^a z^q (truncating)."""
    out = np.zeros_like(A)
    q1, q2, q3 = q
    n = A.shape[0]
    src = A[:n - q1, :n - q2, :n - q3]
    if a >= 0:
        out[q1:, q2:, q3:, a:] = src[..., :NS - a]
    else:
        out[q1:, q2:, q3:, :NS + a] = src[..., -a:]
    return out


def sshift(X, a):
    """multiply by s^a along the last axis (truncating)."""
    out = np.zeros_like(X)
    if a >= 0:
        out[..., a:] = X[..., :NS - a]
    else:
        out[..., :NS + a] = X[..., -a:]
    return out


DEG = np.add.outer(np.add.outer(np.arange(Q + 1), np.arange(Q + 1)), np.arange(Q + 1))
MASK = (DEG <= Q)[..., None]


def pe_fast(eigs, unsigned=False):
    """exact series of prod over letters and adjoint eigenvalues, truncated at q1+q2+q3 <= Q.
    int64 (exact: overflow guarded by the float64 unsigned bound printed below); unsigned=True gives the bound."""
    A = np.zeros((Q + 1, Q + 1, Q + 1, NS), dtype=np.float64 if unsigned else np.int64)
    A[0, 0, 0, SM] = 1
    for ferm, q in LETTERS:
        for c, a in eigs:
            cc = 1 if unsigned else c
            if ferm:
                sign = 1 if unsigned else -1
                S = np.zeros_like(A)
                q1, q2, q3 = q
                S[q1:, q2:, q3:] = sshift(A[:Q + 1 - q1, :Q + 1 - q2, :Q + 1 - q3], a)
                A = A + sign * cc * S
            elif q == (1, 1, 1):
                for k in range(1, Q + 1):
                    A[k, 1:, 1:] += cc * sshift(A[k - 1, :-1, :-1], a)
            else:
                ax = q.index(1)
                for k in range(1, Q + 1):
                    to = [slice(None)] * 3; to[ax] = k
                    fr = [slice(None)] * 3; fr[ax] = k - 1
                    A[tuple(to)] += cc * sshift(A[tuple(fr)], a)
            A = A * MASK
    return A


def ct(A, q, k):
    return A[q[0], q[1], q[2], SM + k]


def hw(weights, lam):
    """multiplicity of SU(3) irrep with hw lam: sum_w sgn(w) mult(lam + rho - w rho)."""
    rho = (2, 1, 0)
    tot = 0
    for w in permutations(range(3)):
        sgn = 1 if sum(1 for i in range(3) for j in range(i + 1, 3) if w[i] > w[j]) % 2 == 0 else -1
        mu = tuple(lam[i] + rho[i] - rho[w[i]] for i in range(3))
        if min(mu) < 0 or max(mu) > Q:
            continue
        tot += sgn * weights.get(mu, 0)
    return tot


def dominant(qmax):
    for d in range(qmax + 1):
        for a in range(d + 1):
            for b in range(d + 1 - a):
                c = d - a - b
                if a >= b >= c:
                    yield (a, b, c)


# ------------------------------------------------------------------ eigenvalues of Ad(t sigma), my own construction
def ad_eigs(N, s0, twist):
    if N == 3:
        t = np.diag([s0, 1, 1 / s0])
    else:
        t = np.diag([s0, 1 / s0])
    J = np.fliplr(np.diag([(-1.) ** a for a in range(N)]))
    basis = []
    for i in range(N):
        for j in range(N):
            if i != j:
                E = np.zeros((N, N), complex); E[i, j] = 1; basis.append(E)
    for i in range(N - 1):
        v = np.zeros(N); v[i] = 1; v[i + 1] = -1; basis.append(np.diag(v).astype(complex))
    B = np.array([b.reshape(-1) for b in basis]).T
    cols = []
    for b in basis:
        X = (-J @ b.T @ np.linalg.inv(J)) if twist else b
        Y = t @ X @ np.linalg.inv(t)
        cols.append(np.linalg.lstsq(B, Y.reshape(-1), rcond=None)[0])
    return np.linalg.eigvals(np.array(cols).T)


s0 = np.exp(0.37j)
EIG3T = [(1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1), (-1, 2), (-1, -2)]
EIG2 = [(1, 0), (1, 2), (1, -2)]
for N, tw, E in [(3, True, EIG3T), (2, False, EIG2)]:
    ev = np.sort_complex(ad_eigs(N, s0, tw))
    pr = np.sort_complex(np.array([c * s0 ** a for c, a in E]))
    print(f'eigs N={N} twist={tw}: max diff {np.max(np.abs(ev - pr)):.1e}')
# SU(2): is sigma (hence C up to inner) trivial on su(2)?
X = np.array([[0.3, 1.7], [-0.4, -0.3]])
J2 = np.fliplr(np.diag([1., -1.]))
print('SU(2): -J X^T J^-1 == X ?', np.allclose(-J2 @ X.T @ np.linalg.inv(J2), X))

# ------------------------------------------------------------------ formula tests on single adjoint letters
def single_letter_test(ferm, eigs, haar, norm, nmax=12):
    """sum_n (+-1)^n tr(C | (Sym or Lambda)^n(adj)^G) t^n via the (twisted) Weyl formula, as integer list."""
    # univariate in t, Laurent in s
    A = np.zeros((nmax + 1, 2 * 2 * nmax + 1), dtype=object); off = 2 * nmax
    A[0, off] = 1
    for c, a in eigs:
        if ferm:
            Bn = A.copy()
            for n in range(1, nmax + 1):
                for k in range(A.shape[1]):
                    if 0 <= k - a < A.shape[1]:
                        Bn[n, k] -= c * A[n - 1, k - a]
            A = Bn
        else:
            for n in range(1, nmax + 1):
                for k in range(A.shape[1]):
                    if 0 <= k - a < A.shape[1]:
                        A[n, k] += c * A[n - 1, k - a]
    out = []
    for n in range(nmax + 1):
        v = sum(h * A[n, off + p] for p, h in haar)
        assert v % norm == 0
        out.append(v // norm)
    return out


HAAR3T = [(0, 2), (4, -1), (-4, -1)]          # (1-s^4)(1-s^-4); overall 2 cancels against the normalisation
HAAR2 = [(0, 2), (2, -1), (-2, -1)]           # (1-s^2)(1-s^-2)
b = single_letter_test(False, EIG3T, HAAR3T, 2)
f = single_letter_test(True, EIG3T, HAAR3T, 2)
import sympy as sp
t = sp.symbols('t')
eb = sp.Poly(sp.series(1 / ((1 - t ** 2) * (1 + t ** 3)), t, 0, 13).removeO(), t).all_coeffs()[::-1]
ef = sp.Poly(sp.expand((1 - t ** 3) * (1 + t ** 5)), t).all_coeffs()[::-1]
ef = ef + [0] * (13 - len(ef))
print('SU(3) twisted, one bosonic adjoint  == 1/((1-t^2)(1+t^3)) :', b == [int(x) for x in eb])
print('SU(3) twisted, one fermionic adjoint == (1-t^3)(1+t^5)     :', f == [int(x) for x in ef], f[:10])
b2 = single_letter_test(False, EIG2, HAAR2, 2)
print('SU(2), one bosonic adjoint == 1/(1-t^2):', b2 == [1 if n % 2 == 0 else 0 for n in range(13)])

# ------------------------------------------------------------------ BMN indices, exact
import time
t0 = time.time()
F3 = pe_fast(EIG3T)
F2 = pe_fast(EIG2)
U3 = pe_fast(EIG3T, unsigned=True)
U2 = pe_fast(EIG2, unsigned=True)
print(f'PE built in {time.time() - t0:.1f}s; max unsigned coefficient (int64 overflow guard, must be < 9.2e18): SU(3) {U3.max():.3e}, SU(2) {U2.max():.3e}')
IC, I2 = {}, {}
for q1 in range(Q + 1):
    for q2 in range(Q + 1 - q1):
        for q3 in range(Q + 1 - q1 - q2):
            q = (q1, q2, q3)
            v = 2 * ct(F3, q, 0) - ct(F3, q, 4) - ct(F3, q, -4)
            assert v % 2 == 0
            if v:
                IC[q] = v // 2
            w = 2 * ct(F2, q, 0) - ct(F2, q, 2) - ct(F2, q, -2)
            assert w % 2 == 0
            if w:
                I2[q] = w // 2
S3 = {lam: (-1) ** sum(lam) * hw(IC, lam) for lam in dominant(Q)}
S2 = {lam: (-1) ** sum(lam) * hw(I2, lam) for lam in dominant(Q)}
import pickle; pickle.dump({l: int(v) for l, v in S2.items()}, open('su2_index_hw.pkl', 'wb'))
print('number of weights with nonzero I_C:', len(IC))
print('SU(3) s = (-1)^|lam| I_C^hw at (7,4,4), (8,5,4), (9,6,6), (10,6,6):', S3[(7, 4, 4)], S3[(8, 5, 4)], S3[(9, 6, 6)], S3[(10, 6, 6)])
print('SU(3) columns |lam|<=22 with s<0:', [(l, v) for l, v in S3.items() if v < 0])
print('weight multiplicities entering (9,6,6):', [(mu, IC.get(mu, 0)) for mu in [(9,6,6),(10,5,6),(9,7,5),(11,5,5),(10,7,4),(11,6,4)]])
print('weight multiplicities entering (10,6,6):', [(mu, IC.get(mu, 0)) for mu in [(10,6,6),(11,5,6),(10,7,5),(12,5,5),(11,7,4),(12,6,4)]])

# ---- compare with stored twisted index data (j <= 30) and with T + D of kernel_parity_j40
tw30 = json.load(open(REPO + 'data/su3/twisted_index_j30.json'))
bad = [(r['lam'], r['twisted_index'], hw(IC, tuple(r['lam']))) for r in tw30['rows'] if r['twisted_index'] != hw(IC, tuple(r['lam']))]
print('twisted_index_j30: rows', len(tw30['rows']), 'mismatches with my I_C^hw:', bad)
okpred = all(r['predicted_signed'] == (-1) ** sum(r['lam']) * r['twisted_index'] for r in tw30['rows'])
print('  predicted_signed == (-1)^|lam| twisted_index in all rows:', okpred,
      '; P-odd counts implied (total - signed)/2 != 0 at:', [(r['lam'], r['total_cohomology'], (r['total_cohomology'] - (-1) ** sum(r['lam']) * r['twisted_index']) // 2) for r in tw30['rows'] if r['total_cohomology'] != (-1) ** sum(r['lam']) * r['twisted_index']])
k40 = json.load(open(REPO + 'data/su3/kernel_parity_j40.json'))
bad = [(r['lam'], r['T'] + r['diag_unsigned'], S3[tuple(r['lam'])]) for r in k40['rows'] if r['T'] + r['diag_unsigned'] != S3[tuple(r['lam'])]]
print('kernel_parity_j40: rows', len(k40['rows']), 'T + D == my s in all rows:', not bad, bad[:3])
negT = [r for r in k40['rows'] if r['T'] < 0]
print('  T<0 rows:', len(negT), ' of which s<0 (index alone negative):', [r['lam'] for r in negT if S3[tuple(r['lam'])] < 0],
      ' min s among T<0 rows:', min(S3[tuple(r['lam'])] for r in negT))

# ------------------------------------------------------------------ SU(2)
neg2 = {l: v for l, v in S2.items() if v < 0}
print('SU(2) columns |lam|<=22 with (-1)^|lam| I^hw < 0:', len(neg2), 'sum', sum(neg2.values()))
k44 = json.load(open(REPO + 'data/su2/kernel_parity_j44.json'))
stored = {tuple(r['lam']): r['T'] for r in k44['rows']}
print('  == stored SU(2) T<0 rows (lam and value):', stored == neg2)
# level sums vs data/su2/bmn_index_j40.json full index
b40 = json.load(open(REPO + 'data/su2/bmn_index_j40.json'))
lev = [0] * 45
for q, v in I2.items():
    lev[2 * sum(q)] += v
print('SU(2) full index level sums == data full_index_unrefined (j<=40):', lev[:41] == b40['full_index_unrefined'])
# closed form (3.4)
num = 1+3*t**4-8*t**6-6*t**10+10*t**12+9*t**14-9*t**16+16*t**18-18*t**20-3*t**22+t**24-3*t**26+9*t**28-2*t**30+3*t**32-3*t**34
Zg = sp.series(num / ((1 - t ** 4) ** 3 * (1 - t ** 8) ** 3), t, 0, 45).removeO()
zg = [int(Zg.coeff(t, j)) for j in range(45)]
print('(3.4) == data graviton_index_unrefined (j<=40):', zg[:41] == b40['graviton_index_unrefined'])
print('(3.4) coefficients j=42,44:', zg[42], zg[44], ' sign pattern (-1)^(j/2) for all even j:',
      all((zg[j] == 0) or (np.sign(zg[j]) == (-1) ** (j // 2)) for j in range(0, 45, 2)), ' odd j zero:', all(zg[j] == 0 for j in range(1, 45, 2)))
# SU(2) T from stored nongraviton reps (j<=40): T = (-1)^|lam| * value ; never positive?
Tng = {}
for js, reps in b40['nongraviton_index_su3_reps'].items():
    j = int(js)
    for rep, v in reps.items():
        a, bb = json.loads(rep)
        l3 = (j // 2 - a - 2 * bb) // 3
        lam = (l3 + a + bb, l3 + bb, l3)
        Tng[lam] = (-1) ** sum(lam) * v
print('SU(2) T from stored nongraviton reps (j<=40):', sorted(Tng.items(), key=lambda x: (sum(x[0]), x[0])))
print('  never positive:', all(v <= 0 for v in Tng.values()), '; equals stored T<0 rows with j<=40:',
      Tng == {l: v for l, v in stored.items() if 2 * sum(l) <= 40})
