"""(3)/(5) Whole twisted-torus scan of the SU(N) C-twisted Cardy exponent, N = 3, 4, 5, 6, ...

usage:  python cardy_torus_sun.py N [dth_frac] [dphi_frac] [ncand]
  dth_frac:  theta grid step in units of pi (default 1/80);  dphi_frac: phi step in units of pi (default 1/480);
  ncand: max number of distinct grid local maxima refined (default 12).
Twisted torus t sigma, t = diag(e^{i th_1}..e^{i th_r}, [1], e^{-i th_r}..e^{-i th_1}), r = N//2, sigma(X) = -J X^T J^-1,
J = antidiag(1,-1,1,..) (code/fortuity/bmn/twistedindex.py).  F(theta, phi) = sum_n tr_adj((t sigma)^n) (1-u^n)^3/n^3 / (N^2-1),
closed form from cardy_lib.su_twisted_data (checked against explicit matrices in cardy_validate.py).
Symmetries used (checked numerically below): th_k -> th_k + pi, th_k -> -th_k, permutations of the th_k, phi -> -phi (complex
conjugation), so the grid covers the cube [0, pi/2]^r x [0, pi] (every face is a reflection symmetry; permutation images of a
maximum appear r! times and are deduplicated).
Involutive points (t^2 central): th = 0 (so-type for N odd, sp-type for N even) and, N even only, th = (pi/2,...,pi/2) (so-type).
Criterion (KKLP's rule, below their (4.14)): compare Re[(-F)^{1/3}] (max-real-part root), relative to S = Re[(-G(e^{2pi i/3}))^{1/3}].
Outputs: protected slice phi = 2pi/3 (max over the torus) and the classical maximum over torus x phi, refined by zooming.
Measured on an M1 Max (one core): N=3 (1/400, 1/1200) 0.8 s; N=4 (1/80, 1/480) 5.5 s; N=5 (1/80, 1/480) 6.5 s.
Estimates (grid cost ~5 us/point for N=4,5, ~9 us for N=6, ~14 us for N=8, plus refinement):
  N=6 (1/40, 1/240): ~30 s;  N=8 (1/20, 1/120): ~1.5-2 min;  N=4 or 5 at (1/240, 1/1440): ~2 min, < 1 GB.
"""
import sys
import time
import itertools
import numpy as np
from cardy_lib import Geven, Godd, ReS, S0, su_twisted_data, F_su_twisted

t0 = time.time()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
dth = eval(sys.argv[2]) if len(sys.argv) > 2 else 1 / 80
dph = eval(sys.argv[3]) if len(sys.argv) > 3 else 1 / 480
ncand = int(sys.argv[4]) if len(sys.argv) > 4 else 12   # max number of distinct grid maxima refined
r = N // 2
even, odd, c0, eps = su_twisted_data(N)
ue, me = np.unique(even, axis=0, return_counts=True)       # group identical frequency rows
uo, mo = np.unique(odd, axis=0, return_counts=True)
dimG = N * N - 1


def F(theta, phi):
    theta = np.atleast_2d(theta)
    out = np.zeros(phi.shape, complex)
    for row, m in zip(ue, me):
        out += m * Geven(theta @ row, phi)
    if c0:
        out += c0 * Godd(0 * phi, phi)
    for row, m in zip(uo, mo):
        out += eps * m * Godd(theta @ row, phi)
    return out / dimG


def val(theta, phi):
    return ReS(F(theta, phi)) / S0


# consistency of the grouped evaluation and of the symmetries
rng = np.random.default_rng(0)
th = rng.uniform(-3, 3, (50, r)); ph = rng.uniform(-3, 3, 50)
chk = np.max(np.abs(F(th, ph) - F_su_twisted(N, th, ph)))
sym = 0.0
for k in range(r):
    sh = th.copy(); sh[:, k] += np.pi
    sym = max(sym, np.max(np.abs(F(sh, ph) - F(th, ph))))
    fl = th.copy(); fl[:, k] *= -1
    sym = max(sym, np.max(np.abs(F(fl, ph) - F(th, ph))))
sym = max(sym, np.max(np.abs(F(th[:, ::-1], ph) - F(th, ph))), np.max(np.abs(F(th, -ph) - np.conj(F(th, ph)))))
print("SU(%d): rank of twisted torus r=%d; grouped-vs-library |dF| = %.1e; symmetry violations max %.1e" % (N, r, chk, sym))

# grid on the cube [0, pi/2]^r x [0, pi]; every face is a reflection symmetry (th -> -th, th -> pi - th, phi -> -phi,
# phi -> 2 pi - phi), so reflect-padding gives exact neighbours and grid local maxima can be read off directly.
n1 = int(round(0.5 / dth))
th1 = np.arange(n1 + 1) * dth * np.pi
nphi = int(round(1 / dph))
PH = np.arange(nphi + 1) * dph * np.pi
TH = np.array(list(itertools.product(th1, repeat=r)))
shape = (n1 + 1,) * r + (nphi + 1,)
print("grid: %d torus points x %d phi values = %d evaluations" % (len(TH), len(PH), len(TH) * len(PH)))
V = np.empty((len(TH), len(PH)))
chunk = max(1, 400000 // len(PH))
for i in range(0, len(TH), chunk):
    T = np.repeat(TH[i:i + chunk], len(PH), axis=0)
    P = np.tile(PH, len(TH[i:i + chunk]))
    V[i:i + chunk] = val(T, P).reshape(-1, len(PH))
V = V.reshape(shape)
t_grid = time.time() - t0


def local_maxima(A):
    Ap = np.pad(A, 1, mode='reflect')
    core = tuple(slice(1, -1) for _ in A.shape)
    ok = np.ones(A.shape, bool)
    for sh in itertools.product((-1, 0, 1), repeat=A.ndim):
        if any(sh):
            sl = tuple(slice(1 + s_, Ap.shape[i] - 1 + s_) for i, s_ in enumerate(sh))
            ok &= A >= Ap[sl]
    return np.argwhere(ok)


def fold(x):
    th = np.mod(x[:r], np.pi)
    th = np.where(th > np.pi / 2, np.pi - th, th)
    th = np.sort(th)[::-1]
    p = np.mod(x[r], 2 * np.pi)
    p = 2 * np.pi - p if p > np.pi else p
    return np.concatenate([th, [p]])


def zoom(x0, w0, fixed_phi=None, m=None, tol=1e-11, maxit=400):
    """Vectorized pattern search: recentre on the best of an m^d grid; shrink by 3 only when the best is interior."""
    x = np.array(x0, float); w = float(w0)
    d = r if fixed_phi is not None else r + 1
    m = m or (9 if d <= 3 else 5)
    offs = np.array(list(itertools.product(np.linspace(-1, 1, m), repeat=d)))
    edge = np.any(np.abs(offs) == 1, axis=1)
    best = -np.inf
    for _ in range(maxit):
        X = x[:d] + w * offs
        v = val(X, np.full(len(X), fixed_phi)) if fixed_phi is not None else val(X[:, :r], X[:, r])
        k = int(np.argmax(v))
        x[:d] = X[k]; best = v[k]
        if not edge[k]:
            w /= 3
        if w < tol:
            break
    if fixed_phi is not None:
        x = np.concatenate([x[:r], [fixed_phi]])
    return fold(x), best


def fmt(x):
    return "theta/pi = (" + ", ".join("%.6f" % (a / np.pi) for a in x[:r]) + "), phi/pi = %.6f" % (x[r] / np.pi)


def refine_all(idx_list, values_at, xs_of, fixed_phi=None, cap=None):
    cap = cap or ncand
    res, seen = [], []
    for idx in sorted(idx_list, key=lambda i: -values_at[tuple(i)]):
        g = values_at[tuple(idx)]
        if any(abs(g - s_) < 1e-9 for s_ in seen):          # permutation images of a refined grid maximum
            continue
        seen.append(g)
        if len(seen) > cap:
            break
        x, v = zoom(xs_of(idx), 2 * dth * np.pi, fixed_phi=fixed_phi)
        if not any(np.max(np.abs(x - y)) < 1e-6 for y, _ in res):
            res.append((x, v))
    return sorted(res, key=lambda t: -t[1])


# special points
specials = {"th=0" + (" (so-type)" if N % 2 else " (sp-type)"): np.zeros(r)}
if N % 2 == 0:
    specials["th=pi/2 (so-type)"] = np.full(r, np.pi / 2)
for nz in range(1, r):
    specials["th=(pi/2)^%d,0^%d (not involutive)" % (nz, r - nz)] = np.array([np.pi / 2] * nz + [0.0] * (r - nz))
for name, x in specials.items():
    pv = val(x[None, :], np.array([2 * np.pi / 3]))[0]
    fine = np.linspace(0, np.pi, 20001)
    vf = val(np.repeat(x[None, :], len(fine), axis=0), fine)
    kf = int(np.argmax(vf))
    print("  %-34s protected %.6f | max over phi %.6f at phi/pi = %.5f" % (name, pv, vf[kf], fine[kf] / np.pi))

# protected slice phi = 2pi/3: torus local maxima, refined
VP = val(TH, np.full(len(TH), 2 * np.pi / 3)).reshape((n1 + 1,) * r)
lm = local_maxima(VP)
print("protected slice phi = 2pi/3: %d grid local maxima (incl. permutation images); refined:" % len(lm))
for x, v in refine_all(lm, VP, lambda idx: np.concatenate([th1[list(idx)], [2 * np.pi / 3]]), fixed_phi=2 * np.pi / 3):
    print("    %.7f at %s" % (v, fmt(x)))

# classical: torus x phi local maxima, refined
lm = local_maxima(V)
print("classical (torus x phi): %d grid local maxima (incl. permutation images); refined:" % len(lm))
for x, v in refine_all(lm, V, lambda idx: np.concatenate([th1[list(idx[:r])], [PH[idx[r]]]])):
    print("    %.7f at %s" % (v, fmt(x)))
print("time: grid %.2f s, total %.2f s" % (t_grid, time.time() - t0))
