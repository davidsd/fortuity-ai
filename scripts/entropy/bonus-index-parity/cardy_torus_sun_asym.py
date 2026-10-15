"""(5) Optional cluster scan: SU(N) C-twisted Cardy exponent over the whole twisted torus AND three independent bonus
phases u_I = e^{i phi_I} (KKLP's 5-parameter index (4.1) without unrefining u = v = w).

usage:  python cardy_torus_sun_asym.py N [dth_frac] [dphi_frac] [cap]
  theta grid on [0, pi/2]^r (faces are reflection symmetries), phi_I grid on [0, 2 pi)^3 (periodic), steps in units of pi.
F = [sum_even-rows m Ge3(row.theta) + c0 Go3(0) + eps sum_odd-rows m Go3(row.theta)]/(N^2-1),
Gz3(alpha; phi) = sum_n e^{i n alpha} prod_I (1 - e^{i n phi_I})/n^3, Ge3 = Gz3(2 alpha; 2 phi)/8, Go3 = Gz3 - Ge3.
Reports the global maxima of Re[(-F)^{1/3}]/S (classical) and on the slice x = 1 (phi_3 = 2pi - phi_1 - phi_2, protected).
Measured (M1 Max, one core): N=3, (1/16, 1/12): 1.7 s;  N=4, (1/4, 1/6), cap 3: 2.5 s (calibration, ~6 us/point).
Estimate for N=4 or 5 at (1/16, 1/18) (9^2 x 36^3 = 3.8M points): grid ~30-40 s, local maxima ~5 s, refinement (5-D, cap 12)
~30 s, i.e. ~1-2 min, memory < 0.5 GB.  Cost scales ~ linearly with the number of grid points.
"""
import sys
import time
import itertools
import numpy as np
from cardy_lib import li3c, ReS, S0, su_twisted_data

t0 = time.time()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
dth = eval(sys.argv[2]) if len(sys.argv) > 2 else 1 / 16
dph = eval(sys.argv[3]) if len(sys.argv) > 3 else 1 / 12
cap = int(sys.argv[4]) if len(sys.argv) > 4 else 12
r = N // 2
even, odd, c0, eps = su_twisted_data(N)
ue, me = np.unique(even, axis=0, return_counts=True)
uo, mo = np.unique(odd, axis=0, return_counts=True)


def Gz3(a, p1, p2, p3):
    return (li3c(a) - li3c(a + p1) - li3c(a + p2) - li3c(a + p3) + li3c(a + p1 + p2) + li3c(a + p1 + p3)
            + li3c(a + p2 + p3) - li3c(a + p1 + p2 + p3))


def Ge3(a, p1, p2, p3):
    return Gz3(2 * a, 2 * p1, 2 * p2, 2 * p3) / 8


def F(theta, P):
    theta = np.atleast_2d(theta)
    p1, p2, p3 = P[:, 0], P[:, 1], P[:, 2]
    out = np.zeros(len(P), complex)
    for row, m in zip(ue, me):
        out += m * Ge3(theta @ row, p1, p2, p3)
    if c0:
        z = 0 * p1
        out += c0 * (Gz3(z, p1, p2, p3) - Ge3(z, p1, p2, p3))
    for row, m in zip(uo, mo):
        a = theta @ row
        out += eps * m * (Gz3(a, p1, p2, p3) - Ge3(a, p1, p2, p3))
    return out / (N * N - 1)


def val(X):          # X = (theta_1..theta_r, phi_1, phi_2, phi_3)
    return ReS(F(X[:, :r], X[:, r:])) / S0


# consistency with the symmetric-phase library function
from cardy_lib import F_su_twisted
rng = np.random.default_rng(3)
th = rng.uniform(0, 3, (20, r)); ph = rng.uniform(0, 3, 20)
print("SU(%d): |F(phi,phi,phi) - F_su_twisted| = %.1e" % (N, np.max(np.abs(F(th, np.stack([ph] * 3, 1)) - F_su_twisted(N, th, ph)))))

n1 = int(round(0.5 / dth)); th1 = np.arange(n1 + 1) * dth * np.pi
n2 = int(round(2 / dph)); ph1 = np.arange(n2) * dph * np.pi
axes = [th1] * r + [ph1] * 3
shape = tuple(len(a) for a in axes)
print("grid %s = %d points" % (shape, np.prod(shape)))
V = np.empty(shape).reshape(len(th1) ** r, -1)
TH = np.array(list(itertools.product(th1, repeat=r)))
PP = np.array(list(itertools.product(ph1, repeat=3)))
for i, t in enumerate(TH):
    V[i] = val(np.hstack([np.repeat(t[None], len(PP), 0), PP]))
V = V.reshape(shape)
tg = time.time() - t0


def local_maxima(A):
    Ap = A
    for ax in range(A.ndim):
        pw = [(0, 0)] * A.ndim; pw[ax] = (1, 1)
        Ap = np.pad(Ap, pw, mode='reflect' if ax < r else 'wrap')
    ok = np.ones(A.shape, bool)
    for sh in itertools.product((-1, 0, 1), repeat=A.ndim):
        if any(sh):
            ok &= A >= Ap[tuple(slice(1 + s_, Ap.shape[i] - 1 + s_) for i, s_ in enumerate(sh))]
    return np.argwhere(ok)


def zoom(f, x0, w, m=5, tol=1e-10, maxit=400):
    x = np.array(x0, float); d = len(x)
    offs = np.array(list(itertools.product(np.linspace(-1, 1, m), repeat=d)))
    edge = np.any(np.abs(offs) == 1, axis=1)
    for _ in range(maxit):
        X = x + w * offs
        v = f(X); k = int(np.argmax(v)); x = X[k]; best = v[k]
        if not edge[k]:
            w /= 3
        if w < tol:
            break
    return x, best


def show(A, f, x_of, label):
    lm = sorted(local_maxima(A), key=lambda i: -A[tuple(i)])
    seen, out = [], []
    for idx in lm:
        g = A[tuple(idx)]
        if any(abs(g - s_) < 1e-9 for s_ in seen):
            continue
        seen.append(g)
        if len(seen) > cap:
            break
        x, v = zoom(f, x_of(idx), 2 * max(dth, dph) * np.pi)
        if not any(abs(v - w_) < 1e-7 for _, w_ in out):
            out.append((x, v))
    print("%s: %d grid local maxima; best refined:" % (label, len(lm)))
    for x, v in sorted(out, key=lambda t: -t[1])[:6]:
        print("    %.7f at theta/pi = %s, phi_I/pi = %s" % (v, np.round(np.mod(x[:r], np.pi) / np.pi, 5), np.round(np.mod(x[r:], 2 * np.pi) / np.pi, 5)))


show(V, val, lambda idx: np.array([axes[k][i] for k, i in enumerate(idx)]), "classical (torus x phi_I)")
# protected slice x = 1: phi_3 = 2 pi - phi_1 - phi_2
P2 = np.array(list(itertools.product(ph1, repeat=2)))
W = np.empty((len(TH), len(P2)))
for i, t in enumerate(TH):
    X = np.hstack([np.repeat(t[None], len(P2), 0), P2, (2 * np.pi - P2.sum(1))[:, None]])
    W[i] = val(X)
W = W.reshape((len(th1),) * r + (len(ph1),) * 2)
fs = lambda X: val(np.hstack([X, (2 * np.pi - X[:, r] - X[:, r + 1])[:, None]]))
show_axes = [th1] * r + [ph1] * 2


def local_maxima_slice(A):
    Ap = A
    for ax in range(A.ndim):
        pw = [(0, 0)] * A.ndim; pw[ax] = (1, 1)
        Ap = np.pad(Ap, pw, mode='reflect' if ax < r else 'wrap')
    ok = np.ones(A.shape, bool)
    for sh in itertools.product((-1, 0, 1), repeat=A.ndim):
        if any(sh):
            ok &= A >= Ap[tuple(slice(1 + s_, Ap.shape[i] - 1 + s_) for i, s_ in enumerate(sh))]
    return np.argwhere(ok)


lm = sorted(local_maxima_slice(W), key=lambda i: -W[tuple(i)])[:cap]
best = max((zoom(fs, np.array([show_axes[k][i] for k, i in enumerate(idx)]), 2 * max(dth, dph) * np.pi) for idx in lm),
           key=lambda t: t[1])
x, v = best
print("protected slice x = 1: best refined %.7f at theta/pi = %s, (phi_1, phi_2)/pi = %s" %
      (v, np.round(np.mod(x[:r], np.pi) / np.pi, 5), np.round(np.mod(x[r:], 2 * np.pi) / np.pi, 5)))
print("time: grid %.1f s, total %.1f s" % (tg, time.time() - t0))
