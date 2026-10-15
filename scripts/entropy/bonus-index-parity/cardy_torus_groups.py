"""(4) Untwisted maximal-torus scan of the bonus-refined Cardy exponent for SO(7), USp(6), G2, SU(3), SU(4).

usage:  python cardy_torus_groups.py GROUP [dth_frac] [dphi_frac]      GROUP in so7, usp6, g2, su3, su4
F(theta, phi) = sum_n tr_adj(t^n)(1-u^n)^3/n^3 / dimG = [rank G(u) + sum_roots Gz(alpha.theta, phi)]/dimG.
At t = 1 (or any central t) F = G(u), so every ratio Re[(-F)^{1/3}]/S is independent of dim G: the identity saddle gives
1.0121344 (max over phi) and 1 (phi = 2pi/3) for every gauge group.  The scan checks that no other holonomy beats it.
Domain: B3/C3: cube [0, pi]^3 (faces are reflection symmetries: theta_i -> -theta_i, theta_i -> 2pi - theta_i);
G2, SU(N): cube [0, 2pi)^rank, periodic.  phi in [0, pi] (phi -> -phi is complex conjugation, the Weyl group contains -1
for B3, C3, G2; for SU(N), theta -> -theta is the outer automorphism which maps tr_adj(t^n) to itself).
Runtime (M1 Max): so7/usp6 with defaults ~4 s; g2 ~2 s; su3 ~1 s; su4 (dth 1/8) ~3 s.
"""
import sys
import time
import itertools
import numpy as np
from cardy_lib import Gz, G, ReS, S0, roots

t0 = time.time()
grp = sys.argv[1] if len(sys.argv) > 1 else 'so7'
dth = eval(sys.argv[2]) if len(sys.argv) > 2 else 1 / 16
dph = eval(sys.argv[3]) if len(sys.argv) > 3 else 1 / 120
if grp in ('su3', 'su4'):
    N = int(grp[2])
    R = []
    for i in range(N):
        for j in range(N):
            if i != j:
                v = np.zeros(N - 1)
                if i < N - 1: v[i] += 1
                if j < N - 1: v[j] -= 1
                R.append(v)               # theta_N = 0 gauge (roots orthogonal to (1,...,1))
    R, rank, nd = np.array(R), N - 1, N - 1
    periodic, L = True, 2 * np.pi
else:
    R, rank, nd = roots(grp)
    if grp == 'g2':
        R, nd = R[:, :2] - R[:, 2:3] * 0, 2   # theta = (x, y, 0): drop the third coordinate
        R = np.array([[r_[0], r_[1]] for r_ in roots('g2')[0]])
        periodic, L = True, 2 * np.pi
    else:
        periodic, L = False, np.pi
dimG = rank + len(R)
Ru, Rm = np.unique(R, axis=0, return_counts=True)


def val(theta, phi):
    theta = np.atleast_2d(theta)
    Fv = rank * G(phi)
    for row, m in zip(Ru, Rm):
        Fv = Fv + m * Gz(theta @ row, phi)
    return ReS(Fv / dimG) / S0


n1 = int(round((L / np.pi) / dth))
th1 = np.arange(n1 if periodic else n1 + 1) * dth * np.pi
nphi = int(round(1 / dph))
PH = np.arange(nphi + 1) * dph * np.pi
TH = np.array(list(itertools.product(th1, repeat=nd)))
print("%s: dim G = %d, rank %d, %d roots; grid %d torus pts x %d phi = %d" % (grp, dimG, rank, len(R), len(TH), len(PH), len(TH) * len(PH)))
V = np.empty((len(TH), len(PH)))
chunk = max(1, 300000 // len(PH))
for i in range(0, len(TH), chunk):
    T = np.repeat(TH[i:i + chunk], len(PH), axis=0)
    V[i:i + chunk] = val(T, np.tile(PH, len(TH[i:i + chunk]))).reshape(-1, len(PH))
V = V.reshape((len(th1),) * nd + (len(PH),))
tg = time.time() - t0


def local_maxima(A):
    # torus directions: wrap (periodic) or reflect (B3/C3 faces); phi direction: reflect
    Ap = A
    for ax in range(A.ndim):
        mode = 'wrap' if (periodic and ax < nd) else 'reflect'
        pw = [(0, 0)] * A.ndim; pw[ax] = (1, 1)
        Ap = np.pad(Ap, pw, mode=mode)
    ok = np.ones(A.shape, bool)
    for sh in itertools.product((-1, 0, 1), repeat=A.ndim):
        if any(sh):
            sl = tuple(slice(1 + s_, Ap.shape[i] - 1 + s_) for i, s_ in enumerate(sh))
            ok &= A >= Ap[sl]
    return np.argwhere(ok)


def zoom(x0, w0, fixed_phi=None, m=7, tol=1e-10, maxit=300):
    x = np.array(x0, float); w = float(w0)
    d = nd if fixed_phi is not None else nd + 1
    offs = np.array(list(itertools.product(np.linspace(-1, 1, m), repeat=d)))
    edge = np.any(np.abs(offs) == 1, axis=1)
    for _ in range(maxit):
        X = x[:d] + w * offs
        v = val(X, np.full(len(X), fixed_phi)) if fixed_phi is not None else val(X[:, :nd], X[:, nd])
        k = int(np.argmax(v)); x[:d] = X[k]; best = v[k]
        if not edge[k]:
            w /= 3
        if w < tol:
            break
    return x, best


def summarize(Vgrid, fixed_phi, label, top=6):
    lm = local_maxima(Vgrid)
    vals = np.array([Vgrid[tuple(i)] for i in lm])
    order = np.argsort(vals)[::-1]
    out, seen = [], []
    for k in order:
        idx = lm[k]
        if any(abs(vals[k] - g) < 1e-9 for g in seen):    # Weyl images of an already refined grid maximum
            continue
        seen.append(vals[k])
        x0 = np.concatenate([th1[list(idx[:nd])], [fixed_phi if fixed_phi is not None else PH[idx[nd]]]])
        x, v = zoom(x0, 2 * dth * np.pi, fixed_phi=fixed_phi)
        if fixed_phi is not None:
            x[nd] = fixed_phi
        if not any(abs(v - w) < 1e-7 for _, w in out):
            out.append((x, v))
        if len(out) >= top:
            break
    print("%s: %d grid local maxima; distinct refined values:" % (label, len(lm)))
    for x, v in out:
        print("    %.7f at theta/pi = %s, phi/pi = %.6f" % (v, np.round(x[:nd] / np.pi, 5), x[nd] / np.pi))


ident = np.zeros(nd)
print("identity: protected %.7f ; max over phi %.7f" % (val(ident[None], np.array([2 * np.pi / 3]))[0],
      val(np.repeat(ident[None], 20001, 0), np.linspace(0, np.pi, 20001)).max()))
kp = int(round((2 / 3) / dph))
VP = V[..., kp] if abs(PH[kp] - 2 * np.pi / 3) < 1e-12 else val(TH, np.full(len(TH), 2 * np.pi / 3)).reshape((len(th1),) * nd)
summarize(VP, 2 * np.pi / 3, "protected slice phi = 2pi/3")
summarize(V, None, "classical (torus x phi)")
print("time: grid %.2f s, total %.2f s" % (tg, time.time() - t0))
