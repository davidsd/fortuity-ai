"""(2) revisited: the untwisted KKLP index at x = -1 (u^3 = -1: phi = pi/3, pi), maximized over the holonomy, for
SU(2), SU(3), SU(4), SO(7), USp(6), G2.  At u = -1, G(u^2) = G(1) = 0, so at any t with t^2 central (Ad_t involutive),
F = tr_adj(t) * 7 zeta(3) / dimG; an element acting as -1 on a Cartan subalgebra has tr_adj(t) = -rank and gives
-F = 7 zeta(3) rank/dimG > 0 (sign-free Cardy growth of `rank' free abelian multiplets), versus the identity's
-F = -7 zeta(3) (cube-root phase pi/3, factor 1/2).  It wins iff rank/dimG > 1/8.  Runs in ~3 s.
"""
import time
import itertools
import numpy as np
from cardy_lib import Gz, G, ReS, S0, ZETA3, roots

t0 = time.time()


def su_roots(N):
    R = []
    for i in range(N):
        for j in range(N):
            if i != j:
                v = np.zeros(N - 1)
                if i < N - 1: v[i] += 1
                if j < N - 1: v[j] -= 1
                R.append(v)
    return np.array(R), N - 1, N - 1


def zoom(f, x0, w, m=9, tol=1e-11):
    x = np.array(x0, float); d = len(x)
    offs = np.array(list(itertools.product(np.linspace(-1, 1, m), repeat=d)))
    edge = np.any(np.abs(offs) == 1, axis=1)
    for _ in range(400):
        X = x + w * offs
        v = f(X); k = int(np.argmax(v)); x = X[k]; best = v[k]
        if not edge[k]:
            w /= 3
        if w < tol:
            break
    return x, best


for grp in ['su2', 'su3', 'su4', 'so7', 'usp6', 'g2']:
    if grp.startswith('su'):
        R, rank, nd = su_roots(int(grp[2]))
    elif grp == 'g2':
        R = np.array([[r_[0], r_[1]] for r_ in roots('g2')[0]]); rank, nd = 2, 2
    else:
        R, rank, nd = roots(grp)
    dimG = rank + len(R)
    Ru, Rm = np.unique(R, axis=0, return_counts=True)

    def val(theta, phi):
        theta = np.atleast_2d(theta)
        Fv = rank * G(phi)
        for row, m in zip(Ru, Rm):
            Fv = Fv + m * Gz(theta @ row, phi)
        return ReS(Fv / dimG) / S0

    n = {1: 720, 2: 144, 3: 40}[nd]
    g = np.arange(n) * 2 * np.pi / n
    TH = np.array(list(itertools.product(g, repeat=nd)))
    line = "%-5s dimG=%2d rank=%d |" % (grp, dimG, rank)
    for phi, lab in [(np.pi, "pi"), (np.pi / 3, "pi/3")]:
        V = val(TH, np.full(len(TH), phi))
        k = int(np.argmax(V))
        x, v = zoom(lambda X: val(X, np.full(len(X), phi)), TH[k], 2 * (g[1] - g[0]))
        vid = val(np.zeros((1, nd)), np.array([phi]))[0]
        line += " phi=%s: identity %.6f, max over torus %.6f at theta/pi=%s |" % (lab, vid, v, np.round(np.mod(x, 2 * np.pi) / np.pi, 4))
    pred = (7 * ZETA3 * rank / dimG) ** (1 / 3) / S0
    line += " (7 zeta3 rank/dimG)^(1/3)/S0 = %.6f%s" % (pred, " [no -1-on-Cartan element in G]" if grp in ('su3', 'su4') else "")
    print(line)
print("time %.2f s" % (time.time() - t0))
