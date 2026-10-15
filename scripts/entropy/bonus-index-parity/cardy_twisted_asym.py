"""Side check for (3): at the involutive twisted points of SU(3), SU(4), SU(5), allow unequal bonus phases
u_I = e^{i th_I} (the 5-parameter KKLP index (4.1) without unrefining u = v = w).  At an involution s:
  F = [c_odd (G3(th) - G3(2th)/8) + (N^2-1) G3(2th)/8]/(N^2-1),   G3(th) = sum_n prod_I (1 - e^{i n th_I})/n^3,
c_odd = 1-N (so-type) or N+1 (sp-type).  (a) maximum over the whole 3-torus, (b) maximum on the slice x = 1 (sum th = 2pi).
Runs in ~3 s.
"""
import time
import itertools
import numpy as np
from cardy_lib import li3c, ReS, S0, ZETA3

t0 = time.time()


def G3(t1, t2, t3):
    return (ZETA3 - li3c(t1) - li3c(t2) - li3c(t3) + li3c(t1 + t2) + li3c(t1 + t3) + li3c(t2 + t3) - li3c(t1 + t2 + t3))


def Ftw(N, codd, t1, t2, t3):
    g1, g2 = G3(t1, t2, t3), G3(2 * t1, 2 * t2, 2 * t3)
    return (codd * (g1 - g2 / 8) + (N * N - 1) * g2 / 8) / (N * N - 1)


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
    return np.mod(x, 2 * np.pi), best


n = 72
g = np.arange(n) * 2 * np.pi / n
T1, T2, T3 = np.meshgrid(g, g, g, indexing='ij')
n2 = 288
g2 = np.arange(n2) * 2 * np.pi / n2
U1, U2 = np.meshgrid(g2, g2, indexing='ij')
for N, typ, codd in [(3, 'so', -2), (4, 'so', -3), (4, 'sp', 5), (5, 'so', -4)]:
    f3 = lambda X: ReS(Ftw(N, codd, X[:, 0], X[:, 1], X[:, 2])) / S0
    A = ReS(Ftw(N, codd, T1, T2, T3)) / S0
    k = np.unravel_index(np.argmax(A), A.shape)
    x, v = zoom(f3, [g[k[0]], g[k[1]], g[k[2]]], 2 * (g[1] - g[0]))
    f2 = lambda X: ReS(Ftw(N, codd, X[:, 0], X[:, 1], 2 * np.pi - X[:, 0] - X[:, 1])) / S0
    B = ReS(Ftw(N, codd, U1, U2, 2 * np.pi - U1 - U2)) / S0
    kb = np.unravel_index(np.argmax(B), B.shape)
    xb, vb = zoom(f2, [g2[kb[0]], g2[kb[1]]], 2 * (g2[1] - g2[0]))
    sym = ReS(Ftw(N, codd, *(np.full(1, 2 * np.pi / 3),) * 3))[0] / S0
    print("SU(%d) %s-type: (a) 3-torus max %.7f at th/pi = %s ;  (b) x=1 slice max %.7f at th/pi = (%.5f, %.5f, %.5f)"
          " [symmetric protected %.7f]" % (N, typ, v, np.round(x / np.pi, 5), vb, xb[0] / np.pi, xb[1] / np.pi,
                                          (2 * np.pi - xb[0] - xb[1]) % (2 * np.pi) / np.pi, sym))
print("time %.2f s" % (time.time() - t0))
