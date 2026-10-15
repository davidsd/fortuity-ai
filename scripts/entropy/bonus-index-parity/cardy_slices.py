"""Side check for (1)-(2): KKLP unrefine u = v = w.  Is the symmetric point also the maximum of
h(theta) = Re[(-G(u1,u2,u3))^{1/3}],  G = sum_n prod_I (1-u_I^n)/n^3 = zeta(3) - sum Li3(u_I) + sum_{I<J} Li3(u_I u_J) - Li3(u1 u2 u3),
u_I = e^{i theta_I}, (a) over the whole 3-torus (classical, all three bonus-type charges free), (b) on the slice
x = u1 u2 u3 = 1 (protected; analytically -G = i theta1 theta2 theta3/2 for theta_I in (0,2pi), sum = 2pi), (c) on x = -1?
Runs in ~3 s.  (Interpretation: a critical point of h on a slice with equal Lagrange multipliers has equal R-charges.)
"""
import time
import itertools
import numpy as np
from cardy_lib import li3c, ReS, S0, ZETA3

t0 = time.time()


def Gfull(t1, t2, t3):
    return (ZETA3 - li3c(t1) - li3c(t2) - li3c(t3) + li3c(t1 + t2) + li3c(t1 + t3) + li3c(t2 + t3) - li3c(t1 + t2 + t3))


def h(t1, t2, t3):
    return ReS(Gfull(t1, t2, t3)) / S0


def local_max_wrap(A):
    Ap = np.pad(A, 1, mode='wrap')
    ok = np.ones(A.shape, bool)
    for sh in itertools.product((-1, 0, 1), repeat=A.ndim):
        if any(sh):
            ok &= A >= Ap[tuple(slice(1 + s_, Ap.shape[i] - 1 + s_) for i, s_ in enumerate(sh))]
    return np.argwhere(ok)


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


def report(label, A, grid, f, top=5):
    lm = local_max_wrap(A)
    seen, out = [], []
    for idx in sorted(lm, key=lambda i: -A[tuple(i)]):
        g = A[tuple(idx)]
        if any(abs(g - s) < 1e-9 for s in seen):
            continue
        seen.append(g)
        x, v = zoom(f, grid[list(idx)], 2 * (grid[1] - grid[0]))
        if not any(abs(v - w) < 1e-7 for _, w in out):
            out.append((np.mod(x, 2 * np.pi), v))
        if len(out) >= top:
            break
    print("%s: %d grid local maxima; distinct refined:" % (label, len(lm)))
    for x, v in out:
        print("    %.7f at theta/pi = %s" % (v, np.round(x / np.pi, 6)))


n = 90
g = np.arange(n) * 2 * np.pi / n
T1, T2, T3 = np.meshgrid(g, g, g, indexing='ij')
A = h(T1, T2, T3)
report("(a) whole 3-torus (classical)", A, g, lambda X: h(X[:, 0], X[:, 1], X[:, 2]))
print("    symmetric reference: phi_0 = 0.708158923 pi -> 1.0121344")
for c, lab in [(2 * np.pi, "(b) slice x = +1 (protected)"), (np.pi, "(c) slice x = -1")]:
    n2 = 360
    g2 = np.arange(n2) * 2 * np.pi / n2
    U1, U2 = np.meshgrid(g2, g2, indexing='ij')
    B = h(U1, U2, c - U1 - U2)
    report(lab, B, g2, lambda X, c=c: h(X[:, 0], X[:, 1], c - X[:, 0] - X[:, 1]), top=4)
print("time %.2f s" % (time.time() - t0))
