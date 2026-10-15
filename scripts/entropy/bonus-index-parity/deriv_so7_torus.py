"""T-038: SO(7) Cardy function over the maximal torus (Weyl chamber pi >= t1 >= t2 >= t3 >= 0), numpy, < 10 s.
F(t, u) = sum_n a_n chi_adj(g^n) = 3 G(1) + sum_{roots al} G(e^{i al.t}),  a_n = -(1-u^n)^3/n^3,  u = e^{i phi},
G(y) = -[Li3(y) - 3 Li3(u y) + 3 Li3(u^2 y) - Li3(u^3 y)]; Li3 on |z| = 1 by a truncated series (error ~ 1/(2 M^2)).
Entropy ~ Re F^{1/3} (best root). Compares the max over the chamber with the identity, at the protected point
phi = 2pi/3 and over a phi grid around phi_0 (KKLP)."""
import time, itertools
import numpy as np
t0 = time.time()
M = 1500
n = np.arange(1, M + 1)
def Li3_circle(psi):                       # psi: array of angles
    psi = np.asarray(psi)[..., None]
    return (np.exp(1j * n * psi) / n ** 3).sum(-1)
R = []
for i, j in itertools.combinations(range(3), 2):
    for s1 in (1, -1):
        for s2 in (1, -1):
            v = [0, 0, 0]; v[i] = s1; v[j] = s2; R.append(v)
for i in range(3):
    for s in (1, -1):
        v = [0, 0, 0]; v[i] = s; R.append(v)
R = np.array(R, dtype=float)               # 18 roots of B3
assert len(R) == 18
K = 13
grid = np.linspace(0, np.pi, K)
pts = np.array([p for p in itertools.product(grid, repeat=3) if p[0] >= p[1] >= p[2]])
def F_all(phi):
    ang = pts @ R.T                         # (P, 18)
    tot = 0
    for c, k in ((1, 0), (-3, 1), (3, 2), (-1, 3)):
        tot = tot + c * Li3_circle(ang + k * phi).sum(-1) + 3 * c * Li3_circle(np.array(k * phi))
    return -tot                            # = 3 G(1) + sum_roots G
def best_re(F):
    r, th = np.abs(F) ** (1 / 3), np.angle(F)
    return np.max([r * np.cos((th + 2 * np.pi * k) / 3) for k in range(3)], axis=0)
prot = best_re(F_all(2 * np.pi / 3))
i0 = np.where(np.all(pts == 0, axis=1))[0][0]
print('chamber points: %d ; protected point: identity %.5f, max %.5f at t/pi = %s' %
      (len(pts), prot[i0], prot.max(), np.round(pts[prot.argmax()] / np.pi, 3)))
best = (-1, None, None)
for phi in np.linspace(0.69, 0.73, 9) * np.pi:
    v = best_re(F_all(phi))
    if v.max() > best[0]:
        best = (v.max(), phi, pts[v.argmax()], v[i0])
print('classical: max over chamber and phi grid %.5f at phi/pi = %.4f, t/pi = %s ; identity value there %.5f' %
      (best[0], best[1] / np.pi, np.round(best[2] / np.pi, 3), best[3]))
print('ratio max_classical / max_protected = %.5f (KKLP 1.01213)' % (best[0] / prot.max()))
print('time %.1fs' % (time.time() - t0))
