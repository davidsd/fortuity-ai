"""Validation of cardy_lib (runs in ~1 s).
(a) li3c vs mpmath.polylog on the unit circle; (b) Gz vs a direct partial sum;
(c) tr_adj((Ad_t o sigma)^n) closed form vs explicit matrices, SU(3)..SU(7), random twisted-torus points, n <= 9;
(d) F_su_twisted at the involutive points vs the closed forms [(N+2)G(u^2)/8 - G(u)]/(N+1) (so) and
    [(N-2)G(u^2)/8 + G(u)]/(N-1) (sp, N even);  (e) untwisted SO(7)/USp(6)/G2 at the identity = G(u)."""
import time
import numpy as np
import mpmath as mp
from cardy_lib import (li3c, Gz, G, ReS, S0, su_twisted_matrix_traces, su_twisted_formula_traces, F_su_twisted,
                       roots, F_untwisted)

t0 = time.time()
rng = np.random.default_rng(1)
ts = np.concatenate([rng.uniform(-20, 20, 400), [0.0, 1e-12, np.pi, -np.pi, 2 * np.pi / 3, 6.28]])
err = max(abs(complex(mp.polylog(3, mp.expj(t))) - li3c(t)) for t in ts)
print("(a) max |li3c - mpmath polylog| over %d points: %.2e" % (len(ts), err))

a, p = 0.7, 2.1
n = np.arange(1, 400001)
direct = np.sum(np.exp(1j * n * a) * (1 - np.exp(1j * n * p)) ** 3 / n ** 3)
print("(b) |Gz - direct sum (n<=4e5)| = %.2e" % abs(Gz(a, p) - direct))

worst = 0
for N in range(3, 8):
    for _ in range(4):
        th = rng.uniform(0, 2 * np.pi, N // 2)
        worst = max(worst, np.max(np.abs(su_twisted_matrix_traces(N, th, 9) - su_twisted_formula_traces(N, th, 9))))
print("(c) max |closed-form - explicit matrix| tr_adj((t sigma)^n), N=3..7, n<=9: %.2e" % worst)
for N in (3, 4, 5, 6):
    tr0 = su_twisted_formula_traces(N, np.zeros(N // 2), 2)
    line = "    N=%d theta=0: tr(sigma)=%+.0f, tr(sigma^2)=%.0f" % (N, tr0[0].real, tr0[1].real)
    if N % 2 == 0:
        trh = su_twisted_formula_traces(N, np.full(N // 2, np.pi / 2), 2)
        line += " ; theta=pi/2: tr=%+.0f, tr^2=%.0f" % (trh[0].real, trh[1].real)
    print(line)

phis = np.array([0.3, 2 * np.pi / 3, 2.2, np.pi])
d = 0
for N in (3, 4, 5, 6, 8):
    r = N // 2
    g1, g2 = G(phis), G(2 * phis)
    so = ((N + 2) * g2 / 8 - g1) / (N + 1)
    if N % 2:
        d = max(d, np.max(np.abs(F_su_twisted(N, np.zeros((4, r)), phis) - so)))
    else:
        sp = ((N - 2) * g2 / 8 + g1) / (N - 1)
        d = max(d, np.max(np.abs(F_su_twisted(N, np.zeros((4, r)), phis) - sp)))
        d = max(d, np.max(np.abs(F_su_twisted(N, np.full((4, r), np.pi / 2), phis) - so)))
print("(d) involutive points vs closed forms (so: theta=0 odd N, theta=pi/2 even N; sp: theta=0 even N): %.2e" % d)

for grp in ('so7', 'usp6', 'g2'):
    R, rk, nd = roots(grp)
    v = F_untwisted(R, rk, np.zeros((4, nd)), phis) - G(phis)
    print("(e) %s: dim=%d, |F(identity) - G| = %.1e" % (grp, rk + len(R), np.max(np.abs(v))))
print("S0 = Re[(4 pi^3 i/27)^{1/3}] = %.15f ; ReS(G(2pi/3)) = %.15f" % (S0, ReS(G(2 * np.pi / 3))))
print("time %.2f s" % (time.time() - t0))
