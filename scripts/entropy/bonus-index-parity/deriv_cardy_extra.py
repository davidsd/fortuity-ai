"""T-038: two small Cardy checks: (1) Z(x) at x = e^{2 pi i k/7} (is m = 8 the smallest m with growth > S?);
(2) closed forms at the trivial twisted holonomy: F_C(phi = pi) = 7 zeta(3) (N-1), ratio (7 zeta(3)/(N+1))^{1/3} / g_prot;
(3) the three branches of x = -1: phi = pi/3, pi, 5pi/3."""
import time, mpmath as mp
from math import gcd
mp.mp.dps = 20
t0 = time.time()
Li3 = lambda z: mp.polylog(3, z)
f = lambda u: -(Li3(1) - 3 * Li3(u) + 3 * Li3(u ** 2) - Li3(u ** 3))
def best(F):
    r, th = mp.cbrt(abs(F)), mp.arg(F)
    return max(r * mp.cos((th + 2 * mp.pi * k) / 3) for k in range(3))
gp = best(f(mp.expj(2 * mp.pi / 3)))
for m in (7, 8):
    vals = [best(f(mp.expj((2 * mp.pi * k / m + 2 * mp.pi * l) / 3))) / gp for k in range(1, m) if gcd(k, m) == 1 for l in range(3)]
    print('m = %d: max growth ratio %.6f' % (m, max(vals)))
print('x = -1 branches phi = pi/3, pi, 5pi/3:', [round(float(best(f(mp.expj(t))) / gp), 5) for t in (mp.pi / 3, mp.pi, 5 * mp.pi / 3)])
print('f(-i pi) = %s  (-7 zeta(3) = %s)' % (mp.nstr(f(mp.mpf(-1)), 12), mp.nstr(-7 * mp.zeta(3), 12)))
for N in (3, 4, 5, 10):
    print('N = %d: (7 zeta(3)/(N+1))^(1/3)/g_prot = %.5f ; ((N+10)/(8(N+1)))^(1/3) = %.5f' %
          (N, mp.cbrt(7 * mp.zeta(3) / (N + 1)) / gp, mp.cbrt(mp.mpf(N + 10) / (8 * (N + 1)))))
print('g_prot = (4 pi^3/27)^(1/3) cos(pi/6) = %.6f ; time %.2fs' % (gp, time.time() - t0))
