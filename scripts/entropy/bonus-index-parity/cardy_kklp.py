"""(1) KKLP phi_0 and the ratio 1.01213;  (2) untwisted values at x = -1.   Runs in ~1 s.

h(phi) = Re[(-G(e^{i phi}))^{1/3}]  (principal = maximal-real-part root), G(u) = Li3(1)-3Li3(u)+3Li3(u^2)-Li3(u^3).
KKLP (4.13)-(4.14): the maximal classical entropy is 3 (N^2 j1 j2)^{1/3} max_phi h(phi); the protected one is
3 (N^2 j1 j2)^{1/3} h(2 pi/3), eq. (4.6).  Here phi_0 is found by mp.findroot on h'(phi) (closed form via Li2),
replacing the 2000-point 30-digit scan of T-038/kklp_cardy.py (68 s).
"""
import time
import numpy as np
import mpmath as mp
from cardy_lib import li3c, ReS

t0 = time.time()
mp.mp.dps = 30
L2 = lambda z: mp.polylog(2, z)
L3 = lambda z: mp.polylog(3, z)


def Gm(phi):
    u = mp.expj(phi)
    return L3(1) - 3 * L3(u) + 3 * L3(u ** 2) - L3(u ** 3)


def dGm(phi):                                   # dG/dphi, using d/dphi Li3(e^{ik phi}) = i k Li2(e^{ik phi})
    u = mp.expj(phi)
    return -3j * (L2(u) - 2 * L2(u ** 2) + L2(u ** 3))


def croot(z):                                   # maximal-real-part cube root = principal root for arg in (-pi, pi]
    return mp.cbrt(z) if mp.im(z) != 0 or mp.re(z) > 0 else mp.cbrt(-z) * mp.expj(mp.pi / 3)


h = lambda phi: mp.re(croot(-Gm(phi)))
dh = lambda phi: mp.re(croot(-Gm(phi)) ** (-2) / 3 * (-dGm(phi)))

S0 = h(2 * mp.pi / 3)
S0_closed = (4 * mp.pi ** 3 / 27) ** (mp.mpf(1) / 3) * mp.cos(mp.pi / 6)
print("G(e^{2 pi i/3}) =", mp.nstr(Gm(2 * mp.pi / 3), 15), "  vs -4 pi^3 i/27 =", mp.nstr(-4 * mp.pi ** 3 / 27, 15), "i")
print("S0 := h(2pi/3) =", mp.nstr(S0, 20), " closed form (4pi^3/27)^{1/3} cos(pi/6) =", mp.nstr(S0_closed, 20))

# coarse float scan (vectorized) only to bracket, then findroot on h'
grid = np.linspace(0, np.pi, 2001)[1:-1]
hv = ReS(li3c(0 * grid) - 3 * li3c(grid) + 3 * li3c(2 * grid) - li3c(3 * grid))
k = int(np.argmax(hv))
phi0 = mp.findroot(dh, mp.mpf(grid[k]))
d2 = mp.diff(h, phi0, 2)
ratio = h(phi0) / S0
print("phi_0 = %s = %s pi   (KKLP: 2.2247 = 0.7081 pi);  h''(phi_0) = %s (< 0: maximum)"
      % (mp.nstr(phi0, 15), mp.nstr(phi0 / mp.pi, 12), mp.nstr(d2, 6)))
print("ratio max_phi h / h(2pi/3) = %s   (KKLP (4.15): 1.01213)" % mp.nstr(ratio, 15))
print("  i.e. S_cl,max - S = %s S;  surviving fraction <= exp(-%s S)" % (mp.nstr(ratio - 1, 6), mp.nstr(ratio - 1, 6)))
# bonus charge at the maximum, KKLP q = 3(R-T) = -dF/dmu with F = 3K f^{1/3}, mu = -i phi:  q/K = i f^{-2/3} G'(phi)
qK = 1j * croot(-Gm(phi0)) ** (-2) * dGm(phi0)
print("  KKLP bonus charge at the maximum: q/(N^2 j1 j2)^{1/3} = %s  (imaginary part %s: real, as (4.13) requires)"
      % (mp.nstr(mp.re(qK), 10), mp.nstr(mp.im(qK), 3)))
print("  branch check: arg(-G(phi_0))/pi = %s (principal root is the max-Re root)" % mp.nstr(mp.arg(-Gm(phi0)) / mp.pi, 8))
print("  other critical points of h on (0, pi):",
      [mp.nstr(mp.findroot(dh, mp.mpf(g)) / mp.pi, 8) for g in grid[1:-1][np.where(np.diff(np.sign(np.diff(hv))) != 0)[0] + 1]])

print("\n(2) untwisted values at x = u^3 = -1 (3 phi = pi mod 2 pi):")
z3 = mp.zeta(3)
Gpi3 = Gm(mp.pi / 3)
print("  G(e^{i pi/3}) = %s ; closed form -(7/12) zeta(3) - i pi^3/54 = %s" % (mp.nstr(Gpi3, 15),
      mp.nstr(-mp.mpf(7) / 12 * z3 - 1j * mp.pi ** 3 / 54, 15)))
print("  G(-1) = %s ; closed form 7 zeta(3) = %s" % (mp.nstr(Gm(mp.pi), 15), mp.nstr(7 * z3, 15)))
for name, phi in [("phi = pi/3 ", mp.pi / 3), ("phi = pi   ", mp.pi), ("phi = 5pi/3", 5 * mp.pi / 3)]:
    print("  %s: S/S_prot = %s" % (name, mp.nstr(h(phi) / S0, 12)))
print("  closed form at phi = pi: sqrt(3) (7 zeta(3)/4)^{1/3} / pi = %s" % mp.nstr(mp.sqrt(3) * (7 * z3 / 4) ** (mp.mpf(1) / 3) / mp.pi, 12))
print("  max over the x = -1 points: phi = pi (u = -1).")
# reference points also used later
print("\n  reference: h at phi = pi/2 (x = -i): %s ;  phi_0/2 = %s pi" % (mp.nstr(h(mp.pi / 2) / S0, 8), mp.nstr(phi0 / 2 / mp.pi, 8)))
print("time %.2f s" % (time.time() - t0))
