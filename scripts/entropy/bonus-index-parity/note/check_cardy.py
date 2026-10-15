"""Leading-Cardy numbers for T-038, independent of the earlier scripts. Runtime about 3 s.
G(u) = Li3(1) - 3Li3(u) + 3Li3(u^2) - Li3(u^3) = sum_n (1-u^n)^3/n^3, KKLP f(mu) = -G(e^{-mu}).
With a holonomy/twist g the exponent is F(g,u) = sum_n tr_adj(g^n)(1-u^n)^3/n^3; the entropy in units of the
protected S of the same group is Re[(-F/dimG)^(1/3)]/S0, principal cube root = maximal real part (KKLP p.19)."""
import numpy as np, mpmath as mp
mp.mp.dps = 25
Li3 = lambda z: mp.polylog(3, z)
G = lambda u: Li3(1) - 3*Li3(u) + 3*Li3(u**2) - Li3(u**3)
root = lambda z: mp.re(z ** (mp.mpf(1)/3))                     # principal root
h = lambda phi: root(-G(mp.exp(1j*phi)))                       # KKLP (4.14)
S0 = h(2*mp.pi/3)
print("S0 =", mp.nstr(S0, 15), " closed form (4pi^3/27)^(1/3) cos(pi/6) =", mp.nstr((4*mp.pi**3/27)**(mp.mpf(1)/3)*mp.cos(mp.pi/6), 15))
print("G(e^{2pi i/3}) =", mp.nstr(G(mp.exp(2j*mp.pi/3)), 12), "; -(4pi^3/27) i =", mp.nstr(-4*mp.pi**3/27, 12), "i")
phi0 = mp.findroot(lambda t: mp.diff(h, t), 2.22)
print("phi0 =", mp.nstr(phi0, 15), "=", mp.nstr(phi0/mp.pi, 12), "pi ; ratio =", mp.nstr(h(phi0)/S0, 15))
print("x=-1, identity: phi=pi/3:", mp.nstr(h(mp.pi/3)/S0, 9), " phi=pi:", mp.nstr(h(mp.pi)/S0, 9),
      " closed sqrt(3)(7z3/4)^(1/3)/pi =", mp.nstr(mp.sqrt(3)*(7*mp.zeta(3)/4)**(mp.mpf(1)/3)/mp.pi, 9))
# window where the classical-index growth exceeds S, and roots of unity x = e^{2 pi i/m}
phi1 = mp.findroot(lambda t: h(t) - S0, 0.76*mp.pi)
print("h > S0 for phi in (2pi/3, %s pi), i.e. arg x in (0, %s)" % (mp.nstr(phi1/mp.pi, 8), mp.nstr(3*phi1 - 2*mp.pi, 6)))
for m in (2, 3, 7, 8, 12, 16, 24):
    c = max(h(2*mp.pi*(k + mp.mpf(1)/m)/3) for k in range(3)) / S0
    print(f"  c_{m} = {mp.nstr(c, 7)}")
# C-trivial groups at u = -1: -1-on-Cartan holonomy gives -F = 7 zeta(3) rank (unsigned gas of rank free multiplets)
z3 = mp.zeta(3)
for name, r, dim in (("SU(2)", 1, 3), ("SU(3)", 2, 8), ("SU(4)", 3, 15), ("SU(5)", 4, 24), ("SO(7)/USp(6)", 3, 21), ("G2", 2, 14)):
    print(f"  (7 zeta3 rank/dimG)^(1/3)/S0 for {name}: {mp.nstr((7*z3*r/dim)**(mp.mpf(1)/3)/S0, 7)}")
# SU(N) C-twist at the so-type involution: F/dimG = [(N+2)G(u^2)/8 - G(u)]/(N+1); sp-type [(N-2)G(u^2)/8 + G(u)]/(N-1)
Fso = lambda N, u: ((N + 2)*G(u**2)/8 - G(u))/(N + 1)
Fsp = lambda N, u: ((N - 2)*G(u**2)/8 + G(u))/(N - 1)
w = mp.exp(2j*mp.pi/3)
for N in (3, 4, 5, 10):
    print(f"  SU({N}) so-type: protected {mp.nstr(root(-Fso(N, w))/S0, 7)} (closed {mp.nstr(((N+10)/(8*mp.mpf(N+1)))**(mp.mpf(1)/3), 7)}),"
          f" u=-1 {mp.nstr(root(-Fso(N, mp.mpf(-1)))/S0, 7)}" + (f"; sp-type protected {mp.nstr(root(-Fsp(N, w))/S0, 7)}" if N % 2 == 0 else ""))
# where the so-type maximum over phi sits, N = 3..22 (numpy Clausen sums on a grid, then mpmath at the grid max)
n = np.arange(1, 40001); ph = np.linspace(0.05, np.pi, 1200)
def Gnp(theta):      # G(e^{i theta}) by direct summation, tail < 1e-9
    E = np.exp(1j*np.outer(theta, n)); return ((1 - E)**3 / n**3).sum(axis=1)
G1, G2 = Gnp(ph), Gnp(2*ph)
rows = []
for N in range(3, 23):
    F = ((N + 2)*G2/8 - G1)/(N + 1)
    val = np.real((-F + 0j)**(1/3)) / float(S0)
    k = int(np.argmax(val)); rows.append((N, ph[k]/np.pi, val[k], float(root(-Fso(N, mp.mpf(-1)))/S0)))
print("  so-type twisted maximum over phi (N, phi/pi, value, value at phi=pi):")
print("   " + "; ".join(f"{N}: {p:.3f}, {v:.4f}, {vpi:.4f}" for N, p, v, vpi in rows))
print("  large-N twisted max = ratio/2 =", mp.nstr(h(phi0)/S0/2, 8), "; Coulomb gas > that iff N+1 <", mp.nstr(7*z3/(h(phi0)/2)**3, 6))
