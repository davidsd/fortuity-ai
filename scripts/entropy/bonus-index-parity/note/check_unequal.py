"""Independent check of reframing (c): KKLP excess delta0(xi) = S_cl,max/S_prot - 1 at unequal R-charges,
leading Cardy order, collapsed (identity) holonomy. Written from scratch (does not import rf/*.py).
S/K = g(mu) + sum_I mu_I xi_I,  g = (-G3)^(1/3) (principal root), K = 3 (dimG j1 j2)^(1/3), xi_I = (R_I - Rbar)/K,
G3 = Li3(1) - sum Li3(u_I) + sum_{I<J} Li3(u_I u_J) - Li3(u1 u2 u3), u_I = e^{-mu_I}.
Protected: sum mu = -2 pi i (closed form: prod (lam + 2 pi i xi_I) = 4 pi^3 i/27, S/K = lam).
Classical maximum over the bonus charge: sum mu = -i Phi, Phi real with Im kappa = 0 (KKLP (4.11)-(4.13)).
BH charges (for labelling only): Q_I/K = cbar + xi_I, cbar = Im(lam)/(2 pi) from Im S_prot = 0.
Usage: python check_unequal.py A B [s1,s2,...] -> ray xi = s*(A, B, -A-B), default s = 0,0.25,0.5,0.75,1 (about 14 s; about 2.8 s per s value)."""
import sys, mpmath as mp
mp.mp.dps = 20
L2 = lambda z: mp.polylog(2, z); L3 = lambda z: mp.polylog(3, z)
def G3(m):
    u = [mp.exp(-x) for x in m]
    return L3(1) - sum(L3(x) for x in u) + L3(u[0]*u[1]) + L3(u[0]*u[2]) + L3(u[1]*u[2]) - L3(u[0]*u[1]*u[2])
def dG3(m):
    u = [mp.exp(-x) for x in m]; P = u[0]*u[1]*u[2]
    return [L2(u[I]) - sum(L2(u[I]*u[J]) for J in range(3) if J != I) + L2(P) for I in range(3)]
def stationary(nu, xi, start):
    """extremize g + sum mu xi over mu with sum mu = nu fixed; returns (S/K, kappa, mu)."""
    def eqs(m1, m2):
        m = [m1, m2, nu - m1 - m2]; g = (-G3(m)) ** (mp.mpf(1)/3); d = dG3(m)
        dg = [-d[I] / (3 * g**2) for I in range(3)]
        return [dg[0] - dg[2] + xi[0] - xi[2], dg[1] - dg[2] + xi[1] - xi[2]]
    m1, m2 = mp.findroot(eqs, start, tol=1e-26, maxsteps=40, verify=False)
    m = [m1, m2, nu - m1 - m2]; g = (-G3(m)) ** (mp.mpf(1)/3); d = dG3(m)
    kappa = -d[2] / (3 * g**2) + xi[2]
    return g + sum(m[I]*xi[I] for I in range(3)), kappa, (m1, m2)
def protected_closed(xi, lam0):
    # roots of prod(lam + 2 pi i xi_I) - 4 pi^3 i/27; pick the one closest to the previous root (continuation)
    c = [2j*mp.pi*x for x in xi]
    co = [1, sum(c), c[0]*c[1] + c[0]*c[2] + c[1]*c[2], c[0]*c[1]*c[2] - 4*mp.pi**3*1j/27]
    return min(mp.polyroots(co), key=lambda r: abs(r - lam0))
A, B = float(sys.argv[1]), float(sys.argv[2])
lam = (4*mp.pi**3/27) ** (mp.mpf(1)/3) * mp.exp(1j*mp.pi/6)
Phi = 3 * mp.mpf('2.22474687153909')                      # 3 phi_0 at xi = 0
cl_start = (-1j*Phi/3, -1j*Phi/3); pr_start = (-2j*mp.pi/3, -2j*mp.pi/3)
for s in ([float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else (0.0, 0.25, 0.5, 0.75, 1.0)):
    xi = [s*A, s*B, -s*(A + B)]
    lam = protected_closed(xi, lam)
    Sp, kp, pr_start = stationary(-2j*mp.pi, xi, pr_start)   # polylog form on sum mu = -2 pi i (cross-check)
    # classical: secant on Im kappa(Phi) = 0
    def imk(P):
        global cl_start
        S, k, cl_start = stationary(-1j*P, xi, cl_start); return mp.im(k), S
    P0, P1 = Phi - mp.mpf('0.01'), Phi + mp.mpf('0.01'); f0, _ = imk(P0); f1, S1 = imk(P1)
    for it in range(12):
        P2 = P1 - f1 * (P1 - P0) / (f1 - f0); P0, f0 = P1, f1; P1 = P2; f1, S1 = imk(P1)
        if abs(f1) < 1e-13: break
    Phi = P1
    cbar = mp.im(lam) / (2*mp.pi); Q = [cbar + x for x in xi]; fr = [q / sum(Q) for q in Q]
    print(f"xi=({xi[0]:+.4f},{xi[1]:+.4f},{xi[2]:+.4f})  BH fractions=({float(fr[0]):.3f},{float(fr[1]):.3f},{float(fr[2]):.3f})"
          f"  ReS_prot/K={float(mp.re(lam)):.10f} (polylog form {float(mp.re(Sp)):.10f})"
          f"  2pi*sqrt(Q1Q2+Q2Q3+Q3Q1)/K={float(2*mp.pi*mp.sqrt(Q[0]*Q[1]+Q[1]*Q[2]+Q[0]*Q[2])):.10f}"
          f"  Phi*/pi={float(Phi/mp.pi):.4f}  delta0={float(mp.re(S1)/mp.re(lam) - 1):.6f}")
