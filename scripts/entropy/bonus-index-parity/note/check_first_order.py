"""First finite-omega coefficient r1 of the KKLP excess at equal charges, delta = delta0 + r1 eps + O(eps^2),
eps = (N^2/j)^(1/3) (j1 = j2 = j). Independent of rf/first_order_check.py. Runtime < 1 s.
log Z = N^2 [f(m) + (w1+w2) H(m)]/(w1 w2) + ..., f = -G3, H = h + (1/3) sum_I d_I f, h = -(1/2) G3^(Li2),
m_I = mu_I - (w1+w2)/3, sum m = nu (bonus potential; nu = -2 pi i protected, Re nu = 0 at the classical maximum).
At the leading omega-saddle: S/K = f^(1/3) + eps (2/3) H/f^(1/3) + O(eps^2) (envelope theorem for the maximum)."""
import mpmath as mp
mp.mp.dps = 25
def parts(u):
    f = -(mp.polylog(3, 1) - 3*mp.polylog(3, u) + 3*mp.polylog(3, u**2) - mp.polylog(3, u**3))
    h = -(mp.polylog(2, 1) - 3*mp.polylog(2, u) + 3*mp.polylog(2, u**2) - mp.polylog(2, u**3)) / 2
    dsum = -(mp.polylog(2, u) - 2*mp.polylog(2, u**2) + mp.polylog(2, u**3))   # (1/3) sum_I d f/d mu_I
    g = f ** (mp.mpf(1)/3)
    return g, (h + dsum) / g
phi0 = mp.mpf('2.22474687153909')
gp, Hp = parts(mp.exp(2j*mp.pi/3)); gc, Hc = parts(mp.exp(1j*phi0))
a, b, c, d = mp.re(gc), mp.re(2*Hc/3), mp.re(gp), mp.re(2*Hp/3)
print("H at the protected point:", mp.nstr(Hp*gp, 12), " expected -2 pi^2/9 =", mp.nstr(-2*mp.pi**2/9, 12))
print("delta0 =", mp.nstr(a/c - 1, 10), "  r1 =", mp.nstr((b*c - a*d)/c**2, 6))
