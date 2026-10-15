"""T-041 (added 2026-10-02 after review): the first-order stable strip in every direction, not only theta = 0.

At finite eps, Re mu_I = Re m_I + (2/3) eps Re g at the protected (intersection black hole) saddle, with
(2/3) Re g = (2/3) S0 = 0.9597 at xi = 0. Linearizing the closed form of closed_forms.py,
   Re m_I = -pi (q_J+q_K)(q_I^2 - q_J q_K) / (alpha (alpha^2 + s1^2)),
at q_I = c0 = 432^(-1/3) gives Re m_I = -kappa xi_I + O(xi^2), kappa = pi/(2 sqrt3 c0). All Re mu_I > 0 then needs
max_I xi_I < (2/3) S0 eps / kappa = 8 c0^2 eps, i.e. the largest charge fraction below 1/3 + (8/3) c0 eps. The radius
of this triangle along the ray theta (xi = r (cos th e1 + sin th e2)) is printed from the linearization and from
the exact closed form at r = 1e-4 (cbar solved from prod_{I<J}(q_I+q_J) = 1/54). Runtime: under a second.
"""
import mpmath as mp

mp.mp.dps = 30
PI = mp.pi
E1 = [2 / mp.sqrt(6), -1 / mp.sqrt(6), -1 / mp.sqrt(6)]
E2 = [mp.mpf(0), 1 / mp.sqrt(2), -1 / mp.sqrt(2)]
C0 = 1 / mp.mpf(432) ** (mp.mpf(1) / 3)
S0 = 2 * PI * mp.sqrt(3) * C0            # S_prot/K at xi = 0, = 2 pi sqrt(sigma2) = 1.4396135


def cbar(xi):
    f = lambda c: (2 * c + xi[0] + xi[1]) * (2 * c + xi[1] + xi[2]) * (2 * c + xi[0] + xi[2]) - mp.mpf(1) / 54
    return mp.findroot(f, C0)


def re_m(xi):
    c = cbar(xi)
    q = [c + x for x in xi]
    s1, s2 = sum(q), q[0] * q[1] + q[1] * q[2] + q[0] * q[2]
    al = mp.sqrt(s2)
    out = []
    for I in range(3):
        J, K = [k for k in range(3) if k != I]
        out.append(-PI * (q[J] + q[K]) * (q[I] ** 2 - q[J] * q[K]) / (al * (al ** 2 + s1 ** 2)))
    return out


A = 2 * S0 / 3
kappa = PI / (2 * mp.sqrt(3) * C0)
print("c0 = %s, S0 = %s, (2/3) S0 = %s, kappa = %s" % (mp.nstr(C0, 10), mp.nstr(S0, 10), mp.nstr(A, 8),
                                                      mp.nstr(kappa, 8)))
# along Q ~ (1+2e, 1-e, 1-e) the largest fraction is 1/3 + 2e/3, so e < (3/2)(8/3) c0 eps = 4 c0 eps
print("first-order stable set: max_I xi_I < %s eps (= 8 c0^2 eps = %s eps); largest fraction < 1/3 + %s eps; "
      "along (1+2e,1-e,1-e): e < 4 c0 eps = %s eps" % (mp.nstr(A / kappa, 6), mp.nstr(8 * C0 ** 2, 6),
                                                      mp.nstr(A / kappa / (3 * C0), 6), mp.nstr(4 * C0, 6)))
print("theta(deg)  r_c/eps (linearized)  r_c/eps (exact Re m at r = 1e-4)")
for th in (0, 10, 20, 30, 40, 50, 60):
    t = mp.radians(th)
    u = [mp.cos(t) * a + mp.sin(t) * b for a, b in zip(E1, E2)]
    r = mp.mpf('1e-4')
    slope = -min(re_m([r * x for x in u])) / r
    print("%6d      %12s          %12s" % (th, mp.nstr(A / (kappa * max(u)), 6), mp.nstr(A / slope, 6)))
