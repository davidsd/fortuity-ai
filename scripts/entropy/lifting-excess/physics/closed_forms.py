"""T-041 physics check 7: closed forms of the protected Cardy saddle at unequal charges (the T-041 physics notes, summarised in summary/fortuity-summary.pdf sec. 7.6).
With q_I = cbar + xi_I (K units), alpha = sqrt(sigma2(q)), s1 = q1+q2+q3, on the Cardy sheet (q1+q2)(q2+q3)(q3+q1) = 1/54:
   Re mu_I = -pi (q_J+q_K)(q_I^2 - q_J q_K) / (alpha (alpha^2 + s1^2)),
   Im mu_I = -pi (q_J+q_K)(q_I + s1) / (alpha^2 + s1^2),
and at first order the convergence margin is Re(omega1+omega2)/3 = (2/3) eps Re g, = 0.9597 eps at xi = 0, so on the
direction q ~ (1+2e, 1-e, 1-e) the stable strip is e < e_c = (2 S0/3)/(pi/sqrt 3) eps = 0.5291 eps.
Checks the closed forms against the saddle reconstructed from the cubic (as in bh_domain.py) on random points of the
black-hole domain. Runtime < 1 s.
"""
import random
import mpmath as mp

mp.mp.dps = 30
PI = mp.pi
random.seed(1)
lam0 = (4 * PI ** 3 / 27) ** (mp.mpf(1) / 3) * mp.expjpi(mp.mpf(1) / 6)
worst = 0
n = 0
for _ in range(40):
    # random direction and radius inside the domain (continue the root from xi = 0)
    th = random.uniform(0, 2 * float(PI))
    rr = random.uniform(0.0, 0.35)
    d = [2 / mp.sqrt(6) * mp.cos(th), -mp.cos(th) / mp.sqrt(6) + mp.sin(th) / mp.sqrt(2),
         -mp.cos(th) / mp.sqrt(6) - mp.sin(th) / mp.sqrt(2)]
    lam = lam0
    ok = True
    for k in range(1, 41):
        xi = [rr * k / 40 * x for x in d]
        F = lambda l: (l + 2j * PI * xi[0]) * (l + 2j * PI * xi[1]) * (l + 2j * PI * xi[2]) - 4 * PI ** 3 * 1j / 27
        lam = mp.findroot(F, lam)
    cbar = mp.im(lam) / (2 * PI)
    q = [cbar + x for x in xi]
    s1 = sum(q)
    s2 = q[0] * q[1] + q[1] * q[2] + q[0] * q[2]
    if s2 <= 0:
        continue
    al = mp.sqrt(s2)
    Lam = 1j * lam / (2 * PI)
    g = -6j * PI / sum(1 / (Lam - x) for x in xi)
    mu = [g / (3 * (Lam - x)) for x in xi]
    for I in range(3):
        J, K = [k for k in range(3) if k != I]
        re_cf = -PI * (q[J] + q[K]) * (q[I] ** 2 - q[J] * q[K]) / (al * (al ** 2 + s1 ** 2))
        im_cf = -PI * (q[J] + q[K]) * (q[I] + s1) / (al ** 2 + s1 ** 2)
        worst = max(worst, abs(mp.re(mu[I]) - re_cf), abs(mp.im(mu[I]) - im_cf))
    n += 1
print("closed forms of Re mu_I, Im mu_I vs reconstructed saddle: %d points, max |diff| = %s" % (n, mp.nstr(worst, 3)))
S0 = mp.re(lam0)
print("first-order margin at xi = 0: (2/3) S0 = %s ;  e_c/eps = (2 S0/3)/(pi/sqrt(3)) = %s" % (
    mp.nstr(2 * S0 / 3, 6), mp.nstr((2 * S0 / 3) / (PI / mp.sqrt(3)), 6)))
