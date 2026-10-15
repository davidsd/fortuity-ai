"""T-041 physics check 1: the black-hole domain of the KKLP charge scan, at leading Cardy order.

Conventions are those of T-038 note sec. 7.1  and
t038/rf/scan_kklp_charges.py: K = 3 (dimG j1 j2)^{1/3}, xi_I = (R_I - Rbar)/K, sum xi = 0, u_I = exp(-mu_I),
S/K = g(mu) + sum mu_I xi_I with g = (-G3)^{1/3}; protected surface sum mu = -2 pi i.
Closed form (T-038 note:497): S_prot/K = lambda with prod_I (lambda + 2 pi i xi_I) = 4 pi^3 i / 27,
cbar = Im(lambda)/(2 pi), q_I = Q_I/K = cbar + xi_I.

This script is independent of the t038 code (written from the formulas only). It
  (a) continues the protected root along rays of the wedge xi1 >= xi2 >= xi3, past Q3 = 0 (the edge where the
      T-038 scan stops) up to the boundary sigma2(q) = 0 of the Cardy-limit black-hole domain;
  (b) checks Re(lambda) = 2 pi sqrt(sigma2(q)) and the Cardy charge relation (q1+q2)(q2+q3)(q3+q1) = 1/54;
  (c) reconstructs the protected saddle mu_I and checks the polylog free energy f = -G3(mu) against the cubic
      mu1 mu2 mu3 / 2 (i.e. that the saddle stays in the canonical chamber of CKKN 1810.12067 eq. (2.24));
  (d) prints Re(mu_I) (the convergence condition |u_I| < 1 is Re mu_I > 0) and compares its sign with the
      closed form Re mu_I ~ -(q_I^2 - q_J q_K) derived in the T-041 physics notes, summarised in summary/fortuity-summary.pdf sec. 7.6.
Runtime: about 2 s.
"""
import mpmath as mp

mp.mp.dps = 30
PI = mp.pi


def G3(m):
    u = [mp.exp(-x) for x in m]
    L = lambda z: mp.polylog(3, z)
    return (mp.zeta(3) - L(u[0]) - L(u[1]) - L(u[2]) + L(u[0] * u[1]) + L(u[0] * u[2]) + L(u[1] * u[2])
            - L(u[0] * u[1] * u[2]))


def lam_root(xi, guess):
    """root of prod(lambda + 2 pi i xi_I) - 4 pi^3 i/27 nearest to guess (continuation)."""
    F = lambda l: (l + 2j * PI * xi[0]) * (l + 2j * PI * xi[1]) * (l + 2j * PI * xi[2]) - 4 * PI ** 3 * 1j / 27
    return mp.findroot(F, guess)


def protected_saddle(xi, lam):
    Lam = 1j * lam / (2 * PI)
    g = -6j * PI / sum(1 / (Lam - x) for x in xi)
    mu = [g / (3 * (Lam - x)) for x in xi]
    return g, mu


def ray(theta_deg, dr=0.01, rmax=1.0):
    e1 = [2 / mp.sqrt(6), -1 / mp.sqrt(6), -1 / mp.sqrt(6)]
    e2 = [0, 1 / mp.sqrt(2), -1 / mp.sqrt(2)]
    th = mp.radians(theta_deg)
    d = [mp.cos(th) * a + mp.sin(th) * b for a, b in zip(e1, e2)]
    lam = (4 * PI ** 3 / 27) ** (mp.mpf(1) / 3) * mp.expjpi(mp.mpf(1) / 6)
    r = mp.mpf(0)
    rows = []
    q3_cross = None
    while r <= rmax:
        xi = [r * x for x in d]
        lam = lam_root(xi, lam)
        cbar = mp.im(lam) / (2 * PI)
        q = [cbar + x for x in xi]
        s2 = q[0] * q[1] + q[1] * q[2] + q[0] * q[2]
        if mp.re(lam) < mp.mpf('1e-6') or s2 <= 0:
            rows.append(('boundary', r, q, s2, lam))
            break
        g, mu = protected_saddle(xi, lam)
        f_poly = -G3(mu)
        f_cub = mu[0] * mu[1] * mu[2] / 2
        rel = (q[0] + q[1]) * (q[1] + q[2]) * (q[2] + q[0])
        pred = [-(q[i] ** 2 - q[(i + 1) % 3] * q[(i + 2) % 3]) for i in range(3)]
        rows.append(('pt', r, q, s2, lam, mu, abs(f_poly - f_cub), rel, pred))
        if q3_cross is None and min(q) <= 0:
            q3_cross = r
        r += dr
    return rows, q3_cross


if __name__ == "__main__":
    import time
    t0 = time.time()
    for th in (0, 30, 60):
        rows, rc = ray(th)
        print("== ray theta = %d deg (0: one large charge, 60: one small charge); Q3 = 0 crossed at r = %s"
              % (th, mp.nstr(rc, 3) if rc is not None else "never"))
        print("   r    fractions Q_I/sumQ             S_prot/K   |ReS-2pi sqrt(s2)|  (q1+q2)(q2+q3)(q3+q1)*54"
              "  |f_poly-f_cubic|  Re mu_I                    sign check  Im mu_I/(2pi)")
        for row in rows:
            if row[0] == 'boundary':
                print("   boundary sigma2 -> 0 reached at r = %s, fractions %s, Re lambda = %s"
                      % (mp.nstr(row[1], 3), [mp.nstr(x / sum(row[2]), 4) for x in row[2]], mp.nstr(mp.re(row[4]), 3)))
                continue
            _, r, q, s2, lam, mu, dfc, rel, pred = row
            if int(mp.nint(r * 100)) % 5 and not (rc is not None and abs(r - rc) < 0.006):
                continue
            sq = sum(q)
            ok = all((mp.re(m) < 0) == (p < 0) for m, p in zip(mu, pred) if abs(p) > 1e-12)
            print("  %5s  (%s)  %8s  %8s  %8s  %8s  (%s)  %s  (%s)" % (
                mp.nstr(r, 3), ", ".join(mp.nstr(x / sq, 4) for x in q), mp.nstr(mp.re(lam), 6),
                mp.nstr(abs(mp.re(lam) - 2 * PI * mp.sqrt(s2)), 2), mp.nstr(rel * 54, 8), mp.nstr(dfc, 2),
                ", ".join(mp.nstr(mp.re(m), 4) for m in mu), ok,
                ", ".join(mp.nstr(mp.im(m) / (2 * PI), 4) for m in mu)))
    print("time %.1f s" % (time.time() - t0))
