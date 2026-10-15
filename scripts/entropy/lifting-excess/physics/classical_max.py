"""T-041 physics check 2: KKLP classical maximum over the bonus at unequal charges, continued into the
part of the black-hole domain that the T-038 scan does not reach (one Q_I < 0, sigma2(Q) > 0).

Independent implementation (formulas of T-038 note sec. 7.1, the T-038 analysis):
  S/K = g(m) + sum_I m_I xi_I + eps (2/3) H(m)/g(m),   g = (-G3)^{1/3} (root of maximal real part, KKLP p. 19),
  H = h + (1/3) sum_I d f/d m_I,  h = -(1/2) G2,  f = -G3,  u_I = exp(-m_I),
  protected: sum m = -2 pi i;  classical maximum over the bonus: sum m = -i Phi, Phi real, maximise Re S(Phi).
delta0 = S_cl,max/S_prot - 1 (eps -> 0);  r1 = d(S_cl,max/S_prot)/d eps at eps = 0.
Also prints Re m_I at both saddles (the trace converges only for Re mu_I > 0, i.e. |u_I| < 1) and
S_cl,max / (K S0) with S0 = S_prot(xi=0)/K = 1.439613509 (the equal-charge value at the same j1, j2).

The classical maximum is the stationary point of Re S(Phi) (Im Lambda = 0, real bonus charge; KKLP (4.11)-(4.13)),
found by continuation from the protected saddle at Phi = 2 pi and a secant, then checked to be a local maximum.

Usage (one core of an M1 Max, about 3 s per point at mp.dps = 16):
  python classical_max.py t038     xi = 0 and the T-038 point (0.828, 0.086, 0.086)    -> classical_max_t038.out (7 s)
  python classical_max.py lune30   ray theta = 30 deg, r = 0.1 ... 0.5                -> classical_max_lune30.out (19 s)
  python classical_max.py lune60   ray theta = 60 deg, r = 0.1 ... 0.4                -> classical_max_lune60.out (17 s)
"""
import sys
import time
import mpmath as mp

mp.mp.dps = 16
PI = mp.pi
S0 = (4 * PI ** 3 / 27) ** (mp.mpf(1) / 3) * mp.cos(PI / 6)


def comb(L, m):
    u = [mp.exp(-x) for x in m]
    return (L(1) - L(u[0]) - L(u[1]) - L(u[2]) + L(u[0] * u[1]) + L(u[0] * u[2]) + L(u[1] * u[2])
            - L(u[0] * u[1] * u[2]))


L3 = lambda z: mp.polylog(3, z)
L2 = lambda z: mp.polylog(2, z)


def f(m):
    return -comb(L3, m)


def dfdm(m):
    u = [mp.exp(-x) for x in m]
    out = []
    for I in range(3):
        J, K = [k for k in range(3) if k != I]
        out.append(-L2(u[I]) + L2(u[I] * u[J]) + L2(u[I] * u[K]) - L2(u[0] * u[1] * u[2]))
    return out


def croot(z):
    r = abs(z) ** (mp.mpf(1) / 3)
    th = mp.arg(z) / 3
    return max([r * mp.expj(th + 2 * PI * k / 3) for k in range(3)], key=mp.re)


def value(m, xi):
    F = f(m)
    g = croot(F)
    return g + sum(a * b for a, b in zip(m, xi)), g, F


def saddle(Phi, xi, guess):
    s = -1j * Phi

    def eqs(m1, m2):
        m = [m1, m2, s - m1 - m2]
        F = f(m)
        g = croot(F)
        d = [g / (3 * F) * x for x in dfdm(m)]
        return [d[0] - d[2] + xi[0] - xi[2], d[1] - d[2] + xi[1] - xi[2]]

    m1, m2 = mp.findroot(eqs, guess, tol=mp.mpf(10) ** -16, maxsteps=60)
    m = [m1, m2, s - m1 - m2]
    v, g, F = value(m, xi)
    return v, m, g


def first_order_C(m, g):
    h = -comb(L2, m) / 2
    H = h + sum(dfdm(m)) / 3
    return (mp.mpf(2) / 3) * H / g


def protected_closed(xi, lam_guess):
    F = lambda l: (l + 2j * PI * xi[0]) * (l + 2j * PI * xi[1]) * (l + 2j * PI * xi[2]) - 4 * PI ** 3 * 1j / 27
    lam = mp.findroot(F, lam_guess)
    Lam = 1j * lam / (2 * PI)
    g = -6j * PI / sum(1 / (Lam - x) for x in xi)
    mu = [g / (3 * (Lam - x)) for x in xi]
    return lam, mu


def lagrange(m, xi):
    """Lambda = dS/d(sum m) at a constrained saddle; dRe S/dPhi = Im Lambda (sum m = -i Phi)."""
    F = f(m)
    g = croot(F)
    return g / (3 * F) * dfdm(m)[2] + xi[2]


def classical_max(xi, m_start, P0=2.10 * PI, P1=2.13 * PI):
    """KKLP (4.11)-(4.13) at unequal charges: Re S(Phi) is stationary where Im Lambda = 0 (real bonus charge).
    Continuation in Phi from the protected saddle at Phi = 2 pi, then secant on Im Lambda; the result is checked
    to be a local maximum of Re S(Phi)."""
    v, m, g = saddle(2 * PI, xi, (m_start[0], m_start[1]))
    # continue to P0 in a few steps
    for k in range(1, 4):
        Pk = 2 * PI + (P0 - 2 * PI) * k / 3
        v, m, g = saddle(Pk, xi, (m[0], m[1]))
    d0 = mp.im(lagrange(m, xi))
    v1, m1, g1 = saddle(P1, xi, (m[0], m[1]))
    d1 = mp.im(lagrange(m1, xi))
    for _ in range(30):
        if abs(d1) < mp.mpf(10) ** -12:
            break
        P2 = P1 - d1 * (P1 - P0) / (d1 - d0)
        if abs(P2 - P1) > 0.05 * PI:
            P2 = P1 + mp.sign(P2 - P1) * 0.05 * PI
        P0, d0 = P1, d1
        v1, m1, g1 = saddle(P2, xi, (m1[0], m1[1]))
        d1 = mp.im(lagrange(m1, xi))
        P1 = P2
    h = mp.mpf('0.002') * PI
    vl = saddle(P1 - h, xi, (m1[0], m1[1]))[0]
    vr = saddle(P1 + h, xi, (m1[0], m1[1]))[0]
    if not (mp.re(v1) >= mp.re(vl) and mp.re(v1) >= mp.re(vr)):
        raise RuntimeError("stationary point is not a maximum")
    return (v1, m1, g1), P1


def point(xi, lam_guess):
    xi = [mp.mpf(x) for x in xi]
    lam, mu = protected_closed(xi, lam_guess)
    # check the polylog saddle at Phi = 2 pi reproduces the closed form
    vp, mp_, gp = saddle(2 * PI, xi, (mu[0], mu[1]))
    (vc, mc, gc), Phi = classical_max(xi, mp_)
    Lp, Lc = mp.re(vp), mp.re(vc)
    Cp, Cc = first_order_C(mp_, gp), first_order_C(mc, gc)
    r1 = (mp.re(Cc) * Lp - Lc * mp.re(Cp)) / Lp ** 2
    cbar = mp.im(lam) / (2 * PI)
    q = [cbar + x for x in xi]
    return dict(lam=lam, q=q, Sprot=Lp, check=abs(vp - lam), Scl=Lc, Phi=Phi, delta0=Lc / Lp - 1, r1=r1,
                Rem_p=[mp.re(x) for x in mp_], Rem_c=[mp.re(x) for x in mc])


def show(p):
    q = p['q']
    sq = sum(q)
    s2 = q[0] * q[1] + q[1] * q[2] + q[0] * q[2]
    print("  (%s)  s2/sq^2=%s  Sprot/K=%s  |closed-polylog|=%s  Scl/K=%s  Phi*/pi=%s  delta0=%s  r1=%s  "
          "Scl/(K S0)=%s  Re m_prot=(%s)  Re m_cl=(%s)" % (
              ", ".join(mp.nstr(x / sq, 4) for x in q), mp.nstr(s2 / sq ** 2, 4), mp.nstr(p['Sprot'], 7),
              mp.nstr(p['check'], 2), mp.nstr(p['Scl'], 7), mp.nstr(p['Phi'] / PI, 6), mp.nstr(p['delta0'], 6),
              mp.nstr(p['r1'], 5), mp.nstr(p['Scl'] / S0, 5),
              ", ".join(mp.nstr(x, 3) for x in p['Rem_p']), ", ".join(mp.nstr(x, 3) for x in p['Rem_c'])),
          flush=True)


def ray(theta_deg, rs):
    e1 = [2 / mp.sqrt(6), -1 / mp.sqrt(6), -1 / mp.sqrt(6)]
    e2 = [0, 1 / mp.sqrt(2), -1 / mp.sqrt(2)]
    th = mp.radians(theta_deg)
    d = [mp.cos(th) * a + mp.sin(th) * b for a, b in zip(e1, e2)]
    lam = (4 * PI ** 3 / 27) ** (mp.mpf(1) / 3) * mp.expjpi(mp.mpf(1) / 6)
    r_prev = mp.mpf(0)
    for r in rs:
        # continue the protected root in small steps
        steps = max(1, int(mp.ceil((r - r_prev) / mp.mpf('0.01'))))
        for k in range(1, steps + 1):
            rr = r_prev + (r - r_prev) * k / steps
            F = lambda l, rr=rr: ((l + 2j * PI * rr * d[0]) * (l + 2j * PI * rr * d[1]) * (l + 2j * PI * rr * d[2])
                                  - 4 * PI ** 3 * 1j / 27)
            lam = mp.findroot(F, lam)
        r_prev = r
        try:
            p = point([r * x for x in d], lam)
            print("theta=%d r=%s" % (theta_deg, mp.nstr(r, 3)), end="")
            show(p)
        except Exception as e:  # noqa: BLE001
            print("theta=%d r=%s FAILED: %s" % (theta_deg, mp.nstr(r, 3), e), flush=True)


if __name__ == "__main__":
    t0 = time.time()
    lam0 = (4 * PI ** 3 / 27) ** (mp.mpf(1) / 3) * mp.expjpi(mp.mpf(1) / 6)
    print("test vs T-038: xi=0 -> delta0 0.012134, r1 0.05507, Phi*/pi 2.1245")
    show(point([0, 0, 0], lam0))
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "t038"):
        print("test vs T-038 (0.828,0.086,0.086): delta0 0.005365, Phi*/pi 2.0728; this is xi = (0.25,-0.125,-0.125)")
        ray(0, [mp.mpf('0.3061862178')])  # |xi| for (0.25,-0.125,-0.125)
    if which in ("all", "lune30"):
        ray(30, [mp.mpf(x) for x in ('0.1', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5')])
    if which in ("all", "lune60"):
        ray(60, [mp.mpf(x) for x in ('0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4')])
    print("time %.1f s" % (time.time() - t0))
