"""Reframing (c) of T-038: charge dependence of the KKLP classical-index excess (arXiv:2606.27955 sec. 4).

KKLP set u = v = w (their (4.2): "For simplicity, we will unrefine this index as u=v=w") and work at leading Cardy order.
This script maps, at large N and in the Cardy regime j1, j2 >> N^2:
  (i)  delta0(xi) = S_cl,max / S_prot - 1 at leading Cardy order, as a function of the R-charge differences xi;
  (ii) r1(xi), the first finite-omega correction:  S_cl,max/S_prot = 1 + delta0 + r1 * eps_eff + O(eps^2),
       eps_eff = (N^2)^{1/3} (j1 + j2) / (2 (j1 j2)^{2/3})   [= (N^2/j)^{1/3} for j1 = j2 = j].
Large-N classical free energy at the collapsed (Cardy) saddle, to first subleading order:
  log Z = N^2 [ f(mu)/(w1 w2) + h(mu) (w1 + w2)/(w1 w2) ] + O(N^2 log w),
  f = -[Li3(1) - sum Li3(u_I) + sum_{I<J} Li3(u_I u_J) - Li3(u1 u2 u3)],  h = same with Li2 and an extra factor 1/2,
  u_I = e^{-mu_I}.  Bonus potential nu = sum mu - w1 - w2 (x = uvw/pq = e^{-nu});  max over the bonus => nu = -i Phi.
With mu_I = m_I + (w1+w2)/3, sum m = -i Phi, the O(1/w) coefficient is H(m) = h(m) + (1/3) sum_I d_I f(m).
Checks built in (selftest): KKLP 1.012134 at xi = 0; protected Re S = 2 pi sqrt(Q1Q2+Q2Q3+Q3Q1) with Im S = 0
(Cardy limit of the BPS black-hole entropy); H on the protected surface equals (m1 m2 + m2 m3 + m3 m1)/6, the
first-order term of the exact large-N cubic N^2 mu1 mu2 mu3 / (2 w1 w2).
Charges: xi_I = (R_I - Rbar)/K, K = 3 (N^2 j1 j2)^{1/3}; physical black holes need Q_I = cbar + xi_I > 0 with
cbar = Im(S_prot/K)/(2 pi) (the Im S = 0 condition fixes the overall R level).

Usage:  python scan_kklp_charges.py selftest          (~5 s)
        python scan_kklp_charges.py scan [n] > out.tsv (n+1 rays over the ordered wedge, radial step 0.02 until a
                                                        BH charge Q_I hits 0; n=12: ~13 x ~15 = ~200 points,
                                                        ~1-3 s each at mp.dps=15 => ~5-10 min on one core)
"""
import sys, time
import mpmath as mp

mp.mp.dps = 15
L3 = lambda z: mp.polylog(3, z)
L2 = lambda z: mp.polylog(2, z)


def comb(L, m):
    u = [mp.e ** (-x) for x in m]
    return L(1) - sum(L(x) for x in u) + L(u[0] * u[1]) + L(u[0] * u[2]) + L(u[1] * u[2]) - L(u[0] * u[1] * u[2])


def f(m):
    return -comb(L3, m)


def h(m):
    return -comb(L2, m) / 2


def df(m, I):
    u = [mp.e ** (-x) for x in m]
    J = [k for k in range(3) if k != I]
    return -(L2(u[I]) - L2(u[I] * u[J[0]]) - L2(u[I] * u[J[1]]) + L2(u[0] * u[1] * u[2]))


def croot(z):  # cube root with maximal real part (KKLP branch rule below their (4.14))
    r = abs(z) ** (mp.mpf(1) / 3)
    th = mp.arg(z) / 3
    return max([r * mp.expj(th + 2 * mp.pi * k / 3) for k in range(3)], key=mp.re)


def saddle(Phi, xi, guess=None):
    """Leading order: ext over m (sum m = -i Phi) of f^{1/3} + sum m xi.  Returns value/K, m, dG/ds."""
    s = -1j * Phi

    def eqs(m1, m2):
        m = [m1, m2, s - m1 - m2]
        F = f(m)
        g = croot(F)
        d = [g / (3 * F) * df(m, I) for I in range(3)]
        return [d[0] - d[2] + xi[0] - xi[2], d[1] - d[2] + xi[1] - xi[2]]

    m1, m2 = mp.findroot(eqs, guess if guess is not None else (s / 3, s / 3))
    m = [m1, m2, s - m1 - m2]
    F = f(m)
    g = croot(F)
    val = g + sum(m[I] * xi[I] for I in range(3))
    dGds = g / (3 * F) * df(m, 2) + xi[2]
    return val, m, dGds, g


def first_order(m, g):
    """Coefficient C with S/K = val + eps_eff * C:  delta S / K = (2/3) eps_eff H / f^{1/3}."""
    H = h(m) + sum(df(m, I) for I in range(3)) / 3
    return (mp.mpf(2) / 3) * H / g, H


def classical_max(xi, P0=2.12 * mp.pi, P1=2.13 * mp.pi):
    """max over the bonus: stationary point of Re G(Phi), i.e. Im dG/ds = 0 (bonus charge real), by secant."""
    v0, m0, d0, g0 = saddle(P0, xi)
    v1, m1, d1, g1 = saddle(P1, xi, (m0[0], m0[1]))
    for _ in range(12):
        if abs(mp.im(d1)) < mp.mpf(10) ** -11:
            break
        P2 = P1 - mp.im(d1) * (P1 - P0) / (mp.im(d1) - mp.im(d0))
        P0, d0 = P1, d1
        v1, m1, d1, g1 = saddle(P2, xi, (m1[0], m1[1]))
        P1 = P2
    return v1, m1, g1, P1


def point(xi, guess_p=None, guess_c=None, Pguess=None):
    xi = [mp.mpf(x) for x in xi]
    vp, mpr, _, gp = saddle(2 * mp.pi, xi, guess_p)
    cbar = mp.im(vp) / (2 * mp.pi)
    Q = [cbar + x for x in xi]
    if Pguess is None:
        vc, mc, gc, Pc = classical_max(xi)
    else:
        vc, mc, gc, Pc = classical_max(xi, Pguess - mp.mpf('0.004') * mp.pi, Pguess + mp.mpf('0.004') * mp.pi)
    Cp, Hp = first_order(mpr, gp)
    Cc, _ = first_order(mc, gc)
    Lp, Lc = mp.re(vp), mp.re(vc)
    delta0 = Lc / Lp - 1
    r1 = (mp.re(Cc) * Lp - Lc * mp.re(Cp)) / Lp ** 2
    return dict(xi=xi, Q=Q, Sprot=Lp, Scl=Lc, Phi=Pc, delta0=delta0, r1=r1, Hp=Hp, mp=mpr, mc=mc)


def selftest():
    t0 = time.time()
    p = point((0, 0, 0))
    print("xi=0: delta0 = %.6f (KKLP 1.01213 -> 0.012134), r1 = %.6f, Phi*/pi = %.5f" % (p['delta0'], p['r1'], p['Phi'] / mp.pi))
    for xi in [(0.1, -0.1, 0), (0.25, -0.125, -0.125)]:
        p = point(xi)
        Q = p['Q']
        e2 = Q[0] * Q[1] + Q[1] * Q[2] + Q[0] * Q[2]
        m = p['mp']
        Hcubic = (m[0] * m[1] + m[1] * m[2] + m[0] * m[2]) / 6
        print("xi=%s: Q/sumQ=(%.3f,%.3f,%.3f) delta0=%.6f r1=%.6f | checks: ReS_prot - 2pi sqrt(e2) = %.1e, H - cubic = %.1e"
              % (tuple(float(x) for x in xi), *(float(q / sum(Q)) for q in Q), p['delta0'], p['r1'],
                 float(p['Sprot'] - 2 * mp.pi * mp.sqrt(e2)), float(abs(p['Hp'] - Hcubic))))
    print("selftest %.1f s" % (time.time() - t0))


def scan(n):
    """Rays in the plane sum(xi)=0 over the ordered wedge xi1 >= xi2 >= xi3 (theta = 0: one large charge,
    theta = 60 deg: one small charge), marching outward with continuation until min Q <= 0."""
    e1 = [2 / mp.sqrt(6), -1 / mp.sqrt(6), -1 / mp.sqrt(6)]
    e2 = [0, 1 / mp.sqrt(2), -1 / mp.sqrt(2)]
    print("theta_deg\tr\txi1\txi2\txi3\tQ1/sQ\tQ2/sQ\tQ3/sQ\tSprot/K\tScl/K\tPhi/pi\tdelta0\tr1")
    for k in range(n + 1):
        th = mp.pi / 3 * k / n
        gp = gc = Pg = None
        r = mp.mpf(0)
        while True:
            xi = [r * (mp.cos(th) * a + mp.sin(th) * b) for a, b in zip(e1, e2)]
            try:
                p = point(xi, gp, gc, Pg)
            except Exception as e:  # noqa: BLE001
                print("# fail theta=%.1f r=%.3f: %s" % (float(th * 180 / mp.pi), float(r), e), flush=True)
                break
            Q = p['Q']
            if min(Q) <= 0:
                break
            sQ = sum(Q)
            print("\t".join("%.5f" % float(x) for x in [th * 180 / mp.pi, r] + list(xi) + [q / sQ for q in Q] +
                             [p['Sprot'], p['Scl'], p['Phi'] / mp.pi, p['delta0'], p['r1']]), flush=True)
            gp, gc, Pg = (p['mp'][0], p['mp'][1]), (p['mc'][0], p['mc'][1]), p['Phi']
            r += mp.mpf('0.02')


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "scan":
        scan(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    else:
        selftest()
