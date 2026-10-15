"""T-041 physics check 3: where is the intersection black hole 'stable' (all Re nu_i >= 0) at large charge?

Exact (finite-charge) large-N formulas of Choi, Jain, Kim, Krishna, Kwon, Lee, Minwalla, Patel,
arXiv:2501.17217v2 (CJKKKLMP), charges in units of N^2 (q_i = Q_i/N^2, j_i = J_i/N^2):
  black-hole sheet (their eq. sh1):  j1 j2/2 + q1 q2 q3 = (q1+q2+q3+1/2)(q1q2+q1q3+q2q3-(j1+j2)/2)
  entropy (entbhc):                  S/N^2 = 2 pi sqrt(q1q2+q2q3+q3q1-(j1+j2)/2)
  chemical potentials (chempotform, lufo; their Re parts are the renormalized BPS chemical potentials):
     z_I = -(S+2 pi i J2)/(S-2 pi i Q_I), z4 = (S+2 pi i J2)/(S+2 pi i J1),
     mu_I = 2 pi i z_I/(1+z1+z2+z3+z4), omega_1 = -2 pi i z4/(...), omega_2 = -2 pi i/(...)
  mu_1 < 0  <=>  (j1+j2)/2 < (q2+q3)(q1+2q1^2-2q2q3)/(1+2q1)     (cvanmp)
The trace defining the index converges only for Re mu_I > 0, Re omega_i > 0 (their eq. condconv; CKKN
1810.12067 after (2.43): "if the solution for any of Delta_I, omega_i has negative real part, that solution will
not describe a good saddle point").

(a) Family q = Q (1+2e, 1-e, 1-e), j1 = j2 = j on the sheet: find e_c(Q) where Re mu_1 changes sign and compare
    with the leading Cardy prediction e_c ~ 1/(3 sqrt(Q)) = (4^{1/3}/3) eps, eps = (N^2/j)^{1/3}
    (the T-041 physics notes, summarised in summary/fortuity-summary.pdf sec. 7.6). Here j is CJKKKLMP's angular momentum J/N^2. The T-041 note's eps uses KKLP's
    j = J + Rbar instead (Rbar = mean R-charge = Q for this family; first_order.py). The two agree as Q -> oo, and
    the last two columns give e_c/eps in KKLP's convention (added 2026-10-02 after review; the other columns are
    unchanged).
(b) CJKKKLMP index algorithm (their sec. 4.3) at the T-038 point with fractions (0.828, 0.086, 0.086) and at
    (0.703, 0.148, 0.148): rank-2 DDBH core with mu_1 = 0; compare its entropy with the intersection black hole and
    with the equal-charge black hole at the same index charges, as Q grows (approach to the Cardy limit).
Runtime: about 3 s.
"""
import time
import mpmath as mp

mp.mp.dps = 40
PI = mp.pi


def j_on_sheet(q):
    """j1 = j2 = j on the sheet for given q (positive root)."""
    s1 = sum(q) + mp.mpf(1) / 2
    s2 = q[0] * q[1] + q[1] * q[2] + q[0] * q[2]
    s3 = q[0] * q[1] * q[2]
    # j^2/2 + s3 = s1 (s2 - j)  ->  j^2/2 + s1 j + s3 - s1 s2 = 0
    return -s1 + mp.sqrt(s1 ** 2 - 2 * (s3 - s1 * s2))


def entropy(q, j1, j2):
    return 2 * PI * mp.sqrt(q[0] * q[1] + q[1] * q[2] + q[0] * q[2] - (j1 + j2) / 2)


def chem(q, j1, j2):
    S = entropy(q, j1, j2)
    z = [-(S + 2j * PI * j2) / (S - 2j * PI * x) for x in q]
    z4 = (S + 2j * PI * j2) / (S + 2j * PI * j1)
    den = 1 + sum(z) + z4
    mu = [2j * PI * x / den for x in z]
    om = [-2j * PI * z4 / den, -2j * PI / den]
    return S, mu, om


def mu1_rhs(q):
    return (q[1] + q[2]) * (q[0] + 2 * q[0] ** 2 - 2 * q[1] * q[2]) / (1 + 2 * q[0])


def part_a():
    print("(a) stable strip around equal charges: q = Q(1+2e, 1-e, 1-e), j1 = j2 on the sheet")
    print("      Q          j        eps=j^-1/3   e_c (Re mu_1 = 0)   e_c*sqrt(Q)  e_c/eps   [Cardy: 1/3, 4^(1/3)/3=0.529]"
          "                            eps_KKLP=(j+Q)^-1/3  e_c/eps_KKLP")
    for Q in (mp.mpf(10) ** k for k in (1, 2, 3, 4, 6, 8)):
        def remu1(e):
            q = [Q * (1 + 2 * e), Q * (1 - e), Q * (1 - e)]
            j = j_on_sheet(q)
            S, mu, om = chem(q, j, j)
            return mp.re(mu[0])
        e_c = mp.findroot(remu1, (mp.mpf('1e-6'), mp.mpf('0.5')), solver='anderson')
        q = [Q * (1 + 2 * e_c), Q * (1 - e_c), Q * (1 - e_c)]
        j = j_on_sheet(q)
        # cross-check with the closed condition (cvanmp)
        chk = j - mu1_rhs(q)
        eps = j ** (-mp.mpf(1) / 3)
        eps_k = (j + sum(q) / 3) ** (-mp.mpf(1) / 3)
        print("  %9s  %11s  %9s  %12s  %10s  %8s   (j - rhs(cvan) = %9s)  %10s  %8s" % (
            mp.nstr(Q, 3), mp.nstr(j, 5), mp.nstr(eps, 4), mp.nstr(e_c, 6), mp.nstr(e_c * mp.sqrt(Q), 5),
            mp.nstr(e_c / eps, 5), mp.nstr(chk, 3), mp.nstr(eps_k, 4), mp.nstr(e_c / eps_k, 5)))
    # signs at e = 0 and at a few e > 0 for one large Q
    Q = mp.mpf(10) ** 4
    print("  sample, Q = 1e4: Re(mu_I), Re(omega_i) at e = 0, 0.002, 0.05:")
    for e in (0, mp.mpf('0.002'), mp.mpf('0.05')):
        q = [Q * (1 + 2 * e), Q * (1 - e), Q * (1 - e)]
        j = j_on_sheet(q)
        S, mu, om = chem(q, j, j)
        print("     e=%s: Re mu = (%s), Re omega = (%s)" % (
            mp.nstr(e, 3), ", ".join(mp.nstr(mp.re(x), 4) for x in mu), ", ".join(mp.nstr(mp.re(x), 4) for x in om)))


def sheet_eq(q, j1, j2):
    return j1 * j2 / 2 + q[0] * q[1] * q[2] - (sum(q) + mp.mpf(1) / 2) * (
        q[0] * q[1] + q[1] * q[2] + q[0] * q[2] - (j1 + j2) / 2)


def part_b():
    print("\n(b) CJKKKLMP index algorithm (sec. 4.3) at fixed index charges, approaching the Cardy limit")
    print("    fractions (index line fixed by a black hole on the sheet with these charge fractions, j1 = j2)")
    for fr in ((mp.mpf('0.828'), mp.mpf('0.086'), mp.mpf('0.086')), (mp.mpf('0.703'), mp.mpf('0.1485'), mp.mpf('0.1485'))):
        print("  fractions %s" % [mp.nstr(x, 4) for x in fr])
        print("      Q_tot     eps=j^-1/3  S_int/N^2     Re mu_1(int)   S_DDBH/N^2   core fractions             "
              "S_eq/N^2     S_int/S_eq  S_DDBH/S_eq")
        for Qt in (mp.mpf(10) ** k for k in (1, 2, 3, 4, 6, 8)):
            q = [Qt * x for x in fr]
            j = j_on_sheet(q)
            S, mu, om = chem(q, j, j)
            # index-line invariants: q_i - q_k and q_i + j; line: (q_i + x, j - x)
            # rank-2 DDBH core: (q1 + x - y, q2 + x, q3 + x, j - x, j - x) on the sheet with mu_1 = 0, y > 0

            def eqs(x, y):
                c = [q[0] + x - y, q[1] + x, q[2] + x]
                jj = j - x
                return [sheet_eq(c, jj, jj), jj - mu1_rhs(c)]
            # initial guess: core at equal charges, Cardy estimate
            y0 = q[0] - q[1]
            x0 = mp.mpf(0)
            try:
                x, y = mp.findroot(eqs, (x0, y0))
                c = [q[0] + x - y, q[1] + x, q[2] + x]
                Sd = entropy(c, j - x, j - x)
                cf = [cc / sum(c) for cc in c]
            except Exception as e:  # noqa: BLE001
                x = y = Sd = mp.nan
                cf = [mp.nan] * 3
            # equal-charge black hole on the same index line: (q_i + x', j - x') with q1 + x' = q2 + x' impossible
            # unless q1 = q2; instead compare with the equal-charge BH with the same j (Cardy-limit statement)
            qe = mp.findroot(lambda t: sheet_eq([t, t, t], j, j), Qt / 3)
            Se = entropy([qe, qe, qe], j, j)
            print("  %10s  %8s  %12s  %12s  %12s  (%s)  %12s  %8s  %8s" % (
                mp.nstr(Qt, 3), mp.nstr(j ** (-mp.mpf(1) / 3), 3), mp.nstr(S, 6), mp.nstr(mp.re(mu[0]), 4),
                mp.nstr(Sd, 6), ", ".join(mp.nstr(t, 4) for t in cf), mp.nstr(Se, 6), mp.nstr(S / Se, 5),
                mp.nstr(Sd / Se, 5)))


if __name__ == "__main__":
    t0 = time.time()
    part_a()
    part_b()
    print("time %.1f s" % (time.time() - t0))
