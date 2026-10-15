"""T-041: KKLP-excess map on the whole Cardy-limit black-hole domain (production script; NOT run here).

Extends `t038/rf/scan_kklp_charges.py scan n` (whose rays stop at min Q_I <= 0, t038/rf/scan_kklp_charges.py:148)
to the boundary sigma2(q) = 0 of the black-hole domain (the T-041 physics notes, summarised in summary/fortuity-summary.pdf sec. 7.6), and adds per point the
convergence/stability data of the T-041 physics notes, summarised in summary/fortuity-summary.pdf sec. 7.6:
  min_I Re m_I at the protected and at the classical saddle (the index trace converges only for Re mu_I > 0;
  at first order Re mu_I = Re m_I + (2/3) eps Re g, so near equal charges the intersection black hole is in the
  stable phase iff eps > -min Re m_I / 0.9597 [first-order estimate, small xi]),
  S_cl,max/(K S0) (the comparison relevant if the BPS side is dressed, scenario S2 of notes sec. 3.2).
Independent of the t038 code: uses physics/classical_max.py (validated against T-038 and scan12 at shared points).

Rays: theta_k = 60 k/n deg in the ordered wedge xi1 >= xi2 >= xi3 (same parametrization as the t038 scan).
  k = 0 (theta = 0) never reaches sigma2 = 0 (Q2 = Q3 ~ K/(162 r^2)); it is marched to r = RMAX0 (default 0.9,
  where the t038 scan failed, t041/outputs/scan12.tsv) or to the first numerical failure.
  k >= 1: marched until S_prot/K < SMIN (default 0.03) or the first failure.
Options: --from-edge   start each ray (k >= 1) at the first grid radius past Q3 = 0 (only the part missing from scan12);
         --step S      radial step (default 0.02).
Output: TSV on stdout, one line per point; '# ...' comment lines for failures and ray ends.

Usage:
  python scan_extended.py selftest                       (2 points, ~8 s)
  python scan_extended.py scan 12 --from-edge > scan_extended_edge.tsv     (~250 points, ~15-25 min, one core)
  python scan_extended.py scan 12 > scan_extended_full.tsv                 (~440 points, ~25-45 min, one core)
  python scan_extended.py ray K 12 [--from-edge]         (one ray, for parallel jobs; ray 1 is the longest: ~80 pts)
"""
import sys
import time
import mpmath as mp
import classical_max as cm

PI = mp.pi
SMIN = mp.mpf('0.03')
RMAX0 = mp.mpf('0.9')


def dirvec(theta_deg):
    e1 = [2 / mp.sqrt(6), -1 / mp.sqrt(6), -1 / mp.sqrt(6)]
    e2 = [0, 1 / mp.sqrt(2), -1 / mp.sqrt(2)]
    th = mp.radians(theta_deg)
    return [mp.cos(th) * a + mp.sin(th) * b for a, b in zip(e1, e2)]


def lam_continue(lam, d, r_from, r_to, h=mp.mpf('0.005')):
    steps = max(1, int(mp.ceil(abs(r_to - r_from) / h)))
    for k in range(1, steps + 1):
        rr = r_from + (r_to - r_from) * k / steps
        F = lambda l, rr=rr: ((l + 2j * PI * rr * d[0]) * (l + 2j * PI * rr * d[1]) * (l + 2j * PI * rr * d[2])
                              - 4 * PI ** 3 * 1j / 27)
        lam = mp.findroot(F, lam)
    return lam


HEADER = ("theta_deg\tr\tQ1/sQ\tQ2/sQ\tQ3/sQ\tsigma2/sQ^2\tSprot/K\tScl/K\tPhi/pi\tdelta0\tr1\t"
          "minRe_m_prot\tminRe_m_cl\tScl/(K*S0)\teps_stable_est")


def row(theta, r, p):
    q = p['q']
    sq = sum(q)
    s2 = (q[0] * q[1] + q[1] * q[2] + q[0] * q[2]) / sq ** 2
    mp_min = min(p['Rem_p'])
    mc_min = min(p['Rem_c'])
    eps_est = max(mp.mpf(0), -mp_min / (2 * cm.S0 / 3))
    vals = [theta, r] + [x / sq for x in q] + [s2, p['Sprot'], p['Scl'], p['Phi'] / PI, p['delta0'], p['r1'],
                                               mp_min, mc_min, p['Scl'] / cm.S0, eps_est]
    return "\t".join(mp.nstr(v, 7) for v in vals)


def run_ray(k, n, step, from_edge, out=sys.stdout):
    theta = mp.mpf(60) * k / n
    d = dirvec(theta)
    lam = (4 * PI ** 3 / 27) ** (mp.mpf(1) / 3) * mp.expjpi(mp.mpf(1) / 6)
    r_prev = mp.mpf(0)
    r = mp.mpf(0)
    if from_edge and k > 0:
        # first grid radius past Q3 = 0
        while True:
            lam_try = lam_continue(lam, d, r_prev, r)
            cbar = mp.im(lam_try) / (2 * PI)
            if cbar + r * d[2] <= 0:
                lam, r_prev = lam_try, r
                break
            lam, r_prev = lam_try, r
            r += step
    t0 = time.time()
    npts = 0
    while True:
        if k == 0 and r > RMAX0 + step / 2:
            print("# ray theta=0 stopped at RMAX0 = %s" % mp.nstr(RMAX0, 3), file=out, flush=True)
            break
        try:
            lam = lam_continue(lam, d, r_prev, r)
            r_prev = r
            if mp.re(lam) < SMIN:
                print("# ray theta=%s ends: S_prot/K = %s < SMIN at r = %s" % (mp.nstr(theta, 4), mp.nstr(mp.re(lam), 3),
                                                                             mp.nstr(r, 4)), file=out, flush=True)
                break
            p = cm.point([r * x for x in d], lam)
        except Exception as e:  # noqa: BLE001
            msg = str(e).replace("\n", " ")
            print("# fail theta=%s r=%s: %s" % (mp.nstr(theta, 4), mp.nstr(r, 4), msg), file=out, flush=True)
            break
        print(row(theta, r, p), file=out, flush=True)
        npts += 1
        r += step
    print("# ray theta=%s: %d points, %.1f s" % (mp.nstr(theta, 4), npts, time.time() - t0), file=out, flush=True)


def selftest():
    t0 = time.time()
    print(HEADER)
    # one interior point (shared with scan12: theta = 30, r = 0.20 -> delta0 0.00899, r1 0.05030)
    d = dirvec(30)
    lam = lam_continue((4 * PI ** 3 / 27) ** (mp.mpf(1) / 3) * mp.expjpi(mp.mpf(1) / 6), d, 0, mp.mpf('0.2'))
    print(row(30, mp.mpf('0.2'), cm.point([mp.mpf('0.2') * x for x in d], lam)))
    # one point in the missing segment (classical_max_lune30.out: r = 0.3 -> delta0 0.00587687)
    lam = lam_continue(lam, d, mp.mpf('0.2'), mp.mpf('0.3'))
    print(row(30, mp.mpf('0.3'), cm.point([mp.mpf('0.3') * x for x in d], lam)))
    print("# selftest %.1f s (expect delta0 = 0.00899 and 0.00588)" % (time.time() - t0))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    from_edge = "--from-edge" in sys.argv
    step = mp.mpf('0.02')
    if "--step" in sys.argv:
        step = mp.mpf(sys.argv[sys.argv.index("--step") + 1])
        args = [a for a in args if a != sys.argv[sys.argv.index("--step") + 1]]
    if not args or args[0] == "selftest":
        selftest()
    elif args[0] == "scan":
        n = int(args[1]) if len(args) > 1 else 12
        print(HEADER, flush=True)
        for k in range(n + 1):
            run_ray(k, n, step, from_edge)
    elif args[0] == "ray":
        k, n = int(args[1]), int(args[2])
        print(HEADER, flush=True)
        run_ray(k, n, step, from_edge)
