"""T-041 physics check 4: extent of the rays of `t038/rf/scan_kklp_charges.py scan n` (closed form, leading Cardy).

The scan marches r = 0, 0.02, 0.04, ... along xi = r (cos th e1 + sin th e2), th = (pi/3) k/n, and stops when
min_I Q_I <= 0 (t038/rf/scan_kklp_charges.py:136-154). For each ray this prints
  r_Q   : where the smallest charge Q3 crosses 0 (where the scan stops),
  r_S   : where sigma2(q) = 0 (boundary of the Cardy-limit black-hole domain; S_prot -> 0),
  points: number of scan points to r_Q (step 0.02) and to r_S.
On th = 0, Q2 = Q3 = K/(162 r^2) + O(r^-4) > 0 for all r (derived in the T-041 physics notes, summarised in summary/fortuity-summary.pdf sec. 7.6), so the stopping
rule never fires; the ray tends to the corner Q = (1,0,0), which lies on the circle sigma2 = 0.
Runtime: under 1 s.
"""
import mpmath as mp

mp.mp.dps = 25
PI = mp.pi


def dirvec(th):
    e1 = [2 / mp.sqrt(6), -1 / mp.sqrt(6), -1 / mp.sqrt(6)]
    e2 = [0, 1 / mp.sqrt(2), -1 / mp.sqrt(2)]
    return [mp.cos(th) * a + mp.sin(th) * b for a, b in zip(e1, e2)]


def cbar(xi):
    """unique root of prod_K (2c - xi_K) = 1/54 with 2c > max xi (all pairwise sums of q positive)."""
    F = lambda c: (2 * c - xi[0]) * (2 * c - xi[1]) * (2 * c - xi[2]) - mp.mpf(1) / 54
    lo = max(xi) / 2 + mp.mpf('1e-30')
    hi = lo + 1
    return mp.findroot(F, (lo, hi), solver='bisect' if False else 'anderson')


def q_of(r, d):
    xi = [r * x for x in d]
    c = cbar(xi)
    return [c + x for x in xi]


if __name__ == "__main__":
    n = 12
    print("scan %d: per ray, r where Q3 = 0 (scan stops) and r where sigma2 = 0 (black-hole domain ends)" % n)
    print("  k  theta   r_Q        r_S        pts to r_Q  pts to r_S   fractions at r_S (approx)")
    for k in range(n + 1):
        th = PI / 3 * k / n
        d = dirvec(th)
        q3 = lambda r: q_of(r, d)[2]
        s2 = lambda r: (lambda q: q[0] * q[1] + q[1] * q[2] + q[0] * q[2])(q_of(r, d))
        if k == 0:
            rq = rs = mp.inf
            fr = "(1, 0, 0) at r -> infinity; Q2 = Q3 ~ 1/(162 r^2)"
            for r in (1, 3, 10, 30):
                q = q_of(mp.mpf(r), d)
                print("     th=0, r=%3d: Q2/K=%s  162 r^2 Q2/K=%s  S_prot/K=%s" % (
                    r, mp.nstr(q[1], 4), mp.nstr(162 * r * r * q[1], 5),
                    mp.nstr(2 * PI * mp.sqrt(q[0] * q[1] + q[1] * q[2] + q[0] * q[2]), 4)))
        else:
            rq = mp.findroot(q3, (mp.mpf('0.01'), mp.mpf(5)), solver='anderson')
            rs = mp.findroot(s2, (rq, mp.mpf(20)), solver='anderson')
            q = q_of(rs * (1 - mp.mpf('1e-9')), d)
            sq = sum(q)
            fr = "(%s)" % ", ".join(mp.nstr(x / sq, 4) for x in q)
        npq = "inf" if rq == mp.inf else str(int(mp.floor(rq / mp.mpf('0.02'))) + 1)
        nps = "inf" if rs == mp.inf else str(int(mp.floor(rs / mp.mpf('0.02'))) + 1)
        print("  %2d  %5s  %9s  %9s  %10s  %10s   %s" % (k, mp.nstr(th * 180 / PI, 3), mp.nstr(rq, 4),
                                                       mp.nstr(rs, 4), npq, nps, fr))
