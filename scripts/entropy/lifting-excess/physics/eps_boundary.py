"""T-041 (added 2026-10-02): how far in eps the first-order expansion can reach near the arc.

At fixed xi and j1 = j2 = j, the exact large-N supersymmetric black hole (the CJKKKLMP sheet of stability_strip.py,
in KKLP's charges: j = J + Rbar, Q_I = K (cbar + xi_I), K = 3 (dim G j^2)^{1/3}, eps = (dim G/j)^{1/3}) gives, in units
of K,
   (S/K)^2/(4 pi^2) = p1 = e2(q) - eps/9 + cbar eps^2/3,        q_I = cbar + xi_I,
with cbar fixed by p0 = p1 p2,  p2 = e1(q) + eps^2/6,  p0 = e3(q) + 1/54 - cbar eps/9 + cbar^2 eps^2/6
(the relations of indep/indep_delta_r1.py bh_finite_check; checked below against the sheet in N^2 units).
At eps = 0 this is the Cardy black hole, S_prot/K = 2 pi sqrt(e2(q)). The black hole at fixed xi exists while p1 > 0.
eps_b is the first eps > 0 with p1 = 0 on the branch cbar(eps) continued from the Cardy solution (the other roots of
the constraint, cubic in cbar, are not examined). There S_prot = 2 pi K sqrt(p1) has a square-root branch point and
this black hole ends, so an expansion at fixed xi is limited to eps < eps_b (delta itself is singular there unless
S_cl,max vanishes in step, which is not computed here), and the first-order statement needs eps << eps_b.
Printed per point: eps_b; the first-order estimate 27 e2(q) = 0.684 (S_prot/K)^2 (from dp1/deps -> -1/27 at the
arc); delta0/r1 (the eps at which r1 eps equals delta0); and the check of the sheet in N^2 units at eps = eps_b/2.
Points: the last point of each ray of outputs/extended, and (5 deg, r = 1.56), with delta0, r1 and S_prot/K read
from there. Runtime: about 20 s on one core.
"""
import os
import mpmath as mp

mp.mp.dps = 30
PI = mp.pi
HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, '..', 'outputs', 'extended')


def xi_of(theta_deg, r):
    e1 = [2 / mp.sqrt(6), -1 / mp.sqrt(6), -1 / mp.sqrt(6)]
    e2 = [0, 1 / mp.sqrt(2), -1 / mp.sqrt(2)]
    t = mp.radians(theta_deg)
    return [r * (mp.cos(t) * a + mp.sin(t) * b) for a, b in zip(e1, e2)]


def sym(q):
    return sum(q), q[0] * q[1] + q[1] * q[2] + q[0] * q[2], q[0] * q[1] * q[2]


def constraint(cb, xi, eps):
    e1, e2, e3 = sym([cb + x for x in xi])
    p1 = e2 - eps / 9 + cb * eps ** 2 / 3
    p2 = e1 + eps ** 2 / 6
    p0 = e3 + mp.mpf(1) / 54 - cb * eps / 9 + cb ** 2 * eps ** 2 / 6
    return p0 - p1 * p2


def p1_of(cb, xi, eps):
    e1, e2, e3 = sym([cb + x for x in xi])
    return e2 - eps / 9 + cb * eps ** 2 / 3


def cbar_cardy(xi):
    # prod_{I<J}(q_I + q_J) = 1/54 is the eps = 0 constraint; start from the equal-charge value and continue in r
    cb = 1 / mp.mpf(432) ** (mp.mpf(1) / 3)
    for t in mp.linspace(0, 1, 41)[1:]:
        cb = mp.findroot(lambda c: constraint(c, [t * x for x in xi], 0), cb)
    return cb


def eps_boundary(xi):
    """Continue cbar(eps) from eps = 0 in steps; bracket the first sign change of p1 and refine."""
    cb = cbar_cardy(xi)
    p_prev, e_prev, c_prev = p1_of(cb, xi, 0), mp.mpf(0), cb
    de = p_prev / 100                     # p1 falls roughly like eps/27 near the arc: ~40 steps to the boundary
    e = mp.mpf(0)
    while True:
        e += de
        c = mp.findroot(lambda c: constraint(c, xi, e), c_prev)
        p = p1_of(c, xi, e)
        if p <= 0:
            def p1_at(eps):
                cc = mp.findroot(lambda c: constraint(c, xi, eps), c_prev)
                return p1_of(cc, xi, eps)
            eb = mp.findroot(p1_at, (e_prev, e), solver='anderson')
            return eb, cb
        p_prev, e_prev, c_prev = p, e, c
        if e > 10:
            return mp.inf, cb


def sheet_check(xi, eps, cb):
    """|LHS - RHS| of the CJKKKLMP sheet in N^2 units (dim G = N^2) at (xi, eps), and S/K from it vs 2 pi sqrt(p1)."""
    Q = [3 * eps ** -2 * (cb + x) for x in xi]              # Q_I/N^2
    J = eps ** -3 - 3 * eps ** -2 * cb                       # J/N^2 = (j - Rbar)/N^2, j1 = j2
    s2 = Q[0] * Q[1] + Q[1] * Q[2] + Q[0] * Q[2]
    lhs = J * J / 2 + Q[0] * Q[1] * Q[2]
    rhs = (sum(Q) + mp.mpf(1) / 2) * (s2 - J)
    SK = 2 * PI * mp.sqrt(s2 - J) / (3 * eps ** -2)
    return abs(lhs - rhs) / abs(lhs), abs(SK - 2 * PI * mp.sqrt(p1_of(cb, xi, eps)))


def ext_rows(k):
    rows, cols = [], None
    for line in open(os.path.join(EXT, "ray%d.tsv" % k)):
        if line.startswith('#') or not line.strip():
            continue
        tok = line.rstrip('\n').split('\t')
        if tok[0] == 'theta_deg':
            cols = tok
            continue
        rows.append(dict(zip(cols, tok)))
    return rows


points = []
for k in range(1, 13):
    rows = ext_rows(k)
    points.append(rows[-1])
    if k == 1:
        points.append(rows[-2])           # (5 deg, r = 1.56)
print("theta  r      S_prot/K    eps_b         27*e2(q)      delta0/r1     eps_b/(delta0/r1)  sheet check at eps_b/2")
for row in points:
    th, r = mp.mpf(row['theta_deg']), mp.mpf(row['r'])
    xi = xi_of(th, r)
    eb, cb = eps_boundary(xi)
    e2 = sym([cb + x for x in xi])[1]
    SK = 2 * PI * mp.sqrt(e2)
    d0r1 = mp.mpf(row['delta0']) / mp.mpf(row['r1'])
    half = eb / 2
    ch = mp.findroot(lambda c: constraint(c, xi, half), cb)
    rel, dS = sheet_check(xi, half, ch)
    print("%5s  %5s  %10s  %12s  %12s  %12s  %12s       rel %.1e, |dS/K| %.1e"
          % (mp.nstr(th, 3), mp.nstr(r, 3), mp.nstr(SK, 6), mp.nstr(eb, 6), mp.nstr(27 * e2, 6), mp.nstr(d0r1, 4),
             mp.nstr(eb / d0r1, 4), float(rel), float(dS)))
    if abs(SK - mp.mpf(row['Sprot/K'])) > 1e-6 * max(1, SK):
        print("   S_prot/K from the closed form differs from EXT's %s" % row['Sprot/K'])
