"""T-041 physics check 6: scenario S2 of the T-041 physics notes, summarised in summary/fortuity-summary.pdf sec. 7.6 (dressing on the BPS side only).
The KKLP-type bound at fixed (j1, j2, xi) then needs S_cl,max(xi) > K S0 (S0 = S_prot(0)/K, the equal-charge value).
Finds the radius r_c where S_cl,max(xi)/(K S0) = 1 on the rays theta = 0, 30, 60 deg (same rays as bh_domain.py),
by linear interpolation between r = 0.06 and 0.07 (one Newton-free bracket per ray). Runtime ~20 s.
Usage: python s2_region.py THETA_DEG
"""
import sys
import mpmath as mp
import classical_max as cm

PI = mp.pi
th = int(sys.argv[1]) if len(sys.argv) > 1 else 60
e1 = [2 / mp.sqrt(6), -1 / mp.sqrt(6), -1 / mp.sqrt(6)]
e2 = [0, 1 / mp.sqrt(2), -1 / mp.sqrt(2)]
t = mp.radians(th)
d = [mp.cos(t) * a + mp.sin(t) * b for a, b in zip(e1, e2)]
lam = (4 * PI ** 3 / 27) ** (mp.mpf(1) / 3) * mp.expjpi(mp.mpf(1) / 6)
vals = []
for r in (mp.mpf('0.06'), mp.mpf('0.07')):
    xi = [r * x for x in d]
    p = cm.point(xi, lam)
    lam = p['lam']
    q = p['q']
    vals.append((r, p['Scl'] / cm.S0 - 1, [x / sum(q) for x in q], p['delta0']))
    print("theta=%d r=%s  fractions (%s)  Scl/(K S0) - 1 = %s  delta0 = %s" % (
        th, mp.nstr(r, 3), ", ".join(mp.nstr(x, 4) for x in vals[-1][2]), mp.nstr(vals[-1][1], 5),
        mp.nstr(p['delta0'], 6)), flush=True)
(r0, y0, f0, _), (r1, y1, f1, _) = vals
rc = r0 - y0 * (r1 - r0) / (y1 - y0)
w = (rc - r0) / (r1 - r0)
fc = [a + w * (b - a) for a, b in zip(f0, f1)]
print("theta=%d: S_cl,max = K S0 at r_c ~ %s, fractions ~ (%s)" % (th, mp.nstr(rc, 4), ", ".join(mp.nstr(x, 3) for x in fc)))
