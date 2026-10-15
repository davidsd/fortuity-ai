import os as _os, sys as _sys
REPO = (_os.environ.get('REPO') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *['..'] * 5))).rstrip('/') + '/'
_sys.path.insert(0, REPO + 'code'); _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
"""Independent re-runs of the dmk claims at a different prime and seeds (read-only use of the repo package)."""
import time
from fractions import Fraction as Fr
import numpy as np
import sympy

from fortuity.bmn import dmk
from fortuity.bmn.blockrestrict import restrict_u1
from fortuity.bmn.evaluate import Points
from fortuity.bmn.words import Q_expr, expr_sector, contents_of_sector

P2 = 33554393
assert sympy.isprime(P2)
t0 = time.time()


def vec(expr_modp, N, sec, seed, p, npts=12):
    degs = sorted({sum(c[3:6]) for c in contents_of_sector(sec)})
    pts = Points(N, npts, max(degs) + 2, degs, seed=seed, p=p)
    return pts.expr_vector(expr_modp) % p


O = dmk.o330()
print('terms in O:', len(O), ' sector', expr_sector(O))
data = dmk.load_rational_json(REPO + 'data/su3/dmk_o330_hw.json')
print('data file == builder:', data == O)
dens = sorted({v.denominator for v in O.values()})
print('distinct denominators in O:', dens[:20], '... max', max(dens))

QO = Q_expr(O)
sec = expr_sector(O)
low = (4,) + sec[1:]
for p, seed in ((P2, 101), (P2, 202)):
    v = vec(dmk.to_modp(QO, p), 3, low, seed, p)
    print(f'N=3 Q O at p={p} seed={seed}: functionals={v.size} nonzero={int(np.count_nonzero(v))}')
v4 = vec(dmk.to_modp(QO, P2), 4, low, 303, P2, npts=4)
print(f'N=4 Q O at p={P2}: functionals={v4.size} nonzero={int(np.count_nonzero(v4))}')
print('time so far %.1fs' % (time.time() - t0))

# x1^3 restriction at the second prime, two seeds
c = Fr(-648, 5)
cmod = c.numerator * pow(c.denominator, P2 - 2, P2) % P2
O224 = dmk.o2_24()
sec2 = (5, 4, 4, 4)
for label, expr in (('readings 1+2+3', O),
                    ('no traceless projection', dmk.o330(project=False)),
                    ('no Koszul sign', dmk.o330(graded_sign=False))):
    img = restrict_u1(dmk.to_modp(expr, P2), 2, (3, 0, 0), p=P2)
    for seed in (41, 43):
        a = vec(img, 2, sec2, seed, P2)
        b = vec(dmk.to_modp(O224, P2), 2, sec2, seed, P2)
        # best-fit ratio a = lam * b mod p (from first nonzero entry), then check proportionality
        nz = np.nonzero(b)[0]
        i0 = int(nz[0])
        lam = int(a[i0]) * pow(int(b[i0]), P2 - 2, P2) % P2
        prop = np.array_equal(a, (b * lam) % P2)
        # rational reconstruction of lam
        rr = None
        for den in range(1, 2000):
            num = lam * den % P2
            if num > P2 // 2:
                num -= P2
            if abs(num) < 10**6:
                rr = Fr(num, den)
                break
        print(f'{label:26s} seed={seed}: n_terms(img)={len(img)} proportional to O224: {prop}; ratio = {rr} '
              f'(== -648/5: {lam == cmod})')
print('time %.1fs' % (time.time() - t0))
