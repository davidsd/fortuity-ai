import os as _os, sys as _sys
REPO = (_os.environ.get('REPO') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *['..'] * 5))).rstrip('/') + '/'
_sys.path.insert(0, REPO + 'code'); _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
"""Skeptic check: dMK's (2.25) is a statement about the SU(3)_R SINGLET part T of the cubic-in-xhat restriction
(their (2.22)-(2.23)), not about the full x1^3 coefficient of the hw component (= T + T^1_1 + T^11_11 + T^111_111).
T = (1/10) sum_{ordered ijk} (1/n_m) [coefficient of x^{m(ijk)} in o330(ijk)], with O^{(ijk)} the symmetrised
component (trace of the symmetriser on Sym^3 C^3 = 10; traceless parts drop out).
Computes T mod p for: readings 1+2 (repo operator), without reading 1 (printed r/R), without reading 2."""
import sys, time
from fractions import Fraction as Fr
from itertools import product
from collections import Counter
import numpy as np
from fortuity.bmn import dmk
from fortuity.bmn.blockrestrict import restrict_u1
from fortuity.bmn.evaluate import Points
from fortuity.bmn.modp import P
from fortuity.bmn.words import contents_of_sector, expr_sector

t0 = time.time()
sec = (5, 4, 4, 4)
degs = sorted({sum(c[3:6]) for c in contents_of_sector(sec)})
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 23
pts = Points(2, 12, max(degs) + 2, degs, seed=seed)
vec = lambda e: pts.expr_vector(e) % P
inv = lambda q: (Fr(q).numerator * pow(Fr(q).denominator, P - 2, P)) % P
b = vec(dmk.to_modp(dmk.o2_24()))
assert b.any()

def T_of(**kw):
    acc = np.zeros_like(b)
    for ijk in product(range(3), repeat=3):
        cnt = Counter(ijk)
        e = (cnt[0], cnt[1], cnt[2])
        n_m = {3: 1, 2: 3, 1: 6}[max(cnt.values())] if len(cnt) < 3 else 6
        n_m = 1 if len(cnt) == 1 else (3 if len(cnt) == 2 else 6)
        X = restrict_u1(dmk.to_modp(dmk.o330(ijk, **kw)), 2, e)
        acc = (acc + vec(X) * inv(Fr(1, 10 * n_m))) % P
    return acc

def ratio_report(name, a):
    # is a = c * b (mod p)?  find c from one nonzero entry, check all
    k = int(np.nonzero(b)[0][0])
    c = (int(a[k]) * pow(int(b[k]), P - 2, P)) % P
    prop = np.array_equal(a, (b * c) % P)
    # rational reconstruction of c (small num/den)
    rec = None
    for den in range(1, 2000):
        num = (c * den) % P
        if num > P // 2:
            num -= P
        if abs(num) < 10**6:
            rec = Fr(num, den); break
    print('%-48s proportional to O^2_24: %s; ratio mod p -> %s   [%.1fs]' % (name, prop, rec if prop else '-', time.time() - t0))

for label, kw in (('T (singlet), readings 1+2 (repo operator)', {}),
                  ('T (singlet), printed r/R (no reading 1)', {'project': False}),
                  ('T (singlet), no Koszul sign (no reading 2)', {'graded_sign': False})):
    ratio_report(label, T_of(**kw))
# full x1^3 coefficient of the hw component, for comparison
for label, kw in (('x1^3 coeff of hw comp., readings 1+2', {}), ('x1^3 coeff of hw comp., printed r/R', {'project': False})):
    ratio_report(label, vec(restrict_u1(dmk.to_modp(dmk.o330((0, 0, 0), **kw)), 2, (3, 0, 0))))
