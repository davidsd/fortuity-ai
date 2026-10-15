import os as _os, sys as _sys
REPO = (_os.environ.get('REPO') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *['..'] * 5))).rstrip('/') + '/'
_sys.path.insert(0, REPO + 'code'); _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
"""Singlet part T of the cubic restriction without reading 1 (printed r/R), at a chosen prime; Wang rational
reconstruction of the ratio T / O^2_24."""
import sys, time
from fractions import Fraction as Fr
from itertools import product
from collections import Counter
import numpy as np
from fortuity.bmn import dmk
from fortuity.bmn.blockrestrict import restrict_u1
from fortuity.bmn.evaluate import Points
from fortuity.bmn.words import contents_of_sector

p = int(sys.argv[1]); seed = int(sys.argv[2]); project = sys.argv[3] == 'proj'
t0 = time.time()
sec = (5, 4, 4, 4)
degs = sorted({sum(c[3:6]) for c in contents_of_sector(sec)})
pts = Points(2, 12, max(degs) + 2, degs, seed=seed, p=p)
vec = lambda e: pts.expr_vector(e) % p
inv = lambda q: (Fr(q).numerator * pow(Fr(q).denominator, p - 2, p)) % p
b = vec(dmk.to_modp(dmk.o2_24(), p))
acc = np.zeros_like(b)
for ijk in product(range(3), repeat=3):
    cnt = Counter(ijk)
    n_m = {1: 1, 2: 3, 3: 6}[len(cnt)]
    X = restrict_u1(dmk.to_modp(dmk.o330(ijk, project=project), p), 2, (cnt[0], cnt[1], cnt[2]), p=p)
    acc = (acc + vec(X) * inv(Fr(1, 10 * n_m))) % p
k = int(np.nonzero(b)[0][0])
c = (int(acc[k]) * pow(int(b[k]), p - 2, p)) % p
prop = np.array_equal(acc, (b * c) % p)

def wang(a, m):
    import math
    B = math.isqrt(m // 2)
    r0, r1, s0, s1 = m, a % m, 0, 1
    while r1 > B:
        q = r0 // r1
        r0, r1, s0, s1 = r1, r0 - q * r1, s1, s0 - q * s1
    if s1 == 0 or abs(s1) > B:
        return None
    return Fr(r1, s1)
print("RESIDUE", c); print('p=%d seed=%d project=%s: T proportional to O^2_24: %s; ratio %s (Wang); -648/5 mod p? %s  [%.1fs]'
      % (p, seed, project, prop, wang(c, p), c == inv(Fr(-648, 5)), time.time() - t0))
