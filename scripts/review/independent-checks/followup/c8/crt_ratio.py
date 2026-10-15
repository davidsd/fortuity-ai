import os, sys, subprocess
from fractions import Fraction as Fr
# Singlet part T of the cubic restriction WITHOUT reading 1 (printed r/R tensors): the ratio T / O^2_24 at two primes,
# combined by CRT and rational reconstruction. With reading 1 the ratio is -648/5 (dMK eq. 2.25); without it, it is not.
HERE = os.path.dirname(os.path.abspath(__file__))
ps, res = (33554393, 50331653), []
for p in ps:
    out = subprocess.run([sys.executable, os.path.join(HERE, 'singlet_T_unproj_c.py'), str(p), '7', 'raw'],
                         capture_output=True, text=True).stdout
    print(out.strip())
    res.append(int([l for l in out.splitlines() if l.startswith('RESIDUE')][0].split()[1]))
M = ps[0] * ps[1]
c = (res[0] * ps[1] * pow(ps[1], -1, ps[0]) + res[1] * ps[0] * pow(ps[0], -1, ps[1])) % M
def wang(a, m):
    import math
    B = math.isqrt(m // 2)
    r0, r1, s0, s1 = m, a % m, 0, 1
    while r1 > B:
        q = r0 // r1
        r0, r1, s0, s1 = r1, r0 - q * r1, s1, s0 - q * s1
    return None if s1 == 0 or abs(s1) > B else Fr(r1, s1)
print('CRT over primes %s: T / O^2_24 without reading 1 = %s (with reading 1: -648/5)' % (ps, wang(c, M)))
