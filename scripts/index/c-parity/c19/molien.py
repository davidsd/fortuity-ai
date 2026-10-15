"""Exact Molien series of the Weyl group S_N acting on the Cartan-valued (traceless diagonal) BMN letters of SU(N).

Hilbert series of (Sym(B (x) V0) (x) Lambda(F (x) V0))^{S_N}, graded by (2J, q1, q2, q3), where V0 = standard
(N-1)-dim rep of S_N and the letters are (summary/fortuity-summary.pdf (section 2), code/fortuity/bmn/words.py line 8):
    phi_m : bosonic,   2J = 0, q = e_m
    psi_m : fermionic, 2J = 1, q = (1,1,1) - e_m
    f     : bosonic,   2J = 2, q = (1,1,1)
Each letter type contributes one copy of V0.  Molien:
    H = (1/N!) sum_sigma prod_bos 1/det(1 - x sigma | V0) * prod_ferm det(1 + x sigma | V0),
with det(1 - t sigma | V0) = prod_cycles (1 - t^len) / (1 - t)  and  det(1 + t sigma | V0) = prod_cycles(1-(-t)^len)/(1+t).
Exact integer arithmetic, truncated at q1+q2+q3 <= qmax.
"""
from math import factorial
from collections import Counter, defaultdict
from itertools import product

LETTERS = [  # (bosonic?, 2J, q)
    (True, 0, (1, 0, 0)), (True, 0, (0, 1, 0)), (True, 0, (0, 0, 1)),
    (False, 1, (0, 1, 1)), (False, 1, (1, 0, 1)), (False, 1, (1, 1, 0)),
    (True, 2, (1, 1, 1)),
]


def partitions(n, maxpart=None):
    maxpart = maxpart or n
    if n == 0:
        yield ()
        return
    for k in range(min(n, maxpart), 0, -1):
        for rest in partitions(n - k, k):
            yield (k,) + rest


def class_size(lam, N):
    c = Counter(lam)
    den = 1
    for l, m in c.items():
        den *= l ** m * factorial(m)
    return factorial(N) // den


def poly_mul(a, b, deg):
    out = [0] * (deg + 1)
    for i, x in enumerate(a):
        if x == 0 or i > deg:
            continue
        for j, y in enumerate(b):
            if i + j > deg:
                break
            out[i + j] += x * y
    return out


def poly_div_exact(num, den):
    """exact polynomial division num/den (lists, low degree first), den[0] = +-1 assumed nonzero; remainder must be 0."""
    num = list(num)
    dd = len(den) - 1
    while dd > 0 and den[dd] == 0:
        dd -= 1
    q = [0] * max(1, len(num) - dd)
    for k in range(len(num) - 1, dd - 1, -1):
        c = num[k]
        if c == 0:
            continue
        assert c % den[dd] == 0
        f = c // den[dd]
        q[k - dd] = f
        for i in range(dd + 1):
            num[k - dd + i] -= f * den[i]
    assert all(v == 0 for v in num), "non-exact division"
    return q


def series_inv(den, deg):
    """1/den as a power series to degree deg (den[0] = 1)."""
    assert den[0] == 1
    out = [0] * (deg + 1)
    out[0] = 1
    for k in range(1, deg + 1):
        s = 0
        for i in range(1, min(k, len(den) - 1) + 1):
            s += den[i] * out[k - i]
        out[k] = -s
    return out


def boson_factor(cyc, deg):
    """1/det(1 - x sigma | V0) = (1 - x) / prod_c (1 - x^l), power series to degree deg."""
    den = [1]
    for l in cyc:
        p = [0] * (l + 1); p[0] = 1; p[l] = -1
        den = poly_mul(den, p, len(den) + l)
    inv = series_inv(den, deg)
    return poly_mul(inv, [1, -1], deg)


def fermion_factor(cyc):
    """det(1 + x sigma | V0) = prod_c (1 - (-x)^l) / (1 + x), a polynomial of degree N-1."""
    num = [1]
    for l in cyc:
        p = [0] * (l + 1); p[0] = 1; p[l] = -((-1) ** l)
        num = poly_mul(num, p, len(num) + l)
    return poly_div_exact(num, [1, 1])


def molien(N, qmax, letters=LETTERS):
    """{(2J, q1, q2, q3): dim of S_N invariants} for q1+q2+q3 <= qmax (zeros omitted)."""
    total = defaultdict(int)
    for cyc in partitions(N):
        cs = class_size(cyc, N)
        ser = {(0, 0, 0, 0): 1}
        for bos, tj, q in letters:
            w = sum(q)
            dmax = qmax // w
            uni = boson_factor(cyc, dmax) if bos else fermion_factor(cyc)
            new = defaultdict(int)
            for key, c in ser.items():
                s = key[1] + key[2] + key[3]
                for n, a in enumerate(uni):
                    if a == 0:
                        continue
                    if s + n * w > qmax:
                        break
                    k2 = (key[0] + n * tj, key[1] + n * q[0], key[2] + n * q[1], key[3] + n * q[2])
                    new[k2] += c * a
            ser = {k: v for k, v in new.items() if v}
        for k, v in ser.items():
            total[k] += cs * v
    out = {}
    nf = factorial(N)
    for k, v in total.items():
        assert v % nf == 0, (k, v)
        if v:
            assert v > 0, (k, v)
            out[k] = v // nf
    return out


if __name__ == '__main__':
    import sys
    N, Q = int(sys.argv[1]), int(sys.argv[2])
    m = molien(N, Q)
    print(len(m), sum(m.values()))
