"""Brute-force check of the Molien series: dimension of S_N-invariant super-polynomials in the traceless diagonal BMN
letters at a fixed weight (2J, q1, q2, q3), by exact rank (Fractions) of the Reynolds images of all monomials.

Coordinates: each letter type L has slot values x_{L,1..N} with sum 0; we use x_{L,1..N-1} as independent coordinates
and x_{L,N} = -sum_{i<N} x_{L,i}.  sigma in S_N acts by x_{L,i} -> x_{L,sigma(i)} (a linear substitution).
Fermionic letter types (psi_m) give Grassmann coordinates; sign conventions handled explicitly.
"""
from fractions import Fraction
from itertools import permutations, product, combinations
from collections import defaultdict
from molien import LETTERS


def compositions(n, k):
    """all tuples of k nonneg ints summing to n"""
    if k == 1:
        yield (n,)
        return
    for a in range(n + 1):
        for rest in compositions(n - a, k - 1):
            yield (a,) + rest


def letter_counts(twoJ, q):
    """all count vectors (n_phi1,n_phi2,n_phi3,n_psi1,n_psi2,n_psi3,n_f) of weight (twoJ, q)."""
    out = []
    for nf in range(twoJ // 2 + 1):
        npsi = twoJ - 2 * nf
        for ps in compositions(npsi, 3):
            phis = [q[I] - nf - (npsi - ps[I]) for I in range(3)]
            if min(phis) < 0:
                continue
            out.append(tuple(phis) + tuple(ps) + (nf,))
    return out


class Space:
    def __init__(self, N):
        self.N = N
        r = N - 1
        self.r = r
        # variable indexing: bosonic letter types b in [phi1,phi2,phi3,f] -> bos index; fermionic psi1..3 -> ferm index
        self.btypes = [0, 1, 2, 6]
        self.ftypes = [3, 4, 5]
        self.nb = 4 * r
        self.nf = 3 * r

    def bvar(self, t, i):
        return self.btypes.index(t) * self.r + i

    def fvar(self, t, i):
        return self.ftypes.index(t) * self.r + i

    def image(self, sigma, i):
        """slot i (0..N-2) -> list of (slot coord j in 0..N-2, coeff) for x_{sigma(i)}."""
        s = sigma[i]
        if s < self.N - 1:
            return [(s, 1)]
        return [(j, -1) for j in range(self.N - 1)]

    def monomials(self, counts):
        """all super-monomials with these letter counts: (bexp tuple, sorted ferm tuple)."""
        r = self.r
        per_type = []
        for t, n in enumerate(counts):
            if t in self.btypes:
                per_type.append([('b', t, c) for c in compositions(n, r)])
            else:
                if n > r:
                    return []
                per_type.append([('f', t, c) for c in combinations(range(r), n)])
        out = []
        for choice in product(*per_type):
            bexp = [0] * self.nb
            fs = []
            for kind, t, c in choice:
                if kind == 'b':
                    for i, e in enumerate(c):
                        bexp[self.bvar(t, i)] += e
                else:
                    fs.extend(self.fvar(t, i) for i in c)
            out.append((tuple(bexp), tuple(sorted(fs))))
        return out

    def act(self, sigma, mono):
        """sigma . mono as dict {(bexp, ferm tuple): coeff}."""
        bexp, fs = mono
        poly = {(tuple([0] * self.nb), ()): 1}
        # bosonic part (commuting)
        for v, e in enumerate(bexp):
            if not e:
                continue
            t = self.btypes[v // self.r]; i = v % self.r
            lin = [(self.bvar(t, j), c) for j, c in self.image(sigma, i)]
            for _ in range(e):
                new = defaultdict(int)
                for (be, ff), c in poly.items():
                    for w, a in lin:
                        b2 = list(be); b2[w] += 1
                        new[(tuple(b2), ff)] += c * a
                poly = {k: x for k, x in new.items() if x}
        # fermionic part, multiplied on the right in the stored (sorted) order
        for v in fs:
            t = self.ftypes[v // self.r]; i = v % self.r
            lin = [(self.fvar(t, j), c) for j, c in self.image(sigma, i)]
            new = defaultdict(int)
            for (be, ff), c in poly.items():
                for w, a in lin:
                    if w in ff:
                        continue
                    sgn = (-1) ** sum(1 for u in ff if u > w)
                    new[(be, tuple(sorted(ff + (w,))))] += c * a * sgn
            poly = {k: x for k, x in new.items() if x}
        return poly

    def invariant_dim(self, twoJ, q):
        monos = []
        for cnt in letter_counts(twoJ, q):
            monos.extend(self.monomials(cnt))
        if not monos:
            return 0
        perms = list(permutations(range(self.N)))
        rows = []
        for m in monos:
            acc = defaultdict(int)
            for s in perms:
                for k, c in self.act(s, m).items():
                    acc[k] += c
            rows.append({k: c for k, c in acc.items() if c})
        return rank(rows)


def rank(rows):
    """exact rank of sparse integer rows (dicts) by fraction-free-ish Gaussian elimination with Fractions."""
    piv = {}   # col -> row(dict) with leading col
    r = 0
    for row in rows:
        row = {k: Fraction(v) for k, v in row.items() if v}
        while row:
            col = min(row)
            if col in piv:
                p = piv[col]
                f = row[col] / p[col]
                for k, v in p.items():
                    nv = row.get(k, 0) - f * v
                    if nv:
                        row[k] = nv
                    else:
                        row.pop(k, None)
            else:
                piv[col] = row
                r += 1
                break
    return r
