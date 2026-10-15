"""Exact symbolic evaluation of BMN-letter multi-trace expressions on traceless N x N matrices.

Independent of fortuity.bmn.{evaluate,words,blockrestrict}: letters 0,1,2 = phi_m (bosonic), 3,4,5 = psi_m (Grassmann),
matrix entries are independent variables except the last diagonal entry = -(sum of the others).
A polynomial is {(bos, ferm): coeff} with bos a sorted tuple (multiset) and ferm a strictly increasing tuple of
Grassmann variable ids (the monomial is bos * ferm[0] ferm[1] ...).  Coefficients are ints / Fractions (exact).
Q is the odd derivation Q psi_m = eps_{mnp} phi_n phi_p (matrix product), Q phi = 0, acting from the left.
"""
from collections import defaultdict
from itertools import combinations, permutations


def is_f(l):
    return 3 <= l <= 5


def eps(a, b, c):
    if len({a, b, c}) < 3:
        return 0
    return 1 if (a, b, c) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1


def clean(d):
    return {k: v for k, v in d.items() if v != 0}


def padd(A, B, c=1):
    out = defaultdict(int, A)
    for k, v in B.items():
        out[k] += c * v
    return clean(out)


def pmul(A, B):
    out = defaultdict(int)
    for (ba, fa), ca in A.items():
        for (bb, fb), cb in B.items():
            if fa and fb:
                sa = set(fa)
                if any(y in sa for y in fb):
                    continue
                inv = sum(1 for y in fb for x in fa if x > y)
                f = tuple(sorted(fa + fb))
                s = -1 if inv & 1 else 1
            else:
                f = fa if fa else fb
                s = 1
            b = tuple(sorted(ba + bb)) if (ba and bb) else (ba or bb)
            out[(b, f)] += s * ca * cb
    return clean(out)


class Alg:
    def __init__(self, N):
        self.N = N
        self.var, self.inv = {}, {}
        n = 0
        for l in range(6):
            for i in range(N):
                for j in range(N):
                    if i == j == N - 1:
                        continue
                    self.var[(l, i, j)] = n
                    self.inv[n] = (l, i, j)
                    n += 1

        def mono(l, v):
            return ((), (v,)) if is_f(l) else ((v,), ())
        self.mat = {}
        for l in range(6):
            M = [[None] * N for _ in range(N)]
            for i in range(N):
                for j in range(N):
                    if i == j == N - 1:
                        M[i][j] = {mono(l, self.var[(l, k, k)]): -1 for k in range(N - 1)}
                    else:
                        M[i][j] = {mono(l, self.var[(l, i, j)]): 1}
            self.mat[l] = M
        # Q psi_m = eps_{mnp} phi_n phi_p
        self.QM = {}
        for m in range(3):
            acc = [[{} for _ in range(N)] for _ in range(N)]
            for nn in range(3):
                for pp in range(3):
                    e = eps(m, nn, pp)
                    if e:
                        prod = self.matmul(self.mat[nn], self.mat[pp])
                        for i in range(N):
                            for j in range(N):
                                acc[i][j] = padd(acc[i][j], prod[i][j], e)
            self.QM[m] = acc
        self._tr = {}

    def matmul(self, A, B):
        N = self.N
        C = [[{} for _ in range(N)] for _ in range(N)]
        for i in range(N):
            for j in range(N):
                acc = defaultdict(int)
                for k in range(N):
                    if A[i][k] and B[k][j]:
                        for key, v in pmul(A[i][k], B[k][j]).items():
                            acc[key] += v
                C[i][j] = clean(acc)
        return C

    def trace(self, w):
        w = tuple(w)
        t = self._tr.get(w)
        if t is not None:
            return t
        M = self.mat[w[0]]
        for l in w[1:]:
            M = self.matmul(M, self.mat[l])
        acc = {}
        for i in range(self.N):
            acc = padd(acc, M[i][i])
        self._tr[w] = acc
        return acc

    def multi(self, ws):
        out = {((), ()): 1}
        for w in ws:
            out = pmul(out, self.trace(w))
            if not out:
                break
        return out

    def expr(self, E):
        acc = defaultdict(int)
        for ws, c in E.items():
            for k, v in self.multi(ws).items():
                acc[k] += c * v
        return clean(acc)

    def Q(self, Pl):
        out = defaultdict(int)
        for (b, f), c in Pl.items():
            for t, g in enumerate(f):
                l, i, j = self.inv[g]
                q = self.QM[l - 3][i][j] if (i, j) != (self.N - 1, self.N - 1) else None
                assert q is not None
                s = -1 if t & 1 else 1
                fr = f[:t] + f[t + 1:]
                for (bq, fq), cq in q.items():
                    assert not fq
                    out[(tuple(sorted(b + bq)), fr)] += s * c * cq
        return clean(out)


# ------------------------------------------------------------------ independent formal U(1) restriction (x2=x3=0)
def canon_word(w):
    """Cyclic word -> (sign, canonical rotation); (0, None) if it vanishes or has length 1 (traceless letters)."""
    L = len(w)
    if L == 1:
        return 0, None
    seen = {}
    cur, s = tuple(w), 1
    for _ in range(L):
        if cur in seen and seen[cur] != s:
            return 0, None
        seen.setdefault(cur, s)
        first, rest = cur[0], cur[1:]
        if is_f(first) and (sum(1 for l in rest if is_f(l)) & 1):
            s = -s
        cur = rest + (first,)
    best = min(seen)
    return seen[best], best


def parity(w):
    return sum(1 for l in w if is_f(l)) & 1


def canon_multi(ws):
    ws = list(ws)
    sign = 1
    # bubble sort with sign for swapping two odd factors
    for i in range(len(ws)):
        for j in range(len(ws) - 1 - i):
            if ws[j] > ws[j + 1]:
                if parity(ws[j]) and parity(ws[j + 1]):
                    sign = -sign
                ws[j], ws[j + 1] = ws[j + 1], ws[j]
    for a, b in zip(ws, ws[1:]):
        if a == b and parity(a):
            return 0, None
    return sign, tuple(ws)


def restrict_x1(E, k, Nsmall=2):
    """Coefficient of x1^k (x2 = x3 = 0) of E under SU(Nsmall+1) > SU(Nsmall) x U(1): phi_m -> diag(A_m + x_m 1, -Nsmall x_m),
    psi -> diag(B, 0).  Returns {SU(Nsmall) multitrace (canonical): coeff} (exact)."""
    out = defaultdict(int)
    cache = {}

    def options(w):
        o = cache.get(w)
        if o is not None:
            return o
        o = []
        pos1 = [i for i, l in enumerate(w) if l == 0]
        for r in range(0, min(len(pos1), k) + 1):
            for S in combinations(pos1, r):
                rem = tuple(l for i, l in enumerate(w) if i not in S)
                if not rem:                               # all letters phi_1, all replaced
                    o.append((r, None, Nsmall + (-Nsmall) ** len(w)))
                    continue
                s, cw = canon_word(rem)
                if s:
                    o.append((r, cw, s))
        cache[w] = o
        return o

    for ws, c in E.items():
        per = [options(w) for w in ws]
        stack = [(0, 0, [], c)]
        while stack:
            i, acc, words, coef = stack.pop()
            if i == len(per):
                if acc == k:
                    s, m = canon_multi(words)
                    if s:
                        out[m] += s * coef
                continue
            for r, cw, val in per[i]:
                if acc + r > k:
                    continue
                stack.append((i + 1, acc + r, words if cw is None else words + [cw], coef * val))
    return clean(out)
