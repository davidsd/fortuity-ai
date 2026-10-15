"""Exact linear algebra over F_p using float64 BLAS matmuls (13-bit split trick).

We take p < 2^26.  A matmul A@B with entries < p is split as
A = A1*2^13 + A0, B = B1*2^13 + B0 with all parts < 2^13, so each partial product
entry is < 2^26 * n, exact in float64 for inner dimension n < 2^27.
"""
import numpy as np

P = 67108859  # prime, 2^26 - 5
HALF = 1 << 13


def _split(A):
    A = np.asarray(A)
    return (A >> 13).astype(np.float64), (A & (HALF - 1)).astype(np.float64)


def matmul_mod(A, B, p=P, block=1024):
    """(A @ B) mod p for integer arrays with entries in [0, p).  B is processed in column blocks so
    that the float64 temporaries stay small even when B is a huge basis."""
    A = np.asarray(A)
    B = np.asarray(B)
    m, n = A.shape[0], B.shape[1]
    out = np.empty((m, n), dtype=np.int64)
    if A.shape[1] == 0:
        out[:] = 0
        return out
    A1, A0 = _split(A)
    hh_f = (HALF * HALF) % p
    for c0 in range(0, n, block):
        Bb = B[:, c0:c0 + block]
        B1, B0 = _split(Bb)
        hh = (A1 @ B1).astype(np.int64); hh %= p
        hl = (A1 @ B0).astype(np.int64); hl += (A0 @ B1).astype(np.int64); hl %= p
        ll = (A0 @ B0).astype(np.int64); ll %= p
        hh *= hh_f; hh %= p; hl *= HALF; hl += ll; hl += hh; hl %= p
        out[:, c0:c0 + block] = hl
    return out


def inv_mod(a, p=P):
    return pow(int(a), p - 2, p)


class RREFBasis:
    """Incrementally maintained reduced row-echelon basis over F_p (int32 storage, p < 2^26).

    B[:rank]: rows, pivots: list of column indices, B[i, piv[i]] = 1, B[i, piv[j]] = 0 (j != i).
    """

    def __init__(self, ncols, p=P, capacity=256):
        self.ncols = ncols
        self.p = p
        self.B = np.zeros((capacity, ncols), dtype=np.int32)
        self.piv = []

    @property
    def rank(self):
        return len(self.piv)

    def _grow(self, need):
        if need <= self.B.shape[0]:
            return
        cap = max(need, 2 * self.B.shape[0])
        newB = np.zeros((cap, self.ncols), dtype=np.int32)
        newB[:self.rank] = self.B[:self.rank]
        self.B = newB

    def reduce(self, X):
        """Return X reduced modulo the current basis (rows of X are vectors)."""
        X = np.asarray(X, dtype=np.int64) % self.p
        r = self.rank
        if r == 0:
            return X
        coeff = X[:, self.piv]
        return (X - matmul_mod(coeff, self.B[:r], self.p)) % self.p

    def add(self, X):
        """Add rows of X to the span.  Returns boolean array: which rows were independent
        (of the basis and of the earlier rows in X)."""
        p = self.p
        X = self.reduce(X)
        m = X.shape[0]
        indep = np.zeros(m, dtype=bool)
        NR = np.zeros((m, self.ncols), dtype=np.int64)
        new_piv = []
        r = 0
        for i in range(m):
            v = X[i]
            if r:
                c = v[new_piv]
                if np.any(c):
                    v = (v - matmul_mod(c[None, :], NR[:r], p)[0]) % p
            nz = np.nonzero(v)[0]
            if nz.size == 0:
                continue
            c0 = int(nz[0])
            v = (v * inv_mod(v[c0], p)) % p
            if r:
                col = NR[:r, c0]
                if np.any(col):
                    NR[:r] = (NR[:r] - np.outer(col, v)) % p
            NR[r] = v
            new_piv.append(c0)
            r += 1
            indep[i] = True
        if r:
            NR = NR[:r]
            R = self.rank
            if R > 0:
                col = self.B[:R, new_piv].astype(np.int64)
                # in-place row-blocked update: B[:R] -= col @ NR   (small temporaries only)
                rb = 256
                for r0 in range(0, R, rb):
                    r1 = min(r0 + rb, R)
                    prod = matmul_mod(col[r0:r1], NR, p)
                    np.subtract(prod, self.B[r0:r1], out=prod)      # prod = col@NR - B
                    np.negative(prod, out=prod)                       # B - col@NR
                    np.remainder(prod, p, out=prod)
                    self.B[r0:r1] = prod
            self._grow(R + r)
            self.B[R:R + r] = NR
            self.piv.extend(new_piv)
        return indep

    def add_batched(self, X, chunk=256):
        X = np.asarray(X, dtype=np.int64)
        out = np.zeros(X.shape[0], dtype=bool)
        for s in range(0, X.shape[0], chunk):
            out[s:s + chunk] = self.add(X[s:s + chunk])
        return out


def rank_mod(X, p=P, chunk=256):
    X = np.asarray(X, dtype=np.int64)
    if X.size == 0:
        return 0
    b = RREFBasis(X.shape[1], p)
    b.add_batched(X, chunk)
    return b.rank
