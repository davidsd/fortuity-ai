"""Random-point evaluation of multi-trace expressions over arbitrary letters (generic version of bmn.evaluate)."""
import numpy as np
from fortuity.bmn.modp import P
from fortuity.bmn.grassmann import gmul_matrix, gmul_scalar, gtrace, gadd, subsets
from .letters import is_ferm


class PointsFull:
    def __init__(self, N, npts, k, degrees, letters, seed=0, p=P):
        self.N, self.P, self.k, self.p = N, npts, k, p
        self.degrees = sorted(degrees)
        rng = np.random.default_rng(seed)
        invN = pow(N, p - 2, p)
        idx = np.arange(N)

        def traceless(shape):
            M = rng.integers(0, p, shape + (N, N), dtype=np.int64)
            tr = np.trace(M, axis1=-2, axis2=-1)
            M[..., idx, idx] = (M[..., idx, idx] - (tr[..., None] * invN) % p) % p
            return M % p

        self.letter = {}
        for L in letters:
            self.letter[L] = {1: traceless((npts, k))} if is_ferm(L) else {0: traceless((npts, 1))}
        self.ncomp = {d: len(subsets(k, d)) for d in self.degrees}
        self.offsets, off = {}, 0
        for d in self.degrees:
            self.offsets[d] = off
            off += self.ncomp[d]
        self.cols_per_point = off
        self.ncols = off * npts
        self._cache = {}

    def necklace(self, w):
        c = self._cache.get(w)
        if c is not None:
            return c
        M = self.letter[w[0]]
        for L in w[1:]:
            M = gmul_matrix(M, self.letter[L], self.k, self.p)
        val = gtrace(M, self.p)
        self._cache[w] = val
        return val

    def multitrace(self, multi):
        if not multi:
            return {0: np.ones((self.P, 1), dtype=np.int64)}
        val = self.necklace(multi[0])
        for w in multi[1:]:
            val = gmul_scalar(val, self.necklace(w), self.k, self.p)
        return val

    def expr_vector(self, expr):
        acc = {}
        for multi, c in expr.items():
            acc = gadd(acc, self.multitrace(multi), self.p, coeff=c % self.p)
        out = np.zeros((self.P, self.cols_per_point), dtype=np.int64)
        for d, a in acc.items():
            if d not in self.offsets:
                if np.any(a):
                    raise ValueError(f"degree {d} outside {self.degrees}")
                continue
            out[:, self.offsets[d]:self.offsets[d] + self.ncomp[d]] = a
        return out.reshape(-1)
