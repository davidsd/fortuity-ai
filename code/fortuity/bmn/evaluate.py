"""Evaluate gauge-invariant expressions at random points (traceless N x N matrices over F_p).

Fermionic letters psi_m are replaced by sum_s theta_s V^(s)_m with k Grassmann generators theta_s and
random numeric matrices V^(s)_m.  A word with d fermions evaluates to a degree-d element of Lambda[theta].
The vector of all components over all points is a faithful linear functional set on the space of
gauge-invariant operators (for enough points).
"""
import numpy as np
from collections import OrderedDict
from .modp import P
from .grassmann import gmul_matrix, gmul_scalar, gtrace, gadd, subsets
from .words import is_ferm, F, PHI, PSI, nferm


def g2_phi():
    """The octonion 3-form phi_{abc} on R^7 (entries 0, +-1), normalised so that phi_acd phi_bcd = 6 delta_ab."""
    triples = [((0, 1, 2), 1), ((0, 3, 4), 1), ((0, 5, 6), 1), ((1, 3, 5), 1), ((1, 4, 6), -1), ((2, 3, 6), -1), ((2, 4, 5), -1)]
    phi = np.zeros((7, 7, 7), dtype=np.int64)
    for (a, b, c), s in triples:
        for (x, y, z), sg in (((a, b, c), 1), ((b, c, a), 1), ((c, a, b), 1), ((b, a, c), -1), ((a, c, b), -1), ((c, b, a), -1)):
            phi[x, y, z] = s * sg
    return phi


def project_g2(M, p):
    """Random elements of g2 in the 7-dimensional representation, over F_p: antisymmetrise M (..., 7, 7) and remove the
    7-dimensional part, X -> X - (1/6) phi_abc phi_ade X_de (so(7) = g2 + 7)."""
    phi = g2_phi()
    X = (M - np.swapaxes(M, -1, -2)) % p
    inv6 = pow(6, p - 2, p)
    Y = np.einsum('ade,...de->...a', phi, X) % p           # 7-component part
    Z = np.einsum('abc,...a->...bc', phi, Y) % p
    return (X - inv6 * Z) % p


class Points:
    CACHE_BYTES = 3 * 1024**3   # LRU budget for cached trace values

    def __init__(self, N, npts, k, degrees, seed=0, p=P, diagonal=False, group='SU'):
        """N: gauge rank, npts: number of points, k: number of Grassmann generators,
        degrees: sorted list of fermion numbers appearing in this sector.
        group: 'SU' (traceless N x N), 'SO_odd' (SO(2N+1): antisymmetric (2N+1) x (2N+1)) or
        'Sp' (Sp(N) = USp(2N): X = -Omega S with S symmetric 2N x 2N, i.e. Omega X + X^T Omega = 0) or
        'G2' (7 x 7, the stabiliser of the octonion 3-form inside so(7); N is ignored)."""
        self.N, self.P, self.k, self.p = N, npts, k, p
        self.group = group
        self.degrees = sorted(degrees)
        rng = np.random.default_rng(seed)
        if group != 'SU' and diagonal:
            raise NotImplementedError("diagonal (Cartan) points only implemented for SU(N)")
        n = {'SU': N, 'SO_odd': 2 * N + 1, 'Sp': 2 * N, 'G2': 7}[group]
        if group == 'Sp':
            Omega = np.zeros((n, n), dtype=np.int64)
            Omega[:N, N:] = np.eye(N, dtype=np.int64)
            Omega[N:, :N] = -np.eye(N, dtype=np.int64)

        def traceless(shape):
            M = rng.integers(0, p, shape + (n, n), dtype=np.int64)
            if group == 'SO_odd':
                return (M - np.swapaxes(M, -1, -2)) % p
            if group == 'G2':
                return project_g2(M, p)
            if group == 'Sp':
                S = (M + np.swapaxes(M, -1, -2)) % p
                return (-np.einsum('ij,...jk->...ik', Omega, S)) % p
            if diagonal:
                M = M * np.eye(N, dtype=np.int64)
            tr = np.trace(M, axis1=-2, axis2=-1)
            # subtract trace/N: multiply by inverse of N mod p
            invN = pow(N, p - 2, p)
            idx = np.arange(N)
            M[..., idx, idx] = (M[..., idx, idx] - (tr[..., None] * invN) % p) % p
            return M % p

        self.letter = {}
        for m in PHI:
            self.letter[m] = {0: traceless((npts, 1))}
        self.letter[F] = {0: traceless((npts, 1))}
        for m in PSI:
            self.letter[m] = {1: traceless((npts, k))}
        self.ncomp = {d: len(subsets(k, d)) for d in self.degrees}
        self.offsets = {}
        off = 0
        for d in self.degrees:
            self.offsets[d] = off
            off += self.ncomp[d]
        self.cols_per_point = off
        self.ncols = off * npts
        self._neck_cache = OrderedDict()
        self._cache_bytes = 0

    def necklace(self, w):
        """Value of tr(word) as {deg: (P, C)}."""
        c = self._neck_cache.get(w)
        if c is not None:
            self._neck_cache.move_to_end(w)
            return {d: a.astype(np.int64) for d, a in c.items()}
        M = self.letter[w[0]]
        for l in w[1:]:
            M = gmul_matrix(M, self.letter[l], self.k, self.p)
        val = gtrace(M, self.p)
        c = {d: a.astype(np.int32) for d, a in val.items()}   # values < 2^26
        nb = sum(a.nbytes for a in c.values())
        self._neck_cache[w] = c
        self._cache_bytes += nb
        while self._cache_bytes > self.CACHE_BYTES and len(self._neck_cache) > 1:
            _, old = self._neck_cache.popitem(last=False)
            self._cache_bytes -= sum(a.nbytes for a in old.values())
        return val

    def multitrace(self, multi):
        if len(multi) == 0:
            return {0: np.ones((self.P, 1), dtype=np.int64)}
        val = self.necklace(multi[0])
        for w in multi[1:]:
            val = gmul_scalar(val, self.necklace(w), self.k, self.p)
        return val

    def expr_vector(self, expr):
        """Flattened evaluation vector (length ncols) of an expression."""
        acc = {}
        for multi, c in expr.items():
            acc = gadd(acc, self.multitrace(multi), self.p, coeff=c % self.p)
        return self.flatten(acc)

    def flatten(self, acc):
        out = np.zeros((self.P, self.cols_per_point), dtype=np.int64)
        for d, a in acc.items():
            if d not in self.offsets:
                if np.any(a):
                    raise ValueError(f"degree {d} not in sector degrees {self.degrees}")
                continue
            o = self.offsets[d]
            out[:, o:o + self.ncomp[d]] = a
        return out.reshape(-1)

    def matrix(self, exprs):
        return np.array([self.expr_vector(e) for e in exprs], dtype=np.int64).reshape(len(exprs), -1)
