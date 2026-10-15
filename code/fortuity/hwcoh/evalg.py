"""Random evaluation of multitrace words (traceless N x N matrices over F_p, fermions in a Grassmann frame).

Same idea as fortuity.bmn.evaluate (psi_m -> sum_s theta_s V^(s)_m with k Grassmann generators), but Grassmann
products use sparse structure tables (only disjoint subset pairs), and words are evaluated in sorted order reusing
common prefixes (necklace prefixes and multitrace prefixes).
Output layout per word: for each point, one block per sector degree d (components = d-subsets of range(k)).
"""
import math
from itertools import combinations
from functools import lru_cache
import numpy as np
from fortuity.bmn.modp import P as PRIME

PHI, PSI, F = (0, 1, 2), (3, 4, 5), 6


@lru_cache(maxsize=None)
def subsets(k, d):
    return tuple(combinations(range(k), d))


@lru_cache(maxsize=None)
def sindex(k, d):
    return {s: i for i, s in enumerate(subsets(k, d))}


@lru_cache(maxsize=None)
def table(k, d1, d2):
    """Pairs (i, j) of disjoint subsets (sizes d1, d2) -> target o, sign; sorted by o.
    Returns I, J, S (int64 arrays) and reduceat starts."""
    tgt = sindex(k, d1 + d2)
    rows = []
    for i, s in enumerate(subsets(k, d1)):
        ss = set(s)
        for j, t in enumerate(subsets(k, d2)):
            if ss.intersection(t):
                continue
            inv = sum(1 for a in s for b in t if a > b)
            rows.append((tgt[tuple(sorted(s + t))], i, j, -1 if inv % 2 else 1))
    rows.sort()
    O = np.array([r[0] for r in rows], dtype=np.int64)
    I = np.array([r[1] for r in rows], dtype=np.int64)
    J = np.array([r[2] for r in rows], dtype=np.int64)
    S = np.array([r[3] for r in rows], dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, O[1:] != O[:-1]])
    assert len(starts) == math.comb(k, d1 + d2)
    return I, J, S, starts


def _cartan(group, N, n, shape, rng, p):
    a = rng.integers(0, p, shape + (max(N, 1),), dtype=np.int64)
    X = np.zeros(shape + (n, n), dtype=np.int64)
    if group == 'SU':
        a = a[..., :N]
        a[..., -1] = (-a[..., :-1].sum(axis=-1)) % p           # traceless
        idx = np.arange(N)
        X[..., idx, idx] = a
    elif group == 'SO_odd':                                  # 2x2 rotation blocks, last row/column zero
        for i in range(N):
            X[..., 2 * i, 2 * i + 1] = a[..., i]
            X[..., 2 * i + 1, 2 * i] = (-a[..., i]) % p
    elif group == 'Sp':                                      # diag(a, -a) in the Omega basis
        idx = np.arange(N)
        X[..., idx, idx] = a
        X[..., idx + N, idx + N] = (-a) % p
    elif group == 'G2':                                      # random combinations of a basis of a Cartan subalgebra
        B = _g2_cartan_basis(p)                              # (2, 7, 7)
        c = rng.integers(0, p, shape + (2,), dtype=np.int64)
        X = (c[..., 0, None, None] * B[0] + c[..., 1, None, None] * B[1]) % p
    else:
        raise NotImplementedError(f"Cartan letters not implemented for {group}")
    return X % p


_G2_CARTAN = {}


def _g2_cartan_basis(p):
    """A basis of a Cartan subalgebra of g2 (7x7, over F_p), with no reference to the form of phi: the centraliser in g2 of
    a fixed random element H of g2 (regular with probability ~1; checked: the centraliser is 2-dimensional)."""
    if p in _G2_CARTAN:
        return _G2_CARTAN[p]
    from fortuity.bmn.evaluate import project_g2
    from .hw import nullspace_left
    from fortuity.bmn.modp import rank_mod
    r = np.random.default_rng(20261001)
    E = []
    for a in range(7):
        for b in range(a + 1, 7):
            M = np.zeros((7, 7), dtype=np.int64); M[a, b] = 1
            E.append(project_g2(M, p))
    E = np.array(E) % p                                      # 21 spanning elements of g2 (rank 14)
    H = project_g2(r.integers(0, p, (7, 7), dtype=np.int64), p)
    comm = np.array([(H @ e - e @ H) % p for e in E]).reshape(len(E), -1) % p
    K = nullspace_left(comm, p)                              # combinations c with [H, sum c_i E_i] = 0
    C = np.einsum('ki,iab->kab', K % p, E) % p
    C = C[[i for i in range(len(C)) if C[i].any()]]
    basis = []
    for M in C:                                              # extract 2 independent elements
        cand = basis + [M.reshape(-1)]
        if rank_mod(np.array(cand) % p, p) == len(cand):
            basis.append(M.reshape(-1))
        if len(basis) == 2:
            break
    assert len(basis) == 2 and rank_mod(C.reshape(len(C), -1) % p, p) == 2, "centraliser is not 2-dimensional"
    B = np.array(basis).reshape(2, 7, 7) % p
    assert not ((B[0] @ B[1] - B[1] @ B[0]) % p).any()
    _G2_CARTAN[p] = B
    return B


def matrix_size(group, N):
    return {'SU': N, 'SO_odd': 2 * N + 1, 'Sp': 2 * N, 'G2': 7}[group]


def algebra_dim(group, N):
    return {'SU': N * N - 1, 'SO_odd': N * (2 * N + 1), 'Sp': N * (2 * N + 1), 'G2': 14}[group]


class Frame:
    def __init__(self, N, npts, k, degrees, seed=0, p=PRIME, group='SU', cartan=False):
        """group: 'SU' (traceless N x N), 'SO_odd' (so(2N+1): antisymmetric) or 'Sp' (usp(2N): X = -Omega S,
        S symmetric), as in fortuity.bmn.evaluate.Points."""
        self.N, self.P, self.k, self.p = N, npts, k, p
        self.group = group
        self.degrees = sorted(degrees)
        rng = np.random.default_rng(seed)
        invN = pow(N, p - 2, p)
        n = matrix_size(group, N)
        if group == 'Sp':
            Omega = np.zeros((n, n), dtype=np.int64)
            Omega[:N, N:] = np.eye(N, dtype=np.int64)
            Omega[N:, :N] = -np.eye(N, dtype=np.int64)

        def traceless(shape):
            M = rng.integers(0, p, shape + (n, n), dtype=np.int64)
            if cartan:
                # letters restricted to a Cartan subalgebra (the Coulomb branch): exact operators vanish there, so the
                # rank of graviton rows at such points counts the Coulomb-type graviton classes
                return _cartan(group, N, n, shape, rng, p)
            if group == 'SO_odd':
                return (M - np.swapaxes(M, -1, -2)) % p
            if group == 'G2':
                from fortuity.bmn.evaluate import project_g2
                return project_g2(M, p)
            if group == 'Sp':
                S = (M + np.swapaxes(M, -1, -2)) % p
                return (-np.einsum('ij,...jk->...ik', Omega, S)) % p
            tr = np.trace(M, axis1=-2, axis2=-1)
            idx = np.arange(N)
            M[..., idx, idx] = (M[..., idx, idx] - (tr[..., None] * invN) % p) % p
            return M

        # deg-0 letters: (P, 1, N, N); psi: (P, k, N, N)
        self.L = {}
        for l in PHI + (F,):
            self.L[l] = (0, traceless((npts, 1)))
        for l in PSI:
            self.L[l] = (1, traceless((npts, k)))
        self.ncomp = {d: math.comb(k, d) for d in self.degrees}
        self.offsets, off = {}, 0
        for d in self.degrees:
            self.offsets[d] = off
            off += self.ncomp[d]
        self.cpp = off
        self.ncols = off * npts

    # --- Grassmann products ---
    def mat_mul_letter(self, A, dA, l):
        """(P, C(k,dA), N, N) times letter l -> (value, degree)."""
        dl, B = self.L[l]
        p = self.p
        if dl == 0:
            return np.matmul(A, B) % p, dA
        I, J, S, st = table(self.k, dA, 1)
        R = np.matmul(A[:, I], B[:, J]) % p          # (P, nnz, N, N)
        R *= S[None, :, None, None]
        R = np.add.reduceat(R, st, axis=1) % p
        return R, dA + 1

    def scal_mul(self, a, da, b, db):
        I, J, S, st = table(self.k, da, db)
        r = (a[:, I] * b[:, J]) % self.p
        r *= S[None, :]
        return np.add.reduceat(r, st, axis=1) % self.p, da + db

    # --- words ---
    def necklace_values(self, necks):
        """dict necklace -> (scalar array (P, C), degree), for a collection of necklaces (prefix reuse)."""
        out = {}
        stack = []   # stack[i] = (value, degree) of prefix of length i+1
        prev = ()
        for w in sorted(set(necks)):
            cp = 0
            while cp < min(len(w), len(prev)) and w[cp] == prev[cp]:
                cp += 1
            del stack[cp:]
            if not stack:
                d0, v0 = self.L[w[0]]
                stack.append((v0, d0))
            for l in w[len(stack):]:
                v, d = stack[-1]
                stack.append(self.mat_mul_letter(v, d, l))
            v, d = stack[-1]
            out[w] = (np.trace(v, axis1=2, axis2=3) % self.p, d)
            prev = w
        return out

    def eval_words(self, words, out=None, dtype=np.int32):
        """Evaluation matrix (len(words), ncols) of multitraces (rows in the given order)."""
        nv = self.necklace_values([n for w in words for n in w])
        M = np.zeros((len(words), self.P, self.cpp), dtype=dtype) if out is None else out
        order = sorted(range(len(words)), key=lambda i: words[i])
        stack, prev = [], ()
        for i in order:
            w = words[i]
            cp = 0
            while cp < min(len(w), len(prev)) and w[cp] == prev[cp]:
                cp += 1
            del stack[cp:]
            if not stack:
                stack.append(nv[w[0]])
            for n in w[len(stack):]:
                v, d = stack[-1]
                a, da = nv[n]
                stack.append(self.scal_mul(v, d, a, da))
            v, d = stack[-1]
            o = self.offsets[d]
            M[i, :, o:o + self.ncomp[d]] = v
            prev = w
        return M.reshape(len(words), -1)


def nferm(w):
    return sum(1 for n in w for l in n if 3 <= l <= 5)


class DegFrames:
    """One Frame per fermion degree d present in a sector (the evaluation space is a direct sum over d, and a
    Grassmann frame gives only C(k, d) functionals per point on degree-d words).  need[d] = number of functionals
    wanted on the degree-d part.  Columns: blocks per degree."""

    def __init__(self, N, need, extra=4, target=256, seed=0, p=PRIME, group='SU'):
        self.N, self.p = N, p
        kmax = 3 * algebra_dim(group, N)
        self.frames, self.off = {}, {}
        off = 0
        for d in sorted(need):
            k = d
            if d > 0:
                k = min(kmax, d + extra)
                while k < kmax and math.comb(k, d) < target:
                    k += 1
            npts = max(1, math.ceil(need[d] / math.comb(k, d)))
            fr = Frame(N, npts, k, [d], seed=seed + 7919 * d, p=p, group=group)
            self.frames[d] = fr
            self.off[d] = off
            off += fr.ncols
        self.ncols = off

    def describe(self):
        return ", ".join(f"d={d}: k={f.k} pts={f.P} cols={f.ncols}" for d, f in self.frames.items())

    def eval_words(self, words):
        M = np.zeros((len(words), self.ncols), dtype=np.int64)
        byd = {}
        for i, w in enumerate(words):
            byd.setdefault(nferm(w), []).append(i)
        for d, idx in byd.items():
            fr = self.frames[d]
            X = fr.eval_words([words[i] for i in idx])
            M[np.array(idx), self.off[d]:self.off[d] + fr.ncols] = X
        return M
