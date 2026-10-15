"""Random evaluation of full-letter multitrace words (generic version of fortuity.hwcoh.evalg.Frame: traceless
N x N matrices over F_p, fermionic letters in a Grassmann frame with k generators, sparse structure tables,
sorted-prefix reuse).  Letters are (a, b, S) tuples; a word is a tuple of necklaces (tuples of letters)."""
import math
import numpy as np
from fortuity.bmn.modp import P as PRIME
from fortuity.hwcoh.evalg import table
from .letters import is_ferm


def nferm(w):
    return sum(1 for n in w for L in n if is_ferm(L))


class Frame:
    def __init__(self, N, npts, k, degrees, letters, seed=0, p=PRIME, cartan=False, u2block=False, group='SU'):
        # SO/Sp: group = 'SU' (traceless N x N), 'SO_odd' (so(2N+1), antisymmetric) or 'Sp' (usp(2N), X = -Omega S),
        # generated exactly as in fortuity.hwcoh.evalg.Frame (cartan=True: Cartan-valued letters, the Coulomb branch)
        self.N, self.P, self.k, self.p = N, npts, k, p
        self.group = group
        self.degrees = sorted(degrees)
        rng = np.random.default_rng(seed)
        invN = pow(N, p - 2, p)
        idx = np.arange(N)
        if group != 'SU':
            from fortuity.hwcoh.evalg import _cartan, matrix_size
            n = matrix_size(group, N)
            if group == 'Sp':
                Omega = np.zeros((n, n), dtype=np.int64)
                Omega[:N, N:] = np.eye(N, dtype=np.int64)
                Omega[N:, :N] = -np.eye(N, dtype=np.int64)

        def group_letter(shape):
            if cartan:
                return _cartan(group, N, n, shape, rng, p)
            M = rng.integers(0, p, shape + (n, n), dtype=np.int64)
            if group == 'SO_odd':
                return (M - np.swapaxes(M, -1, -2)) % p
            if group == 'Sp':
                S = (M + np.swapaxes(M, -1, -2)) % p
                return (-np.einsum('ij,...jk->...ik', Omega, S)) % p
            raise NotImplementedError(group)

        def traceless(shape):
            if group != 'SU':
                return group_letter(shape)
            M = rng.integers(0, p, shape + (N, N), dtype=np.int64)
            if u2block:                      # u(2) subalgebra of su(3): diag(A, -tr A), A a generic 2x2 block
                M = np.zeros(shape + (N, N), dtype=np.int64)
                A = rng.integers(0, p, shape + (2, 2), dtype=np.int64)
                M[..., :2, :2] = A
                M[..., 2, 2] = (-(A[..., 0, 0] + A[..., 1, 1])) % p
                return M
            if cartan:                       # Coulomb branch: diagonal traceless letters (Q_0-exact words vanish)
                M = np.zeros(shape + (N, N), dtype=np.int64)
                a = rng.integers(0, p, shape + (N,), dtype=np.int64)
                a[..., -1] = (-a[..., :-1].sum(axis=-1)) % p
                M[..., idx, idx] = a
                return M
            tr = np.trace(M, axis1=-2, axis2=-1)
            M[..., idx, idx] = (M[..., idx, idx] - (tr[..., None] * invN) % p) % p
            return M

        self.L = {}
        for L in sorted(letters):
            self.L[L] = (1, traceless((npts, k))) if is_ferm(L) else (0, traceless((npts, 1)))
        self.ncomp = {d: math.comb(k, d) for d in self.degrees}
        self.offsets, off = {}, 0
        for d in self.degrees:
            self.offsets[d] = off
            off += self.ncomp[d]
        self.cpp = off
        self.ncols = off * npts

    def mat_mul_letter(self, A, dA, L):
        dl, B = self.L[L]
        p = self.p
        if dl == 0:
            return np.matmul(A, B) % p, dA
        I, J, S, st = table(self.k, dA, 1)
        R = np.matmul(A[:, I], B[:, J]) % p
        R *= S[None, :, None, None]
        R = np.add.reduceat(R, st, axis=1) % p
        return R, dA + 1

    def scal_mul(self, a, da, b, db):
        I, J, S, st = table(self.k, da, db)
        r = (a[:, I] * b[:, J]) % self.p
        r *= S[None, :]
        return np.add.reduceat(r, st, axis=1) % self.p, da + db

    def necklace_values(self, necks):
        out = {}
        stack, prev = [], ()
        for w in sorted(set(necks)):
            cp = 0
            while cp < min(len(w), len(prev)) and w[cp] == prev[cp]:
                cp += 1
            del stack[cp:]
            if not stack:
                d0, v0 = self.L[w[0]]
                stack.append((v0, d0))
            for L in w[len(stack):]:
                v, d = stack[-1]
                stack.append(self.mat_mul_letter(v, d, L))
            v, d = stack[-1]
            out[w] = (np.trace(v, axis1=2, axis2=3) % self.p, d)
            prev = w
        return out

    def eval_words(self, words, dtype=np.int32):
        nv = self.necklace_values([n for w in words for n in w])
        M = np.zeros((len(words), self.P, self.cpp), dtype=dtype)
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
            if d not in self.offsets:
                raise ValueError(f"word of degree {d} in a frame for degrees {self.degrees}")
            o = self.offsets[d]
            M[i, :, o:o + self.ncomp[d]] = v
            prev = w
        return M.reshape(len(words), -1)
