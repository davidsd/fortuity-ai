"""Full BMN-sector superconformal index of SU(N) N=4 SYM by a torus-grid Haar integral.

Index: Z(t,x,y) = Tr (-1)^F t^{2 sum_I (R_I+J)} x^{q1-q3} y^{q3-q2},  q_I = R_I + J.
Letters (adjoint):  phi_I -> t^2 x_I  (bosonic),  psi_I -> -t^4 / x_I  (fermionic),  f -> t^6 (bosonic),
with (x_1, x_2, x_3) = (x, 1/y, y/x).

The SU(N) Haar integral is done exactly by averaging over an M^{N-1} grid of roots of unity
(z_N = 1), exploiting the S_{N-1} symmetry of the integrand (orbits = multisets).
A series in t (to order jmax) with formal x,y is carried as a dense array over (j, a, b).
"""
import math
import itertools
import numpy as np
from functools import lru_cache

# letter list: (j, a, b, fermionic)   with a = q1 - q3, b = q3 - q2
LETTERS = [
    (2, 1, 0, False), (2, 0, -1, False), (2, -1, 1, False),      # phi_1, phi_2, phi_3
    (4, -1, 0, True), (4, 0, 1, True), (4, 1, -1, True),          # psi_1 (q=(0,1,1)), psi_2 (1,0,1), psi_3 (1,1,0)
    (6, 0, 0, False),                                             # f
]


def weight_ab(q):
    return (q[0] - q[2], q[2] - q[1])


class SeriesBatch:
    """Batch of truncated series S[c, j, a, b], a,b in [-A, A]."""

    def __init__(self, C, jmax, A):
        self.C, self.jmax, self.A = C, jmax, A
        self.S = np.zeros((C, jmax + 1, 2 * A + 1, 2 * A + 1), dtype=np.complex128)
        self.S[:, 0, A, A] = 1.0

    def mul_factor(self, c, jm, am, bm, fermionic):
        """Multiply by (1 - c t^jm x^am y^bm)^(-1) (boson) or (1 - c t^jm x^am y^bm) (fermion).
        c: array (C,) complex."""
        A, J = self.A, self.jmax
        S = self.S
        # target a-range [lo, hi) in index space, source shifted by -am
        a_lo, a_hi = max(0, am), min(2 * A + 1, 2 * A + 1 + am)
        b_lo, b_hi = max(0, bm), min(2 * A + 1, 2 * A + 1 + bm)
        if a_lo >= a_hi or b_lo >= b_hi:
            return
        cc = c[:, None, None]
        if fermionic:
            for j in range(J, jm - 1, -1):
                S[:, j, a_lo:a_hi, b_lo:b_hi] -= cc * S[:, j - jm, a_lo - am:a_hi - am, b_lo - bm:b_hi - bm]
        else:
            for j in range(jm, J + 1):
                S[:, j, a_lo:a_hi, b_lo:b_hi] += cc * S[:, j - jm, a_lo - am:a_hi - am, b_lo - bm:b_hi - bm]


def orbits(M, n):
    """Multisets of n residues mod M with their multiplicities (orbit sizes under S_n)."""
    for ms in itertools.combinations_with_replacement(range(M), n):
        cnt = {}
        for r in ms:
            cnt[r] = cnt.get(r, 0) + 1
        size = math.factorial(n)
        for v in cnt.values():
            size //= math.factorial(v)
        yield ms, size


def full_index(N, jmax, chunk=64, verbose=True, unrefined=False):
    """Return array Z[j, a, b] (a,b shifted by A) of the SU(N) BMN index up to t^jmax.
    If unrefined: x=y=1 and the array has A=0."""
    L = jmax // 2            # max number of letters
    M = L + N                # grid size: exponent range per variable <= L + (N-1)
    A = 0 if unrefined else L
    n = N - 1
    omega = np.exp(2j * np.pi / M)
    total = np.zeros((jmax + 1, 2 * A + 1, 2 * A + 1), dtype=np.complex128)
    orbs = list(orbits(M, n))
    if verbose:
        print(f"[full_index] N={N} jmax={jmax} M={M} orbits={len(orbs)} A={A}", flush=True)
    letters = LETTERS if not unrefined else [(j, 0, 0, f) for (j, a, b, f) in LETTERS]
    for s in range(0, len(orbs), chunk):
        block = orbs[s:s + chunk]
        C = len(block)
        # z values: (C, N) with z_N = 1
        R = np.array([list(ms) + [0] for ms, _ in block], dtype=np.int64)  # residues
        w = np.array([sz for _, sz in block], dtype=np.float64)
        sb = SeriesBatch(C, jmax, A)
        # Vandermonde prod_{a!=b} (1 - z_a/z_b): multiply the batch by this scalar at the end
        z = omega ** R  # (C, N)
        vand = np.ones(C, dtype=np.complex128)
        for a in range(N):
            for b in range(N):
                if a != b:
                    vand *= (1 - z[:, a] / z[:, b])
        # roots: z_a/z_b for a != b, and (N-1) zero weights
        roots = [(z[:, a] / z[:, b]) for a in range(N) for b in range(N) if a != b]
        roots += [np.ones(C, dtype=np.complex128)] * (N - 1)
        for (jm, am, bm, ferm) in letters:
            coef = 1.0  # index: fermionic letters give (1 - w) factors
            for r in roots:
                sb.mul_factor(coef * r, jm, am, bm, ferm)
        total += np.einsum('c,cjab->jab', w * vand, sb.S)
        if verbose and (s // chunk) % 20 == 0:
            print(f"  orbit chunk {s}/{len(orbs)}", flush=True)
    total /= (M ** n) * math.factorial(N)
    Zr = np.rint(total.real).astype(np.int64)
    err = np.abs(total - Zr).max()
    if verbose:
        print(f"[full_index] done, max rounding error {err:.2e}", flush=True)
    assert err < 1e-4, err
    return Zr, A


# ---------- SU(3) representation content ----------

def su3_multiplicities(coeffs, A):
    """coeffs[a+A, b+A] = coefficient of x^a y^b of an SU(3) character (virtual).
    Returns dict {(p,q): multiplicity}.  Highest weight of [p,q] is (a,b) = (p+q, -q)."""
    # Weyl vector rho <-> (q1,q2,q3) = (2,1,0); A_rho = sum_sigma sgn(sigma) e^{sigma(2,1,0)}
    Arho = {}
    for perm in itertools.permutations(range(3)):
        sgn = 1
        pl = list(perm)
        for i in range(3):
            for jj in range(i + 1, 3):
                if pl[i] > pl[jj]:
                    sgn = -sgn
        qv = [0, 0, 0]
        vals = (2, 1, 0)
        for i in range(3):
            qv[perm[i]] = vals[i]
        ab = weight_ab(qv)
        Arho[ab] = Arho.get(ab, 0) + sgn
    n = coeffs.shape[0]
    prod = {}
    nz = np.argwhere(coeffs != 0)
    for (ia, ib) in nz:
        a, b = ia - A, ib - A
        c = int(coeffs[ia, ib])
        for (da, db), s in Arho.items():
            key = (a + da, b + db)
            prod[key] = prod.get(key, 0) + s * c
    out = {}
    for (a, b), c in prod.items():
        if c == 0:
            continue
        # a = p+q+2, b = -q-1
        q = -b - 1
        p = a - 2 - q
        if p >= 0 and q >= 0:
            out[(p, q)] = c
    return out


def character_dim(p, q):
    return (p + 1) * (q + 1) * (p + q + 2) // 2


# ---------- unsigned partition function refined by 2J: dim V_(2J; q1,q2,q3) ----------

def partition_dims(N, jmax, chunk=32, verbose=False):
    """Number of gauge-invariant BMN operators in each sector (2J, q1, q2, q3) with j <= jmax,
    by the SU(N) Haar integral of the unsigned partition function (fermions counted with + sign).
    Returns dict {(2J, q1, q2, q3): dim}."""
    L = jmax // 2
    M = L + N
    A = L
    n = N - 1
    omega = np.exp(2j * np.pi / M)
    JM = jmax // 2          # max 2J
    total = np.zeros((jmax + 1, 2 * A + 1, 2 * A + 1, JM + 1), dtype=np.complex128)
    orbs = list(orbits(M, n))
    # letters with 2J: phi (0), psi (1), f (2)
    letters = [(jm, am, bm, ferm, (1 if ferm else (2 if jm == 6 else 0))) for (jm, am, bm, ferm) in LETTERS]
    for s in range(0, len(orbs), chunk):
        block = orbs[s:s + chunk]
        C = len(block)
        R = np.array([list(ms) + [0] for ms, _ in block], dtype=np.int64)
        w = np.array([sz for _, sz in block], dtype=np.float64)
        z = omega ** R
        vand = np.ones(C, dtype=np.complex128)
        for a in range(N):
            for b in range(N):
                if a != b:
                    vand *= (1 - z[:, a] / z[:, b])
        roots = [(z[:, a] / z[:, b]) for a in range(N) for b in range(N) if a != b]
        roots += [np.ones(C, dtype=np.complex128)] * (N - 1)
        S = np.zeros((C, jmax + 1, 2 * A + 1, 2 * A + 1, JM + 1), dtype=np.complex128)
        S[:, 0, A, A, 0] = 1.0
        for (jm, am, bm, ferm, tj) in letters:
            a_lo, a_hi = max(0, am), min(2 * A + 1, 2 * A + 1 + am)
            b_lo, b_hi = max(0, bm), min(2 * A + 1, 2 * A + 1 + bm)
            for r in roots:
                cc = r[:, None, None, None]
                if ferm:   # (1 + c m): descending j, using old values
                    for j in range(jmax, jm - 1, -1):
                        S[:, j, a_lo:a_hi, b_lo:b_hi, tj:] += cc * S[:, j - jm, a_lo - am:a_hi - am, b_lo - bm:b_hi - bm, :JM + 1 - tj]
                else:      # 1/(1 - c m): ascending j
                    for j in range(jm, jmax + 1):
                        S[:, j, a_lo:a_hi, b_lo:b_hi, tj:] += cc * S[:, j - jm, a_lo - am:a_hi - am, b_lo - bm:b_hi - bm, :JM + 1 - tj]
        total += np.einsum('c,cjabt->jabt', w * vand, S)
        if verbose:
            print(f"  orbit chunk {s}/{len(orbs)}", flush=True)
    total /= (M ** n) * math.factorial(N)
    Zr = np.rint(total.real).astype(np.int64)
    assert np.abs(total - Zr).max() < 1e-4
    dims = {}
    for j in range(0, jmax + 1, 2):
        for ia in range(2 * A + 1):
            for ib in range(2 * A + 1):
                for tj in range(JM + 1):
                    d = int(Zr[j, ia, ib, tj])
                    if d:
                        a, b = ia - A, ib - A
                        # q1 - q3 = a, q3 - q2 = b, q1+q2+q3 = j/2
                        q3 = (j // 2 - a + b) / 3   # from q1 = q3 + a, q2 = q3 - b
                        if q3 != int(q3):
                            continue
                        q3 = int(q3); q1 = q3 + a; q2 = q3 - b
                        dims[(tj, q1, q2, q3)] = d
    return dims
