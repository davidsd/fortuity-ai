"""Exact dimensions of gauge-invariant spaces with given full-letter content (SU(N)), by the Weyl integration
formula on a torus grid: dim(c) = int dmu prod_L chi_{Sym^{n_L} adj} (bosonic L) chi_{Lambda^{n_L} adj} (fermionic L).
Sector (weight-space) dimensions are sums over contents; SU(3)_R highest-weight multiplicities follow by Weyl
alternation over the orbit of lam + rho."""
import math
import numpy as np
from itertools import product, permutations
from collections import Counter
from .letters import is_ferm
from .words import contents_of_sector

RHO = (1, 0, -1)
_GRID = {}


def _grid(N, M):
    key = (N, M)
    if key not in _GRID:
        omega = np.exp(2j * np.pi / M)
        ms = np.array(list(product(range(M), repeat=N - 1)), dtype=np.int64)
        z = omega ** ms                                   # (G, N-1)
        zN = 1 / np.prod(z, axis=1)
        z = np.concatenate([z, zN[:, None]], axis=1)      # (G, N)
        vand = np.ones(len(z), dtype=np.complex128)
        for a in range(N):
            for b in range(N):
                if a != b:
                    vand *= (1 - z[:, a] / z[:, b])
        weight = vand / (M ** (N - 1) * math.factorial(N))
        # power sums of the adjoint character: p_k = sum_{a!=b} (z_a/z_b)^k + N - 1
        ratios = np.stack([z[:, a] / z[:, b] for a in range(N) for b in range(N) if a != b], axis=1)
        _GRID[key] = (weight, ratios, N)
    return _GRID[key]


def _sym_ext(N, M, nmax):
    weight, ratios, _ = _grid(N, M)
    G = len(weight)
    p = [None] + [np.sum(ratios ** k, axis=1) + (N - 1) for k in range(1, nmax + 1)]
    h = [np.ones(G, dtype=np.complex128)]
    e = [np.ones(G, dtype=np.complex128)]
    for n in range(1, nmax + 1):
        h.append(sum(p[k] * h[n - k] for k in range(1, n + 1)) / n)
        e.append(sum((-1) ** (k - 1) * p[k] * e[n - k] for k in range(1, n + 1)) / n)
    return weight, h, e


def content_dim(N, content, M=None):
    """Dimension of the SU(N)-invariant subspace of the (graded-symmetric) tensor product of adjoints
    with the given letter multiset."""
    Y = len(content)
    M = M or (N * Y + 2 * N + 1)   # z_1 exponents: up to N-1 per letter, plus the Vandermonde; no aliasing
    cnt = Counter(content)
    nmax = max(cnt.values())
    weight, h, e = _sym_ext(N, M, nmax)
    f = weight.copy()
    for L, n in cnt.items():
        f = f * (e[n] if is_ferm(L) else h[n])
    val = f.sum()
    r = int(round(val.real))
    assert abs(val - r) < 1e-6, val
    return r


def sector_dim(N, sec, Y=None):
    return sum(content_dim(N, c) for c in contents_of_sector(sec) if Y is None or len(c) == Y)


def weyl_orbit_signed(lam2):
    """[(sign, weight)] for w(lam + rho) - rho over the S_3 Weyl group, in DOUBLED coordinates 2q (the q_j are
    half-integral when the total number of derivatives + gauginos is odd)."""
    lr = tuple(l + 2 * r for l, r in zip(lam2, RHO))
    out = []
    for perm in permutations(range(3)):
        inv = sum(1 for i in range(3) for j in range(i + 1, 3) if perm[i] > perm[j])
        w = tuple(lr[perm[i]] - 2 * RHO[i] for i in range(3))
        out.append((-1 if inv % 2 else 1, w))
    return out


def hw_multiplicity(N, twoJ1, twoJ2, lam2, Y=None):
    """Exact multiplicity of the SU(3)_R irrep with highest weight lam2 (doubled q coordinates) in the free word
    space at (2J1, 2J2), optionally at fixed letter number Y."""
    tot = 0
    for sgn, w in weyl_orbit_signed(lam2):
        if min(w) < 0:
            continue
        tot += sgn * sector_dim(N, (twoJ1, twoJ2) + w, Y)
    return tot


def content_dim_u2(content, M=None):
    """Dimension of the U(2)-invariant subspace for letters in gl(2) = adjoint of U(2) (the u(2)-block restriction
    of SU(3) letters): torus grid over U(2), adjoint character chi = sum_{a!=b} z_a/z_b + 2."""
    Y = len(content)
    M = M or (2 * Y + 5)
    key = ('U2', M)
    if key not in _GRID:
        omega = np.exp(2j * np.pi / M)
        ms = np.array(list(product(range(M), repeat=2)), dtype=np.int64)
        z = omega ** ms
        vand = (1 - z[:, 0] / z[:, 1]) * (1 - z[:, 1] / z[:, 0])
        weight = vand / (M ** 2 * 2)
        ratios = np.stack([z[:, 0] / z[:, 1], z[:, 1] / z[:, 0]], axis=1)
        _GRID[key] = (weight, ratios, 2)
    weight, ratios, _ = _GRID[key]
    cnt = Counter(content)
    nmax = max(cnt.values())
    G = len(weight)
    p = [None] + [np.sum(ratios ** k, axis=1) + 2 for k in range(1, nmax + 1)]
    h = [np.ones(G, dtype=np.complex128)]
    e = [np.ones(G, dtype=np.complex128)]
    for n in range(1, nmax + 1):
        h.append(sum(p[k] * h[n - k] for k in range(1, n + 1)) / n)
        e.append(sum((-1) ** (k - 1) * p[k] * e[n - k] for k in range(1, n + 1)) / n)
    f = weight.copy()
    for L, n in cnt.items():
        f = f * (e[n] if is_ferm(L) else h[n])
    val = f.sum()
    r = int(round(val.real))
    assert abs(val - r) < 1e-6, val
    return r
