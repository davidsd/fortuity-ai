"""Exact dimensions of G-invariant spaces of full-letter words for G = SU(N), SO(2N+1), USp(2N) (and G2), by the Weyl
integration formula on a torus grid; SU(3)_R highest-weight multiplicities by Weyl alternation.

Group-general counterpart of fortuity.full.dims (SU(N) only). Letters are the Chang-Lin superfield coefficients
(a, b, S) of fortuity.full.letters; a content is a multiset of letters, every letter in the adjoint of G:
    dim(content) = (1/|W|) * avg over the torus grid of  prod_{alpha}(1 - z^alpha) * prod_L sym_{n_L}(adj eigenvalues),
with sym = complete (bosonic letter) or elementary (fermionic letter) symmetric polynomial of degree n_L in the
eigenvalues {z^w : w weight of the adjoint} (roots plus `rank` zeros).  The grid Z_M^rank with M > (max exponent) is
exact (no aliasing).  2026-10-07 (full-letter SO/USp development code); promoted to code/ in T-048 (2026-10-13).
"""
from collections import Counter
from functools import lru_cache
from itertools import product

import numpy as np

from ..bmn.groupindex import root_system
from .dims import weyl_orbit_signed
from .letters import is_ferm
from .words import contents_of_sector


@lru_cache(maxsize=None)
def _torus(group, N, M):
    roots, r, W = root_system(group, N)
    R = np.array(roots, dtype=np.int64)                      # (nroots, r)
    omega = np.exp(2j * np.pi / M)
    pts = np.array(list(product(range(M), repeat=r)), dtype=np.int64)   # (P, r)
    zr = omega ** ((pts @ R.T) % M)                           # (P, nroots)
    vand = np.prod(1 - zr, axis=1)                            # Weyl measure numerator
    eig = np.concatenate([zr, np.ones((pts.shape[0], r), dtype=np.complex128)], axis=1)   # adjoint eigenvalues
    return eig, vand, W, r


def _sym_polys(eig, nmax):
    """h_n(eig) and e_n(eig) for n = 0..nmax, vectorised over the points (axis 0)."""
    P, D = eig.shape
    e = np.zeros((nmax + 1, P), dtype=np.complex128)
    h = np.zeros((nmax + 1, P), dtype=np.complex128)
    e[0] = 1.0
    h[0] = 1.0
    for d in range(D):
        x = eig[:, d]
        for n in range(min(nmax, d + 1), 0, -1):
            e[n] = e[n] + x * e[n - 1]
    # h_n via Newton-type recursion from power sums
    p = np.array([np.sum(eig ** k, axis=1) for k in range(nmax + 1)])
    for n in range(1, nmax + 1):
        h[n] = sum(p[k] * h[n - k] for k in range(1, n + 1)) / n
    return h, e


def grid_size(group, N, Y):
    roots, r, W = root_system(group, N)
    wmax = max(abs(c) for a in roots for c in a)
    vpos = max(sum(max(0, a[i]) for a in roots) for i in range(r))
    return Y * wmax + vpos + 1


@lru_cache(maxsize=None)
def _sym_cached(group, N, M, nmax):
    eig, vand, W, r = _torus(group, N, M)
    return _sym_polys(eig, nmax)


@lru_cache(maxsize=None)
def content_dim(group, N, content, M=None):
    """Dimension of the G-invariant subspace of the graded-symmetric tensor product of adjoints with the given
    letter multiset (tuple of letters (a, b, S))."""
    Y = len(content)
    M = M or grid_size(group, N, Y)
    eig, vand, W, r = _torus(group, N, M)
    cnt = Counter(content)
    nmax = max(cnt.values())
    h, e = _sym_cached(group, N, M, max(nmax, 4))
    f = vand.copy()
    for L, n in cnt.items():
        f = f * (e[n] if is_ferm(L) else h[n])
    val = f.sum() / (M ** r) / W
    rv = int(round(val.real))
    assert abs(val - rv) < 1e-6, (val, content)
    return rv


def sector_dim(group, N, sec, Y=None):
    return sum(content_dim(group, N, c) for c in contents_of_sector(sec) if Y is None or len(c) == Y)


def hw_multiplicity(group, N, twoJ1, twoJ2, lam2, Y=None):
    """Exact multiplicity of the SU(3)_R irrep with highest weight lam2 (doubled q) in the G-invariant word space
    at (2J1, 2J2), optionally at fixed letter number Y."""
    tot = 0
    for sgn, w in weyl_orbit_signed(lam2):
        if min(w) < 0:
            continue
        tot += sgn * sector_dim(group, N, (twoJ1, twoJ2) + w, Y)
    return tot
