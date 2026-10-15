"""Tests for the BMN-sector subpackage (fast: a few seconds)."""
import numpy as np
from fortuity.bmn.words import (canon_word, Q_expr, graviton_generators, expr_sector, content,
                                words_of_sector, add_term)
from fortuity.bmn.evaluate import Points
from fortuity.bmn.modp import matmul_mod, rank_mod, RREFBasis, P
from fortuity.bmn.fullindex import full_index, partition_dims
from fortuity.bmn.gravindex import GravitonCounter
from collections import defaultdict


def _vec(expr, N, seed=1, npts=6):
    degs = sorted({sum(content(m)[3:6]) for m in expr})
    k = max(degs)
    return Points(N, npts, max(k, 1), degs, seed=seed).expr_vector(expr)


def test_modp_rank():
    rng = np.random.default_rng(0)
    U = rng.integers(0, P, (200, 60)); V = rng.integers(0, P, (60, 150))
    M = matmul_mod(U, V); M[5] = M[3]
    assert rank_mod(M) == 60
    b = RREFBasis(150); b.add_batched(M, 32); assert b.rank == 60


def test_Q_squared_zero():
    rng = np.random.default_rng(0)
    for N in (2, 3):
        for _ in range(15):
            w = tuple(rng.integers(0, 7, rng.integers(2, 6))); w2 = tuple(rng.integers(0, 7, rng.integers(2, 5)))
            s, cw = canon_word(w); s2, cw2 = canon_word(w2)
            if not (s and s2):
                continue
            e = defaultdict(int); add_term(e, (cw, cw2), 1)
            if not e:
                continue
            qq = Q_expr(Q_expr(e))
            if qq:
                assert not np.any(_vec(qq, N))


def test_generators_closed():
    for name, e in graviton_generators(3, 1):
        q = Q_expr(e)
        if q:
            for N in (2, 3):
                assert not np.any(_vec(q, N)), name


def test_su2_index_and_gravitons():
    # unrefined SU(2) BMN index and graviton index through t^16 (Choi et al. closed form: 1,6,-8,18,-30,56,-81,117)
    Z, A = full_index(2, 16, verbose=False)
    assert [int(x) for x in Z.sum(axis=(1, 2))[::2]] == [1, 0, 6, -8, 18, -30, 56, -81, 117]
    G = GravitonCounter(2, 16, verbose=False, npts=64); G.run()
    assert [int(x) for x in G.index_array().sum(axis=(1, 2))[::2]] == [1, 0, 6, -8, 18, -30, 56, -81, 117]


def test_su2_sector_dimensions():
    d = partition_dims(2, 24)
    assert d[(5, 4, 4, 4)] == 153 and d[(4, 4, 4, 4)] == 232 and d[(6, 4, 4, 4)] == 53
