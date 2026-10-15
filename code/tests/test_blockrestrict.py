"""Tests for fortuity.bmn.blockrestrict (block restriction SU(N) > SU(N-1) x U(1), SO(2N+1) > SO(2N+1-2b) x SO(2)^b).

The formal maps on multi-trace words are compared with direct evaluation of the big-group words on block-diagonal
matrices built from small-group points (fermions via the Grassmann frames of fortuity.bmn.evaluate), extracting the
coefficient of the abelian monomial by interpolation in a scale t.  Also: Q commutes with both maps (chain maps), and
two single-block SO(2) steps compose to the two-block map.
"""
import random
from itertools import product
import numpy as np

from fortuity.bmn.evaluate import Points
from fortuity.bmn.words import words_of_sector, nferm, Q_expr, PHI
from fortuity.bmn.modp import P
from fortuity.bmn.blockrestrict import restrict_u1, restrict_so2, embed_points, interpolate_coefficient


def _random_expr(sec, n, seed, keep=None):
    rng = random.Random(seed)
    ws = [w for w in words_of_sector(sec, 100) if keep is None or keep(w)]
    return {w: rng.randrange(1, P) for w in rng.sample(ws, min(n, len(ws)))}


def _movable(m):
    """a trace that can move into an SO(2) block and carries x_1^2 at least"""
    return any(len(w) % 2 == 0 and all(l in PHI for l in w) and w.count(0) >= 2 for w in m)


def _numeric_vs_formal(big_expr, group, Nbig, Nsmall, kind, powers, npts=3, seed=11):
    degs = sorted({sum(nferm(w) for w in m) for m in big_expr})
    k = max(max(degs), 1)
    small = Points(Nsmall, npts, k, degs, seed=seed, group=group)
    maxdeg = max(max(sum(w.count(0) for w in m) for m in big_expr), max(powers))   # polynomial degree in t
    ts = list(range(1, maxdeg + 2))
    vals = []
    for t in ts:
        big = Points(Nbig, npts, k, degs, seed=seed + 1, group=group)
        embed_points(big, small, kind, (t, 0, 0) if kind == 'U1' else [(t, 0, 0)])
        vals.append(big.expr_vector(big_expr))
    nonzero = 0
    for a in powers:
        num = interpolate_coefficient(vals, ts, a)
        formal = restrict_u1(big_expr, Nsmall, (a, 0, 0)) if kind == 'U1' else restrict_so2(big_expr, [(a, 0, 0)])
        fv = small.expr_vector(formal) if formal else np.zeros_like(num)
        assert np.array_equal(num % P, fv % P), (kind, a)
        nonzero += int(np.count_nonzero(num % P))
    assert nonzero > 0, 'vacuous comparison'


def _same(a, b):
    return {m: c % P for m, c in a.items() if c % P} == {m: c % P for m, c in b.items() if c % P}


def test_u1_restriction_matches_block_diagonal_evaluation():
    # SU(3) > SU(2) x U(1): two fermions and five scalars; images of x^0..x^3 have 60/82/63/24 terms
    ex = _random_expr((2, 5, 2, 2), 60, seed=7)
    _numeric_vs_formal(ex, 'SU', 3, 2, 'U1', powers=(0, 1, 2, 3))


def test_so2_restriction_matches_block_diagonal_evaluation():
    # SO(7) > SO(5) x SO(2): two fermions (possibly in two traces) and six scalars; words with a movable trace;
    # images of x^0, x^2, x^4 have 60/40/11 terms
    ex = _random_expr((2, 6, 2, 2), 60, seed=9, keep=_movable)
    _numeric_vs_formal(ex, 'SO_odd', 3, 2, 'SO2', powers=(0, 2, 4))


def test_restrictions_are_chain_maps():
    ex = _random_expr((2, 3, 2, 2), 60, seed=3)
    for t in [(0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0)]:
        assert _same(restrict_u1(Q_expr(ex), 2, t), Q_expr(restrict_u1(ex, 2, t)))
    ex = _random_expr((2, 3, 3, 2), 60, seed=4)
    for targets in [[(2, 0, 0)], [(1, 1, 0)], [(2, 0, 0), (0, 2, 0)]]:
        assert _same(restrict_so2(Q_expr(ex), targets), Q_expr(restrict_so2(ex, targets)))


def test_so2_two_single_steps_compose():
    ex = _random_expr((1, 5, 3, 3), 200, seed=5)
    for t1, t2 in product([(2, 0, 0), (1, 1, 0)], repeat=2):
        assert _same(restrict_so2(ex, [t1, t2]), restrict_so2(restrict_so2(ex, [t1]), [t2]))
