"""Tests for fortuity.bmn.dmk: the dMK brane-fused operator (O^3_30)^{ijk} (arXiv:2412.08695, eq. (2.24)).

Each of the three readings in the module docstring is checked, and with them the operator is:
- Q-closed at N = 3, not at N = 4, and a [3,0] highest-weight vector;
- of SU(2) x U(1) restriction x1^3 (x) (-648/5) O^2_24 exactly (dMK eq. (2.25)), not merely up to exact terms, where
  O^2_24 is a non-graviton (non-exact) class of the SU(2) sector (5;4,4,4), i.e. the SU(2) threshold class.
Closedness and the restriction are checked by random evaluation over F_p.
"""
import os
from collections import defaultdict
from fractions import Fraction as Fr
from itertools import product

import numpy as np
import pytest

from fortuity.bmn import dmk
from fortuity.bmn.blockrestrict import restrict_u1
from fortuity.bmn.evaluate import Points
from fortuity.bmn.modp import P
from fortuity.bmn.words import Q_expr, expr_sector, contents_of_sector
from fortuity.hwcoh.hw import E_multi

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA = os.path.join(REPO, 'data', 'su3', 'dmk_o330_hw.json')


@pytest.fixture(scope='module')
def O():
    return dmk.o330()


def _vec(expr, N, sec, seed, npts=12):
    """evaluation vector of an exact expression at random SU(N) points of the sector's fermion degrees"""
    degs = sorted({sum(c[3:6]) for c in contents_of_sector(sec)})
    pts = Points(N, npts, max(degs) + 2, degs, seed=seed)
    return pts.expr_vector(dmk.to_modp(expr)) % P


def test_relation_2_20_needs_traceless_R():
    assert dmk.relation_2_20() == {}                                   # holds identically with traceless R
    assert len(dmk.relation_2_20(project=False)) > 0                   # as printed it does not


def test_Q_r_equals_R_on_traceless_parts():
    # reading 1: Q r = R at N = 3 for the traceless parts, constant exactly 1 (Q psi = eps phi phi); sample components
    pts = {d: Points(3, 4, d + 2, [d], seed=7) for d in (1, 2)}
    cases = [('r12_03', 3, 1), ('r12_11', 2, 1), ('r12_22', 4, 1), ('r14_13', 4, 2), ('r16_01', 1, 2), ('r16_23', 5, 2)]
    for name, nidx, deg in cases:
        r, R = dmk.tensor(name), dmk.tensor(dmk.R_OF_r[name])
        comps = list(product(range(3), repeat=nidx))[::max(1, 3 ** nidx // 12)]
        for I in comps:
            a = pts[deg].expr_vector(dmk.to_modp(Q_expr(r(*I)))) % P
            b = pts[deg].expr_vector(dmk.to_modp(R(*I))) % P
            assert np.array_equal(a, b), (name, I)
    # as printed, Q r != R on a trace component (r^i_i)
    raw_r, raw_R = dmk.tensor('r12_11', project=False), dmk.tensor('R12_11', project=False)
    a = pts[1].expr_vector(dmk.to_modp(Q_expr(raw_r(0, 0)))) % P
    b = pts[1].expr_vector(dmk.to_modp(raw_R(0, 0))) % P
    assert not np.array_equal(a, b)


def test_o330_closed_at_N3_only_with_both_readings(O):
    sec = expr_sector(O)
    assert sec == (5, 7, 4, 4)
    low = (4,) + sec[1:]
    assert _vec(O, 3, sec, seed=3).any()                                # nonzero at N = 3
    assert not _vec(Q_expr(O), 3, low, seed=5).any()                    # Q-closed at N = 3
    assert _vec(Q_expr(O), 4, low, seed=6, npts=4).any()                # but not at N = 4 (closed only via N=3 relations)
    assert _vec(Q_expr(dmk.o330(graded_sign=False)), 3, low, seed=5).any()   # reading 2 is needed
    assert _vec(Q_expr(dmk.o330(project=False)), 3, low, seed=5).any()      # reading 1 is needed


def test_o330_is_highest_weight(O):
    for i in (1, 2):
        img = defaultdict(Fr)
        for m, v in O.items():
            for m2, c2 in E_multi(m, i).items():
                img[m2] += v * c2
        assert not any(img.values())


def test_x1_cubed_restriction_is_minus_648_over_5_O224(O):
    img = restrict_u1(dmk.to_modp(O), 2, (3, 0, 0))
    O224 = dmk.o2_24()
    sec = (5, 4, 4, 4)
    assert expr_sector(O224) == sec
    a = _vec(img, 2, sec, seed=17)
    b = _vec(O224, 2, sec, seed=17)
    assert b.any()
    c = Fr(-648, 5)
    assert np.array_equal(a, (b * (c.numerator * pow(c.denominator, P - 2, P))) % P)
    # O^2_24 is the SU(2) threshold class: closed, not exact, non-graviton in the certified SU(2) sector (dim 153)
    from fortuity.bmn.classify import TargetSector, verdict
    T = TargetSector(2, 'SU', sec, gen_nmax=3, seed=17, log=lambda *a: None, maxlen=3, exact_dim=153)
    assert verdict(T.classify(dmk.to_modp(O224))) == 'non-graviton'
    # the x1^0 and x1^2 coefficients vanish at N = 2 (as found for the R-007 representative up to exact terms)
    for k in (0, 2):
        assert not _vec(restrict_u1(dmk.to_modp(O), 2, (k, 0, 0)), 2, (5, 7 - k, 4, 4), seed=19).any()


def test_data_file_matches_builder(O):
    if not os.path.exists(DATA):
        pytest.skip('data file not present')
    assert dmk.load_rational_json(DATA) == O
