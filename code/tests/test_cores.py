"""Tests for fortuity.hwcoh.cores: fused capacity of SU(3) > SU(2) x U(1), P-odd counts from the C-twisted index, and
the core bound of T-032 on the stored (8,5,4) column."""
import json
import os

import pytest

from fortuity.hwcoh.cores import gl3_weights, capacity, core_test, nongraviton_content

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# SU(2) non-graviton classes through level 14 (exact, R-001): O_0 at (2J; q) = (5; 4,4,4) and the two classes above it
SU2_LOW = [(12, 5, (4, 4, 4)), (13, 6, (5, 4, 4)), (14, 7, (5, 5, 4))]


def test_gl3_weights():
    assert {mu: sum(gl3_weights(mu).values()) for mu in [(2, 1, 0), (3, 1, 0), (2, 2, 0), (3, 0, 0), (4, 4, 4)]} == \
        {(2, 1, 0): 8, (3, 1, 0): 15, (2, 2, 0): 6, (3, 0, 0): 10, (4, 4, 4): 1}
    assert gl3_weights((2, 1, 0))[(1, 1, 1)] == 2


def test_capacity_channels():
    # R-016: R-007 at (5; 7,4,4) restricts to x^3 (x) O_0; the j = 32 columns of T-042 and the j = 34 column of T-032
    cases = {(7, 4, 4): {5: 1, 6: 1}, (8, 4, 4): {5: 1, 6: 1}, (7, 5, 4): {6: 1, 7: 1}, (8, 5, 4): {6: 1, 7: 1}}
    for lam, want in cases.items():
        cap, src = capacity(lam, SU2_LOW)
        assert dict(cap) == want, lam
    cap, src = capacity((7, 4, 4), SU2_LOW)
    assert (12, 5, (4, 4, 4), 3, 1) in src                      # the x^3 (x) O_0 channel
    assert dict(capacity((6, 5, 4), [(12, 5, (4, 4, 4))])[0]) == {}   # [2,0] x [0,0] has no [1,1]


def test_nongraviton_content_skips_incomplete_columns():
    recs = {(4, 4, 4): {n: dict(m=m, r_up=r, grav=0, check=True) for n, (m, r) in
                        enumerate([(1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (6, 3)])}}
    # (4,4,4) needs every 2J from 0 to 12 to be complete; with 0..5 only it is skipped
    assert nongraviton_content(recs) == []


def _r007_records():
    d = json.load(open(os.path.join(REPO, 'data', 'su3', 'bmn_hw_cohomology_j30.json')))
    cols = {}
    for c in d['columns']:
        if c['lam'] in ([7, 4, 4], [6, 5, 4]):
            recs, prev = {}, 0
            for s in c['sectors']:
                r = s['m'] - s['h'] - prev                      # h = m - r_up(2J) - r_up(2J-1)
                recs[s['twoJ']] = dict(m=s['m'], r_up=r, grav=s['grav'], check=s['check'])
                prev = r
            cols[tuple(c['lam'])] = recs
    return cols


def test_p_odd_on_the_j30_columns():
    cols = _r007_records()
    t = core_test(3, (7, 4, 4), cols[(7, 4, 4)], SU2_LOW)
    assert (t['H_total'], t['twisted'], t['p_odd'], t['BH'], t['euler_ok']) == (12, 10, 1, 1, True)   # R-007: P-odd
    assert t['cores_lower_bound'] == 0                         # within capacity (it is fused, R-016)
    t = core_test(3, (6, 5, 4), cols[(6, 5, 4)], SU2_LOW)
    assert (t['p_odd'], t['BH']) == (0, 0)


def test_t032_column_8_5_4():
    path = os.path.join(REPO, 'data', 'su3', 'hw_cohomology_8_5_4.json')
    if not os.path.exists(path):
        pytest.skip('data file not present')
    d = json.load(open(path))
    recs = {s['n']: s for s in d['sectors']}
    content = [(l, t, tuple(mu)) for l, t, mu in d['su2_nongraviton_content']]
    t = core_test(3, (8, 5, 4), recs, content)
    assert [t['sectors'][n]['BH'] for n in range(10)] == [0, 0, 0, 0, 0, 0, 2, 1, 0, 0]
    assert (t['H_total'], t['euler'], t['index'], t['twisted'], t['p_odd']) == (37, 3, 3, 31, 3)
    assert dict(t['capacity']) == {6: 1, 7: 1}
    assert t['cores_lower_bound'] == 1 and t['cores_lower_bound_column'] == 1
    assert sum(t['sectors'][n]['grav'] for n in range(10)) == 34        # = R-017's Coulomb count D for this column
    for k in ('p_odd', 'BH', 'cores_lower_bound', 'cores_lower_bound_column', 'H_total'):
        assert d[k] == t[k]
