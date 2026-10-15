"""Drivers and checks of R-001 and R-002 run from code/ and data/ alone (review C14)."""
import os

import pytest

from fortuity.bmn import run_su2_direct, r002_check

DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
SU2 = os.path.join(DATA, 'su2', 'bmn_direct_cohomology_j28.json')
SU3 = os.path.join(DATA, 'su3', 'graviton_class_vanishing_on_diagonal_j24.json')


def test_r001_driver_reproduces_stored_data():
    res = run_su2_direct.run(18)
    assert len(res) > 150
    assert run_su2_direct.compare(res, SU2) == []
    s = run_su2_direct.summary(res)
    assert all(v[2] == 0 for v in s.values())          # no non-graviton class below j = 24
    assert s[4][0] == 6 and s[8][0] == 18


def test_r001_compare_detects_a_planted_difference():
    res = run_su2_direct.run(8)
    res[-1] = dict(res[-1], coh=res[-1]['coh'] + 1)
    assert run_su2_direct.compare(res, SU2) != []


@pytest.mark.skipif(not os.environ.get('FORTUITY_SLOW'), reason='~25 s; set FORTUITY_SLOW=1 to run')
def test_r001_threshold_class():
    res = {tuple(r['sector']): r for r in run_su2_direct.run(24)}
    assert res[(5, 4, 4, 4)]['nongrav'] == 1 and res[(5, 4, 4, 4)]['grav'] == 0
    assert sum(r['nongrav'] for r in res.values()) == 1


def test_r002_operator_basic_properties():
    op, sec = r002_check.load_operator(SU3)
    assert tuple(sec) == (3, 4, 4, 4) and len(op) == 740
    closed, zero_diag, nonzero, _ = r002_check.check_basic(op)
    assert closed and zero_diag and nonzero


@pytest.mark.skipif(not os.environ.get('FORTUITY_SLOW'), reason='~5 min; set FORTUITY_SLOW=1 to run')
def test_r002_operator_not_exact():
    op, sec = r002_check.load_operator(SU3)
    r, independent = r002_check.check_not_exact(op, sec)
    assert r == 924 and independent
