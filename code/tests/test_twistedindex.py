"""Tests for fortuity.bmn.cparity (charge conjugation on words, graviton parity) and fortuity.bmn.twistedindex
(C-twisted BMN index by twisted Weyl integration; kernel parity index).  ~20 s."""
import json, os
from fortuity.bmn.words import graviton_generators
from fortuity.bmn.modp import P
from fortuity.bmn.cparity import conj_expr, canon_expr, letters
from fortuity.bmn.twistedindex import SUN, integrate, to_int, hw_multiplicities, kernel_parity
from fortuity.bmn.gravindex import GravitonCounter

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def test_graviton_parity_on_generators():
    # every single-trace graviton component has C = (-1)^(number of letters)
    n = 0
    for name, e in graviton_generators(5, 1):
        e = canon_expr(e)
        if not e:
            continue
        ce, s = conj_expr(e), (-1) ** letters(e)
        assert all((ce.get(m, 0) - s * e.get(m, 0)) % P == 0 for m in set(ce) | set(e)), name
        n += 1
    assert n > 100


def test_su2_twisted_equals_ordinary():
    # C is inner for su(2), so the twisted index is the ordinary one
    G = SUN(2)
    a, _ = to_int(integrate(G, 10, False)[0])
    b, _ = to_int(integrate(G, 10, True)[0])
    assert a == b and a


def test_su3_indices_against_data():
    Q = 8
    G = SUN(3)
    pl, err = to_int(integrate(G, Q, False)[0])
    assert err < 1e-8
    ref = json.load(open(os.path.join(REPO, 'data/su3/bmn_index_j36.json')))['full_index_unrefined']
    assert [sum(v for q, v in pl.items() if sum(q) == d) for d in range(Q + 1)] == ref[0:2 * Q + 1:2]
    # twisted: (-1)^|lam| I_C(lam) = total cohomology of the column (all classes P-even for j <= 22)
    tw, _ = to_int(integrate(G, Q, True)[0])
    H = hw_multiplicities(tw, Q)
    cols = json.load(open(os.path.join(REPO, 'data/su3/bmn_hw_cohomology_j30.json')))['columns']
    n = 0
    for c in cols:
        lam = tuple(c['lam'])
        if sum(lam) <= Q:
            assert (-1) ** sum(lam) * H.get(lam, 0) == sum(s['h'] for s in c['sectors']), lam
            n += 1
    assert n > 30


def test_su3_kernel_parity_first_noncoulomb_graviton():
    # up to j = 24 the only Coulomb-invisible class of SU(3) is the non-Coulomb graviton R_24 at (4,4,4) (R-015):
    # one P-even class with odd Y, so T = +1, O = -1, and zero elsewhere
    dims = GravitonCounter(3, 24, verbose=False, npts=4096).run()
    kp = kernel_parity(3, 12, dims)
    nonzero = {lam: (r['T'], r['O']) for lam, r in kp.items() if r['T'] or r['O']}
    assert nonzero == {(4, 4, 4): (1, -1)}
