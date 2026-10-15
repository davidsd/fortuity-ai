"""Checks of R-002 from code/ and data/ alone (promoted from scripts/bmn/index-and-direct-cohomology/
verify_explicit_class.py; review C14): the explicit SU(3), j = 24 multi-graviton polynomial stored in
data/su3/graviton_class_vanishing_on_diagonal_j24.json is Q-closed, vanishes identically on diagonal matrices, is
nonzero on generic 3x3 matrices, and its Weyl symmetrisation is not Q-exact (independent of the image of Q from
the sector above, in the Weyl-symmetric subspace).

usage: python -m fortuity.bmn.r002_check [DATA.json] [--exact]
  --exact also runs the non-exactness test (rank of Q on the Weyl-symmetric subspace of (4;4,4,4); a few minutes).
"""
import copy
import json
import os
import sys
from collections import defaultdict

import numpy as np

from fortuity.bmn.words import graviton_generators, mul_expr, Q_expr, expr_sector
from fortuity.bmn.symdirect import SymDirect, symmetrize_expr
from fortuity.bmn.evaluate import Points

DATA = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'su3',
                    'graviton_class_vanishing_on_diagonal_j24.json')
_L = 'XYZ'


def _label(name):
    """Readable label used in the data file for a generator name of graviton_generators."""
    if name[0] == 'u':
        return 'u^{%s}' % ''.join(_L[i] for i in name[2])
    if name[0] == 'v':
        return 'v^{%s}_{%s}' % (''.join(_L[i] for i in name[2]), _L[name[3]])
    return 'w^{%s}' % ''.join(_L[i] for i in name[2])


def load_operator(datafile=DATA):
    """The stored polynomial as a multitrace expression {multitrace: coeff}, and its sector."""
    d = json.load(open(datafile))
    gens = {}
    for name, e in graviton_generators(3, 1):
        lab = _label(name)
        assert lab not in gens, lab
        gens[lab] = e
    op = defaultdict(int)
    for term in d['operator']:
        e = {(): 1}
        for g in term['monomial']:
            e = mul_expr(e, gens[g])
        for m, c in e.items():
            op[m] += term['coeff'] * c
    op = {m: c for m, c in op.items() if c}
    sec = expr_sector(op)
    assert list(sec) == d['sector'], (sec, d['sector'])
    return op, sec


def check_basic(op, seed=31):
    """(closed, zero on diagonal, nonzero generic, Weyl-invariant up to the factor 6)."""
    q = Q_expr(op)
    closed = (not q) or not np.any(Points(3, 6, 7, [1, 3], seed=seed).expr_vector(q))
    pd = Points(3, 8, 7, [1, 3], seed=seed + 1, diagonal=True)
    pg = Points(3, 8, 7, [1, 3], seed=seed + 2)
    zero_diag = not np.any(pd.expr_vector(op))
    nonzero = bool(np.any(pg.expr_vector(op)))
    sym = symmetrize_expr(op)
    diff = defaultdict(int)
    for m, c in sym.items():
        diff[m] += c
    for m, c in op.items():
        diff[m] -= 6 * c
    weyl = not {m: c for m, c in diff.items() if c}
    return closed, zero_diag, nonzero, weyl


def check_not_exact(op, sec, seed=2024):
    """(rank of Q from the Weyl-symmetric sector above, symmetrised op independent of that image)."""
    D = SymDirect(N=3, seed=seed, verbose=False)
    D.guard.log = False
    up = (sec[0] + 1,) + tuple(sec[1:])
    r = D.rankQ(up)
    qb, qpts = D._Qrows[up]
    tb = copy.deepcopy(qb)
    v = qpts.expr_vector(symmetrize_expr(op))
    return r, bool(tb.add(v[None, :])[0])


def main(argv):
    args = [a for a in argv[1:] if not a.startswith('--')]
    op, sec = load_operator(args[0] if args else DATA)
    print(f"operator: {len(op)} multitrace terms, sector {tuple(sec)}")
    closed, zero_diag, nonzero, weyl = check_basic(op)
    print(f"Q-closed: {closed}; zero on diagonal matrices: {zero_diag}; nonzero on generic matrices: {nonzero}; "
          f"Weyl-invariant: {weyl}")
    ok = closed and zero_diag and nonzero
    if '--exact' in argv:
        r, indep = check_not_exact(op, sec)
        print(f"rank of Q on the Weyl-symmetric sector above: {r}; operator independent of the exact space: {indep}")
        ok = ok and indep
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
