"""R-009 from data/ and code/ alone: the stored preimage lambda_0 of eq. (4.8) of arXiv:2312.16443 satisfies
12 Q(lambda_0) = 12 (4.8) on random 3x3 traceless matrices, and the identity fails at N = 4 (it rests on SU(3) trace
relations: (4.8) is Q-exact at N = 3 only)."""
import json
import os
from collections import defaultdict

import numpy as np

from fortuity.bmn.evaluate import Points
from fortuity.bmn.words import Q_expr, canon_multi, canon_word

DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'su3')


def canon_expr(expr):
    out = defaultdict(int)
    for m, c in expr.items():
        sign, ws = 1, []
        for w in m:
            s, cw = canon_word(w)
            if s == 0:
                sign = 0
                break
            sign *= s
            ws.append(cw)
        if not sign:
            continue
        s, cm = canon_multi(tuple(ws))
        if s:
            out[cm] += sign * s * c
    return {m: c for m, c in out.items() if c}


def _load(name, key):
    d = json.load(open(os.path.join(DATA, name)))
    return {tuple(tuple(w) for w in m): c for c, m in d[key]}, d


def test_lambda0_is_a_preimage_of_eq_4p8_at_N3_only():
    lam, meta = _load('cckll_eq4p8_preimage_j24.json', 'lambda0_times_D_canonical')
    O, _ = _load('cckll_eq4p8_operator_j24.json', 'terms')
    D = meta['D']
    diff = defaultdict(int)
    for m, v in canon_expr(Q_expr(lam)).items():
        diff[m] += 12 * v                     # 12 D Q(lambda_0)
    for m, v in canon_expr(O).items():
        diff[m] -= D * v                      # - D * 12 (4.8)
    diff = {m: v for m, v in diff.items() if v}
    assert diff, 'the identity should not hold formally (it needs SU(3) trace relations)'
    for seed in (1, 2):
        assert not np.any(Points(3, 40, 7, [3], seed=seed).expr_vector(diff))
    assert np.any(Points(4, 40, 7, [3], seed=4).expr_vector(diff))
