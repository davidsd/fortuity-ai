"""T-019 (SO/Sp part), step 2: one-way block test of the one-loop lifting of the SO(9) class of R-006.
rho_2 = coefficient of x_1^2 under SO(9) > SO(7) x SO(2), phi_1 = diag(A_1, x_1 eps) (R-016; x on phi_1 only, x_2 = x_3 = 0).
On a multitrace: the SO(2) block of a trace is nonzero only if all its letters are phi_1, and tr(eps^2) = -2, so
rho_2(m) = -2 * sum over the factors tr(phi_1 phi_1) of m of (m with that factor removed), read as an SO(7) word.
rho_2 is a Q_0 chain map (x is central), so rho_2(SO(9)-exact) is SO(7)-exact. Hence if
    rho_2(Q_1 O_9)  is not in  span( SO(7) exact , rho_2(Q_1 g) for all SO(9) graviton polynomials g of (1;5,3,3) ),
then Q_1 O_9 is not in (SO(9) exact + Q_1 gravitons): the SO(9) class is lifted at one loop.
Run: cd scripts/oneloop/so-odd-classes && PYTHONPATH=../../../code python so9_block_lift.py"""
import json, random
import numpy as np
from collections import defaultdict
from fortuity.bmn.modp import P
from fortuity.full.words import add_term, Q_expr, canon_multi, sector_of
from fortuity.full.q1 import Q1_expr
from so7_lift import CONV, values
from gravitons_bmn import graviton_exprs, to_full
from sector_words import words
from lift_test import lift_test_span

PHI1 = (0, 0, (0,))
TR11 = (PHI1, PHI1)


def load_rep(path):
    d = json.load(open(path))
    e = defaultdict(int)
    for c, traces in d['terms']:
        e[tuple(tuple(t) for t in traces)] = (e[tuple(tuple(t) for t in traces)] + c) % P
    return to_full({m: c for m, c in e.items() if c}), d['info']


def rho2(expr):
    out = defaultdict(int)
    for m, c in expr.items():
        for a, w in enumerate(m):
            if w == TR11:
                rest = m[:a] + m[a + 1:]
                s, cm = canon_multi(list(rest))
                if s:
                    out[cm] = (out[cm] - 2 * s * c) % P
    return {m: c for m, c in out.items() if c}


if __name__ == "__main__":
    O9, info = load_rep('../../../data/so_sp/so9_nongraviton_rep_1533.json')
    print(f"O_9: {len(O9)} terms, sectors {set(sector_of(m) for m in O9)}; info {info}")
    q0 = Q_expr(O9)
    print("Q_0 O_9 = 0 on so(9):", (not q0) or not values(q0, 4, 'SO_odd').any(),
          "| on so(11):", (not q0) or not values(q0, 5, 'SO_odd').any())
    Q1O9 = Q1_expr(O9, 9, 'SO_odd')
    print(f"Q_1 O_9: {len(Q1O9)} terms; sectors {set(sector_of(m) for m in Q1O9)}")
    r = rho2(Q1O9)
    print(f"rho_2(Q_1 O_9): {len(r)} terms; sectors {set(sector_of(m) for m in r)}; nonzero on so(7):",
          bool(r) and values(r, 3, 'SO_odd').any())
    G9 = [e for _, e in graviton_exprs((1, 5, 3, 3))]
    span_extra = [rho2(Q1_expr(g, 9, 'SO_odd')) for g in G9]
    span_extra = [e for e in span_extra if e]
    print(f"{len(G9)} SO(9) graviton polynomials in (1;5,3,3); nonzero rho_2(Q_1 g): {len(span_extra)}")
    # control: rho_2(Q_1 Q_0 u) for u in SO(9) (2,2;10,6,6), Y=9, D=0 is SO(7)-exact
    U = words((2, 2, 10, 6, 6), 9, 0)
    rng = random.Random(5)
    ctrls = [rho2(Q1_expr(Q_expr({m: rng.randint(1, 50) for m in rng.sample(U, 8)}), 9, 'SO_odd')) for _ in range(2)]
    ctrls = [c for c in ctrls if c]
    print(f"controls with nonzero rho_2: {len(ctrls)}")
    res = lift_test_span([r] + ctrls, span_extra, (0, 0, 6, 6, 6), 8, 2, 3)
    print("SO(9) block test:", res)

    # structure: is rho_2(Q_1 O_9) = c Q_1 O_18 modulo SO(7) exact (the SO(9) class is lifted through its SO(7) core)?
    from so7_lift import O18
    Q1O18 = Q1_expr(O18(), 7, 'SO_odd')
    res2 = lift_test_span([r, Q1O18], [Q1O18], (0, 0, 6, 6, 6), 8, 2, 3, seeds=(13,))
    print("rho_2(Q_1 O_9) in span(exact, Q_1 O_18):", res2['increases'][0] == 0, res2)
