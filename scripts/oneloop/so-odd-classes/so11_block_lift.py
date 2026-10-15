"""T-019 (SO/Sp part), step 3: one-way two-block test of the one-loop lifting of the two SO(11) classes of R-011.
SO(11) > SO(7) x SO(2) x SO(2), phi_m = diag(A_m, x_m eps, y_m eps) (R-016). A trace is block diagonal, so it contributes
either its SO(7) part or its x-block part or its y-block part; tr(eps^2) = -2. The coefficient maps
  [4,0] class: rho(m) = coefficient of x_1^2 y_1^2 = 4 * sum over ordered pairs of distinct factors tr(phi_1 phi_1) (one
               to each block) of (m without them);
  [0,2] class: rho(m) = coefficient of x_1^2 y_2^2 = 4 * sum over pairs (tr(phi_1 phi_1) -> x block, tr(phi_2 phi_2) -> y block);
are Q_0 chain maps into SO(7), landing in (0,0;6,6,6), Y=8, d_BMN=1. As in so9_block_lift.py: rho(Q_1 O) outside
span(SO(7) exact, rho(Q_1 g) for all SO(11) graviton polynomials g of O's sector) proves that O is lifted at one loop.
Run: cd scripts/oneloop/so-odd-classes && PYTHONPATH=../../../code python so11_block_lift.py"""
import random, time
from collections import defaultdict
from fortuity.bmn.modp import P
from fortuity.full.words import Q_expr, canon_multi, sector_of
from fortuity.full.q1 import Q1_expr
from so7_lift import values, O18
from gravitons_bmn import graviton_exprs
from sector_words import words
from lift_test import lift_test_span
from so9_block_lift import load_rep

TR11 = ((0, 0, (0,)), (0, 0, (0,)))
TR22 = ((0, 0, (1,)), (0, 0, (1,)))


def rho_pair(expr, wx, wy):
    out = defaultdict(int)
    for m, c in expr.items():
        for a, w in enumerate(m):
            if w != wx:
                continue
            for b, w2 in enumerate(m):
                if b == a or w2 != wy:
                    continue
                rest = tuple(x for k, x in enumerate(m) if k not in (a, b))
                s, cm = canon_multi(list(rest))
                if s:
                    out[cm] = (out[cm] + 4 * s * c) % P
    return {m: c for m, c in out.items() if c}


def run(path, sec_bmn, wx, wy, label, src_bmn_ctrl):
    t0 = time.time()
    O, info = load_rep(path)
    print(f"\n== {label}: {len(O)} terms, sector {set(sector_of(m) for m in O)}, info {info}", flush=True)
    q0 = Q_expr(O)
    print("Q_0 O = 0 on so(11):", (not q0) or not values(q0, 5, 'SO_odd').any(),
          "| on so(13):", (not q0) or not values(q0, 6, 'SO_odd').any(), flush=True)
    Q1O = Q1_expr(O, 11, 'SO_odd')
    r = rho_pair(Q1O, wx, wy)
    print(f"Q_1 O: {len(Q1O)} terms; rho(Q_1 O): {len(r)} terms, sectors {set(sector_of(m) for m in r)}, "
          f"nonzero on so(7): {bool(r) and values(r, 3, 'SO_odd').any()}  ({time.time() - t0:.0f} s)", flush=True)
    span_extra, ng = [], 0
    for _, g in graviton_exprs(sec_bmn, 10):
        ng += 1
        e = rho_pair(Q1_expr(g, 11, 'SO_odd'), wx, wy)
        if e:
            span_extra.append(e)
    print(f"{ng} SO(11) graviton polynomials; nonzero rho(Q_1 g): {len(span_extra)}  ({time.time() - t0:.0f} s)", flush=True)
    U = words(src_bmn_ctrl, 11, 0)
    rng = random.Random(7)
    ctrls = [rho_pair(Q1_expr(Q_expr({m: rng.randint(1, 50) for m in rng.sample(U, 8)}), 11, 'SO_odd'), wx, wy)
             for _ in range(3)]
    ctrls = [c for c in ctrls if c]
    print(f"controls with nonzero rho: {len(ctrls)}", flush=True)
    res = lift_test_span([r] + ctrls, span_extra, (0, 0, 6, 6, 6), 8, 2, 3)
    print(f"{label} two-block test:", res, flush=True)
    Q1O18 = Q1_expr(O18(), 7, 'SO_odd')
    res2 = lift_test_span([r], [Q1O18], (0, 0, 6, 6, 6), 8, 2, 3, seeds=(13,))
    print(f"{label}: rho(Q_1 O) in span(exact, Q_1 O_18): {res2['increases'][0] == 0}  ({time.time() - t0:.0f} s)", flush=True)
    return res, res2


if __name__ == "__main__":
    run('../../../data/so_sp/so11_nongraviton_rep_1733.json', (1, 7, 3, 3), TR11, TR11, "SO(11) [4,0] (1;7,3,3)",
        (2, 2, 14, 6, 6))
    run('../../../data/so_sp/so11_nongraviton_rep_1553.json', (1, 5, 5, 3), TR11, TR22, "SO(11) [0,2] (1;5,5,3)",
        (2, 2, 10, 10, 6))
