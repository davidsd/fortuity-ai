"""Validation of blockrestrict.py: the formal restriction maps agree with direct numerical evaluation of the big-group
words on block-diagonal points, where the coefficient of the abelian monomial is extracted by interpolation in a
scale t.  Fermions are carried by the repo's Grassmann frames, so this also checks all signs.

Run from the repo root:  PYTHONPATH=code:scripts/bmn/block-restriction/code python3 \
    scripts/bmn/block-restriction/code/test_blockrestrict.py
"""
import random, time, sys, os
import numpy as np
from fortuity.bmn.evaluate import Points
from fortuity.bmn.words import words_of_sector, nferm, Q_expr, contents_of_sector
from fortuity.bmn.modp import P
from fortuity.bmn.blockrestrict import (restrict_u1, restrict_so2, embed_points, interpolate_coefficient, load_expr_json)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
rng = random.Random(7)


def degrees_of(expr):
    return sorted({sum(nferm(w) for w in m) for m in expr})


def check(name, big_expr, big_group, Nbig, small_group, Nsmall, kind, target_list, npts=4, k=None, seed=11):
    """Compare formal restriction with numerics.  target_list: [triple] for U1, [triples per block] for SO2.
    Only the first variable of each block is switched on numerically, so targets must be (a,0,0) per block;
    for two SO(2) blocks both blocks are scaled by the same t (the monomial x_1^a y_1^b then sits at t^(a+b), and
    the formal side is summed over all splits with that total)."""
    degs = degrees_of(big_expr)
    if k is None:
        k = max(max(degs), 1)
    small = Points(Nsmall, npts, k, degs, seed=seed, group=small_group)
    maxphi1 = max(sum(w.count(0) for w in m) for m in big_expr)
    ts = list(range(1, maxphi1 + 2))
    vals = []
    for t in ts:
        big = Points(Nbig, npts, k, degs, seed=seed + 1, group=big_group)
        if kind == 'U1':
            xv = (t, 0, 0)
        else:
            xv = [(t, 0, 0) for _ in target_list]
        embed_points(big, small, kind, xv)
        vals.append(big.expr_vector(big_expr))
    if kind == 'U1':
        a = target_list[0][0]
        num = interpolate_coefficient(vals, ts, a)
        formal = restrict_u1(big_expr, Nsmall if small_group == 'SU' else None, target_list[0])
    else:
        tot = sum(t[0] for t in target_list)
        num = interpolate_coefficient(vals, ts, tot)
        # all splits of the total degree over the blocks (same scale t on every block)
        formal = {}
        from itertools import product as iprod
        b = len(target_list)
        for split in iprod(range(0, tot + 1, 2), repeat=b):
            if sum(split) != tot:
                continue
            part = restrict_so2(big_expr, [(s, 0, 0) for s in split])
            for m, c in part.items():
                formal[m] = (formal.get(m, 0) + c) % P
    fv = small.expr_vector(formal) if formal else np.zeros_like(num)
    ok = np.array_equal(num % P, fv % P)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {len(big_expr)} big terms -> {len(formal)} small terms; "
          f"nonzero numeric entries {int(np.count_nonzero(num))}", flush=True)
    return ok


def random_expr(words, n):
    ws = rng.sample(words, min(n, len(words)))
    return {w: rng.randrange(1, P) for w in ws}


if __name__ == "__main__":
    t0 = time.time()
    allok = True
    # 1. SU(3) > SU(2) x U(1), terms of the T-021 representative (sector (5;7,4,4))
    rep = load_expr_json(os.path.join(REPO, 'scripts/bmn/su3-level30-representative/runs/rep_nmax5_seed11_expr.json.gz'))
    items = list(rep.items())
    sub = dict(rng.sample(items, 120))
    for a in (0, 1, 2, 3, 4):
        allok &= check(f'SU(3)>SU(2)xU(1) coefficient of x1^{a}', sub, 'SU', 3, 'SU', 2, 'U1', [(a, 0, 0)])
    # 2. SO(9) > SO(7) x SO(2), random words of (1;5,3,3) and of a sector with more fermions (3;6,4,4)
    for sec in [(1, 5, 3, 3), (3, 5, 4, 4)]:
        ws = words_of_sector(sec, 100)
        print(f"  SO(9) words in {sec}: {len(ws)}", flush=True)
        ex = random_expr(ws, 120)
        for a in (0, 2, 4):
            allok &= check(f'SO(9)>SO(7)xSO(2) {sec} coefficient of x1^{a}', ex, 'SO_odd', 4, 'SO_odd', 3, 'SO2', [(a, 0, 0)])
    # 3. SO(11) > SO(7) x SO(2)^2 (both blocks scaled together), and composition with two single-block steps
    ws = words_of_sector((1, 7, 3, 3), 100)
    print(f"  SO(11) words in (1;7,3,3): {len(ws)}", flush=True)
    ex = random_expr(ws, 80)
    allok &= check('SO(11)>SO(7)xSO(2)^2 total degree 4', ex, 'SO_odd', 5, 'SO_odd', 3, 'SO2', [(2, 0, 0), (2, 0, 0)])
    two = restrict_so2(ex, [(2, 0, 0), (2, 0, 0)])
    step = restrict_so2(restrict_so2(ex, [(2, 0, 0)]), [(2, 0, 0)])
    comp_ok = {m: c % P for m, c in two.items() if c % P} == {m: c % P for m, c in step.items() if c % P}
    print(f"{'PASS' if comp_ok else 'FAIL'}  composition SO(11)>SO(9)xSO(2)>SO(7)xSO(2)^2 equals the two-block map "
          f"({len(two)} terms)", flush=True)
    allok &= comp_ok
    # 4. chain map: formal Q commutes with the restriction (checked on a random expression, formally)
    ex = random_expr(words_of_sector((2, 5, 3, 3), 100), 200)
    lhs = restrict_so2(Q_expr(ex), [(2, 0, 0)])
    rhs = Q_expr(restrict_so2(ex, [(2, 0, 0)]))
    cm_ok = {m: c % P for m, c in lhs.items() if c % P} == {m: c % P for m, c in rhs.items() if c % P}
    print(f"{'PASS' if cm_ok else 'FAIL'}  SO(2) restriction commutes with Q formally ({len(lhs)} terms)", flush=True)
    allok &= cm_ok
    lhs = restrict_u1(Q_expr(sub), 2, (3, 0, 0))
    rhs = Q_expr(restrict_u1(sub, 2, (3, 0, 0)))
    cm_ok = {m: c % P for m, c in lhs.items() if c % P} == {m: c % P for m, c in rhs.items() if c % P}
    print(f"{'PASS' if cm_ok else 'FAIL'}  U(1) restriction commutes with Q formally ({len(lhs)} terms)", flush=True)
    allok &= cm_ok
    print(f"ALL {'PASS' if allok else 'FAIL'}  ({time.time() - t0:.0f}s)")
    sys.exit(0 if allok else 1)
