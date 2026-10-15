"""T-028 step 2: the SO(2N+1) fortuitous classes of R-005/R-006/R-011 under block restriction
SO(2N+1) > SO(2N+1-2b) x SO(2)^b.

Hypothesis (block-fusion reading of R-004): the first SO(2N+1) - USp(2N) difference is the SO(7) class dressed by N-3
SO(2) blocks, each carrying one scalar pair x_i x_j.  Test: the coefficient of the SO(2) monomial(s) in the restriction
of the SO(9) / SO(11) class has a non-graviton component in the smaller group.

usage (repo root): PYTHONPATH=code:scripts/bmn/block-restriction/code python3 .../run_so_tower.py STAGE
  STAGE = rep9    : SO(9) representative at (1;5,3,3)                     -> runs/so9_rep_1533.json.gz
          test9   : restrictions of the SO(9) class (+ controls)           -> runs/so9_restrictions.json
          rep11   : SO(11) representatives at (1;7,3,3), (1;5,5,3)         -> runs/so11_rep_*.json.gz
          verify  : Q-closedness of the stored representatives at fresh random points
          test11  : two-block restrictions of the SO(11) classes to SO(7) x SO(2)^2, with ratios of the non-graviton
                    components (tests the brane-fusion dressing f(x,y))    -> runs/so11_restrictions_*.json
"""
import os, sys, time, json, pickle, random
import numpy as np
from itertools import combinations_with_replacement
from fortuity.bmn.words import graviton_generators, mul_expr, expr_sector
from fortuity.bmn.modp import P
from fortuity.hwcoh.gravhw import GravitonAlgebra
from fortuity.bmn.blockrestrict import restrict_so2, load_expr_json, save_expr_json
from fortuity.hwcoh.classreps import hw_nongraviton_reps
from fortuity.bmn.classify import TargetSector

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.abspath(os.path.join(HERE, '..', 'runs'))
t0 = time.time()
TSEED = int(os.environ.get('TARGET_SEED', '17'))          # seed of the target-sector functionals (second seed: 29)
SUF = '' if TSEED == 17 else f'_tseed{TSEED}'


def log(*a):
    print(f'[{time.time() - t0:7.0f}s]', *a, flush=True)


def graviton_controls(N, sec, count=6, seed=3, nmax=None):
    """A few products of single-trace graviton generators in the weight sector sec."""
    gens = [(n, e) for n, e in graviton_generators(nmax or min(2 * N + 2, 10), 1) if e]
    secs = [tuple(expr_sector(e)) for _, e in gens]
    idxs = [i for i in range(len(gens)) if all(secs[i][a] <= sec[a] for a in range(4))]
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    monos = []

    def rec(start, rem, chosen):
        if len(monos) >= 200:
            return
        if not any(rem):
            if len(chosen) >= 2:
                monos.append(tuple(chosen))
            return
        for pos in range(start, len(idxs)):
            s = secs[idxs[pos]]
            if all(s[a] <= rem[a] for a in range(4)):
                rec(pos, tuple(rem[a] - s[a] for a in range(4)), chosen + [idxs[pos]])

    rec(0, tuple(sec), [])
    rnd.shuffle(monos)
    out = []
    for combo in monos:
        e = {(): 1}
        for i in combo:
            e = mul_expr(e, gens[i][1])
        if e:
            out.append(([gens[i][0] for i in combo], e))
        if len(out) >= count:
            break
    return out


def classify_word(c):
    if c['non_graviton_component']:
        return 'NON-GRAVITON'
    if c['not_exact']:
        return 'graviton (mod exact)'
    return 'Q-exact' if c['nonzero'] else 'zero'


def run_tests(name, reps, big_N, sec_big, cases, controls):
    """cases: list of (label, targets list, small N, small sector)."""
    results = []
    cache = {}
    for label, targets, Ns, sec_s in cases:
        key = (Ns, sec_s)
        if key not in cache:
            cache[key] = TargetSector(Ns, 'SO_odd', sec_s, gen_nmax=2 * Ns + 2, seed=TSEED, log=log)
        T = cache[key]
        row = dict(case=label, targets=targets, small_group=f'SO({2 * Ns + 1})', small_sector=sec_s,
                   small_dim=T.dimV, small_exact=T.r_exact, small_grav=T.grav)
        for ri, rep in enumerate(reps):
            img = restrict_so2(rep, targets)
            c = T.classify(img) if img else dict(nonzero=False, closed=True, not_exact=False,
                                                  non_graviton_component=False)
            row[f'rep{ri}'] = dict(terms=len(img), verdict=classify_word(c), **c)
        ctrl = []
        for names, e in controls:
            img = restrict_so2(e, targets)
            ctrl.append(classify_word(T.classify(img)) if img else 'zero')
        row['graviton_controls'] = ctrl
        log(f'{name} {label}: ' + json.dumps({k: v for k, v in row.items() if k.startswith('rep') or k == 'graviton_controls'}))
        results.append(row)
    return results


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'rep9':
        dims = pickle.load(open(os.path.join(RUNS, 'cache', 'dims_SO_odd4_j22_2J2.pkl'), 'rb'))
        GA = GravitonAlgebra(10)
        exprs, info = hw_nongraviton_reps(4, 'SO_odd', (5, 3, 3), 1, dims, GA, seed=1, log=log)
        for i, e in enumerate(exprs):
            save_expr_json(e, os.path.join(RUNS, f'so9_rep_1533_{i}.json.gz'), sector=[1, 5, 3, 3], group='SO(9)', info=info)
        json.dump(info, open(os.path.join(RUNS, 'so9_rep_1533_info.json'), 'w'), indent=1)
    elif stage == 'test9':
        # optional case indices (e.g. "test9 0 2 3"); the default runs all four, x^0 (a 4668+ dim SO(7) sector) is slow
        reps = [load_expr_json(os.path.join(RUNS, 'so9_rep_1533_0.json.gz'))]
        ctrls = graviton_controls(4, (1, 5, 3, 3))
        log(f'SO(9) representative: {len(reps[0])} terms; {len(ctrls)} graviton controls')
        cases = [('x^(2,0,0) -> SO(7) (1;3,3,3)', [(2, 0, 0)], 3, (1, 3, 3, 3)),
                 ('x^(0,0,0) -> SO(7) (1;5,3,3) [plain SO(7) < SO(9)]', [(0, 0, 0)], 3, (1, 5, 3, 3)),
                 ('x^(1,1,0) -> SO(7) (1;4,2,3)', [(1, 1, 0)], 3, (1, 4, 2, 3)),
                 ('x^(0,2,0) -> SO(7) (1;5,1,3)', [(0, 2, 0)], 3, (1, 5, 1, 3))]
        sel = [int(a) for a in sys.argv[2:]] or list(range(len(cases)))
        res = run_tests('SO(9)', reps, 4, (1, 5, 3, 3), [cases[i] for i in sel], ctrls)
        tag = '' if len(sel) == len(cases) else '_cases' + ''.join(map(str, sel))
        json.dump(dict(description='SO(9) fortuitous class (R-006) restricted to SO(7) x SO(2)', controls=[n for n, _ in ctrls],
                       results=res, time=round(time.time() - t0)),
                  open(os.path.join(RUNS, f'so9_restrictions{tag}{SUF}.json'), 'w'), indent=1, default=str)
    elif stage == 'rep11':
        dims = pickle.load(open(os.path.join(RUNS, 'cache', 'dims_SO_odd5_j26_2J2.pkl'), 'rb'))
        # Casimir degrees of so(11) are 2..10, so single-trace generators with n <= 10 span all gravitons (R-011 used
        # 12 as a safety margin; building n = 11, 12 is slow on a laptop). Consistency: grav must equal R-011's count.
        GA = GravitonAlgebra(int(os.environ.get('GEN_NMAX', '10')))
        which = sys.argv[2:] or ['7,3,3', '5,5,3']
        for lamstr in which:
            lam = tuple(int(x) for x in lamstr.split(','))
            exprs, info = hw_nongraviton_reps(5, 'SO_odd', lam, 1, dims, GA, seed=1, log=log)
            tag = ''.join(map(str, lam))
            for i, e in enumerate(exprs):
                save_expr_json(e, os.path.join(RUNS, f'so11_rep_1{tag}_{i}.json.gz'), sector=[1, *lam], group='SO(11)', info=info)
            json.dump(info, open(os.path.join(RUNS, f'so11_rep_1{tag}_info.json'), 'w'), indent=1)
    elif stage == 'test11':
        # Two-block restrictions SO(11) > SO(7) x SO(2)^2, all landing in the SO(7) sector (1;3,3,3) (one cheap target).
        # The single-block statement SO(11) > SO(9) x SO(2) follows by composition (validated in test_blockrestrict):
        # if x^a y^b of the class is non-graviton in SO(7), the y^b coefficient is non-graviton in SO(9).
        # Brane-fusion prediction (mod gravitons, by SU(3)_R covariance and block exchange symmetry):
        #   [4,0] at (7,3,3):  f(x,y) = x1^2 y1^2 (+ a (x1^4 + y1^4))
        #   [0,2] at (5,5,3):  f(x,y) = (x1 y2 - x2 y1)^2  ->  coefficients x1^2y2^2 : x1x2y1y2 : x2^2y1^2 = 1 : -2 : 1
        which = sys.argv[2:] or ['7,3,3', '5,5,3']
        T = TargetSector(3, 'SO_odd', (1, 3, 3, 3), gen_nmax=8, seed=TSEED, log=log)
        cases_by = {(7, 3, 3): [('x^(2,0,0) y^(2,0,0)', [(2, 0, 0), (2, 0, 0)]),
                                ('x^(4,0,0) y^0 (one brane carries everything)', [(4, 0, 0), (0, 0, 0)])],
                    (5, 5, 3): [('x^(2,0,0) y^(0,2,0)', [(2, 0, 0), (0, 2, 0)]),
                                ('x^(1,1,0) y^(1,1,0)', [(1, 1, 0), (1, 1, 0)]),
                                ('x^(0,2,0) y^(2,0,0)', [(0, 2, 0), (2, 0, 0)]),
                                ('x^(2,2,0) y^0 (one brane carries everything)', [(2, 2, 0), (0, 0, 0)])]}
        out = {}
        for lamstr in which:
            lam = tuple(int(x) for x in lamstr.split(','))
            tag = ''.join(map(str, lam))
            rep = load_expr_json(os.path.join(RUNS, f'so11_rep_1{tag}_0.json.gz'))
            ctrls = graviton_controls(5, (1,) + lam, count=4)
            log(f'SO(11) {lam} representative: {len(rep)} terms; {len(ctrls)} graviton controls')
            rows, ref = [], None
            for label, targets in cases_by[lam]:
                img = restrict_so2(rep, targets)
                c = T.classify(img) if img else dict(nonzero=False, closed=True, not_exact=False, non_graviton_component=False)
                row = dict(case=label, targets=targets, terms=len(img), verdict=classify_word(c), **c)
                if img and c['non_graviton_component']:
                    r = T.beg.reduce(T.pts.expr_vector(img)[None, :] % P)[0]
                    if ref is None:
                        ref = r
                        row['ratio_to_first_nongraviton'] = 1
                    else:
                        j = int(np.flatnonzero(ref)[0])
                        lamb = int(r[j]) * pow(int(ref[j]), P - 2, P) % P
                        prop = bool(np.array_equal((r - lamb * ref) % P, np.zeros_like(r)))
                        row['ratio_to_first_nongraviton'] = lamb if lamb <= P // 2 else lamb - P
                        row['proportional'] = prop
                ctrl = []
                for names, e in ctrls:
                    im = restrict_so2(e, targets)
                    ctrl.append(classify_word(T.classify(im)) if im else 'zero')
                row['graviton_controls'] = ctrl
                log(f'SO(11) {lam} {label} -> SO(7) (1;3,3,3): ' + json.dumps({k: v for k, v in row.items() if k != 'targets'}))
                rows.append(row)
            out[tag] = rows
        json.dump(dict(description='SO(11) fortuitous classes (R-011) restricted to SO(7) x SO(2)^2; target SO(7) (1;3,3,3) '
                                   f'dim {T.dimV}, exact {T.r_exact}, graviton {T.grav}; ratios = non-graviton components '
                                   'relative to the first non-graviton case (mod p, symmetric representative)',
                       results=out, time=round(time.time() - t0)),
                  open(os.path.join(RUNS, 'so11_restrictions_' + '_'.join(w.replace(',', '') for w in which) + f'{SUF}.json'), 'w'),
                  indent=1, default=str)
    elif stage == 'verify':
        # the stored SO(9)/SO(11) representatives are Q-closed at fresh random points (seed 991) of their own group
        from fortuity.bmn.evaluate import Points
        from fortuity.bmn.words import Q_expr
        for fname, N in [('so9_rep_1533_0.json.gz', 4), ('so11_rep_1733_0.json.gz', 5), ('so11_rep_1553_0.json.gz', 5)]:
            rep = load_expr_json(os.path.join(RUNS, fname))
            q = Q_expr(rep)
            pts = Points(N, 8, 1, [0], seed=991, group='SO_odd')
            ptsr = Points(N, 8, 1, [1], seed=992, group='SO_odd')
            vq, vr = pts.expr_vector(q) % P, ptsr.expr_vector(rep) % P
            log(f'{fname}: {len(rep)} terms, Q(rep) {len(q)} terms; rep nonzero at fresh points: {bool(vr.any())}; '
                f'Q(rep) = 0 at fresh points: {not vq.any()}')
    else:
        raise SystemExit(__doc__)
    log('done')
