"""T-028 step 1: restrict the SU(3) fortuitous class of R-007 ((2J;q) = (5;7,4,4), j = 30, [3,0]) to SU(2) x U(1).

Input: the explicit highest-weight representative (T-021), 201,564 multi-trace words mod p.
For k = 0..7 the coefficient of x_1^k (x_m = U(1) components of phi_m, D = diag(1,1,-2)) is an SU(2) operator in the
sector (5; 7-k, 4, 4); each is classified there (zero / Q-exact / graviton mod exact / non-graviton component).
Prediction (de Mello Koch-Kim-Kim-Lee-Lee, arXiv:2412.08695, eq. (3.1)): the x_1^3 coefficient is a nonzero multiple
of the SU(2) threshold class O_0 in (5;4,4,4) (the fused / "dual-giant-wrapped" structure).
Control: restrictions of SU(3) graviton products in the same sector must never have a non-graviton component (and
in (5;4,4,4), where SU(2) has no graviton monomials at all, they must be Q-exact).

usage (repo root): PYTHONPATH=code:scripts/bmn/block-restriction/code python3 .../run_su3_to_su2.py
"""
import os, sys, time, json, random
from itertools import combinations_with_replacement
from fortuity.bmn.words import graviton_generators, mul_expr, expr_sector
from fortuity.bmn.modp import P
from fortuity.bmn.blockrestrict import restrict_u1, load_expr_json, save_expr_json
from fortuity.bmn.classify import TargetSector
from fortuity.bmn.fullindex import partition_dims

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
RUNS = os.path.join(HERE, '..', 'runs')
t0 = time.time()


def log(*a):
    print(f'[{time.time() - t0:7.0f}s]', *a, flush=True)


rep = load_expr_json(os.path.join(REPO, 'scripts/bmn/su3-level30-representative/runs/rep_nmax5_seed11_expr.json.gz'))
log(f'loaded R-007 representative: {len(rep)} terms, sector {expr_sector(rep)}')

# control expressions: a few SU(3) graviton monomials in the same weight sector (5;7,4,4)
gens = [(n, e) for n, e in graviton_generators(3, 1) if e]
sec_of = {i: tuple(expr_sector(e)) for i, (n, e) in enumerate(gens)}
target_sec = (5, 7, 4, 4)
monos = []
rnd = random.Random(3)
idxs = list(sec_of)
for r in (2, 3, 4):
    for combo in combinations_with_replacement(idxs, r):
        s = tuple(sum(sec_of[i][a] for i in combo) for a in range(4))
        if s == target_sec:
            monos.append(combo)
rnd.shuffle(monos)
controls = []
for combo in monos:
    e = {(): 1}
    for i in combo:
        e = mul_expr(e, gens[i][1])
    if e:
        controls.append(([gens[i][0] for i in combo], e))
    if len(controls) >= 6:
        break
log(f'{len(monos)} graviton monomials in {target_sec}; using {len(controls)} as controls')

DIMS2 = partition_dims(2, 30, chunk=4)   # exact SU(2) weight-sector dimensions (certify the functionals)
TSEED = int(os.environ.get('TARGET_SEED', '17'))          # seed of the target-sector functionals (second seed: 29)
SUF = '' if TSEED == 17 else f'_tseed{TSEED}'
results = []
for k in range(0, 8):
    sec2 = (5, 7 - k, 4, 4)
    img = restrict_u1(rep, 2, (k, 0, 0))
    log(f'k={k}: restricted image has {len(img)} SU(2) terms (target sector {sec2})')
    if not img:
        results.append(dict(k=k, sector=sec2, terms=0, verdict='zero (formally)'))
        continue
    T = TargetSector(2, 'SU', sec2, gen_nmax=3, seed=TSEED, log=log, maxlen=3, exact_dim=DIMS2.get(sec2, 0))
    assert T.dimV == DIMS2.get(sec2, 0), (T.dimV, DIMS2.get(sec2))
    c = T.classify(img)
    verdict = ('non-graviton' if c['non_graviton_component'] else
               'graviton (mod exact)' if c['not_exact'] else ('Q-exact' if c['nonzero'] else 'zero at N=2'))
    res = dict(k=k, sector=sec2, terms=len(img), dim=T.dimV, exact_rank=T.r_exact, grav=T.grav, **c, verdict=verdict)
    ctrl = []
    for names, e in controls:
        ci = restrict_u1(e, 2, (k, 0, 0))
        if not ci:
            ctrl.append('zero')
            continue
        cc = T.classify(ci)
        ctrl.append('NON-GRAVITON(!)' if cc['non_graviton_component'] else
                    'graviton' if cc['not_exact'] else ('exact' if cc['nonzero'] else 'zero'))
    res['controls'] = ctrl
    log(f'k={k}: {res}')
    results.append(res)
    if k == 3 and not SUF:
        save_expr_json(img, os.path.join(RUNS, 'su3_r007_rep_restricted_x1cubed_su2.json.gz'),
                       description='x_1^3 coefficient of the SU(3)>SU(2)xU(1) restriction of the T-021 R-007 '
                                   'representative; SU(2) sector (5;4,4,4)')

out = dict(description='SU(3) R-007 class restricted to SU(2) x U(1): coefficient of x_1^k classified in SU(2) '
                       'sector (5;7-k,4,4)', source='scripts/bmn/su3-level30-representative/runs/rep_nmax5_seed11_expr.json.gz',
           prime=P, target_seed=TSEED, controls=[names for names, _ in controls], results=results,
           time=round(time.time() - t0))
json.dump(out, open(os.path.join(RUNS, f'su3_to_su2xu1{SUF}.json'), 'w'), indent=1, default=str)
log('done')
