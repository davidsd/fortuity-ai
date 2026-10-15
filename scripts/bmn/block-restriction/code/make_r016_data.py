"""Assemble the data/ files of R-016 from this workspace's runs (both target seeds) and write provenance sidecars.
usage (repo root): PYTHONPATH=code python3 scripts/bmn/block-restriction/code/make_r016_data.py COMMIT"""
import os, sys, json, gzip
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
RUNS = os.path.join(REPO, 'scripts/bmn/block-restriction/runs')
COMMIT = sys.argv[1]
WS = 'scripts/bmn/block-restriction/code'
CODE = ('code/fortuity/bmn/blockrestrict.py (restriction maps), code/fortuity/bmn/classify.py (TargetSector), '
        'code/fortuity/hwcoh/classreps.py (hw representatives); tests code/tests/test_blockrestrict.py')
CONV = ('summary/fortuity-summary.pdf (section 2); BMN sector (2J; q1,q2,q3), letters 0,1,2 = phi_m, 3,4,5 = psi_m, 6 = f '
        '(fortuity.bmn.words); coefficients mod p = 2^26-5; SU(N) > SU(N-1)xU(1) with D = diag(1,..,1,-(N-1)), '
        'SO(2N+1) > SO(2N+1-2b)xSO(2)^b with blocks x eps, eps = [[0,1],[-1,0]]; abelian parts on the scalars only')


def load(name):
    return json.load(open(os.path.join(RUNS, name)))


def write(rel, obj, meta):
    path = os.path.join(REPO, rel)
    json.dump(obj, open(path, 'w'), indent=1 if 'terms' not in obj else None, default=str)
    base = {'computed_by': 'Fortuity collaboration (Claude agent), laptop', 'date': '2026-09-29',
            'code': CODE, 'commit': COMMIT, 'conventions': CONV}
    base.update(meta)
    json.dump(base, open(path[:-5] + '.meta.json', 'w'), indent=1)
    print('wrote', rel, os.path.getsize(path))


def rep(gzname, rel, desc, command, checked):
    d = json.load(gzip.open(os.path.join(RUNS, gzname), 'rt'))
    obj = dict(description=desc, prime=d['prime'], format='terms = [[coefficient mod prime, [[letters of trace 1], ...]], ...]',
               info=d.get('info'), terms=d['terms'])
    write(rel, obj, dict(description=desc, command=command, checked=checked))


# SU(3) -> SU(2) x U(1)
a, b = load('su3_to_su2xu1.json'), load('su3_to_su2xu1_tseed29.json')
same = [ra == {**rb} or all(ra.get(k) == rb.get(k) for k in ra if k != 'time') for ra, rb in zip(a['results'], b['results'])]
write('data/su3/r007_block_restriction_su2xu1.json',
      dict(description='x1^k coefficients (k = 0..7) of the SU(3) > SU(2) x U(1) restriction of the R-007 representative '
                       '(T-021, 201,564 words, sector (5;7,4,4), [3,0] highest weight), classified in the SU(2) sector '
                       '(5;7-k,4,4): target dim / Q-exact rank / graviton classes mod exact, image verdict, and the images '
                       'of 6 SU(3) multi-graviton products (controls); two independent target seeds',
           source_representative='scripts/bmn/su3-level30-representative/runs/rep_nmax5_seed11_expr.json.gz',
           results_seed17=a['results'], results_seed29=b['results'], seeds_agree=all(same), controls=a['controls']),
      dict(description='SU(3) R-007 class restricted to SU(2) x U(1) (R-016 (a))',
           command=f'PYTHONPATH=code:{WS} [TARGET_SEED=29] python3 {WS}/run_su3_to_su2.py',
           checked='targets certified (rank of words = exact Haar dim, asserted); seeds 17 and 29 agree in every entry; '
                   'controls never non-graviton; maps validated by code/tests/test_blockrestrict.py'))
d = json.load(gzip.open(os.path.join(RUNS, 'su3_r007_rep_restricted_x1cubed_su2.json.gz'), 'rt'))
write('data/su3/r007_x1cubed_image_su2.json',
      dict(description='x1^3 coefficient of the SU(3) > SU(2) x U(1) restriction of the R-007 representative: an SU(2) '
                       'operator in (5;4,4,4), closed, not Q-exact, hence a nonzero multiple of the SU(2) threshold class '
                       'O_0 (R-001) modulo exact (SU(2) has no gravitons there)',
           prime=d['prime'], format='terms = [[coefficient mod prime, [[letters of trace 1], ...]], ...]', terms=d['terms']),
      dict(description='SU(2) image of the R-007 class under the x1^3 coefficient map (R-016 (a))',
           command=f'PYTHONPATH=code:{WS} python3 {WS}/run_su3_to_su2.py (k = 3)',
           checked='classified in SU(2) (5;4,4,4) at two target seeds: closed, not exact, non-graviton'))
# SO(9) -> SO(7) x SO(2)
a, b = load('so9_restrictions_cases023.json'), load('so9_restrictions_cases023_tseed29.json')
write('data/so_sp/so9_block_restriction_so7xso2.json',
      dict(description='SO(9) > SO(7) x SO(2) restriction of the SO(9) non-graviton representative at (1;5,3,3) [2,0] '
                       '(R-006 class): coefficients of x1^2, x1 x2, x2^2 classified in SO(7) sectors (1;3,3,3), (1;4,2,3), '
                       '(1;5,1,3); controls = 6 SO(9) multi-graviton products; two independent target seeds',
           results_seed17=a['results'], results_seed29=b['results'], controls=a['controls']),
      dict(description='SO(9) R-006 class restricted to SO(7) x SO(2) (R-016 (b))',
           command=f'PYTHONPATH=code:{WS} [TARGET_SEED=29] python3 {WS}/run_so_tower.py test9 0 2 3',
           checked='SO(7) (1;3,3,3) target reproduces R-005 (903 / 559 exact / 123 graviton); seeds 17 and 29 agree'))
# SO(11) -> SO(7) x SO(2)^2
res = {}
for tag in ('733', '553'):
    res[tag] = dict(seed17=load(f'so11_restrictions_{tag}.json')['results'][tag],
                    seed29=load(f'so11_restrictions_{tag}_tseed29.json')['results'][tag])
write('data/so_sp/so11_block_restriction_so7xso2sq.json',
      dict(description='SO(11) > SO(7) x SO(2)^2 restriction of the SO(11) non-graviton representatives at (1;7,3,3) [4,0] '
                       'and (1;5,5,3) [0,2] (R-011 classes): coefficients of x^a y^b (blocks x, y) of total degree 4, '
                       'classified in SO(7) (1;3,3,3); for non-graviton images, the ratio of their components along the '
                       'one-dimensional non-graviton quotient relative to the first case (mod p, symmetric representative); '
                       'controls = 4 SO(11) multi-graviton products; two independent target seeds',
           results=res),
      dict(description='SO(11) R-011 classes restricted to SO(7) x SO(2)^2 (R-016 (c))',
           command=f'PYTHONPATH=code:{WS} [TARGET_SEED=29] python3 {WS}/run_so_tower.py test11 7,3,3 (resp. 5,5,3)',
           checked='seeds 17 and 29 agree including the ratios 1 : -2 : 1; target reproduces R-005'))
# representatives
rep('so9_rep_1533_0.json.gz', 'data/so_sp/so9_nongraviton_rep_1533.json',
    'SO(9) BMN non-graviton representative, sector (2J;q) = (1;5,3,3), SU(3)_R [2,0] highest weight (the R-006 class): '
    'a closed highest-weight vector outside exact + graviton span, as a multi-trace expression mod p',
    f'PYTHONPATH=code:{WS} python3 {WS}/run_so_tower.py rep9  (hw_nongraviton_reps(4, "SO_odd", (5,3,3), 1), '
    'dims partition_dims_nf(4, 22, twoJ_max=2), GravitonAlgebra(10), seed 1)',
    'hw sector m=442, closed=311, exact=305, h=6, graviton=5, non-graviton=1 (faithful functionals); re-extracted with the '
    'library code at this commit: identical; Q(rep) = 0 at fresh random so(9) points; restriction to SO(7) x SO(2) is '
    'non-graviton (R-016 (b))')
rep('so11_rep_1733_0.json.gz', 'data/so_sp/so11_nongraviton_rep_1733.json',
    'SO(11) BMN non-graviton representative, sector (1;7,3,3), [4,0] highest weight (R-011 class)',
    f'PYTHONPATH=code:{WS} GEN_NMAX=10 python3 {WS}/run_so_tower.py rep11 7,3,3 (dims partition_dims_nf(5, 26, twoJ_max=2))',
    'hw sector m=2387, closed=1780, exact=1757, h=23, graviton=22, non-graviton=1 (as R-011; generators n <= 10 suffice, '
    'Casimir degrees of so(11) <= 10); Q(rep) = 0 at fresh random so(11) points; two-block restriction non-graviton')
rep('so11_rep_1553_0.json.gz', 'data/so_sp/so11_nongraviton_rep_1553.json',
    'SO(11) BMN non-graviton representative, sector (1;5,5,3), [0,2] highest weight (R-011 class)',
    f'PYTHONPATH=code:{WS} GEN_NMAX=10 python3 {WS}/run_so_tower.py rep11 5,5,3',
    'hw sector m=2354, closed=1820, exact=1808, h=12, graviton=11, non-graviton=1 (as R-011); Q(rep) = 0 at fresh random '
    'so(11) points; two-block restriction non-graviton with ratios 1 : -2 : 1')
