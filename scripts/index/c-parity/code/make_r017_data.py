"""Assemble the data/ files of R-017 from this workspace's runs and write provenance sidecars.
usage (repo root): python3 scripts/index/c-parity/code/make_r017_data.py COMMIT"""
import os, sys, json
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
RUNS = os.path.join(REPO, 'scripts/index/c-parity/runs')
COMMIT = sys.argv[1]
WS = 'scripts/index/c-parity/code'
CODE = ('code/fortuity/bmn/twistedindex.py (twisted Weyl integration, kernel_parity), code/fortuity/bmn/cparity.py, '
        'code/fortuity/bmn/gravindex.py (diagonal graviton counts); tests code/tests/test_twistedindex.py')
CONV = ('BMN sector (2J; q1,q2,q3) of SU(N), letters phi_m (0;e_m), psi_m (1;(1,1,1)-e_m), f (2;(1,1,1)); index '
        'Tr (-1)^(2J) with fugacities z^q; C: X -> -X^T on all letters; Y = q1+q2+q3-2J (number of letters); '
        'P = C (-1)^Y; per SU(3)_R hw column lam (irrep [lam1-lam2, lam2-lam3]); |lam| = lam1+lam2+lam3')


AMENDED = ('2026-10-02: rigour statements corrected after review C19 (D is a signed Weyl alternation of randomly '
           'evaluated weight dimensions, not a certified lower bound); numbers unchanged. See R-017 Checked by (4) and '
           'summary/fortuity-summary.pdf (sec. 7.1)')


def write(rel, obj, meta):
    path = os.path.join(REPO, rel)
    json.dump(obj, open(path, 'w'), indent=1, default=str)
    base = {'computed_by': 'Fortuity collaboration (Claude agent), laptop', 'date': '2026-09-29',
            'code': CODE, 'commit': COMMIT, 'conventions': CONV}
    base.update(meta)
    json.dump(base, open(path[:-5] + '.meta.json', 'w'), indent=1)
    print('wrote', rel, os.path.getsize(path))


for N, cohf in [(3, 'data/su3/bmn_hw_cohomology_j30.json'), (4, 'data/su4/bmn_hw_cohomology_j28.json + j30.json')]:
    d = json.load(open(os.path.join(RUNS, f'su{N}_twisted_index_j30.json')))
    write(f'data/su{N}/twisted_index_j30.json',
          dict(description=f'SU({N}) BMN C-twisted index I_C = Tr (-1)^(2J) C per SU(3)_R hw column, j <= 30, compared with '
                           f'the prediction (-1)^|lam| I_C = (total cohomology of the column) - 2 #(P-odd classes) from {cohf}; '
                           'the ordinary index from the same integrator is checked against the data files',
               rows=d['rows'], mismatches=d['mismatches'], ordinary_index_check=d.get('ordinary_index_check')),
          dict(description=f'SU({N}) C-twisted BMN index vs cohomology (R-017)',
               command=f'PYTHONPATH=code python3 {WS}/twisted_index.py {N} 15' + (' 10' if N == 4 else ''),
               checked=f'all {len(d["rows"])} columns agree with the prediction; P-odd only at SU(3) (7,4,4) (the R-007 class); '
                       'ordinary index from the same code equals the data index per column; independent SU(3) twisted '
                       'integrator (scripts/oneloop/su-threshold-classes/twisted_index.py) gives I_C = -10 at (7,4,4)'))
k = json.load(open(os.path.join(RUNS, 'su3_kernel_parity_j40.json')))
write('data/su3/kernel_parity_j40.json',
      dict(description='SU(3) kernel parity index per hw column, j <= 40: T = (-1)^|lam| I_C - D = #P+ - #P- among the '
                       'classes invisible on the Coulomb branch (D = unsigned diagonal-graviton count), and O = (-1)^|lam| '
                       '(I - I_grav^diag) = sum of (-1)^Y over them.  Only nonzero columns listed.  If D is not overcounted, '
                       'T < 0 forces P-odd, hence non-graviton (fortuitous), classes: #fortuitous(lam) >= -T.  D is computed by '
                       'random evaluation and a signed Weyl alternation, so for j = 32-40 this is very likely but not certified '
                       '(R-017 Checked by (4)).  j <= 30 rows carry the known content (R-007, R-015)', rows=k['rows']),
      dict(description='SU(3) kernel parity index to j = 40 (R-017)',
           command=f'PYTHONPATH=code python3 {WS}/diag_counts.py 3 40; PYTHONPATH=code python3 {WS}/noncoulomb_parity.py 3 20 '
                   '(workspace driver; its functions are those of fortuity.bmn.twistedindex at this commit)',
           checked='j <= 30 reproduces exactly the known kernel content (4 non-Coulomb graviton columns of R-015, the R-007 '
                   'class with T = -1); diagonal counts reproduce data/su3/bmn_index_j36.json graviton_index_unrefined (same '
                   'code and seed, so not independent); an independent rerun (seed 2026, p = 67108837) gives identical '
                   'weight dimensions in all 28,336 sectors to j = 40; signed content agrees with Table 1 of '
                   'arXiv:2312.16443 through j = 40.  Not certified: D is a signed Weyl alternation of random-evaluation '
                   'weight dimensions, so T < 0 at j = 32-40 is not rigorous (review C19)', amended=AMENDED))
s = json.load(open(os.path.join(RUNS, 'su2_predictions_j44.json')))
write('data/su2/kernel_parity_j44.json',
      dict(description='SU(2) kernel parity index per hw column, j <= 44 (C is inner, so P = (-1)^Y): T = (-1)^|lam| I - D. '
                       'Negative in 17 columns (forcing odd-Y black holes), never positive. If every SU(2) black hole has odd '
                       'Y and there is no non-Coulomb graviton, the number of black-hole classes per column is -T '
                       '(prediction for T-027, confirmed by T-027 through j = 40).  Rigorous: D = 0 in all 17 negative columns, '
                       'and all SU(2) diagonal counts are exact (see meta)', rows=s['rows']),
      dict(description='SU(2) index-only black-hole predictions to j = 44 (R-017)',
           command=f'PYTHONPATH=code python3 {WS}/diag_counts.py 2 44; PYTHONPATH=code python3 {WS}/su2_predictions.py',
           checked='index reproduces data/su2/bmn_index_j40.json; twisted = ordinary for SU(2); the (6,6,6) column computed '
                   'directly (T-029) has 2 odd-Y black holes = -T.  Rigour (2026-10-02): the computed D is 0 in all 17 columns '
                   'with T < 0, so T = (-1)^|lam| I and T_true <= T; moreover the computed diagonal counts equal the closed '
                   'form (3.4) of arXiv:2312.16443 (Groebner basis) at every j <= 44, and all SU(2) Coulomb states have even '
                   'Y, so the per-sector counts are exact and every T here is exact', amended=AMENDED))
