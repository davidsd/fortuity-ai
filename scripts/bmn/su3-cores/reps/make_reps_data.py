"""Collect the run_reps.py summaries (out/reps_*.json) into data/su3/core_restrictions_j30_j34.json (+ .meta.json).

usage (repo root): python3 scripts/bmn/su3-cores/reps/make_reps_data.py
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
rows = []
for f in sorted(glob.glob(os.path.join(HERE, 'out', 'reps_*.json'))):
    d = json.load(open(f))
    s = d['sector']
    rows.append(dict(
        twoJ=s['twoJ'], lam=s['lam'], j=2 * sum(s['lam']), expect=d['expect'], nongraviton_found=d['info']['nongrav'],
        rep_terms=d['info']['rep_terms'], hw_rows=d['info']['hw_rows'], r_down=d['info']['r_down'],
        channels=[dict(alpha=c['alpha'], su2_sector=c['sector'], su2_dim=c['dim'], su2_exact_rank=c['r_exact'],
                       su2_graviton_rank=c['grav'], image_terms=c['terms'], image_closed=[x['closed'] for x in c['classify']],
                       non_graviton_component=c['residue_nonzero']) for c in d['channels']],
        restriction_rank=d['restriction_rank'], cores=d['cores'], core_coefficients=d['core_coefficients'],
        core_terms=d['core_terms'], core_residues_nonzero=d['core_residues_nonzero'], checks=d['checks'],
        rep_content=d['rep_content'], core_content=d['core_content'], seed=d['seed'], target_seed=d['target_seed'],
        prime=d['prime'], time=d['time']))
rows.sort(key=lambda r: (r['j'], r['lam'], r['twoJ']))
dst = os.path.join(REPO, 'data', 'su3', 'core_restrictions_j30_j34.json')
json.dump(dict(rows=rows), open(dst, 'w'), indent=1)
meta = dict(
    description='SU(3) BMN non-graviton classes restricted to SU(2) x U(1) at the representative level: for each sector, '
                'explicit representatives of the non-graviton classes (fortuity.hwcoh.classreps_par), the coefficient of '
                'x^alpha of their block restriction (R-016) in every channel whose SU(2) target holds non-graviton classes '
                '(the fused capacity of fortuity.hwcoh.cores), reduced modulo exact + graviton there; restriction_rank = '
                'number of fused classes, cores = non-graviton classes minus restriction_rank (T-032 follow-up, R-024)',
    computed_by='Fortuity collaboration (Claude agents) on Harvard FASRC (reps.sbatch: the five medium sectors '
                'submitted 2026-10-10, the heavy (6;8,5,4) one 2026-10-11)',
    date='2026-10-12',
    code='code/fortuity/hwcoh/classreps_par.py (commit d88f5c3) via scripts/bmn/su3-cores/reps/run_reps.py '
         '(commit c1e3fe3); SEED=1, MAXLEN=MAXLEN_UP=6, GEN_NMAX=3, TARGET_SEED=17',
    commit='c1e3fe3',
    conventions='D-006 BMN conventions (fortuity.bmn.words); sector (2J; q1,q2,q3); x = U(1) components of the scalars '
                '(fortuity.bmn.blockrestrict.restrict_u1)',
    command='sbatch reps.sbatch (TASKS=tasks_reps_{heavy,medium}.txt); then python3 '
            'scripts/bmn/su3-cores/reps/make_reps_data.py',
    inputs='engine records: our (8,5,4) column (data/su3/hw_cohomology_8_5_4.json), T-042 (7,5,4), (8,4,4) '
           '(commit b08e01f), R-007 column from data/su3/bmn_hw_cohomology_j30.json; exact SU(3) dims to j = 36; exact '
           'SU(2) dims (fortuity.bmn.fullindex.partition_dims) certify the SU(2) target functionals',
    checked='per sector: the lower Q-images reach the certified rank r_down; self-check rank = m_d; compressed hw rows '
            'have rank m_d and their lower images rank r_down; closed space = m - r_down; non-graviton classes found = '
            'the engine count h - grav; every restricted image is closed in SU(2); every representative is closed at '
            'fresh N = 3 points. Cores (only (6;8,5,4) has one): the images of the combined expression have zero residue '
            'in every channel, and the core is closed at fresh N = 3 points. checks.core0_closed_N4 = false refers '
            'to the saved lift only: it is a random combination of the overcomplete hw rows, so it carries random '
            'N = 3 trace relations, which are generically not closed at N = 4; the flag does not test the class. '
            'Validation: R-007 (5;7,4,4) comes out fused, as in R-016, on the laptop (numpy) and on '
            'the cluster (BLIS kernels); unit tests code/tests/test_classreps_par.py (SU(2) O_0; dMK -648/5)',
    caveat='random evaluation over F_p (p = 2^26 - 5) with certified functionals on both sides (SU(3) sectors by the '
           'engine self-checks, SU(2) targets by their exact dimensions), so residues and ranks are exact over F_p; a bad '
           'prime is the usual residual caveat; classical (tree-level) cohomology')
json.dump(meta, open(dst.replace('.json', '.meta.json'), 'w'), indent=1)
for r in rows:
    print(r['j'], r['lam'], r['twoJ'], 'non-graviton', r['nongraviton_found'], 'fused', r['restriction_rank'],
          'cores', r['cores'])
