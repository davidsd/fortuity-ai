"""Write data/su3/hw_cohomology_8_5_4.json (+ .meta.json): the complete SU(3) column (8,5,4), j = 34, from the T-032
FASRC runs, with the core test of fortuity.hwcoh.cores.

usage (repo root): PYTHONPATH=code python3 scripts/bmn/su3-cores/analysis/make_core_data.py COMMIT
  COMMIT: the code commit that contains fortuity.hwcoh.cores (written into the meta file)
"""
import glob
import json
import os
import pickle
import sys

from fortuity.hwcoh.cores import core_test, nongraviton_content
from fortuity.hwcoh.sector import hw_mult_nf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
HERE = os.path.join(REPO, 'scripts', 'bmn', 'su3-cores')
LAM = (8, 5, 4)
KEEP = ('n', 'm', 'm_d', 'rank_C_d', 'check', 'r_up', 'grav', 'grav_rows', 'words', 'words_up', 'ncols', 'maxlen',
        'maxlen_up', 'seed', 'engine', 'time')


def load(paths, N):
    cols = {}
    for p in sorted(paths):
        for line in open(p):
            r = json.loads(line)
            if r.get('N') == N:
                cols.setdefault(tuple(r['lam']), {})[r['n']] = r
    return cols


recs = load(glob.glob(os.path.join(HERE, 'results', 'su3_8_5_4_*.jsonl')), 3)[LAM]
su2 = load(glob.glob(os.path.join(REPO, 'data', 'su2', 'hw_cohomology_j44_records', '*.jsonl')), 2)
content = [e for e in nongraviton_content(su2) if e[0] <= sum(LAM)]
res = core_test(3, LAM, recs, content)
# every fermion degree with a nonzero exact multiplicity has words and is certified (review B.4)
dims = pickle.load(open(os.path.join(HERE, 'cluster', 'dims_nf_su3_36.pkl'), 'rb'))
for n, r in recs.items():
    exact = {d: hw_mult_nf(dims, n, LAM, (n - d) // 2) for d in range(n % 2, n + 1, 2)}
    assert {int(k): v for k, v in r['m_d'].items()} == {d: v for d, v in exact.items() if v}, n
    assert {int(k): v for k, v in r['rank_C_d'].items()} == {d: v for d, v in exact.items() if v}, n
out = dict(
    N=3, lam=list(LAM), j=2 * sum(LAM), rep=[LAM[0] - LAM[1], LAM[1] - LAM[2]],
    sectors=[dict({k: recs[n][k] for k in KEEP if k in recs[n]}, h=res['sectors'][n]['h'], BH=res['sectors'][n]['BH'],
                  E=sum(LAM) - n / 2) for n in sorted(recs)],
    H_total=res['H_total'], euler=res['euler'], index=res['index'], twisted_index_signed=res['twisted'],
    p_odd=res['p_odd'], BH=res['BH'], capacity=res['capacity'],
    capacity_sources=[dict(su2_level=l, twoJ=t, su2_hw=list(mu), x_power=k, multiplicity=c)
                      for l, t, mu, k, c in res['capacity_sources']],
    cores_per_2J=res['cores_per_2J'], cores_lower_bound=res['cores_lower_bound'],
    cores_lower_bound_column=res['cores_lower_bound_column'],
    su2_nongraviton_content=sorted([list(e[:2]) + [list(e[2])] for e in content]))
dst = os.path.join(REPO, 'data', 'su3', 'hw_cohomology_8_5_4.json')
json.dump(out, open(dst, 'w'), indent=1)
meta = dict(
    description='SU(3) BMN-sector highest-weight Q-cohomology of the SU(3)_R column lam = (8,5,4) (j = 34, [3,1]) at every '
                '2J: per sector the hw multiplicity m, rank r_up of Q from 2J+1, graviton rank grav (products of u_n, v_n, '
                'w_n, n <= 3, modulo exact), h = m - r_up(2J) - r_up(2J-1) and BH = h - grav; per column the Euler '
                'characteristic vs the exact index, the C-twisted index and the exact number of P-odd classes (R-017), '
                'the fused capacity of the SU(3) > SU(2) x U(1) restriction (R-016) and the resulting lower bounds on '
                'genuine SU(3) cores (fortuity.hwcoh.cores)',
    computed_by='Fortuity collaboration (Claude agents) on Harvard FASRC',
    date='2026-10-10 (runs 2026-09-30 to 2026-10-02)',
    code='sectors: fortuity.hwcoh.run_sectors with ENGINE=par (fortuity.hwcoh.sectorpar), MAXLEN=6, MAXLEN_UP=6, '
         'GEN_NMAX=3, SEED=1, via scripts/bmn/su3-cores/cluster/sector.sbatch; post-processing: '
         'fortuity.hwcoh.cores (core_test) via scripts/bmn/su3-cores/analysis/make_core_data.py',
    commit=f'post-processing code {sys.argv[1]}; sectors 2J=7,8,9 with the code of 61ed855 (job 49389620), 2J=6 with '
           'ac60633 or later (jobs 49515813/49516090), 2J=0-5 with the code synced on 2026-10-02 (resubmission with '
           'memmaps on global scratch); the engine mathematics is the same in all of them',
    conventions='D-006 (Chang-Lin / Choi et al. BMN conventions as in fortuity.bmn.words); sector (2J; q1,q2,q3), '
                'j = 2 sum q, E = sum q - J; rep [a,b] = [q1-q2, q2-q3]; P = C (-1)^Y',
    command='sbatch (TASKS=tasks_854_{light,medium,heavy}.txt) sector.sbatch; then python3 '
            'scripts/bmn/su3-cores/analysis/make_core_data.py <commit>',
    inputs='SU(2) non-graviton content: T-027 run_sectors output data/su2/hw_cohomology_j44_records/*.jsonl '
           '(commit c902a67; complete columns only; levels <= 17 can contribute); exact SU(3) dims: '
           'fortuity.hwcoh.dims.partition_dims_nf(3, 36) (dims_nf_su3_36.pkl)',
    checked='every sector: rank of the hw vectors at its functionals = exact multiplicity in every fermion degree, and '
            'every degree with a nonzero exact multiplicity has words (review B.4 guard, re-checked here for the '
            'runs made before it existed), so r_up and grav are exact over F_p; column: Euler characteristic 3 = exact '
            'index 3; (H_total - (-1)^|lam| I_C)/2 = (37 - 31)/2 = 3 P-odd classes = BH, as R-017 predicted (>= 3); '
            'the graviton total 34 equals R-017\'s Coulomb count D for this column; the fused capacity reproduces the '
            'R-016 channel of R-007 at (7,4,4) and the saturated j = 32 columns of T-042 (tests: code/tests/test_cores.py)',
    caveat='random evaluation over F_p (p = 2^26 - 5) with certified functionals: exact over F_p; a bad prime could only '
           'add classes (pairs), which would lower the P-odd count. Traces of length <= 6 (spanning certified per '
           'sector). Classical (tree-level) cohomology; non-graviton in the sense of D-005. The core bound is a count: '
           'which classes are the cores is decided by restricting explicit representatives (T-032 follow-up)')
json.dump(meta, open(dst.replace('.json', '.meta.json'), 'w'), indent=1)
print(json.dumps({k: out[k] for k in ('H_total', 'euler', 'index', 'twisted_index_signed', 'p_odd', 'BH', 'capacity',
                                      'cores_per_2J', 'cores_lower_bound', 'cores_lower_bound_column')}))
