"""Collect the two-loop (d_2) results of the SU(2) sectors of T-040 (cluster/t040_d2.sbatch) into one JSON on stdout.
Per case: the checks of d2_make.py (classes/d2_<case>/d2_make.json) and the exactness test of z in T_2
(classes/<T2 case>/fullpar_seed41.json).  usage: python3 collect_t040_d2.py [cases file]"""
import json, os, sys

cases = [l.split() for l in open(sys.argv[1] if len(sys.argv) > 1 else 't040_d2_cases.txt') if l.strip()]
out = []
for case, reps in cases:
    g, twoJ, q = case.split(':')
    q = [int(x) for x in q.split(',')]
    twoJ = int(twoJ)
    Y = sum(q) - twoJ
    D = case.replace(':', '_').replace(',', '-')
    T2D = 'rank_%s_%d-%d-%d-%d-%d_%d' % (g[2:], twoJ, twoJ, 2 * q[0], 2 * q[1], 2 * q[2], Y - 1)
    row = dict(case=case, j=2 * sum(q), Y=Y, reps=reps, T2=T2D)
    fm, fr = f'classes/d2_{D}/d2_make.json', f'classes/{T2D}/fullpar_seed41.json'
    if os.path.exists(fm):
        m = json.load(open(fm))
        row.update({k: m.get(k) for k in ('d1_kernel_dim', 'a', 'x_words', 'y_words', 'z_words', 'check_Q0y_plus_Q1x_nonzero',
                                           'control_Q0y_plus_2Q1x_nonzero', 'check_Q0z_nonzero', 'control_2K2_nonzero',
                                           'check_Q0z_values', 'E_z_nonzero', 'test_rows', 'time')})
    if os.path.exists(fr):
        r = json.load(open(fr))
        if r.get('n_vecs') and 'lifted_increments' in r:
            row.update(T2_hw_multiplicity=sum(r['target_m_d'].values()), rank_B=r['rank_B'], controls_exact=r['controls_exact'],
                       increments=r['lifted_increments'], span=(r.get('span') or {}).get('spans'))
    row['done'] = 'increments' in row and 'test_rows' in row and len(row['increments']) == len(row['test_rows'])
    out.append(row)
json.dump(out, sys.stdout, indent=1)
