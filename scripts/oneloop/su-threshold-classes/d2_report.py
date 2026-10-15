"""Read the d_2 run of a class: checks of d2_make.py, what veval.py evaluated, and the rank increments by row name.
usage: python3 d2_report.py CLASSDIR PREPDIR      (e.g. classes/r007_d2b prep/r007_d2b_T2)"""
import json, os, sys
import numpy as np

cd, pd = sys.argv[1], sys.argv[2]
m = json.load(open(os.path.join(cd, 'd2_make.json')))
rows = m['test_rows']
print('make  :', {k: m.get(k) for k in ('x_from', 'q1x_equals_stored', 'closed_rows', 'x_words', 'y_words', 'Ux_words', 'z_words',
                                         'check_Q0y_plus_Q1x_nonzero', 'control_Q0y_plus_2Q1x_nonzero', 'check_values',
                                         'check_Q0z_nonzero', 'control_2K2_nonzero', 'check_Q0z_values', 'z_nonzero_values',
                                         'E_z_nonzero', 'n_ctrl', 'time')})
meta = json.load(open(os.path.join(pd, 'meta.json')))
nV, nc, c = meta['nV'], meta['n_ctrl'], meta['ncols']
assert nV - nc == len(rows), (nV, nc, len(rows))
ve = meta.get('veval', {})
print('veval :', {k: ve.get(k) for k in ('K', 'n_ctrl', 'n_words', 'foreign_letter_words', 'off_degree_words')},
      'rows touching foreign letters:', ve.get('rows_touching_foreign'))
V = np.fromfile(os.path.join(pd, 'V.i32'), dtype=np.int32).reshape(nV, c)
nzv = [int(np.count_nonzero(V[r])) for r in range(nV)]
print(f'V rows: nonzero entries of {c}: controls {nzv[:nc]}; z {nzv[nc]}; others min {min(nzv[nc + 1:])} max {max(nzv[nc + 1:])}')
for name in ('rank_cpu_ffblas.json', 'rank_gpu.json'):
    f = os.path.join(pd, name)
    if not os.path.exists(f):
        continue
    r = json.load(open(f))
    inc = r['lifted_increments']
    assert len(inc) == len(rows)
    grp = {}
    for nm, i in zip(rows, inc):
        g = 'd1' if nm.startswith('d1_') else 't' if nm.startswith('t_') else nm
        grp.setdefault(g, []).append(i)
    print(f"{name}: rank_B {r['rank_B']}, controls_exact {r['controls_exact']}, time {r.get('time')}s")
    for g, v in grp.items():
        print(f"   {g:10s}: total {sum(v)} of {len(v)} rows; increments {v}")
