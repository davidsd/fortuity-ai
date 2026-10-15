"""SU(3)_R highest-weight check of test vectors: E_1 v = E_2 v = 0 at fresh random su(3) points (C-odd projection, as
the preps see vectors).  The exactness preps contain only Q_0 of hw vectors, so a test row J t must be an exact hw vector
for an increment to mean non-exactness.  usage: python3 hwcheck.py VECS.pkl [VECS2.pkl ...] [--workers W]
(VECS: dict(words, coef, n_ctrl) as written by d1_make / jt_make; also accepts twords.pkl via --t DIR)."""
import argparse, json, os, pickle, time
import numpy as np
import fullpar
import build_cand as bc
import q1apply
try:
    from fortuity.full.hw import E_multi
except ImportError:
    from fortuity.full.hw import E_multi

log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)
ap = argparse.ArgumentParser()
ap.add_argument('vecs', nargs='*')
ap.add_argument('--t', default=None, help='a d1_make output dir: check t itself (twords.pkl, tvec.npz)')
ap.add_argument('--npts', type=int, default=16)
ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '8')))
a = ap.parse_args()
fullpar.CSEC = -1
items = [(f, None) for f in a.vecs] + ([(a.t, 't')] if a.t else [])
res = {}
for f, kind in items:
    if kind == 't':
        words = pickle.load(open(os.path.join(f, 'twords.pkl'), 'rb'))
        C = np.load(os.path.join(f, 'tvec.npz'))['coef']
    else:
        d = pickle.load(open(f, 'rb'))
        words, C = d['words'], d['coef']
    out = {}
    for i in (1, 2):
        img, V = q1apply.apply(words, C, workers=a.workers, chunk=2000, op=lambda m, i=i: E_multi(m, i))
        vals = bc.eval_combo(img, V, 3, a.npts, 5150 + i, False, a.workers)
        out[f'E{i}'] = [int(sum(np.count_nonzero(x[k]) for x in vals.values())) for k in range(C.shape[0])]
        log(f"{f}{' (t)' if kind else ''}: E{i} image {len(img)} words; nonzero values per row {out[f'E{i}']}")
    res[f + (':t' if kind else '')] = out
res['all_hw'] = all(not any(v) for o in res.values() if isinstance(o, dict) for v in o.values())
json.dump(res, open('hwcheck_' + time.strftime('%m%d%H%M') + '.json', 'w'), indent=1)
log(json.dumps(res))
