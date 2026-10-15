"""Checks of the d_1 inputs (d1_make.py) at fresh random points, C-odd projection: Q_0 t_k = 0 (t_k cocycles) and
Q_0 Q_1 t_k = 0 (since Q_0 Q_1 = -Q_1 Q_0 at order g^2; a consistency check of Q_1 on these derivative letters), and
Q_0 of the controls Q_1 Q_0 u = 0.  usage: python3 d1_check.py OUT [--npts 24] [--workers W]  -> OUT/d1_check.json"""
import argparse, json, os, pickle, time
import numpy as np
import fullpar
import build_cand as bc
import q1apply
from fortuity.bmn.modp import P
try:
    from fortuity.full.words import Q_multi
except ImportError:
    from fortuity.full.words import Q_multi

log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)
ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--npts', type=int, default=24)
ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '8')))
a = ap.parse_args()
fullpar.CSEC = -1
T0 = time.time()
rep = {}
for name, (words, C) in (('t', (pickle.load(open(os.path.join(a.out, 'twords.pkl'), 'rb')),
                                np.load(os.path.join(a.out, 'tvec.npz'))['coef'])),
                         ('Q1t_and_ctrl', (lambda d: (d['words'], d['coef']))(
                             pickle.load(open(os.path.join(a.out, 'vecs_Q1.pkl'), 'rb'))))):
    img, V = q1apply.apply(words, C, workers=a.workers, chunk=2000, op=Q_multi)
    vals = bc.eval_combo(img, V, 3, a.npts, 4242, False, a.workers)
    nz = {str(d): [int(np.count_nonzero(x[k])) for k in range(x.shape[0])] for d, x in vals.items()}
    tot = sum(sum(v) for v in nz.values())
    rep[name] = dict(rows=int(C.shape[0]), words=len(words), q0_image_words=len(img), q0_nonzero_by_degree=nz,
                     q0_total_nonzero=int(tot))
    log(f"Q_0 {name}: {len(img)} image words with nonzero coefficient; nonzero values at fresh points {tot} (must be 0) "
        f"({time.time()-T0:.0f}s)")
rep['ok'] = all(v['q0_total_nonzero'] == 0 for v in rep.values() if isinstance(v, dict))
rep['time'] = round(time.time() - T0)
json.dump(rep, open(os.path.join(a.out, 'd1_check.json'), 'w'), indent=1)
log(json.dumps(rep))
