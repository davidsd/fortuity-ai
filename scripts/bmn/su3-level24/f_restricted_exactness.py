"""Does exactness of eq. (4.8) of arXiv:2312.16443 require f in the Q-preimage?
Appendix A of the paper: 'One of our working assumptions is that the maximal number of f's appearing before the
Q-action is the same as that of the trace relation.'  (4.8) has no f.  We test membership of (4.8) in
Q(V_up restricted to n_f <= k), k = 0, 1, 2, in the Weyl-symmetric subspace.
usage: python3 f_restricted_exactness.py <prime> <seed> <npts> <out.json>"""
import sys, json, time, copy, numpy as np
from collections import defaultdict
from fortuity.bmn.symdirect import SymDirect
from fortuity.bmn.words import Q_multi
from fortuity.bmn.evaluate import Points
from fortuity.bmn.modp import RREFBasis
from gens_util import load_O
p, seed, npts, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
sec, up = (3, 4, 4, 4), (4, 4, 4, 4)
t0 = time.time(); log = lambda *a: print('[%6.0fs]' % (time.time() - t0), *a, flush=True)
D = SymDirect(N=3, gen_nmax=3, seed=seed, verbose=False); D.guard.log = False
k, degs, cpp = D._sector_k(sec)
pts = Points(3, npts, k, degs, seed=seed, p=p)
targets = {}
for name, path in (('(4.8) = -6 O_6', 'O_cckll.json'), ('O_1^{(0,3)}', 'O1_control.json'), ('R_24^{(0,3)}', 'R24.json')):
    e, _ = load_O(path); targets[name] = pts.expr_vector(e) % p
sw = D.symwords(up)
nf = [max(sum(1 for w in m for l in w if l == 6) for m in e) for e in sw]
cnt = {k2: sum(1 for x in nf if x == k2) for k2 in (0, 1, 2)}
log('symmetrised words of (4;4,4,4) by number of f:', cnt, '(contents (4,4,0), (5,2,1), (6,0,2))')
rows = {}
for i, e in enumerate(sw):
    q = defaultdict(int)
    for m, c in e.items():
        for m2, c2 in Q_multi(m).items():
            q[m2] += c * c2
    q = {m: c for m, c in q.items() if c}
    rows[i] = pts.expr_vector(q) if q else np.zeros(pts.ncols, dtype=np.int64)
log('Q-images evaluated')
res = {'prime': p, 'counts_by_nf': cnt, 'levels': {}}
basis = RREFBasis(pts.ncols, p)
for kmax in (0, 1, 2):
    idx = [i for i in range(len(sw)) if nf[i] == kmax]
    for s0 in range(0, len(idx), 256):
        basis.add(np.array([rows[i] for i in idx[s0:s0 + 256]], dtype=np.int64))
    verdict = {}
    for name, v in targets.items():
        verdict[name] = bool(not np.any(basis.reduce(v[None, :])[0]))
    log('preimages with n_f <= %d: rank of Q-images = %d ; in span: %s' % (kmax, basis.rank, verdict))
    res['levels'][str(kmax)] = {'rank': int(basis.rank), 'in_span': verdict}
json.dump(res, open(out, 'w'), indent=1); log('saved', out)
