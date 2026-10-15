"""Search for a sparse f-free Q-preimage of eq. (4.8) of arXiv:2312.16443 in the Weyl-symmetric subspace.
Stage A (one prime): evaluate Q-images of all f-free symmetrised words of (4;4,4,4); for several orderings,
greedy-select independent words until (4.8) is in the span; solve; report support sizes; save the best selection."""
import sys, json, time, copy, pickle, numpy as np
from collections import defaultdict
from fortuity.bmn.symdirect import SymDirect
from fortuity.bmn.words import Q_multi
from fortuity.bmn.evaluate import Points
from fortuity.bmn.modp import RREFBasis
from gens_util import load_O
p, seed, npts = 67108859, 20261010, 96
sec, up = (3, 4, 4, 4), (4, 4, 4, 4)
t0 = time.time(); log = lambda *a: print('[%6.0fs]' % (time.time() - t0), *a, flush=True)
D = SymDirect(N=3, gen_nmax=3, seed=seed, verbose=False); D.guard.log = False
k, degs, cpp = D._sector_k(sec)
pts = Points(3, npts, k, degs, seed=seed, p=p)
O, _ = load_O('O_cckll.json'); vO = pts.expr_vector(O) % p
sw_all = D.symwords(up)
ff = [i for i, e in enumerate(sw_all) if max(sum(1 for w in m for l in w if l == 6) for m in e) == 0]
log('f-free symmetrised words:', len(ff))
def rep(e):   # representative multi-trace (smallest key) and its trace-length structure
    m = min(e); return m, tuple(sorted(len(w) for w in m))
info = {i: rep(sw_all[i]) for i in ff}
structs = defaultdict(int)
for i in ff: structs[info[i][1]] += 1
log('trace structures (lengths): ', dict(sorted(structs.items())))
rows = {}
for i in ff:
    q = defaultdict(int)
    for m, c in sw_all[i].items():
        for m2, c2 in Q_multi(m).items():
            q[m2] += c * c2
    q = {m: c for m, c in q.items() if c}
    rows[i] = pts.expr_vector(q) % p if q else np.zeros(pts.ncols, dtype=np.int64)
log('Q-images evaluated')
def solve(order, label):
    basis = RREFBasis(pts.ncols, p); S = []
    for s0 in range(0, len(order), 128):
        chunk = order[s0:s0 + 128]
        ind = basis.add(np.array([rows[i] for i in chunk], dtype=np.int64))
        S += [i for i, f in zip(chunk, ind) if f]
        if not np.any(basis.reduce(vO[None, :])[0]):
            break
    M = np.array([rows[i] for i in S], dtype=np.int64)
    A = np.concatenate([M, np.eye(len(S), dtype=np.int64)], axis=1)
    full = RREFBasis(A.shape[1], p); full.add_batched(A)
    r = full.reduce(np.concatenate([vO, np.zeros(len(S), dtype=np.int64)])[None, :])[0]
    assert not np.any(r[:pts.ncols])
    y = (-r[pts.ncols:]) % p
    T = [(S[j], int(y[j])) for j in range(len(S)) if y[j]]
    st = defaultdict(int)
    for i, _ in T: st[info[i][1]] += 1
    log('%-42s independent words used %4d | support of the preimage %4d | structures %s' % (label, len(S), len(T), dict(sorted(st.items()))))
    return S, T
orders = {
 'enumeration order': list(ff),
 'more traces first': sorted(ff, key=lambda i: (-len(info[i][1]), info[i][1], i)),
 'fewer traces first': sorted(ff, key=lambda i: (len(info[i][1]), info[i][1], i)),
 'structures (2,2,4),(2,3,3),(2,2,2,2) first': sorted(ff, key=lambda i: (0 if info[i][1] in ((2, 2, 4), (2, 3, 3), (2, 2, 2, 2)) else 1, info[i][1], i)),
 '(2,2,4) then (2,3,3) then (2,2,2,2)': sorted(ff, key=lambda i: ({(2, 2, 4): 0, (2, 3, 3): 1, (2, 2, 2, 2): 2}.get(info[i][1], 3), i)),
 '(2,3,3) then (2,2,4) then (2,2,2,2)': sorted(ff, key=lambda i: ({(2, 3, 3): 0, (2, 2, 4): 1, (2, 2, 2, 2): 2}.get(info[i][1], 3), i)),
 '(2,2,2,2) then (2,2,4) then (2,3,3)': sorted(ff, key=lambda i: ({(2, 2, 2, 2): 0, (2, 2, 4): 1, (2, 3, 3): 2}.get(info[i][1], 3), i)),
}
best = None
for label, order in orders.items():
    S, T = solve(order, label)
    if best is None or len(T) < len(best[2]):
        best = (label, S, T)
log('best ordering:', best[0], '| support', len(best[2]))
pickle.dump({'label': best[0], 'S': best[1], 'T': best[2], 'prime': p}, open('runs/preimage_stageA.pkl', 'wb'))
json.dump({'label': best[0], 'support_indices': [i for i, _ in best[2]], 'coeffs_mod_p': [c for _, c in best[2]], 'prime': p,
           'words': {str(i): [[c, [list(w) for w in m]] for m, c in sw_all[i].items()] for i, _ in best[2]}}, open('runs/preimage_stageA.json', 'w'))
log('saved')
