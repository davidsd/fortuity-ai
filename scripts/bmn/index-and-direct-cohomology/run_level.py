"""Worker: compute graviton dims for one level j (a chunk of its sectors) from the checkpoint, write a partial file.
usage: run_level.py N jmax qsum chunk_id nchunks cap_gb"""
import sys, pickle, os, time
import numpy as np
from fortuity.bmn.gravindex import GravitonCounter
from fortuity.bmn.guard import MemGuard
N, jmax, qsum, cid, nch, cap = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6])
guard = MemGuard(cap, log=False)
G = GravitonCounter(N, jmax, verbose=False, npts=4096, k=(int(os.environ['GRAV_K']) if os.environ.get('GRAV_K') else None))
ckf = f'results/ckpt_N{N}.pkl'
if os.path.exists(ckf):
    ck = pickle.load(open(ckf, 'rb'))
    assert ck['N'] == N and ck['k'] == G.k and ck['npts'] == G.npts and ck['seed'] == G.seed
    G.std, G.dims = ck['std'], ck['dims']
    assert ck['qsum_done'] == qsum - 1, (ck['qsum_done'], qsum)
else:
    assert qsum == 1
secs = [(twoJ, q1, q2, qsum - q1 - q2) for q1 in range(qsum + 1) for q2 in range(qsum - q1 + 1) for twoJ in range(0, qsum + 1)]
# sort by expected size (largest first) then deal round-robin to chunks for balance
secs.sort(key=lambda s: -sum(len(v) for k, v in G.std.items() if 2*sum(k[1:]) == 2*qsum - 4 and all(k[i] <= s[i] for i in range(4))))
mine = [s for i, s in enumerate(secs) if i % nch == cid]
out = {}
t0 = time.time()
for s in mine:
    if s in G.dims:
        continue
    G.count_sector(s)
    out[s] = (G.dims[s], G.std[s])
    guard.check(f'sector {s}')
pickle.dump(out, open(f'results/partial_N{N}_q{qsum}_c{cid}.pkl', 'wb'))
print(f'chunk {cid}/{nch} level j={2*qsum}: {len(mine)} sectors in {time.time()-t0:.0f}s, peak rss {guard.peak:.2f} GB', flush=True)
