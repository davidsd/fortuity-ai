"""Per-sector dimensions of the diagonal (Coulomb-branch) graviton space, from the shared GravitonCounter
(fortuity.bmn.gravindex, read-only).  usage: python3 diag_counts.py N JMAX  -> runs/cache/diag_dims_N{N}_j{JMAX}.pkl"""
import sys, os, time, pickle
from fortuity.bmn.gravindex import GravitonCounter
N, jmax = int(sys.argv[1]), int(sys.argv[2])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', 'cache', f'diag_dims_N{N}_j{jmax}.pkl')
t0 = time.time()
G = GravitonCounter(N, jmax, verbose=True, npts=4096)
print('npts', G.npts, 'k', G.k, 'mesons', len(G.mes), flush=True)
dims = G.run(checkpoint=out + '.ckpt')
pickle.dump(dict(N=N, jmax=jmax, dims=dims), open(out, 'wb'))
print(f'done {len(dims)} sectors, {time.time()-t0:.0f}s -> {out}', flush=True)
