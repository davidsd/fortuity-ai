"""Index route: full BMN index (Haar integral) minus graviton index (eigenvalue staircase).
usage: run_index.py N jmax [mem_cap_gb]   (resumable via results/ckpt_N{N}.pkl)"""
import sys, time, pickle, os
import numpy as np
from fortuity.bmn.fullindex import full_index, su3_multiplicities
from fortuity.bmn.gravindex import GravitonCounter
from fortuity.bmn.guard import MemGuard

N = int(sys.argv[1]); jmax = int(sys.argv[2])
cap = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
guard = MemGuard(cap, log=True)
fi = f"results/fullindex_N{N}_j{jmax}.pkl"
if os.path.exists(fi):
    Z, A = pickle.load(open(fi, "rb"))
    print("loaded full index", fi, flush=True)
else:
    t0 = time.time()
    Z, A = full_index(N, jmax, verbose=True)
    pickle.dump((Z, A), open(fi, "wb"))
    print(f"full index done in {time.time()-t0:.1f}s", flush=True)
guard.check("after full index")
print("full index unrefined:", list(Z.sum(axis=(1, 2))), flush=True)
t0 = time.time()
G = GravitonCounter(N, jmax, verbose=True, npts=4096)
print("npts", G.npts, "k", G.k, "mesons", len(G.mes), flush=True)
dims = G.run(checkpoint=f"results/ckpt_N{N}.pkl", guard=guard)
Zg = G.index_array(A)
print(f"graviton counting done in {time.time()-t0:.1f}s", flush=True)
D = Z - Zg
print("Z - Z_grav unrefined:", list(D.sum(axis=(1, 2))), flush=True)
out = {}
for j in range(0, jmax + 1, 2):
    m = {(int(p), int(q)): int(c) for (p, q), c in su3_multiplicities(D[j], A).items()}
    if m: print(f"j={j}: ", {f"[{p},{q}]": c for (p, q), c in sorted(m.items())}, flush=True)
    out[j] = m
pickle.dump(dict(N=N, jmax=jmax, Z=Z, Zg=Zg, A=A, dims=dims, nongrav=out), open(f"results/index_N{N}_j{jmax}.pkl", "wb"))
print("saved", flush=True)
