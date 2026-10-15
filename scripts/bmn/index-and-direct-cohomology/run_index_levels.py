"""Driver: full index + level-by-level graviton counting in subprocesses (memory returned between chunks).
usage: run_index_levels.py N jmax [nchunks] [parallel] [cap_gb]"""
import sys, os, pickle, subprocess, time, glob
import numpy as np
from fortuity.bmn.fullindex import full_index, su3_multiplicities
from fortuity.bmn.gravindex import GravitonCounter
N, jmax = int(sys.argv[1]), int(sys.argv[2])
nch = int(sys.argv[3]) if len(sys.argv) > 3 else 4
par = int(sys.argv[4]) if len(sys.argv) > 4 else 1
cap = float(sys.argv[5]) if len(sys.argv) > 5 else 8.0
fi = f"results/fullindex_N{N}_j{jmax}.pkl"
if os.path.exists(fi):
    Z, A = pickle.load(open(fi, "rb"))
else:
    Z, A = full_index(N, jmax, verbose=True); pickle.dump((Z, A), open(fi, "wb"))
print("full index unrefined:", [int(x) for x in Z.sum(axis=(1, 2))], flush=True)
G = GravitonCounter(N, jmax, verbose=False, npts=4096, k=(int(os.environ['GRAV_K']) if os.environ.get('GRAV_K') else None))   # only for k/npts/seed bookkeeping
ckf = f'results/ckpt_N{N}.pkl'
if os.path.exists(ckf):
    ck = pickle.load(open(ckf, 'rb')); start = ck['qsum_done'] + 1
    std, dims = ck['std'], ck['dims']
    print(f"resuming after j={2*ck['qsum_done']}", flush=True)
else:
    start = 1; std, dims = {(0, 0, 0, 0): {()}}, {(0, 0, 0, 0): 1}
    pickle.dump(dict(N=N, k=G.k, npts=G.npts, seed=G.seed, qsum_done=0, std=std, dims=dims), open(ckf, 'wb'))
env = dict(os.environ, PYTHONPATH='.', OPENBLAS_NUM_THREADS=str(max(1, 4 // par)), OMP_NUM_THREADS=str(max(1, 4 // par)))
for qsum in range(start, jmax // 2 + 1):
    t0 = time.time()
    # partial files from an interrupted run of this level are reused (same nchunks); remove stale ones otherwise
    have = {cid for cid in range(nch) if os.path.exists(f'results/partial_N{N}_q{qsum}_c{cid}.pkl')}
    for f in glob.glob(f'results/partial_N{N}_q{qsum}_c*.pkl'):
        cid = int(f.rsplit('_c', 1)[1].split('.')[0])
        if cid >= nch:
            os.remove(f)
    if have:
        print(f"level j={2*qsum}: reusing finished chunks {sorted(have)}", flush=True)
    procs = []
    for cid in range(nch):
        if cid in have:
            continue
        while len([p for p in procs if p.poll() is None]) >= par:
            time.sleep(2)
        procs.append(subprocess.Popen([sys.executable, 'scripts/run_level.py', str(N), str(jmax), str(qsum), str(cid), str(nch), str(cap)], env=env))
    for p in procs:
        p.wait()
    if any(p.returncode != 0 for p in procs):
        print(f"level j={2*qsum}: a chunk failed (return codes {[p.returncode for p in procs]}); stopping", flush=True)
        sys.exit(1)
    for cid in range(nch):
        part = pickle.load(open(f'results/partial_N{N}_q{qsum}_c{cid}.pkl', 'rb'))
        for s, (d, st) in part.items():
            dims[s] = d; std[s] = st
    nst = sum(len(v) for s, v in std.items() if 2 * sum(s[1:]) == 2 * qsum)
    print(f"[grav N={N}] j={2*qsum:3d} done, #std at this level={nst}, level time {time.time()-t0:.0f}s", flush=True)
    tmp = ckf + '.tmp'
    pickle.dump(dict(N=N, k=G.k, npts=G.npts, seed=G.seed, qsum_done=qsum, std=std, dims=dims), open(tmp, 'wb'))
    os.replace(tmp, ckf)
    # report non-graviton content so far
    Zg = np.zeros_like(Z)
    for sec, d in dims.items():
        j = 2 * sum(sec[1:])
        if j <= jmax:
            Zg[j, sec[1] - sec[3] + A, sec[3] - sec[2] + A] += (-1) ** sec[0] * d
    D = Z - Zg
    m = {(int(p), int(q)): int(c) for (p, q), c in su3_multiplicities(D[2 * qsum], A).items()}
    print(f"j={2*qsum}: Z - Z_grav = ", {f"[{p},{q}]": c for (p, q), c in sorted(m.items())} if m else 0, flush=True)
    pickle.dump(dict(N=N, jmax=2 * qsum, Z=Z, Zg=Zg, A=A, dims=dims), open(f"results/index_N{N}_j{2*qsum}.pkl", "wb"))
print("done", flush=True)
