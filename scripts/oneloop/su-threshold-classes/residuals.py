"""Joint rank of several exactness tests on the same K test vectors: for each prep (fullpar --prep-dir with V = [ctrl;
op_1 t_1 .. op_1 t_K]) reduce the V rows modulo the row space of B (the Q_0-exact functionals), check that the controls
reduce to zero, and stack the residuals of all preps side by side.  rank of the stack = rank of the combined map
t -> ([op_1 t], [op_2 t], ...) on the span of the classes of the t_k.  (CPU numpy; for small sectors.)
usage: python3 residuals.py PREP1 [PREP2 ...]"""
import json, os, sys
import numpy as np
from gpu_rank import ModBasis, P
from fortuity.hwcoh import ffla

res_all, info = [], []
for d in sys.argv[1:]:
    meta = json.load(open(os.path.join(d, 'meta.json')))
    nB, nV, c, nc = meta['nB'], meta['nV'], meta['ncols'], meta['n_ctrl']
    ncart = int(meta.get('ncart', 0) or 0)
    B = np.memmap(os.path.join(d, 'B.i32'), dtype=np.int32, mode='r', shape=(max(nB, 1), c + ncart))[:, :c]
    V = np.asarray(np.memmap(os.path.join(d, 'V.i32'), dtype=np.int32, mode='r', shape=(max(nV, 1), c)), dtype=np.int64)
    basis = ModBasis(np, c)
    for r0 in range(0, nB, 2048):
        basis.add(np.asarray(B[r0:r0 + 2048], dtype=np.int64) % P)
    R = basis.reduce(V.copy() % P)
    ctrl_nz = int(np.count_nonzero(R[:nc]))
    res_all.append(R[nc:])
    rk = int(ffla.rank(R[nc:])) if nV > nc else 0
    info.append(dict(prep=d, rank_B=basis.rank, n_ctrl=nc, ctrl_residual_nonzero=ctrl_nz, n_vec=nV - nc, rank_residuals=rk))
    print(json.dumps(info[-1]), flush=True)
    assert ctrl_nz == 0, f"controls not exact in {d}"
S = np.hstack(res_all)
print(json.dumps(dict(joint_rank=int(ffla.rank(S)), n_vec=S.shape[0], separate=[i['rank_residuals'] for i in info])))
