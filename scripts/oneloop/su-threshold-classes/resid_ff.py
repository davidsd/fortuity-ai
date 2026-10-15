"""Residual sketches of V rows modulo the row space of B, and joint ranks over several exactness tests.

For a prep (B = Q_0-exact functionals of the target, V = [controls; test rows] at the same functionals):
N = right null space of B (nul x c), S = G N for a random G (m x nul), P = V S^T (nV x m).  A row of V lies in the row
space of B iff its row of P vanishes (with probability ~1 - m/p per dimension, for m >= the residual rank), so the
controls must give zero rows, and the sequential rank increments of the test rows of P equal those of cpu_rank.py.
P rows of the same K test vectors in different preps can be put side by side: the rank of [P_1 | P_2 | ...] is the
rank of t -> ([op_1 t], [op_2 t], ...) on the span of the classes of the t_k (joint lowest-weight / d_1 tests).

usage: python3 resid_ff.py PREP [--m 48] [--seed 5]       -> PREP/resid_sketch.npz, PREP/resid.json
       python3 resid_ff.py --joint OUT.json PREP1 PREP2 ...  (uses the stored sketches; test rows only)"""
import json, os, sys, time
import numpy as np
from fortuity.bmn.modp import P as PR, matmul_mod
from fortuity.hwcoh import ffblas, ffla

log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)


def seq_increments(Pm, start=0):
    """rank increments of rows start.. of Pm after rows 0..start-1 (small matrices)."""
    out = []
    base = int(ffla.rank(Pm[:start])) if start else 0
    for r in range(start, Pm.shape[0]):
        rk = int(ffla.rank(Pm[:r + 1]))
        out.append(rk - base)
        base = rk
    return out


def sketch(prep, m=48, seed=5, blk=4096):
    T0 = time.time()
    meta = json.load(open(os.path.join(prep, 'meta.json')))
    nB, nV, c, nc = meta['nB'], meta['nV'], meta['ncols'], meta['n_ctrl']
    ncart = int(meta.get('ncart', 0) or 0)
    B = np.memmap(os.path.join(prep, 'B.i32'), dtype=np.int32, mode='r', shape=(max(nB, 1), c + ncart))[:, :c]
    V = np.fromfile(os.path.join(prep, 'V.i32'), dtype=np.int32).reshape(nV, c).astype(np.int64) % PR
    if ffblas.HAVE_FFBLAS:
        SB = ffblas.StreamBasis(c)
        for r0 in range(0, nB, blk):
            SB.add(np.asarray(B[r0:r0 + blk], dtype=np.int64))
            if (r0 // blk) % 10 == 0:
                log(f"  B rows {min(nB, r0 + blk)}/{nB}: rank {SB.rank} ({time.time()-T0:.0f}s)")
        rB = SB.rank
        N = SB.null_space()
        del SB
        mm = ffblas.matmul
    else:                                   # small preps without the MKL kernel
        N = ffla.nullspace_right(np.asarray(B[:nB], dtype=np.int64) % PR)
        rB = c - N.shape[0]
        mm = matmul_mod
    log(f"rank B = {rB}, null space {N.shape[0]} x {c} ({time.time()-T0:.0f}s)")
    G = np.random.default_rng(seed).integers(0, PR, (m, N.shape[0]), dtype=np.int64)
    S = np.asarray(mm(G, N % PR), dtype=np.int64) % PR
    del N
    Pm = np.asarray(mm(V, np.ascontiguousarray(S.T)), dtype=np.int64) % PR
    ctrl_nz = int(np.count_nonzero(Pm[:nc]))
    inc = seq_increments(Pm, nc)
    res = dict(prep=prep, case=meta.get('case'), rank_B=int(rB), ncols=c, n_ctrl=nc, n_vec=nV - nc, m=m, seed=seed,
               ctrl_residual_nonzero=ctrl_nz, controls_exact=bool(ctrl_nz == 0), increments=inc,
               rank_residuals=int(ffla.rank(Pm[nc:])) if nV > nc else 0, time=round(time.time() - T0))
    np.savez(os.path.join(prep, 'resid_sketch.npz'), P=Pm, n_ctrl=nc)
    json.dump(res, open(os.path.join(prep, 'resid.json'), 'w'), indent=1)
    log(json.dumps(res))
    return res


def joint(out, preps):
    parts, info = [], []
    for d in preps:
        z = np.load(os.path.join(d, 'resid_sketch.npz'))
        Pm, nc = z['P'], int(z['n_ctrl'])
        assert not np.count_nonzero(Pm[:nc]), f"controls not exact in {d}"
        parts.append(Pm[nc:])
        info.append(dict(prep=d, rank=int(ffla.rank(Pm[nc:]))))
    assert len({p.shape[0] for p in parts}) == 1, "different numbers of test rows"
    Sx = np.hstack(parts)
    res = dict(preps=info, n_vec=int(Sx.shape[0]), joint_rank=int(ffla.rank(Sx)))
    # kernel of the joint map on the K test vectors (coefficient vectors c with c^T [P_1 | ...] = 0)
    ker = ffla.nullspace_right(np.ascontiguousarray(Sx.T) % PR)
    res['joint_kernel_dim'] = int(ker.shape[0])
    np.save(out.replace('.json', '_kernel.npy'), ker)
    json.dump(res, open(out, 'w'), indent=1)
    log(json.dumps(res))
    return res


if __name__ == '__main__':
    if sys.argv[1] == '--joint':
        joint(sys.argv[2], sys.argv[3:])
    else:
        args = sys.argv[1:]
        m = int(args[args.index('--m') + 1]) if '--m' in args else 48
        seed = int(args[args.index('--seed') + 1]) if '--seed' in args else 5
        sketch(args[0], m, seed)
