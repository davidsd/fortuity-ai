"""Is an abelian (Coulomb-branch) polynomial the Cartan image of a Q_0-cocycle of T_2?  (R-007 order 2, sec. 7.)

f = eps(t) for a cocycle t  <=>  the row [0 | e_f] (zero on the Q_0-image functionals, Cartan values of f on the
Cartan functionals) lies in the row space of the outbound prep A = [B_out | E].  So rank([A; [0|e_f]]) - rank(A) is
the dimension of span{f} modulo eps(cocycles); a full increment proves that no combination of the f is a Coulomb image.
(One-sided: a cocycle image always lies in the row space, whatever the evaluation points.)

eval:  rebuilds the Cartan frames of the prep exactly (letters of the hw-cache words, seeds, points; checked by
       recomputing the stored Cartan block of one row group), evaluates the polynomials, writes PREP/VC.i32 + VC.json.
rank:  StreamBasis over the rows of B.i32 (with the Cartan columns), then the VC rows one by one.
usage: python3 membership.py eval PREP CACHE LW.pkl CARTAN     |     python3 membership.py rank PREP"""
import json, math, os, pickle, sys, time
import numpy as np
import fullpar
from fullpar import frame_k, Frame, nferm, c_frame, eval_csec
from fortuity.bmn.modp import P, matmul_mod

log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)


def abelian_eval(fr, f, d):
    """Values (npts x C(k,d), point-major) of the abelian polynomial f (monomials of species letters (i, a, b, S),
    i = 1, 2 -> diagonal entries 0, 1 of the Cartan-valued letter matrices) restricted to its fermion degree d."""
    out = np.zeros((fr.P, fr.ncomp[d]), dtype=np.int64)
    for mono, c in f.items():
        if sum(1 for L in mono if L[1:] and len(L[3]) % 2 == 0) != d:
            continue
        v, dv = np.ones((fr.P, 1), dtype=np.int64), 0
        for L in mono:
            dl, M = fr.L[L[1:]]
            x = np.ascontiguousarray(M[:, :, L[0] - 1, L[0] - 1]) % P     # (npts, k) odd, (npts, 1) even
            v, dv = fr.scal_mul(v, dv, x, dl)
        assert dv == d
        out = (out + c * v) % P
    return out.reshape(-1)


def do_eval(prep, cache, lwfile, cartan):
    fullpar.CSEC = -1
    t0 = time.time()
    meta = json.load(open(os.path.join(prep, 'meta.json')))
    ncols, ncart, nB = meta['ncols'], meta['ncart'], meta['nB']
    log(f"loading {cache}")
    ws, groups, ms, stats = pickle.load(open(cache, 'rb'))
    letters_tp = sorted({L for w in ws for n in w for L in n})
    degs = sorted({nferm(ws[int(g['parts'][0][0][0])]) for g in groups})
    cchunks, ccol = [], ncols
    for d in degs:
        k = frame_k(3, d)
        cpp = math.comb(k, d) if d else 1
        npts = max(1, math.ceil((cartan + fullpar.MARGIN) / cpp))
        cchunks.append((d, npts, k, meta['seed'] + 104729 + 31 * d, ccol))
        ccol += npts * cpp
    assert ccol - ncols == ncart, (ccol - ncols, ncart)
    log(f"{len(letters_tp)} letters, Cartan chunks {[(d, n, k) for d, n, k, _, _ in cchunks]} ({time.time()-t0:.0f}s)")
    # check the reconstruction: recompute the stored Cartan block of the smallest row group of every degree
    B = np.memmap(os.path.join(prep, 'B.i32'), dtype=np.int32, mode='r', shape=(nB, ncols + ncart))
    goff, r0 = [], 0
    for g in groups:
        goff.append(r0)
        r0 += g['k']
    mism = 0
    for dg in degs:
        cand = [i for i in range(len(groups)) if nferm(ws[int(groups[i]['parts'][0][0][0])]) == dg]
        gi = min(cand, key=lambda i: groups[i]['rows'])
        g, g0 = groups[gi], goff[gi]
        d, npts, k, sd, c0 = next(c for c in cchunks if c[0] == dg)
        fr = Frame(3, npts, k, [d], letters_tp, seed=sd, cartan=True)
        Z = None
        for widx, H, R in g['parts']:
            Xc = eval_csec(fr, [ws[int(j)] for j in widx], dtype=np.int64)
            part = matmul_mod(R % P, matmul_mod(H.astype(np.int64) % P, Xc))
            Z = part if Z is None else (Z + part) % P
        stored = np.asarray(B[g0:g0 + g['k'], c0:c0 + npts * fr.cpp], dtype=np.int64)
        mm_ = int(np.count_nonzero((Z - stored) % P))
        mism += mm_
        log(f"reconstruction check on group {gi} ({g['k']} rows, degree {d}): {mm_} mismatches of {Z.size} (must be 0)")
    assert mism == 0, "Cartan frames not reproduced"
    if os.environ.get('MEMBERSHIP_CHECK_ONLY'):
        return
    del ws, groups
    LW = pickle.load(open(lwfile, 'rb'))
    space = LW['space']
    missing = [L for L in LW['letters'] if L not in set(letters_tp)]
    assert not missing, f"letters of the polynomials missing from the frames: {missing[:5]}"
    VC = np.zeros((len(space), ncols + ncart), dtype=np.int64)
    for d, npts, k, sd, c0 in cchunks:
        fr = Frame(3, npts, k, [d], letters_tp, seed=sd, cartan=True)
        frc = c_frame(fr)
        for r, f in enumerate(space):
            # the prep's C-odd projection: X + CSEC * C(X), CSEC = -1
            VC[r, c0:c0 + npts * fr.cpp] = (abelian_eval(fr, f, d) - abelian_eval(frc, f, d)) % P
        log(f"  degree {d}: {npts} points x {fr.cpp} ({time.time()-t0:.0f}s)")
    nz = [int(np.count_nonzero(VC[r])) for r in range(len(space))]
    log(f"evaluated {len(space)} polynomials; nonzero Cartan values per row: min {min(nz)}, max {max(nz)}")
    VC.astype(np.int32).tofile(os.path.join(prep, 'VC.i32'))
    json.dump(dict(nVC=len(space), width=ncols + ncart, lw=lwfile, cartan=cartan, letters=len(letters_tp),
                   reconstruction_mismatches=mism), open(os.path.join(prep, 'VC.json'), 'w'), indent=1)


def do_rank(prep):
    from fortuity.hwcoh import ffblas
    assert ffblas.HAVE_FFBLAS
    t0 = time.time()
    meta = json.load(open(os.path.join(prep, 'meta.json')))
    vmeta = json.load(open(os.path.join(prep, 'VC.json')))
    nB, w = meta['nB'], meta['ncols'] + meta['ncart']
    assert vmeta['width'] == w
    B = np.memmap(os.path.join(prep, 'B.i32'), dtype=np.int32, mode='r', shape=(nB, w))
    VC = np.fromfile(os.path.join(prep, 'VC.i32'), dtype=np.int32).reshape(vmeta['nVC'], w)
    SB = ffblas.StreamBasis(w)
    for r0 in range(0, nB, 4096):
        SB.add(np.asarray(B[r0:r0 + 4096], dtype=np.int64))
        if (r0 // 4096) % 5 == 0:
            log(f"  rows {min(nB, r0 + 4096)}/{nB}: rank {SB.rank} ({time.time()-t0:.0f}s)")
    rA = SB.rank
    inc = []
    for r in range(VC.shape[0]):
        b = SB.rank
        SB.add(np.asarray(VC[r:r + 1], dtype=np.int64))
        inc.append(int(SB.rank - b))
    res = dict(rank_A=int(rA), n_poly=int(VC.shape[0]), increments=inc, total_increment=int(sum(inc)),
               time=round(time.time() - t0))
    log(json.dumps(res))
    json.dump(res, open(os.path.join(prep, 'membership_rank.json'), 'w'), indent=1)


if __name__ == '__main__':
    if sys.argv[1] == 'eval':
        do_eval(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    else:
        do_rank(sys.argv[2])
