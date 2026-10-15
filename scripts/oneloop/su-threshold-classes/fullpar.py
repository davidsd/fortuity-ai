"""T-031: parallel full-letter exactness test for the one-loop lifting of Q_0-cohomology classes.

Question: for Q_0-closed hw representatives v_k of a sector SEC (letter number Y), is Q_1 v_k Q_0-exact in the
target sector T = (2J1-1, 2J2-1, q; Y)?  The exact subspace of T is Q_0 of the SU(3)_R hw vectors (weight q) of the
preimage sector TP = (2J1, 2J2, q; Y-1).  Q_1 v_k is a hw vector of T, so exactness within hw vectors is equivalent
to exactness in the whole weight space.

Pipeline (same mathematics as oneloop_test.py stage 4, built for sizes ~1e5 as in fortuity.hwcoh.sectorpar):
  1. enumerate the words of TP (full letters: fullsector wordenum, contents at fixed Y via fastcontents);
  2. hw vectors per shape, two-stage (ker E1 inside E1-classes, then ker E2), a port of fortuity.hwcoh.hw._hw_shape
     to full letters (BLIS null spaces when libffblas is present);
  3. compress: words of different flavour-stripped contents are linearly independent at every N, so the hw rows of
     one stripped content span at most m_s dimensions at rank N, m_s = the exact hw multiplicity of that stripped
     content (Weyl integration + Weyl alternation).  Replace them by m_s + 8 random combinations (same span w.h.p.);
  4. sparse Q_0 of the TP words; target words in play = Q_0 images, Q_1 v_k, and the controls;
  5. functionals: per fermion degree d of T, m_T,d + margin evaluation points, in column chunks; words evaluated by
     forked workers into a memmap, B rows = H (Q X), V rows = (Q_1 v) X;
  6. streaming row-echelon basis (BLIS StreamBasis; numpy RREFBasis fallback): rank of B, then the control rows
     Q_1 Q_0 u (must not raise the rank), then each Q_1 v_k (a rank increase = class k lifts, provided the TP words
     span; a zero increase = Q_1 v_k exact, which needs no spanning assumption).
Optional spanning certificate: rank of the TP hw rows per stripped content at TP functionals = m_s.
"""
import math, os, sys, time, json, pickle, gc
import multiprocessing as mp
import numpy as np
from collections import defaultdict
from fastcontents import prime_cache
from fortuity.bmn.modp import P, RREFBasis, matmul_mod
from fortuity.hwcoh import ffblas
from fortuity.hwcoh.hw import nullspace_left, left_kernel, _mm, BIG_M2, STREAM_CHUNK
from fortuity.full.sector import words_of, hw_mult_by_degree, frame_k, contents_cached
from fortuity.full.sector import frame_k as _frame_k_capped, KMAX_TARGET


def frame_k(N, d, extra=2):
    """Grassmann generators for degree-d words.  fortuity.full.sector.frame_k caps k at 3 dim g, which is exact for
    BMN letters (only psi_1,2,3 are fermionic) but not for full letters: psi's together with lambdabar or derivative
    letters can carry more than 3 dim g fermions without vanishing, and with k < d they would evaluate to zero."""
    k = _frame_k_capped(N, d)
    if d and k < d + extra:
        k = d + extra
        while math.comb(k, d) < KMAX_TARGET:
            k += 1
    return k

from fortuity.full.words import Q_multi
from fortuity.full.hw import E_multi
from fortuity.full.evalg import Frame, nferm
from fortuity.full.letters import is_ferm
from fortuity.full.dims import content_dim, weyl_orbit_signed

log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)
MARGIN = 64
CSEC = 0     # charge-conjugation sector: 0 = no restriction, +1 = C-even, -1 = C-odd (set by --csec; SU(3) only)


def c_frame(fr):
    """The same random points with C applied: every letter matrix X -> -X^T (an automorphism of su(N)), so that
    u(C X) = (C u)(X) and u(X) + CSEC u(C X) = ((1 + CSEC C) u)(X) projects onto the CSEC sector."""
    import copy
    frc = copy.copy(fr)
    frc.L = {L: (dl, (-np.swapaxes(M, -1, -2)) % fr.p) for L, (dl, M) in fr.L.items()}
    return frc


def eval_csec(fr, words, dtype=np.int64):
    """Word values at the frame's functionals, projected onto the CSEC sector (plain values if CSEC == 0)."""
    X = fr.eval_words(words, dtype=np.int64)
    if CSEC:
        X = (X + CSEC * c_frame(fr).eval_words(words, dtype=np.int64)) % P
    return X.astype(dtype)


def csec_mult(m, t):
    """multiplicity in the CSEC sector from the total m and the trace t of C (t = #even - #odd)."""
    if not CSEC:
        return m
    v2 = m + CSEC * t
    assert v2 % 2 == 0 and v2 >= 0, (m, t)
    return v2 // 2


# ------------------------------------------------------------------ hw vectors (two-stage, full letters)
def _min_rot(t):
    return min(t[i:] + t[:i] for i in range(len(t)))


def _lab1(L):
    """E1 relabels theta index 1 -> 0; merge the doublet: (a, b, #(S & {0,1}), 2 in S)."""
    a, b, S = L
    return (a, b, (0 in S) + (1 in S), 2 in S)


def e1_class(multi):
    return tuple(sorted(_min_rot(tuple(_lab1(L) for L in w)) for w in multi))


def stripped_content(multi):
    """Flavour-stripped letter content (a, b, |S|) as a sorted tuple: independent sectors at every N."""
    return tuple(sorted((L[0], L[1], len(L[2])) for w in multi for L in w))


def shape(multi):
    return tuple(sorted(_min_rot(tuple((L[0], L[1], len(L[2])) for L in w)) for w in multi))


def hw_shape(words, idxs):
    """hw vectors (ker E1 and ker E2) of one shape block: (idxs, H int64) or None.  Port of hwcoh.hw._hw_shape."""
    pos = {j: a for a, j in enumerate(idxs)}
    classes = defaultdict(list)
    for j in idxs:
        classes[e1_class(words[j])].append(j)
    K1 = []
    for js in classes.values():
        imgs = [E_multi(words[j], 1) for j in js]
        cols = {}
        for d in imgs:
            for m in d:
                cols.setdefault(m, len(cols))
        if cols:
            M = np.zeros((len(js), len(cols)), dtype=np.int64)
            for r, d in enumerate(imgs):
                for m, c in d.items():
                    M[r, cols[m]] = c % P
            N1 = nullspace_left(M)
        else:
            N1 = np.eye(len(js), dtype=np.int64)
        if N1.shape[0]:
            K1.append((js, N1))
    if not K1:
        return None
    nk = sum(N1.shape[0] for _, N1 in K1)
    cols, blocksM = {}, []
    for js, N1 in K1:
        ent, loc = [], {}
        for a, j in enumerate(js):
            for m, c in E_multi(words[j], 2).items():
                cc = cols.setdefault(m, len(cols))
                b = loc.setdefault(cc, len(loc))
                ent.append((a, b, c))
        if loc:
            E2c = np.zeros((len(js), len(loc)), dtype=np.int64)
            for a, b, c in ent:
                E2c[a, b] = (E2c[a, b] + c) % P
            blocksM.append((_mm(N1 % P, E2c), np.fromiter(loc.keys(), dtype=np.int64)))
        else:
            blocksM.append((None, None))
    if cols and ffblas.HAVE_FFBLAS and nk * len(cols) > BIG_M2:
        T = len(cols)
        SB = ffblas.StreamBasis(nk)
        chunk = max(256, int(STREAM_CHUNK // max(nk, 1)))
        offs, r0 = [], 0
        for (js, N1), (prod, cidx) in zip(K1, blocksM):
            if prod is not None:
                order = np.argsort(cidx)
                offs.append((r0, cidx[order], prod[:, order]))
            r0 += N1.shape[0]
        for c0 in range(0, T, chunk):
            c1 = min(T, c0 + chunk)
            MT = np.zeros((c1 - c0, nk), dtype=np.int64)
            for rr, cs, pr in offs:
                a, b = np.searchsorted(cs, c0), np.searchsorted(cs, c1)
                if b > a:
                    MT[cs[a:b] - c0, rr:rr + pr.shape[0]] = pr[:, a:b].T
            SB.add(MT)
            del MT
        N2 = SB.null_space()
        del SB
    elif cols:
        M2 = np.zeros((nk, len(cols)), dtype=np.int64)
        r0 = 0
        for (js, N1), (prod, cidx) in zip(K1, blocksM):
            if prod is not None:
                M2[r0:r0 + N1.shape[0], cidx] = prod
            r0 += N1.shape[0]
        N2 = left_kernel(M2)
    else:
        N2 = np.eye(nk, dtype=np.int64)
    if not N2.shape[0]:
        return None
    H = np.zeros((N2.shape[0], len(idxs)), dtype=np.int64)
    r0 = 0
    for js, N1 in K1:
        cidx = [pos[j] for j in js]
        H[:, cidx] = (H[:, cidx] + _mm(N2[:, r0:r0 + N1.shape[0]] % P, N1 % P)) % P
        r0 += N1.shape[0]
    return np.array(idxs), H


def hw_mult_by_stripped(N, sec, Yv):
    """{stripped content: exact multiplicity of the SU(3)_R irrep with highest weight q = sec[2:]/2}."""
    out = defaultdict(int)
    for sgn, w in weyl_orbit_signed(sec[2:]):
        if min(w) < 0:
            continue
        s2 = tuple(sec[:2]) + tuple(w)
        prime_cache(s2, Yv)
        for c in contents_cached(s2, Yv):
            out[tuple(sorted((L[0], L[1], len(L[2])) for L in c))] += sgn * content_dim(N, c)
    return {k: v for k, v in out.items() if v}


def hw_trC_by_stripped(sec, Yv):
    """{stripped content: trace of C on the multiplicity space of the SU(3)_R irrep with highest weight sec[2:]/2}."""
    from twisted_index import trace_C
    out = defaultdict(int)
    for sgn, w in weyl_orbit_signed(sec[2:]):
        if min(w) < 0:
            continue
        s2 = tuple(sec[:2]) + tuple(w)
        prime_cache(s2, Yv)
        for c in contents_cached(s2, Yv):
            out[tuple(sorted((L[0], L[1], len(L[2])) for L in c))] += sgn * trace_C(c)
    return dict(out)


def hw_trC_by_degree(sec, Yv):
    from twisted_index import trace_C
    out = defaultdict(int)
    for sgn, w in weyl_orbit_signed(sec[2:]):
        if min(w) < 0:
            continue
        s2 = tuple(sec[:2]) + tuple(w)
        prime_cache(s2, Yv)
        for c in contents_cached(s2, Yv):
            out[sum(1 for L in c if is_ferm(L))] += sgn * trace_C(c)
    return dict(out)


def compressed_hw(N, sec, Yv, maxlen, seed=5, cache=None):
    """Words of (sec, Yv) and hw vectors grouped by stripped content s: groups = list of dicts with k = m_s + 8 and
    parts = [(word indices, H_b int32 (rows_b x n_b), R_b int64 (k x rows_b) random)], one part per shape.  The
    compressed rows of s are sum_b R_b H_b (never formed: B_s = sum_b R_b (H_b (Q X)) in the combine stage)."""
    if cache and os.path.exists(cache):
        log(f"hw: loading {cache}")
        return pickle.load(open(cache, 'rb'))
    t = time.time()
    prime_cache(sec, Yv)
    ws = words_of(sec, Yv, maxlen)
    log(f"hw: {sec} Y={Yv}: {len(ws)} words ({time.time()-t:.0f}s)")
    ms = hw_mult_by_stripped(N, sec, Yv)
    if CSEC:
        ts = hw_trC_by_stripped(sec, Yv)
        ms = {k: csec_mult(v, ts.get(k, 0)) for k, v in ms.items()}
        ms = {k: v for k, v in ms.items() if v}
    log(f"hw: exact multiplicities (C sector {CSEC}) of {len(ms)} stripped contents, total {sum(ms.values())} ({time.time()-t:.0f}s)")
    by_shape = defaultdict(list)
    for i, w in enumerate(ws):
        by_shape[shape(w)].append(i)
    by_s = defaultdict(list)
    for sh, idxs in by_shape.items():
        by_s[stripped_content(ws[idxs[0]])].append(idxs)
    rng = np.random.default_rng(seed)
    groups, stats = [], []
    t = time.time()
    done = 0
    for s, shapes in sorted(by_s.items(), key=lambda kv: -sum(len(x) for x in kv[1])):
        m = ms.get(s, 0)
        done += sum(len(x) for x in shapes)
        if m == 0:
            stats.append((s, 0, 0, 0))
            continue
        k = m + 8
        parts, rows = [], 0
        for idxs in shapes:
            r = hw_shape(ws, idxs)
            if r is None:
                continue
            widx, H = r
            parts.append((widx, H.astype(np.int32), rng.integers(0, P, (k, H.shape[0]), dtype=np.int64)))
            rows += H.shape[0]
        if rows:
            groups.append(dict(s=s, k=k, m=m, rows=rows, parts=parts))
        stats.append((s, rows, m, k if rows else 0))
        if len(stats) % 20 == 0:
            log(f"  hw: {len(stats)}/{len(by_s)} stripped contents, {done}/{len(ws)} words ({time.time()-t:.0f}s)")
    log(f"hw: {len(by_s)} stripped contents, {len(by_shape)} shapes; raw hw rows {sum(x[1] for x in stats)}, "
        f"exact multiplicity {sum(ms.values())}, compressed rows {sum(g['k'] for g in groups)} ({time.time()-t:.0f}s)")
    out = (ws, groups, ms, stats)
    if cache:
        pickle.dump(out, open(cache, 'wb'), protocol=4)
    return out


# ------------------------------------------------------------------ evaluation (forked workers, memmap)
_G = {}


def _eval_task(task):
    """Evaluate words order[i0:i1] (degree d) at one frame; write rows i0:i1 of the memmap."""
    fname, nrows, ncols, i0, i1, frargs = task
    words, order, letters = _G['words'], _G['order'], _G['letters']
    N, npts, k, d, seed = frargs
    fr = Frame(N, npts, k, [d], letters, seed=seed)
    X = np.memmap(fname, dtype=np.int32, mode='r+', shape=(nrows, ncols))
    for j0 in range(i0, i1, 20000):
        j1 = min(i1, j0 + 20000)
        X[j0:j1] = eval_csec(fr, [words[i] for i in order[j0:j1]], dtype=np.int32)
    del X
    return i1 - i0


class Rows:
    """Sparse rows over a word list: (row, word, coeff) entries sorted by row -> dense evaluation Y = rows @ X."""
    def __init__(self, n, rs, js, cs):
        o = np.argsort(rs, kind='stable')
        self.n, self.rs, self.js, self.cs = n, np.asarray(rs)[o], np.asarray(js)[o], np.asarray(cs, dtype=np.int64)[o] % P

    def apply(self, X, loc, zero_row, max_entries=50_000_000):
        """Y (n x w) int64 with Y[r] = sum_e cs[e] * X[loc[js[e]]] mod P (loc < 0 -> zero row)."""
        w = X.shape[1]
        Y = np.zeros((self.n, w), dtype=np.int64)
        step = max(1, max_entries // max(w, 1))
        for e0 in range(0, len(self.rs), step):
            e1 = min(len(self.rs), e0 + step)
            r = self.rs[e0:e1]
            li = loc[self.js[e0:e1]]
            keep = li >= 0
            if not keep.any():
                continue
            r, li, c = r[keep], li[keep], self.cs[e0:e1][keep]
            vals = X[li].astype(np.int64) * c[:, None] % P
            starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
            Y[r[starts]] = (Y[r[starts]] + np.add.reduceat(vals, starts, axis=0)) % P
        return Y


def run(N, SEC, Y, ML, vecs, ctrl, out, workers=8, workdir=None, xcap=20e9, factor=1.0, seed=41, hw_cache=None,
        span_check=False, prep_dir=None, case=None, cartan=0, col_slice=None):
    """vecs: list of {word: coeff} (Q_1 v_k, words in T); ctrl: list of {word: coeff} that must be exact."""
    T0 = time.time()
    res = dict(N=N, SEC=list(SEC), Y=Y, maxlen=ML, seed=seed, n_vecs=len(vecs), n_ctrl=len(ctrl), p=int(P), case=case)
    TGT = (SEC[0] - 1, SEC[1] - 1) + tuple(SEC[2:])
    TP = tuple(SEC)
    workdir = workdir or os.environ.get('TMPDIR') or '/tmp'
    # 1-3: preimage words, compressed hw rows
    ws, groups, ms, stats = compressed_hw(N, TP, Y - 1, ML, cache=hw_cache)
    res.update(tp_words=len(ws), tp_m=int(sum(ms.values())), tp_rows=int(sum(g['k'] for g in groups)),
               tp_raw_rows=int(sum(x[1] for x in stats)))
    # 4: sparse Q_0 of the preimage words that occur in hw rows; target word index
    t = time.time()
    tw, tidx = [], {}

    def tix(w):
        j = tidx.get(w)
        if j is None:
            j = tidx[w] = len(tw)
            tw.append(w)
        return j
    used = (np.unique(np.concatenate([p[0] for g in groups for p in g['parts']])) if groups
            else np.zeros(0, dtype=np.int64))
    qrs, qjs, qcs = [], [], []
    for a, i in enumerate(used):
        for m2, c in Q_multi(ws[int(i)]).items():
            if c:
                qrs.append(a); qjs.append(tix(m2)); qcs.append(c)
    upos = np.full(len(ws), -1, dtype=np.int64)
    upos[used] = np.arange(len(used))
    vrs, vjs, vcs = [], [], []
    for r, d in enumerate(list(ctrl) + list(vecs)):
        for w, c in d.items():
            if c % P:
                vrs.append(r); vjs.append(tix(w)); vcs.append(c)
    Vrows = Rows(len(ctrl) + len(vecs), vrs, vjs, vcs)
    del vrs, vjs, vcs
    # sparse Q_0 over the final target word list (float64 data: small integer coefficients)
    import scipy.sparse as sp
    Qcsr = sp.csr_matrix((np.asarray(qcs, dtype=np.float64), (np.asarray(qrs, dtype=np.int64),
                          np.asarray(qjs, dtype=np.int64))), shape=(len(used), len(tw)))
    del qrs, qjs, qcs
    # exactness of the float64 sparse product: |sum_j c_ij x_j| <= (max row l1 norm) * p < 2^53
    l1 = float(np.asarray(abs(Qcsr).sum(axis=1)).max()) if Qcsr.nnz else 0.0
    assert l1 * P < 2.0 ** 53, l1
    res['q0_nnz'], res['q0_max_row_l1'] = int(Qcsr.nnz), l1
    tdeg = np.array([nferm(w) for w in tw], dtype=np.int64)
    letters = sorted({L for w in tw for n in w for L in n})
    log(f"Q_0: {len(used)} preimage words -> {len(tw)} target words in play ({time.time()-t:.0f}s)")
    res['target_words'] = len(tw)
    # 5: functionals
    prime_cache(TGT, Y)
    mT = hw_mult_by_degree(N, TGT, Y)
    if CSEC:
        tT = hw_trC_by_degree(TGT, Y)
        mT = {d: csec_mult(v, tT.get(d, 0)) for d, v in mT.items()}
        mT = {d: v for d, v in mT.items() if v}
    res['csec'] = CSEC
    res['target_m_d'] = {str(k): int(v) for k, v in mT.items()}
    chunks, col = [], 0
    for d, md in sorted(mT.items()):
        k = frame_k(N, d)
        cpp = math.comb(k, d)
        if cpp == 0:
            continue
        npts = max(1, math.ceil(factor * (md + MARGIN) / cpp))
        nd = max(1, int((tdeg == d).sum()))
        per = max(1, int(xcap // (4 * nd * cpp)))
        for ci, p0 in enumerate(range(0, npts, per)):
            np_ = min(per, npts - p0)
            chunks.append((d, np_, k, seed + 7919 * d + 1000003 * ci, col))
            col += np_ * cpp
    ncols = col
    nB = sum(g['k'] for g in groups)
    # optional Coulomb test of the preimage sector: its rows evaluated directly (not through Q_0) at Cartan-valued
    # points; rank([B | E]) - rank(B) = rank of the Cartan evaluation on ker Q_0 = number of Coulomb-type classes
    # (exact vectors vanish on the Cartan), when the preimage sector is the one whose cohomology is wanted.
    cchunks, ccol = [], ncols
    if cartan:
        for d in sorted({g['deg'] for g in groups} if groups and 'deg' in groups[0] else
                        {nferm(ws[int(g['parts'][0][0][0])]) for g in groups}):
            k = frame_k(N, d)
            cpp = math.comb(k, d) if d else 1
            npts = max(1, math.ceil((cartan + MARGIN) / cpp))
            cchunks.append((d, npts, k, seed + 104729 + 31 * d, ccol))
            ccol += npts * cpp
    ncart = ccol - ncols
    res['ncart'] = ncart
    log(f"functionals {ncols} in {len(chunks)} chunks (target hw multiplicity {sum(mT.values())} {mT}); "
        f"B rows {nB}, V rows {Vrows.n}; workers {workers}")
    res['ncols'] = ncols
    # column slices (prep mode, one node each; merge_slices.py reassembles B and V): this run evaluates the chunks with
    # list index = i mod K, and slice 0 also the Cartan block.  Every slice computes the same word lists and chunk
    # layout (all deterministic), so the functionals are the ones of the unsliced run.
    if col_slice:
        si, sk = col_slice
        assert prep_dir, '--col-slice needs --prep-dir'
        mine = [gi for gi in range(len(chunks)) if gi % sk == si]
        cmine = list(range(len(cchunks))) if si == 0 else []
    else:
        mine, cmine = list(range(len(chunks))), list(range(len(cchunks)))
    width = lambda ch: ch[1] * (math.comb(ch[2], ch[0]) if ch[0] else 1)     # points x Grassmann components
    loc_off, lc = {}, 0
    for gi in mine:
        loc_off[gi] = lc
        lc += width(chunks[gi])
    ncols_loc = lc
    cloc_off = {}
    for cj in cmine:
        cloc_off[cj] = lc
        lc += width(cchunks[cj])
    ncart_loc = lc - ncols_loc
    if col_slice:
        res.update(col_slice=[si, sk], ncols_local=ncols_loc, ncart_local=ncart_loc,
                   slice_chunks=[[gi, chunks[gi][4], width(chunks[gi]), loc_off[gi]] for gi in mine],
                   slice_cchunks=[[cj, cchunks[cj][4], width(cchunks[cj]), cloc_off[cj]] for cj in cmine])
        log(f"column slice {si}/{sk}: {len(mine)} of {len(chunks)} chunks, {ncols_loc} of {ncols} columns"
            f"{', Cartan block' if cmine else ''}")
    tag = f"{os.getpid()}_{seed}"
    fB = os.path.join(workdir, f"B_{tag}.i32")
    fV = os.path.join(workdir, f"V_{tag}.i32")
    fX = os.path.join(workdir, f"X_{tag}.i32")
    with open(fB, 'wb') as fh:
        fh.truncate(4 * max(nB, 1) * (ncols_loc + ncart_loc))
    with open(fV, 'wb') as fh:
        fh.truncate(4 * max(Vrows.n, 1) * max(ncols_loc, 1))
    Bfull = np.memmap(fB, dtype=np.int32, mode='r+', shape=(max(nB, 1), ncols_loc + ncart_loc))
    Bm = Bfull[:, :ncols_loc]
    Vm = np.memmap(fV, dtype=np.int32, mode='r+', shape=(max(Vrows.n, 1), max(ncols_loc, 1)))
    # group row offsets; per stripped content one random matrix R_s = [R_b1 | R_b2 | ...] (k_s x rows_s), so that
    # B_s = R_s Z_s with Z_s = stacked H_b Y_b is a single GEMM per content (the per-shape accumulation was memory-bound)
    goff, r0 = [], 0
    for g in groups:
        goff.append(r0)
        r0 += g['k']
        if len(g['parts'][0]) == 3:
            g['R'] = np.ascontiguousarray(np.hstack([R for _, _, R in g['parts']]))
            g['parts'] = [(widx, H) for widx, H, _ in g['parts']]
        g['roff'] = np.cumsum([0] + [H.shape[0] for _, H in g['parts']])
        g['deg'] = nferm(ws[int(g['parts'][0][0][0])])          # one fermion degree per stripped content
    mm = ffblas.matmul if ffblas.HAVE_FFBLAS else matmul_mod
    BIG = 2_000_000                                              # rows_b * n_b above this: MKL (threaded) product
    from concurrent.futures import ThreadPoolExecutor
    t_eval = t_comb = 0.0
    order_d, loc_d, Q_d = {}, {}, {}
    for d in sorted({chunks[gi][0] for gi in mine}):
        wd = np.flatnonzero(tdeg == d)
        order_d[d] = np.array(sorted(wd.tolist(), key=lambda i: tw[i]), dtype=np.int64)   # prefix reuse in Frame
        loc = np.full(len(tw), -1, dtype=np.int64)
        loc[order_d[d]] = np.arange(order_d[d].size)
        loc_d[d] = loc
        Q_d[d] = Qcsr[:, order_d[d]].tocsr()        # Q_0 restricted to target words of degree d, columns in X order
    del Qcsr
    for gi in mine:
        d, np_, k, sd, c0 = chunks[gi]
        lc0 = loc_off[gi]                          # column offset in this run's B and V (= c0 unless sliced)
        cpp = math.comb(k, d)
        w = np_ * cpp
        order, loc = order_d[d], loc_d[d]
        t = time.time()
        if order.size:
            with open(fX, 'wb') as fh:
                fh.truncate(4 * order.size * w)
            _G.update(words=tw, order=order, letters=letters)
            ng = min(order.size, max(1, workers))
            tasks = [(fX, order.size, w, g * order.size // ng, (g + 1) * order.size // ng, (N, np_, k, d, sd))
                     for g in range(ng)]
            if workers > 1:
                from concurrent.futures import ProcessPoolExecutor
                with ProcessPoolExecutor(workers, mp_context=mp.get_context('fork')) as pool:
                    list(pool.map(_eval_task, tasks))
            else:
                for tk in tasks:
                    _eval_task(tk)
            X = np.memmap(fX, dtype=np.int32, mode='r', shape=(order.size, w))
        else:
            X = np.zeros((1, w), dtype=np.int32)
        t_eval += time.time() - t
        t = time.time()
        # Yall = Q_0(word) evaluated, for every preimage word (sparse float64 product, exact, in column pieces), int32
        Yall = np.zeros((len(used), w), dtype=np.int32)
        if order.size:
            for s0 in range(0, w, 256):
                s1 = min(w, s0 + 256)
                Yall[:, s0:s1] = (np.fmod(Q_d[d] @ np.asarray(X[:, s0:s1], dtype=np.float64), P).astype(np.int64) % P)
        # B_s = R_s Z_s, Z_s = stacked H_b Y_b.  Q_0 changes the fermion number by one, so only stripped contents of
        # degree d-1 or d+1 have nonzero images in these (degree-d) columns: skip the others (their B entries stay 0).
        # Large shape blocks: threaded MKL product; small ones: numpy in parallel threads (matmul releases the GIL).
        with ThreadPoolExecutor(max_workers=max(1, workers)) as tp:
            for g, g0 in zip(groups, goff):
                if g['deg'] not in (d - 1, d + 1):
                    continue
                Zs = np.empty((int(g['roff'][-1]), w), dtype=np.int64)
                small = []
                for i, (widx, H) in enumerate(g['parts']):
                    if ffblas.HAVE_FFBLAS and H.shape[0] * H.shape[1] >= BIG:
                        Zs[g['roff'][i]:g['roff'][i + 1]] = mm(H.astype(np.int64), Yall[upos[widx]].astype(np.int64))
                    else:
                        small.append(i)

                def job(i, g=g, Zs=Zs):
                    widx, H = g['parts'][i]
                    Zs[g['roff'][i]:g['roff'][i + 1]] = matmul_mod(H, Yall[upos[widx]])
                list(tp.map(job, small))
                Bm[g0:g0 + g['k'], lc0:lc0 + w] = mm(g['R'], Zs).astype(np.int32)
                del Zs
        del Yall
        if Vrows.n:
            Vm[:, lc0:lc0 + w] = Vrows.apply(X, loc, None).astype(np.int32)
        t_comb += time.time() - t
        del X
        if order.size:
            os.remove(fX)
        log(f"  chunk d={d} cols {c0}:{c0 + w} done (eval {t_eval:.0f}s, combine {t_comb:.0f}s)")
    if cmine:
        t = time.time()
        letters_tp = sorted({L for w in ws for n in w for L in n})
        for cj in cmine:
            d, npts, k, sd, c0 = cchunks[cj]
            lc0 = cloc_off[cj]
            fr = Frame(N, npts, k, [d], letters_tp, seed=sd, cartan=True)
            for g, g0 in zip(groups, goff):
                if g['deg'] != d:
                    continue
                Zs = np.empty((int(g['roff'][-1]), npts * fr.cpp), dtype=np.int64)
                for i, (widx, H) in enumerate(g['parts']):
                    Xc = eval_csec(fr, [ws[int(j)] for j in widx], dtype=np.int64)
                    Zs[g['roff'][i]:g['roff'][i + 1]] = matmul_mod(H.astype(np.int64), Xc)
                Bfull[g0:g0 + g['k'], lc0:lc0 + npts * fr.cpp] = mm(g['R'], Zs).astype(np.int32)
        log(f"Cartan block: {ncart} columns ({time.time()-t:.0f}s)")
    Bm.flush(); Vm.flush()
    res['timing_eval'], res['timing_combine'] = round(t_eval), round(t_comb)
    res.update(nB=int(nB), nV=int(Vrows.n))
    if prep_dir:
        # hand B and V to the GPU rank job (gpu_rank.py): int32 row-major files plus metadata
        os.makedirs(prep_dir, exist_ok=True)
        del Bm, Vm
        import shutil
        shutil.move(fB, os.path.join(prep_dir, 'B.i32'))
        shutil.move(fV, os.path.join(prep_dir, 'V.i32'))
        res['prep_dir'] = prep_dir
        res['time'] = round(time.time() - T0)
        json.dump(res, open(os.path.join(prep_dir, 'meta.json'), 'w'), indent=1)
        if not col_slice:
            json.dump(res, open(out, 'w'), indent=1)
        log(f"prep done: B {nB} x {ncols}, V {Vrows.n} x {ncols} in {prep_dir} ({res['time']}s)")
        return res
    # 6: streaming rank
    t = time.time()
    if ffblas.HAVE_FFBLAS:
        SB = ffblas.StreamBasis(ncols)
        add = lambda R: SB.add(np.asarray(R, dtype=np.int64))
        rank = lambda: SB.rank
    else:
        SB = RREFBasis(ncols)
        add = lambda R: SB.add_batched(np.asarray(R, dtype=np.int64) % P)
        rank = lambda: SB.rank
    for r0 in range(0, nB, 4096):
        add(Bm[r0:min(nB, r0 + 4096)])
    rB = rank()
    log(f"rank B (Q_0-exact subspace of the target, hw) = {rB} ({time.time()-t:.0f}s)")
    if ncart:
        SB2 = ffblas.StreamBasis(ncols + ncart) if ffblas.HAVE_FFBLAS else RREFBasis(ncols + ncart)
        for r0 in range(0, nB, 4096):
            R_ = np.asarray(Bfull[r0:min(nB, r0 + 4096)], dtype=np.int64)
            SB2.add(R_) if ffblas.HAVE_FFBLAS else SB2.add_batched(R_ % P)
        rBE = SB2.rank
        del SB2
        res.update(rank_B_with_cartan=int(rBE), coulomb_rank=int(rBE - rB))
        log(f"rank [B | Cartan] = {rBE}: Cartan rank on ker Q_0 (Coulomb-type classes of the preimage sector) = {rBE - rB}")
    nc = len(ctrl)
    if nc:
        add(Vm[:nc])
    rC = rank()
    lifted = []
    for kk in range(len(vecs)):
        before = rank()
        add(Vm[nc + kk:nc + kk + 1])
        lifted.append(int(rank() - before))
    res.update(rank_B=int(rB), rank_after_controls=int(rC), controls_exact=bool(rC == rB), lifted_increments=lifted,
               n_lifted=int(sum(lifted)), timing_rank=round(time.time() - t))
    log(f"controls: rank {rB} -> {rC} ({'exact, OK' if rC == rB else 'NOT EXACT: pipeline error'}); "
        f"Q_1 v rank increments per class {lifted}")
    del Bm, Vm, SB
    for f in (fB, fV):
        try:
            os.remove(f)
        except OSError:
            pass
    # optional: spanning certificate of the preimage words (needed only to turn a rank increase into a proof)
    if span_check:
        res['span'] = span_certificate(N, TP, Y - 1, ws, groups, ms)
    res['time'] = round(time.time() - T0)
    json.dump(res, open(out, 'w'), indent=1)
    log(f"wrote {out} ({res['time']}s)")
    return res


def span_certificate(N, sec, Yv, ws, groups, ms, seed=777, slice_=None):
    """Rank at rank-N points of the compressed hw rows of each stripped content vs its exact multiplicity m_s: all equal
    <=> the words span the hw subspace (needed only to turn a rank increase into a proof of lifting)."""
    from fortuity.hwcoh import ffla
    mm = ffblas.matmul if ffblas.HAVE_FFBLAS else matmul_mod
    letters = sorted({L for w in ws for n in w for L in n})
    tot_r = 0
    bad = []
    if slice_ is not None:                        # (i, K): only groups with index = i mod K (array jobs)
        groups = [g for a, g in enumerate(groups) if a % slice_[1] == slice_[0]]
        ms = {g['s']: g['m'] for g in groups}
    for g in groups:
        d = nferm(ws[int(g['parts'][0][0][0])])        # one fermion degree per stripped content
        k = frame_k(N, d)
        cpp = math.comb(k, d) if d else 1
        if cpp == 0:          # more fermions than Grassmann components: such words vanish identically (m = 0)
            continue
        # several points: the Grassmann components of a single point are not independent enough functionals
        # a word that is linear in one fermion letter gives at most dim g independent values per point, whatever C(k, d)
        # is, so size the points by min(cpp, dim g) (with C(k, d) the SU(2) degree-1 certificate saw 16 x 3 = 48 < 51)
        eff = min(cpp, N * N - 1) if d else 1
        fr = Frame(N, max(16, math.ceil(2 * (g['m'] + 16) / eff)), k, [d], letters, seed=seed + d)
        acc = None
        if len(g['parts'][0]) == 3:
            parts = list(g['parts'])
        else:      # run() folded the random matrices into g['R'] (columns per part at g['roff'])
            parts = [(widx, H, g['R'][:, g['roff'][i]:g['roff'][i + 1]]) for i, (widx, H) in enumerate(g['parts'])]
        for widx, H, R in parts:
            X = eval_csec(fr, [ws[int(i)] for i in widx], dtype=np.int64)
            Z = mm(R, mm(H.astype(np.int64), X))
            acc = Z if acc is None else (acc + Z) % P
        r = (ffblas.rank_pair32(acc.astype(np.int32))[0] if ffblas.HAVE_FFBLAS else int(ffla.rank(acc)))
        tot_r += r
        if r != g['m']:
            bad.append((str(g['s']), int(r), int(g['m'])))
    return dict(rank=int(tot_r), m=int(sum(ms.values())), spans=not bad and tot_r == sum(ms.values()), deficits=bad[:20])


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('case')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    ap.add_argument('--seed', type=int, default=41)
    ap.add_argument('--xcap', type=float, default=20e9)
    ap.add_argument('--span', action='store_true')
    ap.add_argument('--prep-dir', default=None, help='write B, V, meta.json there for gpu_rank.py and stop')
    ap.add_argument('--workdir', default=None)
    ap.add_argument('--factor', type=float, default=1.0, help='multiply the number of evaluation points (rank-stability '
                    'check: ranks must not change)')
    ap.add_argument('--maxlen', type=int, default=None, help='trace-length cap (default 3 for SU(2), 6 for SU(3))')
    ap.add_argument('--span-slice', default=None, help='i/K: certificate for the groups with index = i mod K only')
    ap.add_argument('--span-only', action='store_true', help='only the spanning certificate of the preimage words '
                    '(per stripped content: rank of the compressed hw rows = exact multiplicity), from the hw cache')
    ap.add_argument('--cartan', type=int, default=0, help='append this many (+margin) Cartan functionals per degree '
                    '(Coulomb rank of the preimage sector cohomology)')
    ap.add_argument('--csec', choices=['all', 'even', 'odd'], default='all',
                    help='restrict to one charge-conjugation sector (SU(3); C: X -> -X^T on every letter)')
    ap.add_argument('--col-slice', default=None, help='i/K: prep only the column chunks with index = i mod K (one node '
                    'per slice; merge_slices.py reassembles B and V)')
    ap.add_argument('--hw-only', action='store_true', help='compute and cache the compressed hw rows of the preimage '
                    'sector, then stop (a later run with the same flags loads the cache)')
    args = ap.parse_args()
    CSEC = {'all': 0, 'even': 1, 'odd': -1}[args.csec]
    ctag = ('' if not CSEC else f'_c{args.csec}') + (f'_cartan' if args.cartan else '')
    CASES = {
        'su2_j36': dict(N=2, SEC=(9, 9, 12, 12, 12), Y=9, ML=3),
        'su3_j24': dict(N=3, SEC=(3, 3, 8, 8, 8), Y=9, ML=6),
        'su3_j30': dict(N=3, SEC=(5, 5, 14, 8, 8), Y=10, ML=6),
        'su2_j24': dict(N=2, SEC=(5, 5, 8, 8, 8), Y=7, ML=3),
    }
    # rank-only cases for the all-orders analysis of R-007: the order-n target T_n = (4,4;14,8,8) at Y = 11-n.
    # r007_o{n}_in : rank of Q_0 from (5,5;14,8,8) at Y = 10-n into T_n;  r007_o{n}_out: rank of Q_0 from T_n into
    # (3,3;14,8,8) at Y = 12-n.  Then dim H(T_n) on [3,0] hw vectors = m(T_n) - rank_in - rank_out.
    for n in (2, 3, 4):
        CASES[f'r007_o{n}_in'] = dict(N=3, SEC=(5, 5, 14, 8, 8), Y=11 - n, ML=6)
        CASES[f'r007_o{n}_out'] = dict(N=3, SEC=(4, 4, 14, 8, 8), Y=12 - n, ML=6)
    # validation of the Coulomb count: SU(2) partner sectors of O_0 at order n (Y = 8 - n), known from oneloop_test.py
    # (runs/run_su2p_Y*.txt): Y=6: h = 27, all Coulomb-type; Y=4: h = 4, all Coulomb-type
    for y in (6, 4):
        CASES[f'su2p{y}_in'] = dict(N=2, SEC=(5, 5, 8, 8, 8), Y=y, ML=3)
        CASES[f'su2p{y}_out'] = dict(N=2, SEC=(4, 4, 8, 8, 8), Y=y + 1, ML=3)
    if args.case in CASES:
        C = CASES[args.case]
    else:
        # generic rank-only case 'rank:<N>:<2J1>,<2J2>,<2q1>,<2q2>,<2q3>:<Y>': rank of Q_0 from (SEC; Y-1) into
        # (2J1-1, 2J2-1, q; Y); the source of a one-loop test 'su<N>:<2J>:<q1>,<q2>,<q3>' uses classes/<that case>/q1.pkl
        parts = args.case.split(':')
        if parts[0] == 'rank':
            C = dict(N=int(parts[1]), SEC=tuple(int(x) for x in parts[2].split(',')), Y=int(parts[3]),
                     ML={2: 3, 3: 6}[int(parts[1])])
        else:
            N_, twoJ, q = int(parts[0][2:]), int(parts[1]), tuple(int(x) for x in parts[2].split(','))
            C = dict(N=N_, SEC=(twoJ, twoJ) + tuple(2 * x for x in q), Y=sum(q) - twoJ, ML={2: 3, 3: 6}[N_])
    D = f'classes/{args.case.replace(":", "_").replace(",", "-")}'
    D0 = D                                   # the class vectors (q1.pkl, ctrl.pkl) live in the unsuffixed directory
    if args.maxlen:
        C = dict(C, ML=args.maxlen)
        D += f'_ml{args.maxlen}'
    if args.factor != 1.0:
        D += f'_f{args.factor:g}'
    os.makedirs(D, exist_ok=True)
    vecs = pickle.load(open(f'{D0}/q1.pkl', 'rb')) if os.path.exists(f'{D0}/q1.pkl') else []
    ctrl = pickle.load(open(f'{D0}/ctrl.pkl', 'rb')) if os.path.exists(f'{D0}/ctrl.pkl') else []
    os.makedirs('cache', exist_ok=True)
    if args.span_only:
        hwc = f'cache/fullpar_hw_{D.split("/")[-1]}{ctag}.pkl'
        if not os.path.exists(hwc):          # the prep runs of the rank-only cases cached under their own tag
            alt = [f for f in os.listdir('cache') if f.startswith(f'fullpar_hw_{D.split("/")[-1]}_c')]
            hwc = os.path.join('cache', sorted(alt)[0]) if alt else hwc
        ws, groups, ms, stats = compressed_hw(C['N'], tuple(C['SEC']), C['Y'] - 1, C['ML'], cache=hwc)
        sl = tuple(int(x) for x in args.span_slice.split('/')) if args.span_slice else None
        sc = span_certificate(C['N'], tuple(C['SEC']), C['Y'] - 1, ws, groups, ms, slice_=sl)
        out = f'{D}/span_seed{args.seed}{ctag}' + (f'_slice{sl[0]}of{sl[1]}' if sl else '') + '.json'
        json.dump(dict(case=args.case, csec=CSEC, hw_cache=hwc, **sc), open(out, 'w'), indent=1)
        log(f"span certificate ({hwc}): rank {sc['rank']} of exact multiplicity {sc['m']}, spans {sc['spans']}, "
            f"deficits {sc['deficits']} -> {out}")
        sys.exit(0)
    if args.hw_only:
        compressed_hw(C['N'], tuple(C['SEC']), C['Y'] - 1, C['ML'], cache=f'cache/fullpar_hw_{D.split("/")[-1]}{ctag}.pkl')
        log(f"hw rows cached for {args.case}")
        sys.exit(0)
    run(C['N'], C['SEC'], C['Y'], C['ML'], vecs, ctrl, f'{D}/fullpar_seed{args.seed}{ctag}.json', workers=args.workers,
        seed=args.seed, xcap=args.xcap, factor=args.factor, hw_cache=f'cache/fullpar_hw_{D.split("/")[-1]}{ctag}.pkl', span_check=args.span,
        prep_dir=args.prep_dir, workdir=args.workdir, case=args.case, cartan=args.cartan,
        col_slice=tuple(int(x) for x in args.col_slice.split('/')) if args.col_slice else None)
