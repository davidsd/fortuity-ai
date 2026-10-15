"""T-030: one-loop fate of the singlet Q_0-cohomology of a BMN sector, with the full one-loop supercharge Q_1
(Budzik-Kulp) of the full-letter code (fortuity.full).

Usage:  python3 oneloop_test.py <case> <stage>
  case:  su3_j24  (SU(3), sector (3,3;8,8,8) Y=9: the 29 classes of R-002, incl. the non-Coulomb R_24 of R-009)
         su2_j36  (SU(2), sector (9,9;12,12,12) Y=9: KKLP's O_1 (lifted) and O_1' (unlifted), plus gravitons)
  stage: reps | coulomb | q1 | target | all

Sector labels (2J1, 2J2, 2q1, 2q2, 2q3) at letter number Y (fullsector conventions). Q_0 maps (2J1,2J2,q;Y) to
(2J1-1,2J2-1,q;Y+1); Q_1 maps it to (2J1-1,2J2-1,q;Y) and raises d_BMN by one. A source class v lifts at one loop
iff Q_1 v is not Q_0-exact in the target sector T = (2J1-1,2J2-1,q;Y), whose exact subspace is Q_0 of the
(2J1,2J2,q;Y-1) sector. Everything is restricted to SU(3)_R singlets (hw vectors of weight (q,q,q)); Q_0 and Q_1
commute with SU(3)_R. Ranks are over F_p (p = 2^26-5) with random evaluation functionals, as in fortuity.hwcoh.
"""
import math, os, sys, time, pickle, math, json
import numpy as np
from collections import defaultdict
from fastcontents import prime_cache
from fortuity.bmn.modp import RREFBasis, matmul_mod, P
from fortuity.hwcoh import ffla
from fortuity.full.sector import words_of, hw_mult_by_degree, frame_k
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
from fortuity.full.evalg import Frame, nferm
from fortuity.full.hw import nullspace_left, hw_vectors
from fortuity.full.q1 import Q1_multi

CASES = {
    'su3_j24': dict(N=3, SEC=(3, 3, 8, 8, 8), Y=9, ML=6),
    'su2_j36': dict(N=2, SEC=(9, 9, 12, 12, 12), Y=9, ML=3),
    'su3_j30': dict(N=3, SEC=(5, 5, 14, 8, 8), Y=10, ML=6),     # R-007: [3,0] hw, 4 gravitons + the fortuitous class
    'su2_j24': dict(N=2, SEC=(5, 5, 8, 8, 8), Y=7, ML=3),       # SU(2) threshold class O_0 (KKLP)
    **{f'su2p_Y{y}': dict(N=2, SEC=(4, 4, 8, 8, 8), Y=y, ML=3) for y in (7, 6, 5, 4)},   # O_0's partner sectors
}
case, stage = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'all')


def parse_case(case):
    """Generic BMN source sector 'su<N>:<2J>:<q1>,<q2>,<q3>' -> (2J,2J;2q) at Y = sum q - 2J (maxlen 3 for SU(2),
    6 for SU(3)); named cases from CASES otherwise."""
    if case in CASES:
        return CASES[case]
    g, twoJ, q = case.split(':')
    N = int(g[2:])
    q = tuple(int(x) for x in q.split(','))
    return dict(N=N, SEC=(int(twoJ), int(twoJ)) + tuple(2 * x for x in q), Y=sum(q) - int(twoJ), ML={2: 3, 3: 6}[N])


C = parse_case(case)
N, SEC, Y, ML = C['N'], C['SEC'], C['Y'], C['ML']
UP = (SEC[0] + 1, SEC[1] + 1) + SEC[2:]       # Q_0-preimage sector of SEC (Y-1)
LOW = (SEC[0] - 1, SEC[1] - 1) + SEC[2:]      # Q_0-image sector of SEC (Y+1), also the Q_1 target sector (Y)
MARGIN = 64
CHUNK = int(os.environ.get('CHUNK', '512'))
OUT = f'classes/{case.replace(":", "_").replace(",", "-")}'
os.makedirs(OUT, exist_ok=True)
log = lambda *a: print(*a, flush=True)
T0 = time.time()


def hw_blocks(sec, Yv):
    """Singlet hw vectors of (sec, Yv), cached in this workspace."""
    prime_cache(sec, Yv)
    fn = f'cache/hw_{"_".join(map(str, sec))}_Y{Yv}_ml{ML}_N{N}.pkl'
    ws = words_of(sec, Yv, ML)
    if os.path.exists(fn):
        nw, blocks = pickle.load(open(fn, 'rb'))
        if nw == len(ws):
            return ws, [(i, H.astype(np.int64)) for i, H in blocks]
    blocks = hw_vectors(ws)
    pickle.dump((len(ws), [(i, H.astype(np.int32)) for i, H in blocks]), open(fn, 'wb'), protocol=4)
    return ws, blocks


def frames_for(m_d, letters, factor=1.0, seed=11, cartan=False, margin=MARGIN):
    """[(degree, Frame)] with >= factor*(m_d + margin) functionals per fermion degree."""
    out = []
    for d, md in sorted(m_d.items()):
        need = int(factor * (md + margin))
        k = frame_k(N, d)
        npts = max(1, math.ceil(need / math.comb(k, d)))
        out.append((d, Frame(N, npts, k, [d], letters, seed=seed + 7919 * d, cartan=cartan)))
    return out


def evaluate(words, frames):
    ncols = sum(fr.ncols for _, fr in frames)
    X = np.zeros((len(words), ncols), dtype=np.int64)
    deg = np.array([nferm(w) for w in words], dtype=np.int64)
    col = 0
    for d, fr in frames:
        idx = np.flatnonzero(deg == d)
        if idx.size:
            X[idx, col:col + fr.ncols] = fr.eval_words([words[i] for i in idx])
        col += fr.ncols
    return X


def eval_dicts(dicts, frames, chunk=20000):
    """Evaluate a list of sparse vectors {word: coeff} at the frames -> (len(dicts), ncols)."""
    ncols = sum(fr.ncols for _, fr in frames)
    allw = sorted({w for d in dicts for w in d})
    out = np.zeros((len(dicts), ncols), dtype=np.int64)
    for s in range(0, len(allw), chunk):
        part = allw[s:s + chunk]
        Xc = evaluate(part, frames)
        pidx = {w: i for i, w in enumerate(part)}
        for r, d in enumerate(dicts):
            rows = [pidx[w] for w in d if w in pidx]
            if rows:
                cs = np.array([d[part[i]] for i in rows], dtype=np.int64) % P
                out[r] = (out[r] + matmul_mod(cs[None, :], Xc[np.array(rows)])[0]) % P
    return out


def hw_eval(ws, blocks, frames):
    ncols = sum(fr.ncols for _, fr in frames)
    nA = sum(H.shape[0] for _, H in blocks)
    A = np.zeros((nA, ncols), dtype=np.int64)
    r0 = 0
    for widx, H in blocks:
        A[r0:r0 + H.shape[0]] = matmul_mod(H % P, evaluate([ws[int(i)] for i in widx], frames))
        r0 += H.shape[0]
    return A


def hw_dicts(ws, blocks):
    out = []
    for widx, H in blocks:
        for r in range(H.shape[0]):
            nz = np.flatnonzero(H[r])
            out.append({ws[int(widx[c])]: int(H[r, c]) for c in nz})
    return out


def apply(op, v):
    out = defaultdict(int)
    for w, c in v.items():
        for m2, c2 in op(w).items():
            out[m2] = (out[m2] + c * c2) % P
    return {m: c for m, c in out.items() if c}


_Q0C, _Q1C = {}, {}
def _q0w(w):
    r = _Q0C.get(w)
    if r is None:
        r = _Q0C[w] = dict(Q_multi(w))
    return r
def _q1w(w):
    r = _Q1C.get(w)
    if r is None:
        r = _Q1C[w] = Q1_multi(w, N)
    return r
Q0 = lambda v: apply(_q0w, v)
Q1 = lambda v: apply(_q1w, v)


def letters_of(*collections):
    return sorted({L for coll in collections for w in coll for n in w for L in n})


# ---------------------------------------------------------------- stage 1: cohomology representatives in SEC
if stage in ('reps', 'all'):
    ws, blocks = hw_blocks(SEC, Y)
    m_d = hw_mult_by_degree(N, SEC, Y)
    # the frames must know every letter that occurs, including letters that only appear in Q_0-images of UP words
    ws_up0, blocks_up0 = hw_blocks(UP, Y - 1) if Y > 1 else ([], [])
    up_letters = letters_of([m for w in ws_up0 for m in Q_multi(w)]) if ws_up0 else []
    F0 = frames_for(m_d, sorted(set(letters_of(ws)) | set(up_letters)))
    A = hw_eval(ws, blocks, F0)
    basis = RREFBasis(A.shape[1])
    indep = np.zeros(A.shape[0], dtype=bool)
    for s in range(0, A.shape[0], 256):
        indep[s:s + 256] = basis.add(A[s:s + 256])
    sel = np.flatnonzero(indep)
    m = sum(m_d.values())
    log(f"[{case}] SEC {SEC} Y={Y}: words {len(ws)}, hw rows {A.shape[0]}, independent {len(sel)} vs exact m {m} {m_d} "
        f"-> {'SPAN' if len(sel) == m else 'NO SPAN'} ({time.time()-T0:.0f}s)")
    assert len(sel) == m, "hw vectors of the word set do not span the singlet space"
    Hd = hw_dicts(ws, blocks)
    Hsel, Asel = [Hd[i] for i in sel], A[sel]
    del A, basis
    # exact subspace of SEC: Q_0 of the singlet hw vectors of UP (Y-1)
    ws_up, blocks_up = hw_blocks(UP, Y - 1)
    m_up = sum(hw_mult_by_degree(N, UP, Y - 1).values())
    Ed = [Q0(v) for v in hw_dicts(ws_up, blocks_up)]
    E = eval_dicts(Ed, F0)
    ebasis = RREFBasis(E.shape[1]); ebasis.add_batched(E)
    log(f"exact: UP {UP} Y={Y-1}: words {len(ws_up)}, hw rows {len(Ed)} (exact m_up {m_up}); rank of Q_0 images {ebasis.rank} ({time.time()-T0:.0f}s)")
    # closed combinations: Q_0 of the selected hw vectors, evaluated in LOW (Y+1)
    Qd = [Q0(v) for v in Hsel]
    m_low = hw_mult_by_degree(N, LOW, Y + 1)
    prime_cache(LOW, Y + 1)
    F1 = frames_for(m_low, letters_of([w for d in Qd for w in d]), factor=1.3, seed=23)
    M = eval_dicts(Qd, F1)
    K = nullspace_left(M)
    log(f"closedness: Q_0 image rank {ffla.rank(M)} (m_low {sum(m_low.values())}); closed hw combinations {K.shape[0]} ({time.time()-T0:.0f}s)")
    Cm = matmul_mod(K % P, Asel)
    reps = [r for r in range(Cm.shape[0]) if ebasis.add(Cm[r:r + 1])[0]]
    h = len(reps)
    log(f"cohomology dimension h = {h} = closed {K.shape[0]} - exact rank {ebasis.rank - h} ({time.time()-T0:.0f}s)")
    vecs = []
    for r in reps:
        v = defaultdict(int)
        for j in np.flatnonzero(K[r]):
            for w, c in Hsel[j].items():
                v[w] = (v[w] + int(K[r, j]) * c) % P
        vecs.append({w: c for w, c in v.items() if c})
    log("representative sizes (words):", [len(v) for v in vecs])
    pickle.dump(dict(vecs=vecs, m_d=m_d, m_low=m_low, exact_rank=ebasis.rank - h, h=h), open(f'{OUT}/reps.pkl', 'wb'))
    # closedness re-check at fresh points (words obey trace identities, so Q_0 v is a nonzero formal combination)
    qs = [Q0(v) for v in vecs]
    fr = frames_for(m_low, letters_of([w for q in qs for w in q]), factor=0.2, seed=4242)
    V = eval_dicts(qs, fr)
    log(f"closedness at fresh points: nonzero entries {int(np.count_nonzero(V))} of {V.size} (must be 0)")

# ---------------------------------------------------------------- stage 2: Coulomb-branch (Cartan) restriction
if stage in ('coulomb', 'all'):
    R = pickle.load(open(f'{OUT}/reps.pkl', 'rb'))
    vecs = R['vecs']
    letters = letters_of([w for v in vecs for w in v])
    degs = sorted({nferm(w) for v in vecs for w in v})
    frames = [(d, Frame(N, 40, max(d + 4, 6) if d else 0, [d], letters, seed=99 + d, cartan=True)) for d in degs]
    V = eval_dicts(vecs, frames)
    rc = ffla.rank(V)
    Kc = nullspace_left(V)
    log(f"Coulomb: rank of the Cartan values of the {len(vecs)} classes = {rc}; non-Coulomb combinations {Kc.shape[0]} "
        f"(functionals {V.shape[1]})")
    pickle.dump(dict(rank=rc, K=Kc), open(f'{OUT}/coulomb.pkl', 'wb'))

# ---------------------------------------------------------------- stage 3: Q_1 of the representatives
if stage in ('q1', 'all'):
    R = pickle.load(open(f'{OUT}/reps.pkl', 'rb'))
    q1s = []
    for k, v in enumerate(R['vecs']):
        t = time.time()
        q1s.append(Q1(v))
        log(f"rep {k}: {len(v)} words -> Q_1 v: {len(q1s[-1])} words ({time.time()-t:.0f}s)")
    pickle.dump(q1s, open(f'{OUT}/q1.pkl', 'wb'))

# ---------------------------------------------------------------- stage 4: is Q_1 v Q_0-exact in the target sector?
def colblocks_for(m_d, letters, factor, seed, chunk=CHUNK):
    """Functionals >= factor*(m_d + margin) per fermion degree, split into frames of <= chunk columns."""
    out = []
    for d, md in sorted(m_d.items()):
        need = int(factor * (md + MARGIN))
        k = frame_k(N, d)
        cpp = math.comb(k, d)
        total = max(1, math.ceil(need / cpp))
        per = max(1, chunk // cpp)
        for c0 in range(0, total, per):
            out.append((d, Frame(N, min(per, total - c0), k, [d], letters, seed=seed + 7919 * d + c0)))
    return out


def word_matrix(rows_dicts, index):
    """Sparse rows {word: coeff} -> list of (cols array, coeffs array) over a word index."""
    return [(np.array([index[w] for w in d], dtype=np.int64), np.array([d[w] % P for w in d], dtype=np.int64)) for d in rows_dicts]


if stage in ('target', 'all'):
    R = pickle.load(open(f'{OUT}/reps.pkl', 'rb'))
    q1s = pickle.load(open(f'{OUT}/q1.pkl', 'rb'))
    TGT, TUP = LOW, SEC                       # target (Y), its Q_0-preimage sector (Y-1)
    prime_cache(TGT, Y)
    mT = hw_mult_by_degree(N, TGT, Y)
    ws_tu, blocks_tu = hw_blocks(TUP, Y - 1)
    m_tu = hw_mult_by_degree(N, TUP, Y - 1)
    log(f"target {TGT} Y={Y}: singlet mult {sum(mT.values())} {mT}; preimage {TUP} Y={Y-1}: words {len(ws_tu)}, "
        f"hw rows {sum(H.shape[0] for _, H in blocks_tu)}, singlet mult {sum(m_tu.values())} {m_tu} ({time.time()-T0:.0f}s)")
    # (a) spanning check of the preimage words: rank of their singlet hw vectors = exact multiplicity, per degree
    deg_tu = np.array([nferm(w) for w in ws_tu], dtype=np.int64)
    Ftu = colblocks_for(m_tu, letters_of(ws_tu), 1.0, 31)
    nA = sum(H.shape[0] for _, H in blocks_tu)
    rk_tu = {}
    for d in sorted(m_tu):
        cbs = [fr for dd, fr in Ftu if dd == d]
        A = np.zeros((nA, sum(fr.ncols for fr in cbs)), dtype=np.int64)
        col = 0
        wd = np.flatnonzero(deg_tu == d)
        for fr in cbs:
            X = np.zeros((len(ws_tu), fr.ncols), dtype=np.int64)
            if wd.size:
                X[wd] = fr.eval_words([ws_tu[i] for i in wd])
            r0 = 0
            for widx, H in blocks_tu:
                A[r0:r0 + H.shape[0], col:col + fr.ncols] = matmul_mod(H % P, X[widx])
                r0 += H.shape[0]
            col += fr.ncols
            del X
        rows = np.flatnonzero(A.any(axis=1))
        rk_tu[d] = int(ffla.rank(A[rows])) if rows.size else 0
        del A
    span_tu = all(rk_tu.get(d, 0) == m for d, m in m_tu.items())
    log(f"(a) preimage hw rank per degree {rk_tu} vs {m_tu} -> {'SPAN' if span_tu else 'NO SPAN'} ({time.time()-T0:.0f}s)")
    # (b) Q_0 of the preimage hw vectors, as dense blocks over target words (as in sector_full.Sector.prepare)
    twords, tidx = [], {}
    def tix(w):
        j = tidx.get(w)
        if j is None:
            j = tidx[w] = len(twords); twords.append(w)
        return j
    qblocks = []
    for widx, H in blocks_tu:
        rs, js, cs = [], [], []
        for r, wi in enumerate(widx):
            for m2, c in Q_multi(ws_tu[int(wi)]).items():
                rs.append(r); js.append(tix(m2)); cs.append(c % P)
        if not rs:
            continue
        rs, js, cs = (np.array(x, dtype=np.int64) for x in (rs, js, cs))
        tg, jl = np.unique(js, return_inverse=True)
        Qm = np.zeros((len(widx), len(tg)), dtype=np.int64)
        np.add.at(Qm, (rs, jl), cs)
        Cm = matmul_mod(H % P, Qm % P)
        keep = np.flatnonzero(Cm.any(axis=0))
        if keep.size:
            qblocks.append((tg[keep], Cm[:, keep]))
    # Q_1 v and the controls Q_1 Q_0 u (u singlet hw vectors of UP) over the same target word list
    ws_up, blocks_up = hw_blocks(UP, Y - 1)
    ups = hw_dicts(ws_up, blocks_up)[:3]
    ctrl = [Q1(Q0(u)) for u in ups]
    vq = [(np.array([tix(w) for w in d], dtype=np.int64), np.array([d[w] % P for w in d], dtype=np.int64)) for d in q1s]
    vc = [(np.array([tix(w) for w in d], dtype=np.int64), np.array([d[w] % P for w in d], dtype=np.int64)) for d in ctrl]
    tdeg = np.array([nferm(w) for w in twords], dtype=np.int64)
    nB = sum(Cm.shape[0] for _, Cm in qblocks)
    log(f"(b) target words in play {len(twords)}; exact rows {nB}; Q_1 v sizes {[len(d) for d in q1s]}; controls {[len(d) for d in ctrl]} ({time.time()-T0:.0f}s)")
    results = {}
    for seed in (41, 43):
        FT = colblocks_for(mT, letters_of(twords), 1.2, seed)
        ncols = sum(fr.ncols for _, fr in FT)
        B = np.zeros((nB, ncols), dtype=np.int64)
        Vq = np.zeros((len(vq), ncols), dtype=np.int64)
        Vc = np.zeros((len(vc), ncols), dtype=np.int64)
        col = 0
        for d, fr in FT:
            # evaluate only the words of fermion degree d; other words vanish on these functionals
            wd = np.flatnonzero(tdeg == d)
            loc = np.full(len(twords), -1, dtype=np.int64); loc[wd] = np.arange(wd.size)
            X = np.zeros((wd.size + 1, fr.ncols), dtype=np.int64)          # last row: zero (other degrees)
            for s0 in range(0, wd.size, 200000):
                part = wd[s0:s0 + 200000]
                X[s0:s0 + part.size] = fr.eval_words([twords[i] for i in part])
            zero = wd.size
            def rows(ix):
                l = loc[ix]
                return np.where(l < 0, zero, l)
            r0 = 0
            for tg, Cm in qblocks:
                B[r0:r0 + Cm.shape[0], col:col + fr.ncols] = matmul_mod(Cm, X[rows(tg)])
                r0 += Cm.shape[0]
            for r, (ix, cs) in enumerate(vq):
                if ix.size:
                    Vq[r, col:col + fr.ncols] = matmul_mod(cs[None, :], X[rows(ix)])[0]
            for r, (ix, cs) in enumerate(vc):
                if ix.size:
                    Vc[r, col:col + fr.ncols] = matmul_mod(cs[None, :], X[rows(ix)])[0]
            col += fr.ncols
            del X, loc
        img = RREFBasis(ncols); img.add_batched(B)
        res = img.reduce(Vq)
        resc = img.reduce(Vc)
        nz = [int(np.count_nonzero(r)) for r in res]
        rk = int(ffla.rank(res)) if res.size else 0
        log(f"[seed {seed}] target functionals {ncols}; exact rank {img.rank} ({time.time()-T0:.0f}s)")
        log(f"[seed {seed}] Q_1 v residual nonzero entries per class: {nz}")
        log(f"[seed {seed}] RANK of residuals = number of classes lifted at one loop: {rk}")
        log(f"[seed {seed}] controls Q_1 Q_0 u residuals (must be 0): {[int(np.count_nonzero(r)) for r in resc]}")
        results[seed] = dict(rank=rk, nonzero=nz, exact_rank=int(img.rank), ncols=int(ncols),
                             control=[int(np.count_nonzero(r)) for r in resc], residuals=res)
        if os.path.exists(f'{OUT}/coulomb.pkl'):
            Kc = pickle.load(open(f'{OUT}/coulomb.pkl', 'rb'))['K']
            if Kc.shape[0]:
                rn = matmul_mod(Kc % P, res % P)
                log(f"[seed {seed}] residual on the non-Coulomb combinations: nonzero entries {[int(np.count_nonzero(r)) for r in rn]}")
        del B
    pickle.dump(dict(results=results, mT=mT, m_tu=m_tu, rk_tu=rk_tu, span_tu=span_tu), open(f'{OUT}/target.pkl', 'wb'))
    log(f"done ({time.time()-T0:.0f}s)")
