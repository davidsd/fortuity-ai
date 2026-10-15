"""Explicit representatives of the [3,0] hw Q_0-cohomology of SU(3) at (5,5;14,8,8), Y=10 (the R-007 class sector),
then Q_1 on them and the Coulomb-branch (Cartan) test: a Q_0-exact operator vanishes on commuting letters, so a
nonzero Cartan value of Q_1 v proves Q_1 v is not Q_0-exact, i.e. the class v lifts at one loop."""
import os, sys, time, pickle, math, json
import numpy as np
from collections import defaultdict
from fortuity.bmn.modp import RREFBasis, matmul_mod, P
from fortuity.hwcoh import ffla
from fortuity.full.sector import words_of, hw_mult_by_degree, frame_k
from fortuity.full.words import Q_multi
from fortuity.full.evalg import Frame, nferm
from fortuity.full.hw import nullspace_left
from fortuity.full.q1 import Q1_multi
from fortuity.full.letters import is_ferm, name

N, ML = 3, 6
SEC, Y = (5, 5, 14, 8, 8), 10
UP, LOW = (6, 6, 14, 8, 8), (4, 4, 14, 8, 8)
MARGIN = 64
log = lambda *a: print(*a, flush=True)
T0 = time.time()


def load_hw(sec, Yv):
    nw, blocks = pickle.load(open(f'cache/hw_{"_".join(map(str, sec))}_Y{Yv}_ml{ML}.pkl', 'rb'))
    return [(i, H.astype(np.int64)) for i, H in blocks]


def frames_for(sec, Yv, letters, m_d, factor=1.0, seed=11, cartan=False):
    """List of (degree, Frame) giving >= factor*(m_d + margin) functionals per fermion degree."""
    out = []
    for d, md in sorted(m_d.items()):
        need = int(factor * (md + MARGIN))
        k = frame_k(N, d)
        cpp = math.comb(k, d)
        npts = max(1, math.ceil(need / cpp))
        out.append((d, Frame(N, npts, k, [d], letters, seed=seed + 7919 * d, cartan=cartan)))
    return out


def evaluate(words, frames):
    """(len(words), ncols) int64 evaluation matrix, degree by degree."""
    ncols = sum(fr.ncols for _, fr in frames)
    X = np.zeros((len(words), ncols), dtype=np.int64)
    deg = np.array([nferm(w) for w in words])
    col = 0
    for d, fr in frames:
        idx = np.flatnonzero(deg == d)
        if idx.size:
            X[idx, col:col + fr.ncols] = fr.eval_words([words[i] for i in idx])
        col += fr.ncols
    return X


def hw_rows_eval(words, blocks, frames):
    """Evaluate hw vectors (rows of the blocks) at the frames: returns matrix (sum rows, ncols)."""
    ncols = sum(fr.ncols for _, fr in frames)
    nA = sum(H.shape[0] for _, H in blocks)
    A = np.zeros((nA, ncols), dtype=np.int64)
    r0 = 0
    for widx, H in blocks:
        X = evaluate([words[int(i)] for i in widx], frames)
        A[r0:r0 + H.shape[0]] = matmul_mod(H % P, X)
        r0 += H.shape[0]
    return A


def hw_as_dicts(words, blocks):
    """hw vectors as sparse dicts {word: coeff}."""
    out = []
    for widx, H in blocks:
        for r in range(H.shape[0]):
            nz = np.flatnonzero(H[r])
            out.append({words[int(widx[c])]: int(H[r, c]) for c in nz})
    return out


def Q_of_dict(v):
    out = defaultdict(int)
    for w, c in v.items():
        for m2, c2 in Q_multi(w).items():
            out[m2] = (out[m2] + c * c2) % P
    return {m: c for m, c in out.items() if c}


stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
os.makedirs('classes', exist_ok=True)

if stage in ('all', 'reps'):
    ws = words_of(SEC, Y, ML)
    blocks = load_hw(SEC, Y)
    letters = sorted({L for w in ws for n in w for L in n})
    m_d = hw_mult_by_degree(N, SEC, Y)
    log(f"words {len(ws)}, hw rows {sum(H.shape[0] for _, H in blocks)}, m_d {m_d} ({time.time()-T0:.0f}s)")
    F0 = frames_for(SEC, Y, letters, m_d)
    A = hw_rows_eval(ws, blocks, F0)
    log(f"A {A.shape} ({time.time()-T0:.0f}s)")
    # independent hw rows
    basis = RREFBasis(A.shape[1])
    indep = np.zeros(A.shape[0], dtype=bool)
    for s in range(0, A.shape[0], 256):
        indep[s:s + 256] = basis.add(A[s:s + 256])
    sel = np.flatnonzero(indep)
    log(f"independent hw rows {len(sel)} (exact m {sum(m_d.values())}) ({time.time()-T0:.0f}s)")
    hwd = hw_as_dicts(ws, blocks)
    Hsel = [hwd[i] for i in sel]
    Asel = A[sel]
    del A, basis
    # exact rows: Q-images of hw(UP, Y-1) evaluated at F0
    ws_up = words_of(UP, Y - 1, ML)
    blocks_up = load_hw(UP, Y - 1)
    hwd_up = hw_as_dicts(ws_up, blocks_up)
    Ed = [Q_of_dict(v) for v in hwd_up]
    ewords = sorted({w for d in Ed for w in d})
    eidx = {w: i for i, w in enumerate(ewords)}
    XE = evaluate(ewords, F0)
    E = np.zeros((len(Ed), XE.shape[1]), dtype=np.int64)
    for r, d in enumerate(Ed):
        if d:
            rows = np.array([eidx[w] for w in d]); cs = np.array([c for c in d.values()], dtype=np.int64)
            E[r] = matmul_mod(cs[None, :], XE[rows])[0]
    del XE
    ebasis = RREFBasis(E.shape[1]); ebasis.add_batched(E)
    log(f"exact rows {E.shape[0]}, rank {ebasis.rank} ({time.time()-T0:.0f}s)")
    # closedness: Q-images of the selected hw vectors, evaluated at frames of LOW (Y+1)
    Qd = [Q_of_dict(v) for v in Hsel]
    qwords = sorted({w for d in Qd for w in d})
    qidx = {w: i for i, w in enumerate(qwords)}
    m_low = hw_mult_by_degree(N, LOW, Y + 1)
    letters_low = sorted({L for w in qwords for n in w for L in n})
    F1 = frames_for(LOW, Y + 1, letters_low, m_low, factor=1.3, seed=23)
    XQ = evaluate(qwords, F1)
    M = np.zeros((len(Qd), XQ.shape[1]), dtype=np.int64)
    for r, d in enumerate(Qd):
        if d:
            rows = np.array([qidx[w] for w in d]); cs = np.array([c for c in d.values()], dtype=np.int64)
            M[r] = matmul_mod(cs[None, :], XQ[rows])[0]
    del XQ
    rM = ffla.rank(M)
    K = nullspace_left(M)
    log(f"Q-image matrix {M.shape}, rank {rM} (m_low {m_low}); closed hw vectors {K.shape[0]} ({time.time()-T0:.0f}s)")
    C = matmul_mod(K % P, Asel)
    # cohomology representatives: rows of C independent of the exact space
    reps = []
    for r in range(C.shape[0]):
        if ebasis.add(C[r:r + 1])[0]:
            reps.append(r)
    log(f"cohomology dimension {len(reps)} (expected 5) ({time.time()-T0:.0f}s)")
    vecs = []
    for r in reps:
        v = defaultdict(int)
        for j in np.flatnonzero(K[r]):
            for w, c in Hsel[j].items():
                v[w] = (v[w] + int(K[r, j]) * c) % P
        vecs.append({w: c for w, c in v.items() if c})
    log("representative sizes (words):", [len(v) for v in vecs])
    pickle.dump(dict(vecs=vecs, reps=reps, m_d=m_d, m_low=m_low, rM=rM, exact_rank=ebasis.rank), open('classes/reps_5_5_14_8_8_Y10.pkl', 'wb'))

if stage in ('all', 'check'):
    # closedness is checked numerically at fresh random points: words obey trace identities for N x N matrices,
    # so Q_0 v is in general a nonzero formal combination of words that vanishes as an operator
    R = pickle.load(open('classes/reps_5_5_14_8_8_Y10.pkl', 'rb'))
    qs = [Q_of_dict(v) for v in R['vecs']]
    qw = sorted({w for q in qs for w in q})
    letters = sorted({L for w in qw for n in w for L in n})
    degs = sorted({nferm(w) for w in qw})
    frames = [(d, Frame(N, 6, frame_k(N, d), [d], letters, seed=4242 + d)) for d in degs]
    X = evaluate(qw, frames); qi = {w: i for i, w in enumerate(qw)}
    for k, q in enumerate(qs):
        rows = np.array([qi[w] for w in q]); cs = np.array(list(q.values()), dtype=np.int64)
        val = matmul_mod(cs[None, :], X[rows])[0]
        log(f"rep {k}: {len(R['vecs'][k])} words, Q_0 v: {len(q)} formal terms, nonzero at fresh points: {int(np.count_nonzero(val))} of {val.size} (must be 0)")

if stage in ('all', 'q1'):
    R = pickle.load(open('classes/reps_5_5_14_8_8_Y10.pkl', 'rb'))
    q1s = []
    for k, v in enumerate(R['vecs']):
        t = time.time()
        out = defaultdict(int)
        for w, c in v.items():
            for m2, c2 in Q1_multi(w, N).items():
                out[m2] = (out[m2] + c * c2) % P
        q1 = {m: c for m, c in out.items() if c}
        q1s.append(q1)
        log(f"rep {k}: Q_1 v has {len(q1)} words ({time.time()-t:.0f}s)")
    pickle.dump(q1s, open('classes/q1_reps.pkl', 'wb'))

if stage in ('all', 'cartan'):
    R = pickle.load(open('classes/reps_5_5_14_8_8_Y10.pkl', 'rb'))
    q1s = pickle.load(open('classes/q1_reps.pkl', 'rb'))
    allw = sorted({w for q in q1s for w in q})
    letters = sorted({L for w in allw for n in w for L in n})
    degs = sorted({nferm(w) for w in allw})
    # Cartan frames: a few points per degree, generous frame size
    frames = [(d, Frame(N, 8, max(d + 4, 6) if d else 0, [d], letters, seed=99 + d, cartan=True)) for d in degs]
    X = evaluate(allw, frames)
    widx = {w: i for i, w in enumerate(allw)}
    vals = []
    for k, q in enumerate(q1s):
        rows = np.array([widx[w] for w in q]); cs = np.array(list(q.values()), dtype=np.int64)
        val = matmul_mod(cs[None, :], X[rows])[0]
        vals.append(val)
        log(f"rep {k}: Q_1 v on the Cartan: nonzero entries {int(np.count_nonzero(val))} of {val.size}")
    V = np.array(vals)
    log("rank of the Cartan values of Q_1 on the cohomology:", ffla.rank(V) if V.size else 0)
    # sanity: Q_0-exact words vanish on the Cartan
    ws = words_of(SEC, Y, ML)
    import random
    random.seed(5)
    sample = random.sample(ws, 20)
    qw = sorted({m for w in sample for m in Q_multi(w)})
    qletters = sorted({L for w in qw for n in w for L in n})
    qdegs = sorted({nferm(w) for w in qw})
    qframes = [(d, Frame(N, 8, max(d + 4, 6) if d else 0, [d], qletters, seed=99 + d, cartan=True)) for d in qdegs]
    XQ = evaluate(qw, qframes); qi = {w: i for i, w in enumerate(qw)}
    bad = 0
    for w in sample:
        q = Q_multi(w)
        rows = np.array([qi[m] for m in q]); cs = np.array(list(q.values()), dtype=np.int64) % P
        bad += int(np.any(matmul_mod(cs[None, :], XQ[rows])[0]))
    log(f"sanity: Q_0(word) on the Cartan nonzero for {bad} of {len(sample)} words (must be 0)")


if stage in ('su2',):
    # Restriction test: letters restricted to an su(2) subalgebra (upper-left block).  Restriction is a chain map, so a
    # Q_0-exact SU(3) operator restricts to a Q_0-exact operator of the SU(2) theory.  Contrapositive: if Q_1 v
    # restricted is not exact in SU(2), then Q_1 v is not exact in SU(3) and the class v lifts.
    from fortuity.full.dims import content_dim
    from fortuity.full.sector import contents_cached
    N2, ML2 = 2, 3
    q1s = pickle.load(open('classes/q1_reps.pkl', 'rb'))
    dim_low = sum(content_dim(N2, c) for c in contents_cached(LOW, Y))          # SU(2) weight space (4,4;10)
    ws2 = words_of(LOW, Y, ML2)                                                  # SU(2) words of the target sector
    src2 = words_of(SEC, Y - 1, ML2)                                             # SU(2) words of the source (5,5;9)
    log(f"SU(2): target words {len(ws2)} (dim {dim_low}), source words {len(src2)}")
    deg_counts = defaultdict(int)
    for w in ws2:
        deg_counts[nferm(w)] += 1
    letters = sorted({L for q in q1s for w in q for n in w for L in n} | {L for w in ws2 for n in w for L in n})
    # functionals: proportional to the number of target words per degree, total ~ 1.3 * dim + margin
    tot_need = int(1.3 * dim_low) + 3 * MARGIN
    frames = []
    for d, cnt in sorted(deg_counts.items()):
        need = max(MARGIN, int(tot_need * cnt / len(ws2)))
        k = frame_k(N2, d)
        cpp = math.comb(k, d)
        frames.append((d, Frame(N2, max(1, math.ceil(need / cpp)), k, [d], letters, seed=777 + d)))
    ncols = sum(fr.ncols for _, fr in frames)
    X2 = evaluate(ws2, frames)
    r_target = ffla.rank(X2)
    log(f"SU(2) target words rank {r_target} vs exact dim {dim_low} at {ncols} functionals ({time.time()-T0:.0f}s)")
    del X2
    # Q_0 image
    qd = [Q_multi(w) for w in src2]
    qwords = sorted({m for d in qd for m in d})
    qi = {w: i for i, w in enumerate(qwords)}
    XQ = evaluate(qwords, frames)
    img = RREFBasis(ncols)
    batch = []
    for d in qd:
        if not d:
            continue
        rows = np.array([qi[w] for w in d]); cs = np.array(list(d.values()), dtype=np.int64) % P
        batch.append(matmul_mod(cs[None, :], XQ[rows])[0])
        if len(batch) >= 256:
            img.add(np.array(batch)); batch = []
    if batch:
        img.add(np.array(batch))
    del XQ
    log(f"SU(2) Q_0-image rank {img.rank} (source words {len(src2)}) ({time.time()-T0:.0f}s)")
    # Q_1 v restricted
    vals = []
    for k, q in enumerate(q1s):
        items = list(q.items())
        val = np.zeros(ncols, dtype=np.int64)
        for s in range(0, len(items), 20000):
            chunk = items[s:s + 20000]
            Xc = evaluate([w for w, _ in chunk], frames)
            cs = np.array([c for _, c in chunk], dtype=np.int64) % P
            val = (val + matmul_mod(cs[None, :], Xc)[0]) % P
        res = img.reduce(val[None, :])[0]
        vals.append(res)
        log(f"rep {k}: Q_1 v restricted to SU(2): nonzero {int(np.count_nonzero(val))}; residual mod Q_0-exact nonzero entries {int(np.count_nonzero(res))} ({time.time()-T0:.0f}s)")
    V = np.array(vals)
    log("rank of the residuals (number of classes proven to lift by the SU(2) restriction):", ffla.rank(V))
    pickle.dump(dict(vals=vals, img_rank=img.rank, ncols=ncols, r_target=r_target, dim_low=dim_low), open('classes/su2_restriction.pkl', 'wb'))


if stage in ('u2',):
    # Restriction to the u(2) block diag(A, -tr A) of su(3) (a Lie subalgebra; Q_0 acts by graded commutators, so
    # restriction is a chain map).  Exactness is tested in the restricted theory, whose invariant space is that of
    # the U(2) theory with letters in gl(2).  Zero residual: evidence for exactness; nonzero: proof of lifting
    # (provided the source words span, see the rank report).
    from fortuity.full.dims import content_dim_u2
    from fortuity.full.sector import contents_cached
    ML_SRC = int(os.environ.get('ML_SRC', '4'))
    q1s = pickle.load(open('classes/q1_reps.pkl', 'rb'))
    dim_low = sum(content_dim_u2(c) for c in contents_cached(LOW, Y))
    dim_src = sum(content_dim_u2(c) for c in contents_cached(SEC, Y - 1))
    src = words_of(SEC, Y - 1, ML_SRC)
    log(f"u(2) restriction: target dim {dim_low}, source dim {dim_src}, source words (maxlen {ML_SRC}) {len(src)}")
    deg_counts = defaultdict(int)
    for c in contents_cached(LOW, Y):
        deg_counts[sum(1 for L in c if is_ferm(L))] += content_dim_u2(c)
    letters = sorted({L for q in q1s for w in q for n in w for L in n} | {L for w in src for n in w for L in n}
                     | {L for c in contents_cached(LOW, Y) for L in c})
    tot_need = int(1.25 * dim_low) + 3 * MARGIN
    frames = []
    for d, cnt in sorted(deg_counts.items()):
        need = max(MARGIN, int(tot_need * cnt / dim_low))
        k = frame_k(N, d)
        cpp = math.comb(k, d)
        frames.append((d, Frame(N, max(1, math.ceil(need / cpp)), k, [d], letters, seed=888 + d, u2block=True)))
    ncols = sum(fr.ncols for _, fr in frames)
    log(f"functionals {ncols}")
    # source spanning check (restricted theory): rank of the source words vs dim_src, at source-sector frames
    src_deg = defaultdict(int)
    for c in contents_cached(SEC, Y - 1):
        src_deg[sum(1 for L in c if is_ferm(L))] += content_dim_u2(c)
    sframes = []
    for d, cnt in sorted(src_deg.items()):
        need = max(MARGIN, int((int(1.25 * dim_src) + 3 * MARGIN) * cnt / dim_src))
        k = frame_k(N, d)
        sframes.append((d, Frame(N, max(1, math.ceil(need / math.comb(k, d))), k, [d], letters, seed=999 + d, u2block=True)))
    r_src = 0
    sb = RREFBasis(sum(fr.ncols for _, fr in sframes))
    for s0 in range(0, len(src), 4096):
        sb.add_batched(evaluate(src[s0:s0 + 4096], sframes))
    r_src = sb.rank
    del sb
    log(f"source words rank {r_src} vs U(2) dim {dim_src} -> {'span' if r_src == dim_src else 'DO NOT SPAN (a nonzero residual would not be a proof)'} ({time.time()-T0:.0f}s)")
    # Q_0 image
    img = RREFBasis(ncols)
    batch = []
    for s0 in range(0, len(src), 2000):
        qd = [Q_multi(w) for w in src[s0:s0 + 2000]]
        qwords = sorted({m for d in qd for m in d})
        qi = {w: i for i, w in enumerate(qwords)}
        XQ = evaluate(qwords, frames)
        for d in qd:
            if not d:
                continue
            rows = np.array([qi[w] for w in d]); cs = np.array(list(d.values()), dtype=np.int64) % P
            batch.append(matmul_mod(cs[None, :], XQ[rows])[0])
            if len(batch) >= 256:
                img.add(np.array(batch)); batch = []
        del XQ
    if batch:
        img.add(np.array(batch))
    log(f"u(2) Q_0-image rank {img.rank} ({time.time()-T0:.0f}s)")
    vals = []
    for k, q in enumerate(q1s):
        items = list(q.items())
        val = np.zeros(ncols, dtype=np.int64)
        for s0 in range(0, len(items), 20000):
            chunk = items[s0:s0 + 20000]
            Xc = evaluate([w for w, _ in chunk], frames)
            cs = np.array([c for _, c in chunk], dtype=np.int64) % P
            val = (val + matmul_mod(cs[None, :], Xc)[0]) % P
        res = img.reduce(val[None, :])[0]
        vals.append(res)
        log(f"rep {k}: Q_1 v restricted to u(2): nonzero {int(np.count_nonzero(val))}; residual mod Q_0-exact nonzero entries {int(np.count_nonzero(res))} ({time.time()-T0:.0f}s)")
    V = np.array(vals)
    log("rank of the residuals (classes proven to lift by the u(2) restriction, if the source words span):", ffla.rank(V))
    pickle.dump(dict(vals=vals, img_rank=img.rank, ncols=ncols, r_src=r_src, dim_src=dim_src, dim_low=dim_low), open('classes/u2_restriction.pkl', 'wb'))


if stage in ('u1mono',):
    # Coefficient of the abelian monomial x_1^k1 x_2^k2 x_3^k3 (phi_m -> phi_m + x_m D) of the restricted Q_1 v, tested
    # for Q_0-exactness in the SU(2) sector (4,4; 14-2k1, 8-2k2, 8-2k3) at Y = 10 - k (chain map, R-016)
    #.  k = (3,0,0) is the monomial in which the R-007 class itself restricts to the SU(2) threshold class.
    from fortuity.full.restrict import restrict_monomial
    from fortuity.full.dims import content_dim
    from fortuity.full.sector import contents_cached
    target = tuple(int(x) for x in os.environ.get('MONO', '3,0,0').split(','))
    kk = sum(target)
    N2, ML2 = 2, int(os.environ.get('ML2', '3'))
    LOW2 = (4, 4, 14 - 2 * target[0], 8 - 2 * target[1], 8 - 2 * target[2])
    SRC2 = (5, 5) + LOW2[2:]
    Y2 = Y - kk
    q1s = pickle.load(open('classes/q1_reps.pkl', 'rb'))
    dim_low = sum(content_dim(N2, c) for c in contents_cached(LOW2, Y2))
    dim_src = sum(content_dim(N2, c) for c in contents_cached(SRC2, Y2 - 1))
    src = words_of(SRC2, Y2 - 1, ML2)
    log(f"monomial x^{target}: SU(2) target {LOW2} Y={Y2} dim {dim_low}; source {SRC2} Y={Y2-1} dim {dim_src}, words {len(src)}")
    # restricted Q_1 v
    restr = []
    for k, q in enumerate(q1s):
        t = time.time()
        acc = defaultdict(int)
        for w, c in q.items():
            for m2, c2 in restrict_monomial(w, N, target).items():
                acc[m2] = (acc[m2] + c * c2) % P
        r = {m: c for m, c in acc.items() if c}
        restr.append(r)
        log(f"rep {k}: restricted coefficient has {len(r)} SU(2) words ({time.time()-t:.0f}s)")
    letters = sorted({L for r in restr for w in r for n in w for L in n} | {L for w in src for n in w for L in n}
                     | {L for c in contents_cached(LOW2, Y2) for L in c})
    deg_counts = defaultdict(int)
    for c in contents_cached(LOW2, Y2):
        deg_counts[sum(1 for L in c if is_ferm(L))] += content_dim(N2, c)
    tot_need = int(1.3 * dim_low) + 3 * MARGIN
    frames = []
    for d, cnt in sorted(deg_counts.items()):
        need = max(MARGIN, int(tot_need * cnt / max(dim_low, 1)))
        k = frame_k(N2, d)
        frames.append((d, Frame(N2, max(1, math.ceil(need / math.comb(k, d))), k, [d], letters, seed=555 + d + 31 * kk)))
    ncols = sum(fr.ncols for _, fr in frames)
    # source spanning (at source-sector frames) and Q_0 image
    sdeg = defaultdict(int)
    for c in contents_cached(SRC2, Y2 - 1):
        sdeg[sum(1 for L in c if is_ferm(L))] += content_dim(N2, c)
    sframes = []
    for d, cnt in sorted(sdeg.items()):
        need = max(MARGIN, int((int(1.3 * dim_src) + 3 * MARGIN) * cnt / max(dim_src, 1)))
        k = frame_k(N2, d)
        sframes.append((d, Frame(N2, max(1, math.ceil(need / math.comb(k, d))), k, [d], letters, seed=666 + d + 31 * kk)))
    sb = RREFBasis(sum(fr.ncols for _, fr in sframes))
    for s0 in range(0, len(src), 4096):
        sb.add_batched(evaluate(src[s0:s0 + 4096], sframes))
    r_src = sb.rank
    del sb
    log(f"source words rank {r_src} vs exact dim {dim_src} -> {'span' if r_src == dim_src else 'DO NOT SPAN (raise ML2)'} ({time.time()-T0:.0f}s)")
    img = RREFBasis(ncols)
    batch = []
    for s0 in range(0, len(src), 2000):
        qd = [Q_multi(w) for w in src[s0:s0 + 2000]]
        qwords = sorted({m for d in qd for m in d})
        qi = {w: i for i, w in enumerate(qwords)}
        XQ = evaluate(qwords, frames)
        for d in qd:
            if not d:
                continue
            rows = np.array([qi[w] for w in d]); cs = np.array(list(d.values()), dtype=np.int64) % P
            batch.append(matmul_mod(cs[None, :], XQ[rows])[0])
            if len(batch) >= 256:
                img.add(np.array(batch)); batch = []
    if batch:
        img.add(np.array(batch))
    log(f"SU(2) Q_0-image rank {img.rank} at {ncols} functionals ({time.time()-T0:.0f}s)")
    vals = []
    for k, r in enumerate(restr):
        if not r:
            vals.append(np.zeros(ncols, dtype=np.int64)); log(f"rep {k}: restricted coefficient is 0"); continue
        items = list(r.items())
        val = np.zeros(ncols, dtype=np.int64)
        for s0 in range(0, len(items), 20000):
            chunk = items[s0:s0 + 20000]
            Xc = evaluate([w for w, _ in chunk], frames)
            cs = np.array([c for _, c in chunk], dtype=np.int64) % P
            val = (val + matmul_mod(cs[None, :], Xc)[0]) % P
        res = img.reduce(val[None, :])[0]
        vals.append(res)
        log(f"rep {k}: x^{target} coefficient: nonzero {int(np.count_nonzero(val))}; residual mod Q_0-exact nonzero entries {int(np.count_nonzero(res))} ({time.time()-T0:.0f}s)")
    V = np.array(vals)
    log(f"rank of the residuals for x^{target} (classes proven to lift if the source words span): {ffla.rank(V)}")
    pickle.dump(dict(vals=vals, img_rank=img.rank, ncols=ncols, r_src=r_src, dim_src=dim_src, dim_low=dim_low, target=target),
                open(f'classes/u1mono_{"_".join(map(str, target))}.pkl', 'wb'))
