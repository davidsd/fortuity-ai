"""SU(3)_R highest-weight vectors in the free multitrace word space, shape by shape.

Raising operators (Lie algebra action commuting with Q, see fortuity.bmn.words for Q):
  E1 = phi_1 d/dphi_2 - psi_2 d/dpsi_1     (weight + (1,-1,0))
  E2 = phi_2 d/dphi_3 - psi_3 d/dpsi_2     (weight + (0,1,-1))
Both are even derivations that only relabel flavours, so they preserve the flavour-stripped shape of a multitrace.
hw vectors of weight lam = common kernel of E1, E2 on W_lam; computed per shape as a left null space mod p.
"""
import numpy as np
from collections import defaultdict
from fortuity.bmn.words import canon_word, canon_multi
from fortuity.bmn.modp import P, matmul_mod, RREFBasis
from .wordenum import shape
from . import ffla

EMAP = {1: {1: (1, 0), 3: (-1, 4)},
        2: {2: (1, 1), 4: (-1, 5)}}


_CANON = {}
_PAR = {}


def _canon(w):
    r = _CANON.get(w)
    if r is None:
        r = canon_word(w)
        _CANON[w] = r
        if len(_CANON) > 5_000_000:
            _CANON.clear()
    return r


def _par(w):
    r = _PAR.get(w)
    if r is None:
        r = sum(1 for l in w if 3 <= l <= 5) & 1
        _PAR[w] = r
    return r


def _sorted_sign(ws):
    """(sign, sorted tuple) for a list of canonical necklaces; sign from reordering odd ones; 0 if an odd repeats."""
    odd = [w for w in ws if _par(w)]
    srt = tuple(sorted(ws))
    if len(odd) < 2:
        return 1, srt
    if len(set(odd)) < len(odd):
        return 0, None
    order = sorted(range(len(odd)), key=lambda i: odd[i])
    inv = 0
    for a in range(len(order)):
        oa = order[a]
        for b in range(a + 1, len(order)):
            if oa > order[b]:
                inv += 1
    return (-1 if inv & 1 else 1), srt


def E_multi(multi, i):
    """E_i acting on a canonical multitrace -> dict {multitrace: coeff}."""
    out = {}
    mp = EMAP[i]
    for a, w in enumerate(multi):
        for pos, l in enumerate(w):
            if l in mp:
                c, nl = mp[l]
                s, cw = _canon(w[:pos] + (nl,) + w[pos + 1:])
                if not s:
                    continue
                sm, cm = _sorted_sign(multi[:a] + (cw,) + multi[a + 1:])
                if not sm:
                    continue
                out[cm] = out.get(cm, 0) + c * s * sm
    return {m: c for m, c in out.items() if c}


def nullspace_left_slow(M, p=P):
    """Reference implementation (pivot by pivot) of the left null space mod p."""
    r, c = M.shape
    A = np.concatenate([M % p, np.eye(r, dtype=np.int64)], axis=1)
    row = 0
    for col in range(c):
        if row == r:
            break
        nz = np.nonzero(A[row:, col])[0]
        if nz.size == 0:
            continue
        piv = row + nz[0]
        if piv != row:
            A[[row, piv]] = A[[piv, row]]
        inv = pow(int(A[row, col]), p - 2, p)
        A[row] = (A[row] * inv) % p
        f = A[:, col].copy()
        f[row] = 0
        nzr = np.nonzero(f)[0]
        if nzr.size:
            A[nzr] = (A[nzr] - np.outer(f[nzr], A[row])) % p
        row += 1
    return A[row:, c:]


def nullspace_left(M, p=P):
    """Basis (rows) of {v : v M = 0 mod p} (FLINT null space of M^T)."""
    r, c = M.shape
    if r == 0:
        return np.zeros((0, 0), dtype=np.int64)
    if c == 0:
        return np.eye(r, dtype=np.int64)
    MT = np.ascontiguousarray(np.asarray(M).T)
    if _ffblas is not None and r * c >= BIG_NULLSPACE:
        return _ffblas.nullspace_right(MT, p)       # BLIS kernel (threaded) for large matrices
    return ffla.nullspace_right(MT, p)


# entries; above this the BLIS kernel is used for null spaces.  On SU(4) j=28 (2;6,4,4) this took the hw stage from
# 1183 s to 397 s (serial, 32 threads); forked pools over shapes were slower than serial (535-1241 s), so the parallel
# engine keeps hw vectors serial when FLINT is present.
BIG_NULLSPACE = 1_000_000
BIG_M2 = 300_000_000        # stage-2 matrices above this many entries are streamed (see _hw_shape)
STREAM_CHUNK = 200_000_000  # entries per streamed chunk


E1CLASS = (10, 10, 2, 11, 11, 5, 6)   # merge the E1 doublets: phi1,phi2 -> A ; psi1,psi2 -> B


def e1_class(multi):
    return tuple(sorted(_min_rot(tuple(E1CLASS[l] for l in w)) for w in multi))


def _min_rot(t):
    return min(t[i:] + t[:i] for i in range(len(t)))


def left_kernel(M, p=P, rng=None):
    """Left null space of M (r x c) mod p; when c > r + 16 the columns are first compressed by a dense random
    projection (BLIS kernel).  A failed projection can only enlarge the computed kernel; callers verify the final
    result (rank = m check, two-sided)."""
    r, c = M.shape
    if c > r + 16:
        rng = rng or np.random.default_rng(12345)
        R = rng.integers(0, p, (c, r + 16), dtype=np.int64)
        M = _mm(M % p, R)
    return nullspace_left(M)


def _mm(A, B):
    """mod-p matmul: BLIS kernel if built, else numpy (fortuity.bmn.modp.matmul_mod)."""
    if _ffblas is not None:
        return _ffblas.matmul(A, B)
    return matmul_mod(A, B)


from . import ffblas as _ffblas
if not _ffblas.HAVE_FFBLAS:
    _ffblas = None


def _E_rows(words, idxs, i):
    """Sparse E_i images of the given words: list of dicts."""
    return [E_multi(words[j], i) for j in idxs]


def hw_vectors(words, stats=None, workers=1):
    """words: canonical multitraces spanning W_lam (one weight).  Returns a list of blocks (word_indices, H) with
    H (k x len(word_indices)) int64 mod p whose rows span the hw vectors of that shape.
    Stage 1: ker E1 inside E1-classes (words related by 1<->2 relabelling); stage 2: ker E2 on that, per shape.
    workers > 1: shapes are processed by a pool of forked processes."""
    by_shape = defaultdict(list)
    for idx, w in enumerate(words):
        by_shape[shape(w)].append(idx)
    groups = sorted(by_shape.values(), key=len, reverse=True)
    if workers > 1 and len(groups) > 1:
        # Fork the pool of single-threaded workers BEFORE any call of the threaded kernel in this process, and do every
        # shape (the largest included, largest first) inside it. The previous order (largest shape in the parent with
        # the kernel's full thread count, then fork the pool) deadlocked on PI Symmetry at j=30, SO(7) (6,5,4) 2J=3:
        # the 40 forked workers inherited the parent's live MKL/OpenMP runtime and slept for 4 h at 0% CPU (job 574592,
        # 2026-10-14). The medium sector escaped only because its largest shape stayed below BIG_NULLSPACE.
        import multiprocessing as mp
        global _HW_WORDS
        _HW_WORDS = words
        with mp.get_context('fork').Pool(workers, initializer=_single_thread) as pool:
            res = pool.map(_hw_shape_task, groups, chunksize=1)
        _HW_WORDS = None
    else:
        res = [_hw_shape(words, g) for g in groups]
    blocks = []
    for r in res:
        if r is None:
            continue
        idxs, H, st = r
        if stats is not None:
            stats.append(st)
        if H.shape[0]:
            blocks.append((np.array(idxs), H))
    return blocks


_HW_WORDS = None


def _hw_shape_task(idxs):
    return _hw_shape(_HW_WORDS, idxs)


def _single_thread():
    if _ffblas is not None:
        _ffblas.set_threads(1)


def _hw_shape(words, idxs):
    """hw vectors of one shape block: returns (idxs, H, stats) or None."""
    if True:
        pos = {j: a for a, j in enumerate(idxs)}
        # stage 1
        classes = defaultdict(list)
        for j in idxs:
            classes[e1_class(words[j])].append(j)
        K1 = []                                  # list of (word indices, coefficient rows)
        for cl, js in classes.items():
            imgs = _E_rows(words, js, 1)
            cols = {}
            for d in imgs:
                for m in d:
                    cols.setdefault(m, len(cols))
            if cols:
                M = np.zeros((len(js), len(cols)), dtype=np.int64)
                for r, d in enumerate(imgs):
                    for m, c in d.items():
                        M[r, cols[m]] = c
                N1 = nullspace_left(M)
            else:
                N1 = np.eye(len(js), dtype=np.int64)
            if N1.shape[0]:
                K1.append((js, N1))
        if not K1:
            return None
        # stage 2: E2 on the K1 vectors, assembled class by class (K1 is block-sparse, E2 rows are sparse)
        nk = sum(N1.shape[0] for _, N1 in K1)
        cols = {}
        blocksM = []
        for js, N1 in K1:
            ent = []
            loc = {}
            for a, j in enumerate(js):
                for m, c in E_multi(words[j], 2).items():
                    cc = cols.setdefault(m, len(cols))
                    b = loc.setdefault(cc, len(loc))
                    ent.append((a, b, c))
            if loc:
                E2c = np.zeros((len(js), len(loc)), dtype=np.int64)
                for a, b, c in ent:
                    E2c[a, b] += c
                blocksM.append((_mm(N1, E2c % P), np.fromiter(loc.keys(), dtype=np.int64)))
            else:
                blocksM.append((None, None))
        if cols and _ffblas is not None and nk * len(cols) > BIG_M2:
            # huge shape: never form M2 (nk x #images); stream M2^T in chunks of image columns into a row-echelon
            # basis (BLIS kernel) and take its null space = left kernel of M2 (exact, no projection)
            T = len(cols)
            SB = _ffblas.StreamBasis(nk)
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
        H = np.zeros((N2.shape[0], len(idxs)), dtype=np.int64)
        if N2.shape[0]:
            r0 = 0
            for js, N1 in K1:
                cidx = [pos[j] for j in js]
                H[:, cidx] = (H[:, cidx] + _mm(N2[:, r0:r0 + N1.shape[0]], N1)) % P
                r0 += N1.shape[0]
        return idxs, H.astype(np.int32), (len(idxs), nk, H.shape[0])   # int32: entries < p < 2^31 (memory)
