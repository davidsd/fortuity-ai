"""One production unit: sector (N, lam, n = 2J) of the SU(3)_R highest-weight complex.

Computes, with functionals of sector n only:
  rank_C  = rank of the hw vectors of W_(n, lam)            (must equal m_n: exact self-check, per fermion degree)
  r_up    = rank of Q on hw(V_(n+1, lam))                    (Q-images of hw vectors of W_(n+1, lam))
  grav    = rank of hw graviton vectors modulo Q-exact       (in hw(H_(n, lam)))
Then h_n = m_n - r_n - r_{n+1} and BH_n = h_n - grav_n once r_n (from sector n-1) is known.

Evaluation is split by fermion degree d (the evaluation space is a direct sum over d) and into point chunks, so the
full word-evaluation matrix is never stored.  need[d] = m_(n, d) + margin functionals per degree, doubled on failure.
"""
import math
import time
import numpy as np
from fortuity.bmn.words import contents_of_sector, Q_multi
from fortuity.bmn.modp import matmul_mod, P
from . import wordenum as enum, ffla
from .hw import hw_vectors
from .evalg import Frame, nferm, algebra_dim
from .column import SHIFTS

_QW = None


def _q_chunk(idxs):
    return [Q_multi(_QW[i]) for i in idxs]


KMAX_TARGET = 12   # functionals per point; small = many bosonic points (one point is not faithful)


_DIMS_LEVEL = {}


def _dims_level_max(dims_nf):
    """Highest level j = 2 sum(q) present in an nf-refined dims table (keys (2J, q1, q2, q3, nf))."""
    key = id(dims_nf)
    if key not in _DIMS_LEVEL or _DIMS_LEVEL[key][0] is not dims_nf:
        _DIMS_LEVEL[key] = (dims_nf, max(2 * sum(k[1:4]) for k in dims_nf))
    return _DIMS_LEVEL[key][1]


def hw_mult_nf(dims_nf, n, lam, f):
    tot = 0
    for sh, s in SHIFTS:
        q = tuple(a + b for a, b in zip(lam, sh))
        if min(q) < 0:
            continue
        tot += s * dims_nf.get((n,) + tuple(sorted(q, reverse=True)) + (f,), 0)
    return tot


def _hw_cached(sec, maxlen, words, workers):
    """hw_vectors(words), optionally cached on disk (env HW_CACHE = directory). hw vectors depend only on the formal
    words of the sector, not on the gauge group or the seed, so one cache serves every group and seed. The cache stores
    the word list too and is used only if it matches exactly."""
    import os
    import pickle
    d = os.environ.get('HW_CACHE')
    fn = None
    if d:
        fn = os.path.join(d, 'hw_' + '_'.join(map(str, sec)) + f'_ml{maxlen}.pkl')
        if os.path.exists(fn):
            try:
                with open(fn, 'rb') as fh:
                    cw, blocks = pickle.load(fh)
                if cw == list(words):
                    return [(idx, H.astype(np.int64)) for idx, H in blocks]
            except Exception:
                pass
    blocks = hw_vectors(words, workers=workers)
    if fn:
        os.makedirs(d, exist_ok=True)
        tmp = f'{fn}.tmp{os.getpid()}'
        with open(tmp, 'wb') as fh:
            pickle.dump((list(words), [(idx, H.astype(np.int32)) for idx, H in blocks]), fh, protocol=4)
        os.replace(tmp, fn)
    return blocks


def frame_k(N, d, extra=2, target=KMAX_TARGET, group='SU'):
    kmax = 3 * algebra_dim(group, N)
    if d == 0:
        return 0
    k = min(kmax, d + extra)
    while k < kmax and math.comb(k, d) < target:
        k += 1
    return k


class Sector:
    def __init__(self, N, lam, n, dims_nf, margin=32, chunk_cols=4096, seed=1, log=print, graviton_algebra=None, maxlen=None, pre=None, group='SU', maxlen_up=None):
        # the exact-dimension table must reach this sector's level (one level up too, for the Q-images): a table computed
        # to a lower j gives m = 0 and a vacuous self-check (review B.4)
        lvl = 2 * sum(lam)
        if dims_nf and lvl > _dims_level_max(dims_nf):
            raise ValueError(f"dims table reaches j = {_dims_level_max(dims_nf)} < {lvl} needed by sector {n} {tuple(lam)}")
        self.N, self.lam, self.n = N, tuple(lam), n
        self.group = group
        self.dims_nf, self.margin, self.chunk_cols, self.seed, self.log = dims_nf, margin, chunk_cols, seed, log
        self.GA = graviton_algebra
        self.maxlen = maxlen
        # trace-length cap for the words of sector n+1 (Q-images).  Valid when sector n+1's own spanning check passed
        # with a cap <= maxlen_up (a larger cap only adds words); defaults to maxlen.
        self.maxlen_up = maxlen if maxlen_up is None else maxlen_up
        self.pre = pre           # (words, hw blocks) of this sector, computed by the previous sector's job
        self.timing = {}

    def _t(self, key, t0):
        self.timing[key] = self.timing.get(key, 0) + time.time() - t0

    def prepare(self):
        n, lam = self.n, self.lam
        t = time.time()
        if self.pre is not None:
            self.ws = list(self.pre[0])
        else:
            self.ws = enum.words_of_sector((n,) + lam, self.maxlen)
        self.idx = {w: i for i, w in enumerate(self.ws)}
        self.deg = np.array([nferm(w) for w in self.ws], dtype=np.int64)
        up = (n + 1,) + lam
        self.ws_up = enum.words_of_sector(up, self.maxlen_up) if contents_of_sector(up) else []
        self._t('enumerate', t)
        self.log(f"  words: n={n}: {len(self.ws)}, n+1: {len(self.ws_up)} ({time.time()-t:.1f}s)")
        t = time.time()
        hw_workers = getattr(self, 'hw_workers', 1)
        self.hw = self.pre[1] if self.pre is not None else _hw_cached((n,) + lam, self.maxlen, self.ws, hw_workers)
        self.hw_up = _hw_cached(up, self.maxlen_up, self.ws_up, hw_workers) if self.ws_up else []
        self._t('hw', t)
        self.log(f"  hw vectors: n={n}: {sum(H.shape[0] for _, H in self.hw)}, "
                 f"n+1: {sum(H.shape[0] for _, H in self.hw_up)} ({time.time()-t:.1f}s)")
        # Q-images of hw vectors of n+1 as sparse combinations of words of n: blocks (target word idx, C)
        t = time.time()
        self.qblocks, self.qsparse = [], []
        qimg = self._q_images(sorted({int(wi) for widx, _ in self.hw_up for wi in widx}))
        for widx, H in self.hw_up:
            rs, js, cs = [], [], []
            for r, wi in enumerate(widx):
                for m2, c in qimg.pop(int(wi)).items():
                    j = self.idx.get(m2)
                    if j is None:                  # Q lengthens traces: extra evaluation-only words
                        j = self.idx[m2] = len(self.ws)
                        self.ws.append(m2)
                    rs.append(r)
                    js.append(j)
                    cs.append(c % P)
            if not rs:
                continue
            rs, js, cs = (np.array(x, dtype=np.int64) for x in (rs, js, cs))
            if getattr(self, 'sparse_q', False):
                # keep Q sparse: B rows = H (Q X), never the dense Q matrix or H Q (see sectorpar)
                self.qsparse.append((H, rs, js, cs))
            else:
                tg, jl = np.unique(js, return_inverse=True)
                Qm = np.zeros((len(widx), len(tg)), dtype=np.int64)
                np.add.at(Qm, (rs, jl), cs)
                C = matmul_mod(H, Qm % P)
                keep = np.flatnonzero(C.any(axis=0))
                if keep.size:
                    self.qblocks.append((tg[keep], C[:, keep]))
        del qimg
        self._t('qimages', t)
        self.deg = np.array([nferm(w) for w in self.ws], dtype=np.int64)
        # graviton hw blocks
        if self.GA is not None:
            t = time.time()
            self.gblocks = self.GA.hw_blocks((n,) + lam)
            self._t('grav_hw', t)
        present = {int(d) for d in np.unique(self.deg)}
        exact = {d: hw_mult_nf(self.dims_nf, n, lam, (n - d) // 2) for d in range(n % 2, n + 1, 2)}
        missing = {d: v for d, v in exact.items() if v and d not in present}
        if missing:
            # a fermion degree with nonzero exact multiplicity but no words (e.g. MAXLEN too small) used to drop out of
            # the self-check, which then passed vacuously (review B.4)
            raise RuntimeError(f"sector {n} {lam}: no words of fermion degree(s) {sorted(missing)} although the exact "
                               f"multiplicity is {missing}: the word set does not span (maxlen too small?)")
        self.m_d = {d: exact[d] for d in sorted(present)}
        self.m = sum(self.m_d.values())

    def evaluate(self, factor=1, attempt=0):
        """Fill A (hw rows), B (Q rows), G (graviton rows) column blocks, degree by degree, chunk by chunk."""
        N = self.N
        colblocks = []          # (d, frame)
        for d, md in sorted(self.m_d.items()):
            need = factor * (md + self.margin)
            k = frame_k(N, d, group=self.group)
            cpp = math.comb(k, d)
            if cpp == 0:
                # more fermions than Grassmann components of the algebra (d > 3 dim g): these words vanish
                # identically, so the exact multiplicity must be 0 and no functionals are needed
                assert md == 0, (d, md)
                continue
            total_pts = max(1, math.ceil(need / cpp))
            per = max(1, self.chunk_cols // cpp)
            for c0 in range(0, total_pts, per):
                npts = min(per, total_pts - c0)
                colblocks.append((d, Frame(N, npts, k, [d], seed=self.seed + 7919 * d + 104729 * attempt + c0, group=self.group)))
        ncols = sum(fr.ncols for _, fr in colblocks)
        nA = sum(H.shape[0] for _, H in self.hw)
        nB = sum(C.shape[0] for _, C in self.qblocks)
        A = np.zeros((nA, ncols), dtype=np.int64)
        B = np.zeros((nB, ncols), dtype=np.int64)
        gb = self.gblocks if self.GA is not None else []
        nG = sum(H.shape[0] for _, H in gb)
        G = np.zeros((nG, ncols), dtype=np.int64)
        self.log(f"  functionals: {ncols} in {len(colblocks)} chunks; rows A {nA}, B {nB}, G {nG}")
        col = 0
        for d, fr in colblocks:
            t = time.time()
            wd = np.flatnonzero(self.deg == d)
            X = np.zeros((len(self.ws), fr.ncols), dtype=np.int64)
            X[wd] = fr.eval_words([self.ws[i] for i in wd])
            self._t('eval', t)
            t = time.time()
            r0 = 0
            for widx, H in self.hw:
                A[r0:r0 + H.shape[0], col:col + fr.ncols] = matmul_mod(H, X[widx])
                r0 += H.shape[0]
            r0 = 0
            for tidx, C in self.qblocks:
                B[r0:r0 + C.shape[0], col:col + fr.ncols] = matmul_mod(C, X[tidx])
                r0 += C.shape[0]
            self._t('combine', t)
            if nG:
                t = time.time()
                from .evalg import DegFrames
                df = DegFrames.__new__(DegFrames)
                df.frames, df.off, df.ncols = {d: fr}, {d: 0}, fr.ncols
                G[:, col:col + fr.ncols] = self.GA.rows_from_blocks(gb, df)
                self._t('grav_eval', t)
            col += fr.ncols
            del X
        self.colblocks = colblocks
        return A, B, G

    def _q_images(self, idxs):
        """{word index in ws_up: Q(word)} (parallel over words when hw_workers > 1)."""
        W = getattr(self, 'q_workers', 1)
        if W <= 1 or len(idxs) < 1000:
            return {i: Q_multi(self.ws_up[i]) for i in idxs}
        import multiprocessing as mp
        global _QW
        _QW = self.ws_up
        chunks = [idxs[k::W * 4] for k in range(W * 4)]
        with mp.get_context('fork').Pool(W) as pool:
            parts = pool.map(_q_chunk, chunks)
        _QW = None
        out = {}
        for ch, part in zip(chunks, parts):
            out.update(zip(ch, part))
        return out

    def handoff(self):
        """(words, hw blocks) of sector n+1, for the next sector's job."""
        return (self.ws_up, self.hw_up) if self.ws_up else None

    def run(self, max_attempts=5):
        t0 = time.time()
        self.prepare()
        res = dict(N=self.N, lam=list(self.lam), n=self.n, maxlen=self.maxlen, maxlen_up=self.maxlen_up, m=self.m, m_d={str(k): v for k, v in self.m_d.items()},
                   words=len(self.ws), words_up=len(self.ws_up))
        if self.m == 0:
            res.update(rank_C=0, check=True, r_up=0, grav=0, time=time.time() - t0)
            return res
        factor = 1
        prev = None
        for attempt in range(max_attempts):
            A, B, G = self.evaluate(factor, attempt)
            t = time.time()
            # per-degree check: columns of degree d
            ok, col, rk = True, 0, {}
            for d in sorted(self.m_d):
                w = sum(fr.ncols for dd, fr in self.colblocks if dd == d)
                rows = np.flatnonzero(A[:, col:col + w].any(axis=1))
                rk[d] = ffla.rank(A[rows, col:col + w]) if rows.size else 0
                ok &= (rk[d] == self.m_d[d])
                col += w
            self._t('rank', t)
            self.log(f"  check (attempt {attempt}): rank per degree {rk} vs m_d {self.m_d} -> {'OK' if ok else 'FAIL'}")
            if ok:
                break
            if prev == rk:
                self.log("  rank did not grow with more functionals: the word set does not span (maxlen too small?)")
                res['failure'] = 'spanning'
                break
            prev = rk
            factor *= 2
        res['rank_C_d'] = {str(k): v for k, v in rk.items()}
        res['check'] = bool(ok)
        del A
        t = time.time()
        res['r_up'] = ffla.rank(B) if B.shape[0] else 0
        if G.shape[0]:
            res['grav'] = ffla.rank(np.concatenate([B, G], axis=0)) - res['r_up']
            res['grav_rows'] = int(G.shape[0])
        else:
            res['grav'] = 0
        self._t('rank', t)
        res['ncols'] = int(B.shape[1])
        res['timing'] = {k: round(v, 1) for k, v in self.timing.items()}
        res['time'] = round(time.time() - t0, 1)
        return res
