"""One production unit of the full-letter SU(3)_R highest-weight complex: sector s = (2J1, 2J2, 2q1, 2q2, 2q3) at
fixed letter number Y (Q raises Y by one and lowers 2J1, 2J2 by one).  Computes, with functionals of s only:
  rank_C = rank of the hw vectors of W_s (must equal the exact hw multiplicity m_s, per fermion degree)
  r_up   = rank of Q on hw(W_{s+}), s+ = (2J1+1, 2J2+1, 2q) at Y-1   (Q-images evaluated in s)
so that h_hw(s) = m_s - r_s - r_up(s), with r_s = r_up of the sector below (2J1-1, 2J2-1, 2q; Y+1).
Follows fortuity.hwcoh.sector; words are restricted to traces of length <= maxlen and the rank = m check
verifies that they span the hw subspace.
group = 'SU' (default), 'SO_odd' (SO(2N+1)) or 'Sp' (USp(2N)) (G2 letters are not in full.evalg yet): exact multiplicities from fortuity.full.gdims and
letters in that Lie algebra (the full-letter SO/USp development code, 2026-10-08; promoted in T-048). The complex, the word
enumeration, the hw vectors and Q are group independent."""
import math, os, pickle, time
import numpy as np
from fortuity.bmn.modp import matmul_mod, P
from fortuity.hwcoh import ffla
from .words import Q_multi, contents_of_sector
from .wordenum import multitraces_with_content
from .hw import hw_vectors
from .evalg import Frame, nferm
from .dims import content_dim, weyl_orbit_signed
from .letters import is_ferm
from ..hwcoh.evalg import algebra_dim

KMAX_TARGET = 12


def frame_k(N, d, extra=2, target=KMAX_TARGET, group='SU'):
    kmax = 3 * (N * N - 1) if group == 'SU' else 3 * algebra_dim(group, N)
    if d == 0:
        return 0
    k = min(kmax, d + extra)
    while k < kmax and math.comb(k, d) < target:
        k += 1
    return k


# Disk cache for letter contents and hw vectors.  None (default) = no disk cache.  Set the environment variable
# FORTUITY_FULL_CACHE, assign fortuity.full.sector.CACHE_DIR, or pass cache= / hw_cache= explicitly.
CACHE_DIR = os.environ.get('FORTUITY_FULL_CACHE')


def contents_cached(sec, Y, cache=None):
    cache = CACHE_DIR if cache is None else cache
    if cache is None:
        return [c for c in contents_of_sector(sec) if len(c) == Y]
    fn = os.path.join(cache, f'contents_{"_".join(map(str, sec))}_Y{Y}.pkl')
    if os.path.exists(fn):
        return pickle.load(open(fn, 'rb'))
    cs = [c for c in contents_of_sector(sec) if len(c) == Y]
    os.makedirs(cache, exist_ok=True)
    pickle.dump(cs, open(fn, 'wb'))
    return cs


def words_of(sec, Y, maxlen):
    out = []
    for c in contents_cached(sec, Y):
        out += multitraces_with_content(c, maxlen)
    return out


def hw_mult_by_degree(N, sec, Y):
    """{fermion degree: exact multiplicity of the SU(3)_R irrep with highest weight q = sec[2:]/2 in W_(sec, Y)}."""
    out = {}
    for sgn, w in weyl_orbit_signed(sec[2:]):
        if min(w) < 0:
            continue
        s2 = sec[:2] + w
        for c in contents_cached(s2, Y):
            d = sum(1 for L in c if is_ferm(L))
            out[d] = out.get(d, 0) + sgn * content_dim(N, c)
    return {d: m for d, m in out.items() if m}


def hw_mult_by_degree_group(group, N, sec, Y):
    """hw_mult_by_degree for any group of fortuity.full.gdims (SO(2N+1), USp(2N); SU(N) too)."""
    from .gdims import content_dim as gdim
    out = {}
    for sgn, w in weyl_orbit_signed(sec[2:]):
        if min(w) < 0:
            continue
        s2 = tuple(sec[:2]) + tuple(w)
        for c in contents_cached(s2, Y):
            d = sum(1 for L in c if is_ferm(L))
            out[d] = out.get(d, 0) + sgn * gdim(group, N, c)
    return {d: m for d, m in out.items() if m}


class Sector:
    def __init__(self, N, sec, Y, maxlen, margin=32, chunk_cols=4096, seed=1, log=print, hw_cache=None, group='SU'):
        self.N, self.sec, self.Y, self.maxlen, self.group = N, tuple(sec), Y, maxlen, group
        self.margin, self.chunk_cols, self.seed, self.log = margin, chunk_cols, seed, log
        self.hw_cache = CACHE_DIR if hw_cache is None else hw_cache
        self.timing = {}

    def _t(self, key, t0):
        self.timing[key] = self.timing.get(key, 0) + time.time() - t0

    def _hw(self, sec, Y, words):
        if self.hw_cache is None:
            return hw_vectors(words)
        fn = os.path.join(self.hw_cache, f'hw_{"_".join(map(str, sec))}_Y{Y}_ml{self.maxlen}.pkl')
        if os.path.exists(fn):
            nw, blocks = pickle.load(open(fn, 'rb'))
            if nw == len(words):
                return [(i, H.astype(np.int64)) for i, H in blocks]
        blocks = hw_vectors(words)
        os.makedirs(self.hw_cache, exist_ok=True)
        pickle.dump((len(words), [(i, H.astype(np.int32)) for i, H in blocks]), open(fn, 'wb'), protocol=4)
        return blocks

    def prepare(self):
        sec, Y = self.sec, self.Y
        t = time.time()
        self.ws = words_of(sec, Y, self.maxlen)
        self.idx = {w: i for i, w in enumerate(self.ws)}
        up = (sec[0] + 1, sec[1] + 1) + sec[2:]
        self.ws_up = words_of(up, Y - 1, self.maxlen) if Y > 1 else []
        self._t('enumerate', t)
        self.log(f"  words: {sec} Y={Y}: {len(self.ws)}; up {up} Y={Y-1}: {len(self.ws_up)} ({time.time()-t:.1f}s)")
        t = time.time()
        self.hw = self._hw(sec, Y, self.ws)
        self.hw_up = self._hw(up, Y - 1, self.ws_up) if self.ws_up else []
        self._t('hw', t)
        self.log(f"  hw vectors: {sum(H.shape[0] for _, H in self.hw)}; up: {sum(H.shape[0] for _, H in self.hw_up)} ({time.time()-t:.1f}s)")
        t = time.time()
        self.qblocks = []
        for widx, H in self.hw_up:
            rs, js, cs = [], [], []
            for r, wi in enumerate(widx):
                for m2, c in Q_multi(self.ws_up[int(wi)]).items():
                    j = self.idx.get(m2)
                    if j is None:                      # Q lengthens traces: evaluation-only words
                        j = self.idx[m2] = len(self.ws)
                        self.ws.append(m2)
                    rs.append(r); js.append(j); cs.append(c % P)
            if not rs:
                continue
            rs, js, cs = (np.array(x, dtype=np.int64) for x in (rs, js, cs))
            tg, jl = np.unique(js, return_inverse=True)
            Qm = np.zeros((len(widx), len(tg)), dtype=np.int64)
            np.add.at(Qm, (rs, jl), cs)
            C = matmul_mod(H, Qm % P)
            keep = np.flatnonzero(C.any(axis=0))
            if keep.size:
                self.qblocks.append((tg[keep], C[:, keep]))
        self._t('qimages', t)
        self.deg = np.array([nferm(w) for w in self.ws], dtype=np.int64)
        self.letters = sorted({L for w in self.ws for n in w for L in n})
        self.m_d = (hw_mult_by_degree(self.N, sec, Y) if self.group == 'SU'
                    else hw_mult_by_degree_group(self.group, self.N, sec, Y))
        self.m = sum(self.m_d.values())
        self.log(f"  exact hw multiplicity m = {self.m} by fermion degree {self.m_d}")

    def evaluate(self, factor=1, attempt=0):
        colblocks = []
        for d, md in sorted(self.m_d.items()):
            need = factor * (md + self.margin)
            k = frame_k(self.N, d, group=self.group)
            cpp = math.comb(k, d)
            total_pts = max(1, math.ceil(need / cpp))
            per = max(1, self.chunk_cols // cpp)
            for c0 in range(0, total_pts, per):
                npts = min(per, total_pts - c0)
                colblocks.append((d, Frame(self.N, npts, k, [d], self.letters, group=self.group,
                                           seed=self.seed + 7919 * d + 104729 * attempt + c0)))
        ncols = sum(fr.ncols for _, fr in colblocks)
        nA = sum(H.shape[0] for _, H in self.hw)
        nB = sum(C.shape[0] for _, C in self.qblocks)
        A = np.zeros((nA, ncols), dtype=np.int64)
        B = np.zeros((nB, ncols), dtype=np.int64)
        self.log(f"  functionals: {ncols} in {len(colblocks)} chunks; rows A {nA}, B {nB}")
        col = 0
        for d, fr in colblocks:
            t = time.time()
            wd = np.flatnonzero(self.deg == d)
            X = np.zeros((len(self.ws), fr.ncols), dtype=np.int64)
            if wd.size:
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
            col += fr.ncols
            del X
        self.colblocks = colblocks
        return A, B

    def run(self, max_attempts=5):
        t0 = time.time()
        self.prepare()
        res = dict(N=self.N, group=self.group, sec=list(self.sec), Y=self.Y, maxlen=self.maxlen, m=self.m,
                   m_d={str(k): v for k, v in self.m_d.items()}, words=len(self.ws), words_up=len(self.ws_up))
        if self.m == 0:
            res.update(rank_C=0, check=True, r_up=0, time=time.time() - t0)
            return res
        factor, prev = 1, None
        for attempt in range(max_attempts):
            A, B = self.evaluate(factor, attempt)
            t = time.time()
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
        self._t('rank', t)
        res['ncols'] = int(B.shape[1])
        res['timing'] = {k: round(v, 1) for k, v in self.timing.items()}
        res['time'] = round(time.time() - t0, 1)
        return res
