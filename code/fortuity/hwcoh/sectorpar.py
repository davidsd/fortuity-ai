"""Parallel, memory-lean version of hwcoh.sector.Sector (same mathematics, same self-checks).

Differences from Sector:
  * the row matrices A (hw vectors of sector n), B (Q-images of hw vectors of n+1), G (graviton hw vectors) are
    int32 memory-mapped files on node-local scratch ($TMPDIR), never fully in RAM;
  * point chunks are evaluated by a pool of forked worker processes, each writing its column slice;
  * a chunk of fermion degree d only touches the hw blocks and Q-image columns of degree d (all other entries of
    that column slice vanish), which removes most of the combine work;
  * ranks use the BLIS-based kernel (ffblas): per-degree checks on strided views of A, and one pass over [B; G]
    giving r_up = rank B and grav = rank [B; G] - rank B.
"""
import math
import os
import time
import multiprocessing as mp
import numpy as np
from fortuity.bmn.modp import P
from . import ffblas
from .sector import Sector, frame_k
from .evalg import Frame, DegFrames

_W = None      # the SectorPar instance, inherited by forked workers


def _mmap(fname, rows, cols, mode):
    """int32 memmap of shape (rows, cols); an in-memory empty array when the size is zero (mmap refuses it)."""
    if rows * cols == 0:
        return np.zeros((rows, cols), dtype=np.int32)
    return np.memmap(fname, dtype=np.int32, mode=mode, shape=(rows, cols))


def _sparse_rows(nrows, rs, xr, cs, X, w, max_entries=100_000_000):
    """Y (nrows x w, int32) with Y[r] = sum over entries e with rs[e] = r of cs[e] * X[xr[e]]  (mod P).
    rs sorted; processed in chunks of entries so that the temporary stays below max_entries elements."""
    Y = np.zeros((nrows, w), dtype=np.int64)
    step = max(1, max_entries // max(w, 1))
    for e0 in range(0, len(rs), step):
        e1 = min(len(rs), e0 + step)
        r = rs[e0:e1]
        vals = X[xr[e0:e1]].astype(np.int64)
        vals *= cs[e0:e1, None]
        vals %= P
        starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
        Y[r[starts]] += np.add.reduceat(vals, starts, axis=0)
        Y[r[starts]] %= P
    return Y.astype(np.int32)


def _worker_init(threads):
    ffblas.set_threads(threads)


def _eval_group(task):
    """Evaluate a contiguous (sorted) slice of the degree-d words at all points of degree d; write rows of X_d."""
    key, i0, i1 = task
    d = key[0]
    S = _W
    npts, seed, c0 = S.chunk_frames[key]
    fr = Frame(S.N, npts, frame_k(S.N, d, group=S.group), [d], seed=seed, group=S.group)
    X = _mmap(S.fX, len(S.order_d[d]), fr.ncols, 'r+')
    t = time.time()
    B = S.batch
    for j0 in range(i0, i1, B):
        j1 = min(i1, j0 + B)
        idx = S.order_d[d][j0:j1]
        X[j0:j1] = fr.eval_words([S.ws[i] for i in idx], dtype=np.int32)
    del X
    return time.time() - t


class SectorPar(Sector):
    def __init__(self, *a, workers=None, threads_per_worker=1, workdir=None, batch=20000, xcap=40e9, **kw):
        super().__init__(*a, **kw)
        cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
        self.workers = workers or int(os.environ.get('HWPAR_WORKERS', 0)) or max(1, cpus // threads_per_worker)
        self.tpw = threads_per_worker
        self.cpus = cpus
        self.workdir = workdir or os.environ.get('TMPDIR') or '/tmp'
        self.batch = int(os.environ.get('HWPAR_BATCH', batch))   # words per evaluation batch (memory per worker)
        self.xcap = xcap
        # hw vectors: in the parent with threaded FLINT (forked FLINT pools deadlock); without FLINT the numpy fallback is
        # single-threaded, so fork a pool over shapes instead (safe: no FLINT thread pool exists)
        # (Linux only: forking after macOS system frameworks have started threads can deadlock; seen on the laptop)
        import sys
        from . import ffla
        self.hw_workers = 1 if (ffla.HAVE_FLINT or sys.platform == 'darwin') else self.workers
        self.q_workers = self.workers
        self.sparse_q = True        # Sector.prepare keeps Q sparse (no dense Q or H Q blocks)

    def prepare(self):
        super().prepare()
        deg = self.deg
        self.pos = np.full(len(self.ws), -1, dtype=np.int64)
        self.words_d = {}
        for d in np.unique(deg):
            wd = np.flatnonzero(deg == d)
            self.words_d[int(d)] = wd
            self.pos[wd] = np.arange(len(wd))
        # hw rows grouped by degree (all words of a shape share the fermion number)
        self.hw_d, row, self.rows_d = {}, 0, {}
        for d in sorted(self.words_d):
            r0 = row
            for widx, H in self.hw:
                if int(deg[widx[0]]) == d:
                    self.hw_d.setdefault(d, []).append((row, widx, H))
                    row += H.shape[0]
            self.rows_d[d] = (r0, row)
        self.nA = row
        # Q-image blocks (sparse Q, see Sector.prepare) split by target degree: (row0, H, r, target, coeff), sorted by r
        self.q_d, row = {}, 0
        for H, rs, js, cs in self.qsparse:
            td = deg[js]
            for d in np.unique(td):
                m = td == d
                o = np.argsort(rs[m], kind='stable')
                self.q_d.setdefault(int(d), []).append((row, H, rs[m][o], js[m][o], cs[m][o]))
            row += H.shape[0]
        self.nB = row
        self.qsparse = None
        self.gblocks = self.gblocks if self.GA is not None else []
        self.nG = sum(H.shape[0] for _, H in self.gblocks)

    def evaluate(self, factor=1, attempt=0):
        """Degree by degree, column chunk by column chunk: evaluate all words of degree d at the chunk's points
        (parallel over word slices) into a node-local memmap X, combine into A, B, G, delete X.  The size of X is
        capped at xcap bytes by the chunking."""
        global _W
        self.cols_d, self.chunk_frames, col = {}, {}, 0
        self.order_d = {d: np.array(sorted(self.words_d.get(d, []), key=lambda i: self.ws[i]), dtype=np.int64)
                        for d in self.m_d}
        for d, md in sorted(self.m_d.items()):
            need = factor * (md + self.margin)
            cpp = math.comb(frame_k(self.N, d, group=self.group), d)
            if cpp == 0:
                # d > 3 dim g: words of this fermion degree vanish identically (exact multiplicity 0)
                assert md == 0, (d, md)
                self.cols_d[d] = (col, col)
                continue
            npts = max(1, math.ceil(need / cpp))
            nd = max(len(self.order_d[d]), 1)
            per = max(1, int(self.xcap // (4 * nd * cpp)))
            c0 = col
            for ci, p0 in enumerate(range(0, npts, per)):
                np_ = min(per, npts - p0)
                self.chunk_frames[(d, ci)] = (np_, self.seed + 7919 * d + 104729 * attempt + 1000003 * ci, col)
                col += np_ * cpp
            self.cols_d[d] = (c0, col)
        self.ncols = col
        tag = f"{os.getpid()}_{self.n}_{attempt}"
        self.fA, self.fB, self.fG = (os.path.join(self.workdir, f"{x}_{tag}.i32") for x in "ABG")
        self.fX = os.path.join(self.workdir, f"X_{tag}.i32")
        rowpos = np.full(len(self.ws), -1, dtype=np.int64)
        for d, od in self.order_d.items():
            rowpos[od] = np.arange(len(od))
        for f, n in ((self.fA, self.nA), (self.fB, self.nB), (self.fG, self.nG)):
            with open(f, 'wb') as fh:
                fh.truncate(4 * n * self.ncols)
        self.log(f"  functionals: {self.ncols} in {len(self.chunk_frames)} column chunks; rows A {self.nA}, "
                 f"B {self.nB}, G {self.nG}; {self.workers} workers; memmaps in {self.workdir}")
        A = _mmap(self.fA, self.nA, self.ncols, 'r+')
        B = _mmap(self.fB, self.nB, self.ncols, 'r+') if self.nB else None
        G = _mmap(self.fG, self.nG, self.ncols, 'r+') if self.nG else None
        for key in sorted(self.chunk_frames):
            d = key[0]
            od = self.order_d[d]
            npts, seed, c0 = self.chunk_frames[key]
            cpp = math.comb(frame_k(self.N, d, group=self.group), d)
            w = npts * cpp
            if len(od):
                with open(self.fX, 'wb') as fh:
                    fh.truncate(4 * len(od) * w)
                ng = min(len(od), self.workers)
                tasks = [(key, g * len(od) // ng, (g + 1) * len(od) // ng) for g in range(ng)]
                _W = self
                t = time.time()
                # ProcessPoolExecutor raises BrokenProcessPool if a worker dies (e.g. OOM) instead of hanging
                from concurrent.futures import ProcessPoolExecutor
                with ProcessPoolExecutor(self.workers, mp_context=mp.get_context('fork'), initializer=_worker_init,
                                         initargs=(1,)) as pool:
                    self.timing['eval_cpu'] = self.timing.get('eval_cpu', 0) + sum(pool.map(_eval_group, tasks))
                _W = None
                self._t('eval', t)
                t = time.time()
                ffblas.set_threads(self.cpus)
                X = _mmap(self.fX, len(od), w, 'r')
                for row0, widx, H in self.hw_d.get(d, []):
                    ffblas.matmul_into32(H, np.ascontiguousarray(X[rowpos[widx]]), A[row0:row0 + H.shape[0], c0:c0 + w])
                if B is not None:
                    for row0, H, rs, js, cs in self.q_d.get(d, []):
                        Y = _sparse_rows(H.shape[1], rs, rowpos[js], cs, X, w)       # Q X  (shape words x w)
                        ffblas.matmul_into32(H, Y, B[row0:row0 + H.shape[0], c0:c0 + w])
                del X
                os.remove(self.fX)
                self._t('combine', t)
            if G is not None:
                fr = Frame(self.N, npts, frame_k(self.N, d, group=self.group), [d], seed=seed, group=self.group)
                df = DegFrames.__new__(DegFrames)
                df.frames, df.off, df.ncols = {d: fr}, {d: 0}, fr.ncols
                G[:, c0:c0 + w] = self.GA.rows_from_blocks(self.gblocks, df).astype(np.int32)
        del A, B, G

    def _cleanup(self):
        for f in (self.fA, self.fB, self.fG, getattr(self, 'fX', '')):
            try:
                os.remove(f)
            except OSError:
                pass

    def run(self, max_attempts=5):
        t0 = time.time()
        self.prepare()
        res = dict(N=self.N, lam=list(self.lam), n=self.n, maxlen=self.maxlen, maxlen_up=self.maxlen_up, m=self.m,
                   m_d={str(k): v for k, v in self.m_d.items()}, words=len(self.ws), words_up=len(self.ws_up),
                   engine='sectorpar', group=self.group)
        if self.m == 0:
            res.update(rank_C=0, check=True, r_up=0, grav=0, time=time.time() - t0)
            return res
        ffblas.set_threads(self.cpus)
        factor, prev = 1, None
        for attempt in range(max_attempts):
            self.evaluate(factor, attempt)
            ffblas.set_threads(self.cpus)
            t = time.time()
            A = _mmap(self.fA, self.nA, self.ncols, 'r')
            ok, rk = True, {}
            for d in sorted(self.m_d):
                r0, r1 = self.rows_d.get(d, (0, 0))
                c0, c1 = self.cols_d[d]
                rk[d] = ffblas.rank_pair32(A[r0:r1, c0:c1])[0] if r1 > r0 else 0
                ok &= (rk[d] == self.m_d[d])
            del A
            self._t('rank', t)
            self.log(f"  check (attempt {attempt}): rank per degree {rk} vs m_d {self.m_d} -> {'OK' if ok else 'FAIL'}")
            if ok:
                break
            self._cleanup()
            if prev == rk:
                self.log("  rank did not grow with more functionals: the word set does not span (maxlen too small?)")
                res['failure'] = 'spanning'
                break
            prev = rk
            factor *= 2
        res['rank_C_d'] = {str(k): v for k, v in rk.items()}
        res['check'] = bool(ok)
        if ok:
            # free the large combine-stage structures before the final rank (only hw_up/ws_up are handed on)
            for attr in ('qblocks', 'qsparse', 'q_d', 'hw_d', 'gblocks', 'order_d', 'pos', 'words_d'):
                if hasattr(self, attr):
                    setattr(self, attr, None)
            import gc
            gc.collect()
            t = time.time()
            B = _mmap(self.fB, self.nB, self.ncols, 'r')
            G = _mmap(self.fG, self.nG, self.ncols, 'r')
            rB, rBG = ffblas.rank_pair32(B, G)
            res['r_up'], res['grav'] = rB, rBG - rB
            res['grav_rows'] = int(self.nG)
            del B, G
            self._t('rank', t)
            self._cleanup()
        res['ncols'] = int(self.ncols)
        res['timing'] = {k: round(v, 1) for k, v in self.timing.items()}
        res['time'] = round(time.time() - t0, 1)
        return res
