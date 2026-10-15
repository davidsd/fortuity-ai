"""Q-cohomology of the SU(3)_R highest-weight subcomplex at fixed weight lam = (q1, q2, q3), q1 >= q2 >= q3.

Complex: C_n = hw_lam(V_(n; lam)), n = 2J, Q: C_n -> C_{n-1}.  m_n = dim C_n is known exactly from the
partition function (Weyl formula).  r_n = rank(Q|C_n) is computed from spanning hw vectors of the free word space,
evaluated with ~ m_{n-1} + margin random functionals of sector n-1.  h_n = m_n - r_n - r_{n+1} is the multiplicity
of the SU(3)_R irrep with highest weight lam in the cohomology at 2J = n.
Self-check per sector: rank(hw vectors evaluated at the sector's functionals) must equal m_n.
"""
import math
import time
import numpy as np
from fortuity.bmn.words import contents_of_sector, Q_multi
from fortuity.bmn.modp import RREFBasis, matmul_mod, P
from . import wordenum as enum, ffla
from .hw import hw_vectors
from .evalg import Frame, DegFrames, nferm, algebra_dim

SHIFTS = [((0, 0, 0), 1), ((1, -1, 0), -1), ((0, 1, -1), -1), ((2, -1, -1), 1), ((1, 1, -2), 1), ((2, 0, -2), -1)]


def dimV(dims, n, q):
    if min(q) < 0:
        return 0
    return dims.get((n,) + tuple(sorted(q, reverse=True)), 0)


def hw_mult(dims, n, lam):
    return sum(s * dimV(dims, n, tuple(a + b for a, b in zip(lam, sh))) for sh, s in SHIFTS)


def sector_degrees(sec):
    return sorted({sum(c[3:6]) for c in contents_of_sector(sec)})


def make_frame(N, sec, nfunc, extra=4, seed=0, group='SU'):
    degs = sector_degrees(sec)
    k = min(max(degs) + extra, 3 * algebra_dim(group, N))
    cpp = sum(math.comb(k, d) for d in degs)
    npts = max(1, math.ceil(nfunc / cpp))
    return Frame(N, npts, k, degs, seed=seed, group=group)


def stack_rows(rows_iter, ncols):
    Rs = [R for R in rows_iter if R.shape[0]]
    return np.concatenate(Rs, axis=0) if Rs else np.zeros((0, ncols), dtype=np.int64)


class Column:
    def __init__(self, N, lam, dims, margin=32, extra=2, target=24, seed=1, verbose=True, log=print, group='SU'):
        self.N, self.lam, self.dims = N, tuple(lam), dims
        self.group = group
        self.target = target
        self.margin, self.extra, self.seed, self.verbose, self.log = margin, extra, seed, verbose, log
        self.ns = [n for n in range(0, 3 * sum(lam) + 1) if contents_of_sector((n,) + self.lam)]
        self.m = {n: hw_mult(dims, n, self.lam) for n in self.ns}
        self._words, self._index, self._hw = {}, {}, {}

    def words(self, n):
        if n not in self._words:
            t = time.time()
            ws = enum.words_of_sector((n,) + self.lam) if n in self.ns else []
            self._words[n] = ws
            self._index[n] = {w: i for i, w in enumerate(ws)}
            if self.verbose:
                self.log(f"  words(2J={n}) = {len(ws)} ({time.time()-t:.1f}s)")
        return self._words[n]

    def hw(self, n):
        if n not in self._hw:
            t = time.time()
            blocks = hw_vectors(self.words(n))
            self._hw[n] = blocks
            if self.verbose:
                self.log(f"  hw(2J={n}): {sum(H.shape[0] for _, H in blocks)} vectors in {len(blocks)} shapes "
                         f"(m = {self.m.get(n, 0)}) ({time.time()-t:.1f}s)")
        return self._hw[n]

    def q_rows(self, n, X):
        """Rows H_{n} Q (words of 2J=n) evaluated with X = evaluations of the words of 2J=n-1."""
        idx = self._index[n - 1]
        for widx, H in self.hw(n):
            tg = {}
            ent = []
            for r, wi in enumerate(widx):
                for m2, c in Q_multi(self._words[n][wi]).items():
                    j = tg.setdefault(idx[m2], len(tg))
                    ent.append((r, j, c))
            if not tg:
                continue
            Qm = np.zeros((len(widx), len(tg)), dtype=np.int64)
            for r, j, c in ent:
                Qm[r, j] += c
            C = matmul_mod(H, Qm % P)
            yield matmul_mod(C, X[np.fromiter(tg.keys(), dtype=np.int64)])

    def sector(self, n, grav_rows=None):
        """Work at functionals of 2J = n: check rank of C_n, compute r_{n+1} = rank Q|C_{n+1},
        and optionally gravitons modulo Q-exact.  Returns dict."""
        t0 = time.time()
        sec = (n,) + self.lam
        mn = self.m[n]
        res = dict(n=n, m=mn)
        if mn == 0:
            res.update(check=True, r_up=0)
            return res
        ws = self.words(n)
        nd = {}
        for w in ws:
            d = nferm(w)
            nd[d] = nd.get(d, 0) + 1
        factor = 1
        for attempt in range(6):
            need = {d: factor * (min(mn, c) + self.margin) for d, c in nd.items()}
            fr = DegFrames(self.N, need, self.extra, target=self.target, seed=self.seed + 1000 * n + 17 * attempt, group=self.group)
            t = time.time()
            X = fr.eval_words(ws)
            if self.verbose:
                self.log(f"  eval 2J={n}: {len(ws)} words x {fr.ncols} functionals [{fr.describe()}] "
                         f"({time.time()-t:.1f}s)")
            rC = ffla.rank(stack_rows((matmul_mod(H, X[widx]) for widx, H in self.hw(n)), fr.ncols))
            if rC == mn:
                break
            self.log(f"  2J={n}: rank of hw vectors {rC} < m {mn}; doubling the functionals")
            factor *= 2
        res['rank_C'] = rC
        res['check'] = (rC == mn)
        res['ncols'] = fr.ncols
        if not res['check']:
            self.log(f"  WARNING 2J={n}: rank of hw vectors {rC} != m {mn} after retries")
        up = n + 1
        if up in self.ns and self.m.get(up, 0) > 0:
            self.words(up)
            QR = stack_rows(self.q_rows(up, X), fr.ncols)
        else:
            QR = np.zeros((0, fr.ncols), dtype=np.int64)
        res['r_up'] = ffla.rank(QR)
        if grav_rows is not None:
            G = grav_rows(n, fr)
            res['n_grav_rows'] = int(G.shape[0])
            res['grav'] = (ffla.rank(np.concatenate([QR, G], axis=0)) - res['r_up']) if G.shape[0] else 0
        res['time'] = time.time() - t0
        if self.verbose:
            self.log(f"SECTOR 2J={n} lam={self.lam}: m={mn} rankC={res['rank_C']} r_up={res['r_up']}"
                     + (f" grav={res['grav']}" if 'grav' in res else "") + f" ({res['time']:.0f}s)")
        return res

    def run(self, grav_rows=None, ns=None):
        out = {}
        for n in (ns or self.ns):
            out[n] = self.sector(n, grav_rows)
        return self.assemble(out)

    def assemble(self, out):
        rows = []
        for n in sorted(out):
            if (n - 1) in out:
                r_n = out[n - 1]['r_up']
            elif (n - 1) not in self.ns or self.m.get(n - 1, 0) == 0:
                r_n = 0          # Q maps C_n to an empty (or hw-empty) sector
            else:
                r_n = None
            r_up = out[n]['r_up']
            h = out[n]['m'] - r_up - r_n if r_n is not None else None
            row = dict(out[n], r=r_n, h=h)
            if h is not None and 'grav' in out[n]:
                row['BH'] = h - out[n]['grav']
            rows.append(row)
        return rows
