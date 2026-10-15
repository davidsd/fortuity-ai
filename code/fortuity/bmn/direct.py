"""Direct Q-cohomology in the BMN sector (Chang-Lin style), sector by sector.

For a sector s = (2J, q1, q2, q3):
  dim V_s          = rank of evaluations of all multi-trace words in s
  r_s = rank(Q|V_s) (image in s- = (2J-1, q))
  dim H_s          = dim V_s - r_s - r_{s+}
  gravitons in H_s = rank([multi-graviton products in s ; Q V_{s+}]) - r_{s+}
"""
import math
import time
import numpy as np
from collections import defaultdict
from .words import (words_of_sector, contents_of_sector, Q_multi, graviton_generators, expr_sector,
                    content, sector_j)
from .evaluate import Points
from .modp import RREFBasis, rank_mod, P
from .grassmann import gmul_scalar


class Direct:
    def __init__(self, N, maxlen, gen_nmax=None, seed=12345, margin=16, verbose=True):
        self.N = N
        self.maxlen = maxlen
        self.gen_nmax = gen_nmax if gen_nmax is not None else N
        self.seed = seed
        self.margin = margin
        self.verbose = verbose
        self.gens = graviton_generators(self.gen_nmax, 1)
        self.gen_sec = [expr_sector(e) for _, e in self.gens]
        self.gen_par = [sum(content(next(iter(e)))[3:6]) % 2 for _, e in self.gens]
        self._words = {}
        self._points = {}
        self._rankQ = {}
        self._Qimg_rows = {}  # sector s -> evaluation matrix of Q(words of s) at points(s-)
        self._gen_val = {}

    # ----- words and points -----
    def words(self, sec):
        if sec not in self._words:
            self._words[sec] = words_of_sector(sec, self.maxlen)
        return self._words[sec]

    def points(self, sec, need_rows=None):
        conts = contents_of_sector(sec)
        if not conts:
            return None
        degs = sorted({sum(c[3:6]) for c in conts})
        k = max(degs)
        if sec not in self._points:
            nw = len(self.words(sec))
            cpp = sum(math.comb(k, d) for d in degs)
            npts = max(2, math.ceil((nw + self.margin) / cpp))
            h = hash(sec) % (2**31)
            self._points[sec] = Points(self.N, npts, k, degs, seed=self.seed + h)
        return self._points[sec]

    # ----- ranks -----
    def rankQ(self, sec):
        """rank of Q: V_sec -> V_{sec-}; also caches the evaluated image rows."""
        if sec in self._rankQ:
            return self._rankQ[sec]
        ws = self.words(sec)
        if not ws or sec[0] == 0:
            self._rankQ[sec] = 0
            self._Qimg_rows[sec] = None
            return 0
        tgt = (sec[0] - 1,) + sec[1:]
        pts = self.points(tgt)
        if pts is None:
            self._rankQ[sec] = 0
            self._Qimg_rows[sec] = None
            return 0
        rows = []
        for m in ws:
            q = Q_multi(m)
            rows.append(pts.expr_vector(q) if q else np.zeros(pts.ncols, dtype=np.int64))
        M = np.array(rows, dtype=np.int64)
        self._Qimg_rows[sec] = M
        r = rank_mod(M)
        self._rankQ[sec] = r
        return r

    def dimV(self, sec):
        ws = self.words(sec)
        if not ws:
            return 0
        pts = self.points(sec)
        M = pts.matrix([{m: 1} for m in ws])
        return rank_mod(M)

    # ----- gravitons -----
    def gen_value(self, gi, sec):
        key = (gi, sec)
        if key not in self._gen_val:
            pts = self.points(sec)
            acc = {}
            from .grassmann import gadd
            for multi, c in self.gens[gi][1].items():
                acc = gadd(acc, pts.multitrace(multi), pts.p, coeff=c % pts.p)
            self._gen_val[key] = acc
        return self._gen_val[key]

    def graviton_monomials(self, sec):
        """Monomials (tuples of generator indices, nondecreasing) with total sector = sec."""
        target = np.array(sec)
        out = []
        ng = len(self.gens)
        gs = [np.array(s) for s in self.gen_sec]

        def rec(start, remaining, chosen):
            if not remaining.any():
                out.append(tuple(chosen))
                return
            for i in range(start, ng):
                if np.all(gs[i] <= remaining):
                    nxt = i + 1 if self.gen_par[i] else i
                    rec(nxt, remaining - gs[i], chosen + [i])

        rec(0, target, [])
        return out

    def graviton_matrix(self, sec):
        pts = self.points(sec)
        monos = self.graviton_monomials(sec)
        rows = []
        cache = {}
        for mono in monos:
            # incremental products with prefix caching
            val = None
            for t in range(len(mono)):
                pref = mono[:t + 1]
                if pref in cache:
                    val = cache[pref]
                    continue
                g = self.gen_value(mono[t], sec)
                val = g if val is None else gmul_scalar(val, g, pts.k, pts.p)
                cache[pref] = val
            rows.append(pts.flatten(val))
        if not monos:
            return np.zeros((0, pts.ncols), dtype=np.int64), 0
        return np.array(rows, dtype=np.int64).reshape(len(monos), -1), len(monos)

    # ----- full sector analysis -----
    def analyse(self, sec):
        t0 = time.time()
        ws = self.words(sec)
        res = dict(sector=sec, j=sector_j(sec), nwords=len(ws))
        if not ws:
            res.update(dimV=0, rQ=0, rQup=0, coh=0, grav=0, nongrav=0, nmono=0)
            return res
        dV = self.dimV(sec)
        rQ = self.rankQ(sec)
        up = (sec[0] + 1,) + sec[1:]
        rQup = self.rankQ(up) if contents_of_sector(up) else 0
        coh = dV - rQ - rQup
        # gravitons
        G, nmono = self.graviton_matrix(sec)
        if rQup > 0:
            M = np.vstack([G, self._Qimg_rows[up]])
        else:
            M = G
        grav = (rank_mod(M) if M.size else 0) - rQup
        res.update(dimV=dV, rQ=rQ, rQup=rQup, coh=coh, grav=grav, nongrav=coh - grav, nmono=nmono,
                   time=time.time() - t0)
        if self.verbose:
            print(f"sec={sec} j={res['j']:2d} words={len(ws):6d} dimV={dV:5d} closed={dV-rQ:5d} "
                  f"exact={rQup:5d} coh={coh:4d} grav={grav:4d} BH={coh-grav:3d}  ({res['time']:.1f}s)",
                  flush=True)
        return res


def sectors_up_to(jmax, sorted_q=True):
    """All sectors (2J, q1, q2, q3) with j = 2(q1+q2+q3) <= jmax, q1<=q2<=q3 if sorted_q."""
    out = []
    for qsum in range(1, jmax // 2 + 1):
        for q1 in range(qsum + 1):
            for q2 in range(q1 if sorted_q else 0, qsum - q1 + 1):
                q3 = qsum - q1 - q2
                if sorted_q and q3 < q2:
                    continue
                for twoJ in range(0, qsum + 1):
                    sec = (twoJ, q1, q2, q3)
                    if contents_of_sector(sec):
                        out.append(sec)
    return out
