"""Direct Q-cohomology per full-letter sector: dim H_s = dim V_s - rank(Q|V_s) - rank(Q|V_{s+}), s+ = (2J1+1, 2J2+1, q).
Ranks by random evaluation over F_p with adaptive point doubling (as in fortuity.bmn.symdirect)."""
import math, time
import numpy as np
from collections import defaultdict
from fortuity.bmn.modp import RREFBasis
from .letters import is_ferm
from .words import words_of_sector, contents_of_sector, Q_multi
from .evalpoints import PointsFull


class FullDirect:
    def __init__(self, N, maxlen, margin=32, extra_frame=4, seed=777, verbose=False):
        self.N, self.maxlen, self.margin, self.extra, self.seed, self.verbose = N, maxlen, margin, extra_frame, seed, verbose
        self._words, self._rankQ = {}, {}

    def words(self, sec):
        if sec not in self._words:
            self._words[sec] = words_of_sector(sec, self.maxlen)
        return self._words[sec]

    def _frame(self, sec):
        conts = contents_of_sector(sec)
        degs = sorted({sum(1 for L in c if is_ferm(L)) for c in conts})
        k = max(degs) + self.extra
        cpp = sum(math.comb(k, d) for d in degs)
        letters = {L for c in conts for L in c}
        return k, degs, cpp, letters

    def _basis_doubling(self, sec, nrows, row_fn):
        k, degs, cpp, letters = self._frame(sec)
        n = max(2, math.ceil(min(nrows // 6 + self.margin, 128) / cpp))
        prev = None
        while True:
            pts = PointsFull(self.N, n, k, degs, letters, seed=self.seed + n)
            basis = RREFBasis(pts.ncols)
            batch = []
            for v in row_fn(pts):
                batch.append(v)
                if len(batch) >= 256:
                    basis.add(np.array(batch, dtype=np.int64)); batch = []
            if batch:
                basis.add(np.array(batch, dtype=np.int64))
            r = basis.rank
            if prev is not None and r == prev and r < pts.ncols - self.margin:
                return r
            prev = r
            n *= 2

    def dimV(self, sec):
        ws = self.words(sec)
        if not ws:
            return 0
        return self._basis_doubling(sec, len(ws), lambda p: (p.expr_vector({w: 1}) for w in ws))

    def rankQ(self, sec):
        """rank of Q: V_sec -> V_{sec-}."""
        if sec in self._rankQ:
            return self._rankQ[sec]
        ws = self.words(sec)
        low = (sec[0] - 1, sec[1] - 1) + sec[2:]
        if not ws or not contents_of_sector(low):
            self._rankQ[sec] = 0
            return 0
        qs = [Q_multi(w) for w in ws]
        if not any(qs):
            self._rankQ[sec] = 0
            return 0
        def rows(p):
            for q in qs:
                yield p.expr_vector(q) if q else np.zeros(p.ncols, dtype=np.int64)
        r = self._basis_doubling(low, len(ws), rows)
        self._rankQ[sec] = r
        return r

    def dimH(self, sec):
        t = time.time()
        dV = self.dimV(sec)
        if dV == 0:
            return 0
        up = (sec[0] + 1, sec[1] + 1) + sec[2:]
        h = dV - self.rankQ(sec) - (self.rankQ(up) if contents_of_sector(up) else 0)
        if self.verbose:
            print(f"  sec {sec}: words {len(self.words(sec))} dimV {dV} h {h} ({time.time()-t:.1f}s)", flush=True)
        return h
