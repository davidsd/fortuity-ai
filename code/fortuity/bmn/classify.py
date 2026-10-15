"""Classify a given operator in one BMN weight sector (2J; q) of a group: zero, Q-exact, multi-graviton (mod exact),
or with a non-graviton component (T-028, result R-016).

TargetSector evaluates the sector at faithful random functionals (point doubling until the rank of all words is
stable, or equals a known exact dimension), builds the span of Q-exact operators (Q of every word of 2J+1) and of
graviton polynomials (fortuity.bmn.symdirect graviton rows, D-005), and tests where a vector lands.  Used to classify
the images of block restrictions (fortuity.bmn.blockrestrict).  Random evaluation over F_p can only underestimate
ranks; certify the sector by passing exact_dim (e.g. from fortuity.bmn.fullindex.partition_dims).
"""
import copy, math, time
import numpy as np

from fortuity.bmn.words import Q_multi, Q_expr, contents_of_sector
from fortuity.bmn.modp import P, RREFBasis
from fortuity.bmn.symdirect import SymDirect


class TargetSector:
    def __init__(self, N, group, sec, gen_nmax, seed=5, margin=48, log=print, maxlen=100, exact_dim=None):
        self.N, self.group, self.sec, self.log = N, group, tuple(sec), log
        self.sd = SymDirect(N, verbose=False, symmetrize=False, seed=seed, group=group, gen_nmax=gen_nmax,
                            maxlen=maxlen)
        self.sd.guard.log = False
        self.margin = margin
        self.exact_dim = exact_dim
        self._build()

    def _rows(self, pts, exprs, batch=256):
        b = RREFBasis(pts.ncols)
        buf = []
        for e in exprs:
            buf.append(pts.expr_vector(e))
            if len(buf) >= batch:
                b.add(np.array(buf, dtype=np.int64))
                buf = []
        if buf:
            b.add(np.array(buf, dtype=np.int64))
        return b

    def _build(self):
        t0 = time.time()
        sd, sec = self.sd, self.sec
        words = sd.symwords(sec)                                    # list of {multi: 1}
        up = (sec[0] + 1,) + sec[1:]
        low = (sec[0] - 1,) + sec[1:]
        upwords = sd.symwords(up) if contents_of_sector(up) else []
        k, degs, cpp = sd._sector_k(sec)
        target = self.exact_dim if self.exact_dim else len(words)
        npts = max(2, math.ceil((min(target, len(words)) + self.margin) / cpp))
        prev = None
        while True:
            pts = sd.points_n(sec, npts)
            bw = self._rows(pts, words)
            r = bw.rank
            self.log(f"  target {self.group} N={self.N} {sec}: npts={npts} ncols={pts.ncols} rank(words)={r}")
            done = r < pts.ncols - self.margin and (r == prev or (self.exact_dim is not None and r == self.exact_dim))
            if done:
                break
            prev = r
            npts *= 2
        self.pts, self.dimV = pts, r
        # Q-exact span: Q of every word of the sector above
        qexprs = []
        for e in upwords:
            (m, c), = e.items()
            q = Q_multi(m)
            if q:
                qexprs.append(q)
        self.be = self._rows(pts, qexprs)
        self.r_exact = self.be.rank
        G, nm = sd.graviton_rows(sec, pts)
        self.beg = copy.deepcopy(self.be)
        if G.size:
            self.beg.add_batched(G % P)
        self.grav = self.beg.rank - self.r_exact
        # closedness test points one step down
        if contents_of_sector(low):
            kl, degl, cppl = sd._sector_k(low)
            self.pts_low = sd.points_n(low, max(2, math.ceil((self.dimV + self.margin) / cppl)))
        else:
            self.pts_low = None
        self.log(f"  target {self.group} N={self.N} {sec}: dim={self.dimV} exact={self.r_exact} "
                 f"graviton={self.grav} ({time.time() - t0:.0f}s)")

    def classify(self, expr):
        """dict(nonzero, closed, not_exact, non_graviton_component) for an expression in this sector."""
        v = self.pts.expr_vector(expr) % P
        nonzero = bool(v.any())
        closed = None
        if self.pts_low is not None:
            closed = not bool(self.pts_low.expr_vector(Q_expr(expr)).any())
        b1 = copy.deepcopy(self.be)
        not_exact = bool(b1.add(v[None, :])[0])
        b2 = copy.deepcopy(self.beg)
        not_graviton = bool(b2.add(v[None, :])[0])
        return dict(nonzero=nonzero, closed=closed, not_exact=not_exact, non_graviton_component=not_graviton)

    def residue(self, expr):
        """The vector of expr reduced modulo exact + graviton span (zero iff no non-graviton component).  Residues of
        closed operators lie in the image of the non-graviton cohomology, so for a one-dimensional quotient any two
        of them are proportional; their ratio compares non-graviton components."""
        v = self.pts.expr_vector(expr) % P
        return self.beg.reduce(v[None, :])[0]


def verdict(c):
    if c['non_graviton_component']:
        return 'non-graviton'
    if c['not_exact']:
        return 'graviton (mod exact)'
    return 'Q-exact' if c['nonzero'] else 'zero'


def ratio(r, ref, p=P):
    """lambda with r = lambda ref (mod p), as a symmetric representative, or None if not proportional."""
    nz = np.flatnonzero(ref % p)
    if nz.size == 0:
        return None
    j = int(nz[0])
    lam = int(r[j]) * pow(int(ref[j]), p - 2, p) % p
    if np.array_equal((r - lam * ref) % p, np.zeros_like(r)):
        return lam if lam <= p // 2 else lam - p
    return None
