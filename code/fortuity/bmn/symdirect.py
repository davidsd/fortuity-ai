"""Direct cohomology restricted to the Weyl(S_3)-invariant subspace of sectors with q1=q2=q3.

The SU(3) Weyl group acts on letters by permuting flavours; for odd permutations the SU(3) Weyl
element also flips a sign, which on a word amounts to (-1)^{n_phi + n_psi}.  SU(3) singlets are
Weyl-invariant, so the cohomology of the invariant subcomplex contains all singlet cohomologies
(this is the 'permutation property' of Choi-Choi-Kim-Lee-Lee, Table 2).
"""
import math
import time
import numpy as np
from itertools import permutations
from collections import defaultdict
from .words import (words_of_sector, contents_of_sector, Q_multi, canon_word, canon_multi, add_term,
                    graviton_generators, expr_sector, content, sector_j, is_ferm, PHI, PSI, F)
from .evaluate import Points
from .modp import rank_mod, RREFBasis
import copy
from .grassmann import gmul_scalar, gadd
from .guard import MemGuard

PERMS = list(permutations(range(3)))


def perm_sign(s):
    return -1 if sum(1 for i in range(3) for j in range(i + 1, 3) if s[i] > s[j]) % 2 else 1


def relabel_letter(l, s):
    if l in PHI:
        return PHI[s[l]]
    if l in PSI:
        return PSI[s[l - 3]]
    return l


def weyl_multi(multi, s):
    """Weyl action on a multitrace: returns (sign, canonical multitrace) or (0, None)."""
    sgn = perm_sign(s)
    nlet = sum(1 for w in multi for l in w if l != F)
    sign = -1 if (sgn == -1 and nlet % 2 == 1) else 1
    new = []
    for w in multi:
        sw, cw = canon_word(tuple(relabel_letter(l, s) for l in w))
        if sw == 0:
            return 0, None
        sign *= sw
        new.append(cw)
    sm, cm = canon_multi(new)
    if sm == 0:
        return 0, None
    return sign * sm, cm


def symmetrize_multi(multi):
    out = defaultdict(int)
    for s in PERMS:
        sg, m = weyl_multi(multi, s)
        if sg:
            out[m] += sg
    return {m: c for m, c in out.items() if c}


def sym_words(sec):
    """Orbit representatives of words under the Weyl group and their symmetrized expressions."""
    ws = words_of_sector(sec, 100)
    seen = set()
    out = []
    for w in ws:
        if w in seen:
            continue
        orbit = set()
        for s in PERMS:
            sg, m = weyl_multi(w, s)
            if sg:
                orbit.add(m)
        seen |= orbit
        e = symmetrize_multi(w)
        if e:
            out.append(e)
    return out


def symmetrize_expr(expr):
    out = defaultdict(int)
    for m, c in expr.items():
        for m2, c2 in symmetrize_multi(m).items():
            out[m2] += c * c2
    return {m: c for m, c in out.items() if c}


class SymDirect:
    def __init__(self, N, gen_nmax=None, seed=777, margin=32, verbose=True, symmetrize=True, maxlen=100, group='SU', gen_qmax=None):
        """group: 'SU' (SU(N)), 'SO_odd' (SO(2N+1)) or 'Sp' (USp(2N)); see evaluate.Points."""
        self.N = N
        self.group = group
        self.symmetrize = symmetrize
        self.maxlen = maxlen
        if gen_nmax is None:                     # top Casimir degree of the group (review B.3; was N for every group)
            from .gravindex import default_nmax
            gen_nmax = default_nmax(group, N)
        self.gen_nmax = gen_nmax
        self.seed, self.margin, self.verbose = seed, margin, verbose
        self.extra_frame = 4
        self.dimV = {}      # optional exact sector dimensions from fullindex.partition_dims
        self.gens = graviton_generators(self.gen_nmax, 1, qmax=gen_qmax)
        self.gen_sec = [expr_sector(e) for _, e in self.gens]
        self.gen_par = [sum(content(next(iter(e)))[3:6]) % 2 for _, e in self.gens]
        self._sym = {}
        self._points = {}
        self._rankQ = {}
        self._Qrows = {}
        import os
        self.guard = MemGuard(float(os.environ.get('MEMCAP_GB', '10')), log=True)

    def symwords(self, sec):
        if sec not in self._sym:
            t = time.time()
            if self.symmetrize:
                self._sym[sec] = sym_words(sec)
            else:
                self._sym[sec] = [{w: 1} for w in words_of_sector(sec, self.maxlen)]
            if self.verbose:
                print(f"  symwords{sec}: {len(self._sym[sec])} ({time.time()-t:.1f}s)", flush=True)
        return self._sym[sec]

    def _sector_k(self, sec):
        conts = contents_of_sector(sec)
        degs = sorted({sum(c[3:6]) for c in conts})
        # frame larger than the max fermion degree: more independent functionals per point,
        # hence far fewer points (and a much smaller necklace cache)
        dimg = {'SU': self.N * self.N - 1, 'SO_odd': self.N * (2 * self.N + 1), 'Sp': self.N * (2 * self.N + 1), 'G2': 14}[self.group]
        k = min(max(degs) + self.extra_frame, 3 * dimg)
        cpp = sum(math.comb(k, d) for d in degs)
        return k, degs, cpp

    def points_n(self, sec, npts):
        """Points object for sector sec with npts points; older point sets of the same sector are dropped."""
        key = (sec, npts)
        if key not in self._points:
            for k2 in list(self._points):
                if k2[0] == sec:
                    del self._points[k2]
            k, degs, cpp = self._sector_k(sec)
            self._points[key] = Points(self.N, npts, k, degs, seed=self.seed + hash(sec) % 1000 + npts, group=self.group)
        return self._points[key]

    def _basis_doubling(self, sec, nrows, row_fn, tag, bound=None):
        """Build an RREF basis of the rows produced by row_fn(pts) (an iterable of vectors) using
        adaptively many points: start small (or from a known upper bound on the rank) and double until
        the rank is stable and unsaturated."""
        k, degs, cpp = self._sector_k(sec)
        if bound is not None:
            n = max(2, math.ceil((bound + self.margin) / cpp))
        else:
            n = max(2, math.ceil(min(nrows // 6 + self.margin, 128) / cpp))
        prev = None
        while True:
            pts = self.points_n(sec, n)
            basis = RREFBasis(pts.ncols)
            batch = []
            for v in row_fn(pts):
                batch.append(v)
                if len(batch) >= 256:
                    basis.add(np.array(batch, dtype=np.int64))
                    batch = []
            if batch:
                basis.add(np.array(batch, dtype=np.int64))
            r = basis.rank
            self.guard.check(f'{tag}{sec} n={n} rank={r}')
            if prev is not None and r == prev and r < pts.ncols - self.margin:
                return basis, pts
            prev = r
            n *= 2

    def rankQ(self, sec):
        """rank of Q on the symmetric subspace of sec (image in sec-); stores the reduced basis and points."""
        if sec in self._rankQ:
            return self._rankQ[sec]
        sw = self.symwords(sec)
        low = (sec[0] - 1,) + sec[1:]
        if not sw or sec[0] == 0 or not contents_of_sector(low):
            self._rankQ[sec] = 0
            self._Qrows[sec] = None
            return 0
        self.symwords(low)
        t = time.time()
        qexprs = []
        for e in sw:
            q = defaultdict(int)
            for m, c in e.items():
                for m2, c2 in Q_multi(m).items():
                    q[m2] += c * c2
            qexprs.append({m: c for m, c in q.items() if c})

        def rows(pts):
            for q in qexprs:
                yield pts.expr_vector(q) if q else np.zeros(pts.ncols, dtype=np.int64)

        bound = None
        if sec in self.dimV and low in self.dimV:
            bound = min(self.dimV[sec], self.dimV[low])
        basis, pts = self._basis_doubling(low, len(sw), rows, 'Q', bound=bound)
        r = basis.rank
        self._rankQ[sec] = r
        self._Qrows[sec] = (basis, pts)
        if self.verbose:
            print(f"  rankQ{sec} = {r}  (rows {len(sw)}, cols {pts.ncols}, {time.time()-t:.1f}s)", flush=True)
        return r

    def graviton_rows(self, sec, pts):
        target = np.array(sec)
        gs = [np.array(s) for s in self.gen_sec]
        monos = []

        def rec(start, remaining, chosen):
            if not remaining.any():
                monos.append(tuple(chosen))
                return
            for i in range(start, len(self.gens)):
                if np.all(gs[i] <= remaining):
                    nxt = i + 1 if self.gen_par[i] else i
                    rec(nxt, remaining - gs[i], chosen + [i])

        rec(0, target, [])
        # generator values at pts
        gval = {}
        rows = []
        # symmetrize each monomial at the level of expressions is costly; instead symmetrize by
        # applying the Weyl action to generator names.
        name_index = {name: i for i, (name, _) in enumerate(self.gens)}

        def weyl_gen(gi, s):
            name = self.gens[gi][0]
            sgn = perm_sign(s)
            if name[0] == 'u':
                _, n, idx = name
                new = ('u', n, tuple(sorted(s[i] for i in idx)))
                sign = sgn ** n
            elif name[0] == 'v':
                _, n, idx, j = name
                new = ('v', n, tuple(sorted(s[i] for i in idx)), s[j])
                sign = sgn ** n
            else:
                _, n, idx = name
                new = ('w', n, tuple(sorted(s[i] for i in idx)))
                sign = sgn ** (n - 1)
            return sign, name_index[new]

        def value(gi):
            if gi not in gval:
                acc = {}
                for m, c in self.gens[gi][1].items():
                    acc = gadd(acc, pts.multitrace(m), pts.p, coeff=c % pts.p)
                gval[gi] = acc
            return gval[gi]

        perms = PERMS if self.symmetrize else [tuple(range(3))]
        for mono in monos:
            acc = {}
            for s in perms:
                sign = 1
                imgs = []
                for gi in mono:
                    sg, g2 = weyl_gen(gi, s)
                    sign *= sg
                    imgs.append(g2)
                # reorder fermionic generators into sorted order with sign
                odd = [g for g in imgs if self.gen_par[g]]
                if len(set(odd)) < len(odd):
                    continue
                # sign of sorting the odd ones
                order = sorted(range(len(odd)), key=lambda i: odd[i])
                inv = sum(1 for a in range(len(order)) for b in range(a + 1, len(order)) if order[a] > order[b])
                if inv % 2:
                    sign = -sign
                val = None
                for g in sorted(imgs):
                    v = value(g)
                    val = v if val is None else gmul_scalar(val, v, pts.k, pts.p)
                acc = gadd(acc, val, pts.p, coeff=sign % pts.p)
            rows.append(pts.flatten(acc))
        if not monos:
            return np.zeros((0, pts.ncols), dtype=np.int64), 0
        return np.array(rows, dtype=np.int64), len(monos)

    def analyse(self, sec):
        t0 = time.time()
        sw = self.symwords(sec)
        res = dict(sector=sec, j=sector_j(sec), nsym=len(sw))
        if not sw:
            res.update(basis=0, closed=0, exact=0, coh=0, grav=0, BH=0)
            return res
        if sec in self.dimV and not self.symmetrize:
            basis = self.dimV[sec]
            pts = None
            if self.verbose:
                print(f"  dim V{sec} = {basis} (from partition function)", flush=True)
        else:
            eb, pts = self._basis_doubling(sec, len(sw), lambda p: (p.expr_vector(e) for e in sw), 'E')
            basis = eb.rank
            del eb
        rQ = self.rankQ(sec)
        up = (sec[0] + 1,) + sec[1:]
        rQup = self.rankQ(up) if contents_of_sector(up) else 0
        closed = basis - rQ
        coh = closed - rQup
        if rQup > 0:
            qb, qpts = self._Qrows[up]
            gb = copy.deepcopy(qb)
        else:
            if pts is None:
                pts = self.points_n(sec, max(2, math.ceil((basis + self.margin) / self._sector_k(sec)[2])))
            qpts = pts
            gb = RREFBasis(qpts.ncols)
        G, nm = self.graviton_rows(sec, qpts)
        grav = int(gb.add_batched(G).sum()) if G.size else 0
        del gb
        self._Qrows.pop(up, None)
        for key in list(self._points):
            if key[0][0] < sec[0]:
                del self._points[key]
        self.guard.check(f'end{sec}')
        res.update(basis=basis, closed=closed, exact=rQup, coh=coh, grav=grav, BH=coh - grav, nmono=nm,
                   time=time.time() - t0)
        if self.verbose:
            print(f"SEC {sec} j={res['j']} R={(sec[1]-sec[0]/2)} J={sec[0]/2}: #sym={len(sw)} basis={basis} "
                  f"closed={closed} exact={rQup} coh={coh} grav={grav} BH={coh-grav} ({res['time']:.0f}s)", flush=True)
        return res


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Direct BMN cohomology counts in given sectors (2J,q1,q2,q3), as JSON")
    ap.add_argument("group", choices=["SU", "SO_odd", "Sp", "G2"])
    ap.add_argument("N", type=int, help="SU(N), SO(2N+1) or USp(2N)")
    ap.add_argument("out")
    ap.add_argument("--sector", action="append", required=True, help="e.g. 1,3,3,3 (repeatable)")
    ap.add_argument("--gen-nmax", type=int, default=None)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--symmetrize", action="store_true", help="restrict to the S_3 Weyl-invariant subspace")
    a = ap.parse_args()
    secs = [tuple(int(x) for x in s.split(",")) for s in a.sector]
    qmax = tuple(max(s[i] for s in secs) for i in range(4))
    if a.symmetrize:
        qmax = (qmax[0],) + (max(qmax[1:]),) * 3   # Weyl images of generators must also be available
    sd = SymDirect(a.N, gen_nmax=a.gen_nmax, seed=a.seed, symmetrize=a.symmetrize, group=a.group, gen_qmax=qmax)
    res = []
    for s in a.sector:
        r = sd.analyse(tuple(int(x) for x in s.split(",")))
        res.append({k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()})
    with open(a.out, "w") as fh:
        json.dump({"group": a.group, "N": a.N, "seed": a.seed, "gen_nmax": sd.gen_nmax,
                   "symmetrize": a.symmetrize, "results": res}, fh, indent=1)
